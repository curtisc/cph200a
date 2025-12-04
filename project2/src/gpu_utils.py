"""
GPU utilities for detecting NVLink and configuring distributed training.
"""

import torch
import subprocess
import re
from typing import Tuple, List, Optional


# Cache for NVLink detection to avoid repeated subprocess calls
_nvlink_cache: Optional[Tuple[bool, str]] = None

def detect_nvlink(use_cache: bool = True) -> Tuple[bool, str]:
    """
    Detect if GPUs have NVLink connectivity.

    Args:
        use_cache: If True, return cached result from previous detection

    Returns:
        Tuple[bool, str]: (has_nvlink, detection_message)
            - has_nvlink: True if NVLink is detected between GPUs
            - detection_message: Human-readable detection result
    """
    global _nvlink_cache

    # Return cached result if available and requested
    if use_cache and _nvlink_cache is not None:
        return _nvlink_cache

    if not torch.cuda.is_available():
        result = (False, "CUDA not available")
        _nvlink_cache = result
        return result

    device_count = torch.cuda.device_count()

    if device_count < 2:
        result = (False, f"Only {device_count} GPU detected (NVLink requires multiple GPUs)")
        _nvlink_cache = result
        return result

    # Try to detect NVLink using nvidia-smi
    try:
        # Query NVLink status using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=10  # Increased timeout from 5s to 10s
        )

        if result.returncode == 0:
            output = result.stdout.lower()

            # Check if any active NVLink connections exist
            # Look for bandwidth indicators (e.g., "50 gb/s") which indicate active links
            # or status words like "active" or "up"
            if 'active' in output or 'up' in output or 'gb/s' in output:
                # Count number of active links by looking for bandwidth or "active" status
                # Pattern 1: "Link X: YY GB/s" (B200 format)
                bandwidth_links = len(re.findall(r'link\s+\d+:\s+\d+\s+gb/s', output, re.IGNORECASE))
                # Pattern 2: "Link X ... active" (older format)
                active_links = len(re.findall(r'link\s+\d+.*active', output, re.IGNORECASE))

                total_links = bandwidth_links + active_links
                if total_links > 0:
                    result = (True, f"NVLink detected: {total_links} active links between GPUs")
                    _nvlink_cache = result
                    return result

            result = (False, "NVLink supported but no active connections found")
            _nvlink_cache = result
            return result

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        # nvidia-smi nvlink command not available or failed
        # Don't cache this error, just continue to fallback
        pass

    # Fallback: Try using PyTorch's NCCL P2P access check
    try:
        if device_count >= 2:
            # Check if GPUs can do P2P (peer-to-peer) memory access
            # NVLink enables P2P, but P2P can also work over PCIe (slower)
            gpu0 = torch.cuda.device(0)
            gpu1 = torch.cuda.device(1)

            # Try to detect if P2P is available
            # Note: This doesn't definitively prove NVLink, but it's an indicator
            with gpu0:
                tensor0 = torch.randn(1, device='cuda:0')

            # If we can access the tensor from gpu1, P2P is working
            can_access = torch.cuda.can_device_access_peer(0, 1)

            if can_access:
                # We know P2P works, but we can't confirm NVLink without nvidia-smi
                return False, "P2P access available (could be NVLink or PCIe), but cannot confirm NVLink - assuming PCIe"

    except Exception:
        pass

    # Default: assume no NVLink
    return False, f"No NVLink detected between {device_count} GPUs (will use PCIe for DDP)"


def get_recommended_ddp_backend(force_check: bool = False) -> str:
    """
    Get recommended DDP backend based on NVLink availability.

    Args:
        force_check: Force re-checking NVLink status

    Returns:
        str: Recommended backend ('nccl' if NVLink, 'gloo' if PCIe only)
    """
    has_nvlink, message = detect_nvlink()

    if has_nvlink:
        return 'nccl'  # NCCL is optimal with NVLink
    else:
        # Without NVLink, NCCL over PCIe can have issues
        # Consider using gloo or single GPU
        return 'gloo'


def print_gpu_info():
    """Print detailed GPU information including NVLink status."""
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)

    if not torch.cuda.is_available():
        print("CUDA not available")
        print("="*80 + "\n")
        return

    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")

    # Check NVLink
    has_nvlink, message = detect_nvlink()
    print(f"\nNVLink Status: {message}")

    if device_count >= 2:
        # Check P2P access matrix
        print(f"\nP2P Access Matrix:")
        print("     ", end="")
        for j in range(device_count):
            print(f"GPU{j:2d} ", end="")
        print()

        for i in range(device_count):
            print(f"GPU{i:2d} ", end="")
            for j in range(device_count):
                if i == j:
                    can_access = "  -  "
                else:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    can_access = " YES " if can_access else "  NO "
                print(can_access, end="")
            print()

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR DISTRIBUTED TRAINING")
    print("="*80)

    if device_count == 1:
        print("Single GPU detected - use single-GPU training")
    elif has_nvlink:
        print("✓ NVLink detected - safe to use DDP with NCCL backend")
        print("  Recommended strategy: 'ddp' with backend='nccl'")
    else:
        print("⚠ No NVLink detected - DDP over PCIe may be slow")
        print("  Options:")
        print("  1. Use single GPU training for better performance")
        print("  2. Use 'ddp' with backend='gloo' (slower than NCCL+NVLink)")
        print("  3. Use DataParallel instead of DDP (not recommended)")
        print("  4. Consider model parallelism instead of data parallelism")

    print("="*80 + "\n")


def configure_ddp_strategy(devices: Optional[int] = None, force_single_gpu: bool = False) -> Tuple[str, int]:
    """
    Configure DDP strategy based on GPU capabilities.
    Sets optimal NCCL environment variables for both NVLink and PCIe configurations.

    Args:
        devices: Number of devices to use (None = auto-detect)
        force_single_gpu: Force single GPU even if multiple available

    Returns:
        Tuple[str, int]: (strategy, num_devices)
            - strategy: 'auto' or 'ddp'
            - num_devices: Number of devices to use
    """
    import os

    if not torch.cuda.is_available():
        return 'auto', 1  # CPU training

    device_count = torch.cuda.device_count()

    # Convert devices to int if it's a string (from CLI parsing)
    if isinstance(devices, str):
        if devices.lower() == 'auto':
            devices = None
        else:
            devices = int(devices)

    if devices is None:
        devices = device_count

    # Force single GPU if requested
    if force_single_gpu or devices == 1:
        return 'auto', 1

    # Check NVLink for multi-GPU
    if devices > 1:
        # Check if NCCL environment variables are already configured (e.g., from shell script)
        nccl_already_configured = any([
            os.environ.get('NCCL_IB_DISABLE'),
            os.environ.get('NCCL_P2P_DISABLE'),
            os.environ.get('NCCL_TIMEOUT')
        ])

        if nccl_already_configured:
            print(f"\nConfiguring distributed training:")
            print(f"  NCCL environment variables already set externally:")
            for key in ['NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 'NCCL_NET_GDR_LEVEL',
                       'NCCL_SOCKET_NTHREADS', 'NCCL_NSOCKS_PERTHREAD', 'NCCL_TIMEOUT']:
                val = os.environ.get(key)
                if val:
                    print(f"    {key}={val}")
            print(f"  Using DDP with {devices} GPUs (NCCL config from environment)")
            return 'ddp', devices

        # Auto-detect and configure NCCL
        has_nvlink, message = detect_nvlink()

        print(f"\nConfiguring distributed training:")
        print(f"  {message}")

        if has_nvlink:
            # Optimize NCCL for NVLink (B200/H100 systems)
            os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand (using NVLink instead)
            os.environ['NCCL_NET_GDR_LEVEL'] = '0'  # Disable GPU Direct RDMA (NVLink handles P2P)
            os.environ['NCCL_SOCKET_NTHREADS'] = '4'  # More socket threads for large scale
            os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'  # More sockets per thread
            # Increase timeout for large models (default is 600s, set to 1800s = 30min)
            os.environ['NCCL_TIMEOUT'] = '1800'
            print(f"  Optimizing NCCL for NVLink: IB_DISABLE=1, GDR_LEVEL=0, TIMEOUT=1800s")
        else:
            # Disable P2P and InfiniBand when NVLink is not present
            os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P, use socket communication
            os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand (not present)
            print(f"  Setting NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 for DDP over PCIe")

        print(f"  Using DDP with {devices} GPUs")
        return 'ddp', devices

    return 'auto', devices
