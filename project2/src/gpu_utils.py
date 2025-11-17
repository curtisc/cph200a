"""
GPU utilities for detecting NVLink and configuring distributed training.
"""

import torch
import subprocess
import re
from typing import Tuple, List, Optional


def detect_nvlink() -> Tuple[bool, str]:
    """
    Detect if GPUs have NVLink connectivity.

    Returns:
        Tuple[bool, str]: (has_nvlink, detection_message)
            - has_nvlink: True if NVLink is detected between GPUs
            - detection_message: Human-readable detection result
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    device_count = torch.cuda.device_count()

    if device_count < 2:
        return False, f"Only {device_count} GPU detected (NVLink requires multiple GPUs)"

    # Try to detect NVLink using nvidia-smi
    try:
        # Query NVLink status using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output = result.stdout.lower()

            # Check if any active NVLink connections exist
            if 'active' in output or 'up' in output:
                # Count number of active links
                active_links = len(re.findall(r'link\s+\d+.*active', output, re.IGNORECASE))
                if active_links > 0:
                    return True, f"NVLink detected: {active_links} active links between GPUs"

            return False, "NVLink supported but no active connections found"

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        # nvidia-smi nvlink command not available or failed
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
    Sets NCCL environment variables when NVLink is not present.

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
        has_nvlink, message = detect_nvlink()

        print(f"\nConfiguring distributed training:")
        print(f"  {message}")

        if not has_nvlink:
            # Disable P2P and InfiniBand when NVLink is not present
            os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P, use socket communication
            os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand (not present)
            print(f"  Setting NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 for DDP over PCIe")

        print(f"  Using DDP with {devices} GPUs")
        return 'ddp', devices

    return 'auto', devices
