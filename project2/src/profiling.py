"""
Comprehensive profiling utilities for identifying training bottlenecks.

This module provides tools to profile:
- Disk I/O (data loading from disk)
- PCIe bus transfers (CPU <-> GPU)
- CPU utilization and processing time
- RAM and GPU memory usage
"""

import time
import psutil
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional
import threading
import json


class BottleneckProfiler:
    """
    Comprehensive profiler for identifying bottlenecks in model training.

    Tracks:
    - Disk I/O: Time spent loading data from disk
    - PCIe transfers: Time spent moving data to/from GPU
    - CPU processing: Time spent on CPU computations
    - Memory: RAM and GPU memory usage
    """

    def __init__(self, enabled: bool = True, log_interval: int = 10):
        """
        Args:
            enabled: Whether profiling is enabled
            log_interval: Log summary every N steps
        """
        self.enabled = enabled
        self.log_interval = log_interval
        self.reset()

    def reset(self):
        """Reset all profiling metrics."""
        self.metrics = defaultdict(list)
        self.step_count = 0
        self.epoch_count = 0

        # Per-component timing
        self.timings = {
            'disk_io': [],
            'data_transform': [],
            'cpu_to_gpu': [],
            'gpu_forward': [],
            'gpu_backward': [],
            'gpu_to_cpu': [],
            'optimizer_step': [],
            'metric_computation': [],
            'data_loading_total': [],
        }

        # Memory tracking
        self.memory = {
            'cpu_ram_mb': [],
            'gpu_mem_allocated_mb': [],
            'gpu_mem_reserved_mb': [],
            'gpu_mem_peak_mb': [],
        }

        # System resource tracking
        self.system = {
            'cpu_percent': [],
            'cpu_freq_mhz': [],
            'disk_read_mb': [],
            'disk_write_mb': [],
            'network_sent_mb': [],
            'network_recv_mb': [],
        }

        # Disk I/O stats
        self._last_disk_io = psutil.disk_io_counters()
        self._last_net_io = psutil.net_io_counters()

    @contextmanager
    def profile_section(self, section_name: str):
        """
        Context manager to profile a specific section of code.

        Args:
            section_name: Name of the section (e.g., 'disk_io', 'gpu_forward')

        Example:
            with profiler.profile_section('disk_io'):
                data = load_from_disk()
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()

        # Track memory before
        if section_name in ['gpu_forward', 'gpu_backward', 'cpu_to_gpu']:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_gpu_mem = torch.cuda.memory_allocated()

        yield

        # Track memory after and record timing
        if section_name in ['gpu_forward', 'gpu_backward', 'cpu_to_gpu']:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        # Initialize the list if this section hasn't been seen before
        if section_name not in self.timings:
            self.timings[section_name] = []
        self.timings[section_name].append(elapsed * 1000)  # Convert to ms

    def record_memory(self):
        """Record current memory usage (CPU RAM and GPU memory)."""
        if not self.enabled:
            return

        # CPU RAM
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
        self.memory['cpu_ram_mb'].append(cpu_mem)

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

            self.memory['gpu_mem_allocated_mb'].append(gpu_mem_alloc)
            self.memory['gpu_mem_reserved_mb'].append(gpu_mem_reserved)
            self.memory['gpu_mem_peak_mb'].append(gpu_mem_peak)

    def record_system_metrics(self):
        """Record system-level metrics (CPU, disk I/O, network)."""
        if not self.enabled:
            return

        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0)
        self.system['cpu_percent'].append(cpu_percent)

        # CPU frequency
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            self.system['cpu_freq_mhz'].append(cpu_freq.current)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and self._last_disk_io:
            read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 ** 2)
            write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 ** 2)
            self.system['disk_read_mb'].append(read_mb)
            self.system['disk_write_mb'].append(write_mb)
            self._last_disk_io = disk_io

        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io and self._last_net_io:
            sent_mb = (net_io.bytes_sent - self._last_net_io.bytes_sent) / (1024 ** 2)
            recv_mb = (net_io.bytes_recv - self._last_net_io.bytes_recv) / (1024 ** 2)
            self.system['network_sent_mb'].append(sent_mb)
            self.system['network_recv_mb'].append(recv_mb)
            self._last_net_io = net_io

    def step(self):
        """Call at the end of each training step."""
        if not self.enabled:
            return

        self.step_count += 1
        self.record_memory()
        self.record_system_metrics()

        # Debug: print on first call
        if self.step_count == 1:
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Rank {rank}] Profiler.step() called for the first time. Timings collected: {[k for k,v in self.timings.items() if v]}")
            except:
                print(f"Profiler.step() called for the first time. Timings collected: {[k for k,v in self.timings.items() if v]}")

        # Log summary periodically
        if self.step_count % self.log_interval == 0:
            self.print_summary()

    def epoch_end(self):
        """Call at the end of each epoch."""
        if not self.enabled:
            return

        self.epoch_count += 1
        print(f"\n{'='*80}")
        print(f"EPOCH {self.epoch_count} PROFILING SUMMARY")
        print(f"{'='*80}")
        self.print_summary(detailed=True)

    def print_summary(self, detailed: bool = False):
        """Print profiling summary."""
        if not self.enabled or self.step_count == 0:
            return

        # In DDP mode, only print from rank 0
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass  # Not using distributed training

        print(f"\n{'='*80}")
        print(f"PROFILING SUMMARY (Step {self.step_count})")
        print(f"{'='*80}")

        # Timing breakdown
        print("\n📊 TIMING BREAKDOWN (per step, recent batch):")
        print("-" * 80)

        total_time = 0
        timing_summary = {}

        for name, times in self.timings.items():
            if times:
                recent = times[-min(10, len(times)):]  # Last 10 measurements
                mean_time = np.mean(recent)
                std_time = np.std(recent)
                timing_summary[name] = mean_time
                total_time += mean_time

        # Debug: show how many measurements we have
        if not timing_summary:
            print("  ⚠️  No timing data collected yet. Profiler may not be integrated correctly.")
            print(f"  Debug info: step_count={self.step_count}, timing sections with data: {[k for k,v in self.timings.items() if v]}")
            return

        # Sort by time (descending)
        for name in sorted(timing_summary, key=timing_summary.get, reverse=True):
            mean_time = timing_summary[name]
            pct = (mean_time / total_time * 100) if total_time > 0 else 0
            print(f"  {name:25s}: {mean_time:8.2f} ms ({pct:5.1f}%)")

        print(f"  {'TOTAL':25s}: {total_time:8.2f} ms")

        # Identify bottleneck
        if timing_summary:
            bottleneck = max(timing_summary, key=timing_summary.get)
            bottleneck_time = timing_summary[bottleneck]
            bottleneck_pct = (bottleneck_time / total_time * 100) if total_time > 0 else 0

            print(f"\n⚠️  BOTTLENECK: {bottleneck} ({bottleneck_pct:.1f}% of total time)")

            # Provide recommendations
            print("\n💡 RECOMMENDATIONS:")
            if 'disk_io' in bottleneck or 'data_loading' in bottleneck:
                print("  - Disk I/O is the bottleneck!")
                print("    → Increase num_workers in DataLoader")
                print("    → Use faster storage (NVMe SSD)")
                print("    → Cache data in RAM if possible")
                print("    → Consider data preprocessing/compression")
            elif 'cpu_to_gpu' in bottleneck:
                print("  - PCIe transfer to GPU is the bottleneck!")
                print("    → Enable pin_memory=True in DataLoader")
                print("    → Reduce batch size to decrease transfer size")
                print("    → Consider using mixed precision training")
            elif 'gpu_forward' in bottleneck or 'gpu_backward' in bottleneck:
                print("  - GPU computation is the bottleneck!")
                print("    → This is typically ideal (GPU should be the bottleneck)")
                print("    → Consider increasing batch size to maximize GPU usage")
                print("    → Use mixed precision (fp16/bf16) for faster computation")
            elif 'data_transform' in bottleneck:
                print("  - Data augmentation/transforms are the bottleneck!")
                print("    → Simplify augmentation pipeline")
                print("    → Use GPU-based augmentations (e.g., Kornia)")
                print("    → Increase num_workers")

        # Memory usage
        print(f"\n💾 MEMORY USAGE (current):")
        print("-" * 80)

        if self.memory['cpu_ram_mb']:
            cpu_mem = self.memory['cpu_ram_mb'][-1]
            print(f"  CPU RAM:            {cpu_mem:10.1f} MB")

        if torch.cuda.is_available() and self.memory['gpu_mem_allocated_mb']:
            gpu_alloc = self.memory['gpu_mem_allocated_mb'][-1]
            gpu_reserved = self.memory['gpu_mem_reserved_mb'][-1]
            gpu_peak = self.memory['gpu_mem_peak_mb'][-1]
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

            print(f"  GPU Memory (alloc): {gpu_alloc:10.1f} MB ({gpu_alloc/gpu_total*100:.1f}% of {gpu_total:.0f} MB)")
            print(f"  GPU Memory (rsrvd): {gpu_reserved:10.1f} MB ({gpu_reserved/gpu_total*100:.1f}%)")
            print(f"  GPU Memory (peak):  {gpu_peak:10.1f} MB ({gpu_peak/gpu_total*100:.1f}%)")

        # System metrics
        if detailed:
            print(f"\n🖥️  SYSTEM METRICS (recent average):")
            print("-" * 80)

            if self.system['cpu_percent']:
                recent_cpu = self.system['cpu_percent'][-min(10, len(self.system['cpu_percent'])):]
                cpu_avg = np.mean(recent_cpu)
                print(f"  CPU Utilization:    {cpu_avg:10.1f}%")

            if self.system['cpu_freq_mhz']:
                recent_freq = self.system['cpu_freq_mhz'][-min(10, len(self.system['cpu_freq_mhz'])):]
                freq_avg = np.mean(recent_freq)
                print(f"  CPU Frequency:      {freq_avg:10.1f} MHz")

            if self.system['disk_read_mb']:
                recent_read = self.system['disk_read_mb'][-min(10, len(self.system['disk_read_mb'])):]
                read_avg = np.mean(recent_read)
                print(f"  Disk Read:          {read_avg:10.2f} MB/step")

            if self.system['disk_write_mb']:
                recent_write = self.system['disk_write_mb'][-min(10, len(self.system['disk_write_mb'])):]
                write_avg = np.mean(recent_write)
                print(f"  Disk Write:         {write_avg:10.2f} MB/step")

        print(f"\n{'='*80}\n")

    def get_summary_dict(self) -> Dict:
        """Get profiling summary as a dictionary (for logging to wandb, etc)."""
        summary = {}

        # Average timings
        for name, times in self.timings.items():
            if times:
                recent = times[-min(10, len(times)):]
                summary[f'profile/timing/{name}_ms'] = np.mean(recent)

        # Memory
        if self.memory['cpu_ram_mb']:
            summary['profile/memory/cpu_ram_mb'] = self.memory['cpu_ram_mb'][-1]

        if torch.cuda.is_available() and self.memory['gpu_mem_allocated_mb']:
            summary['profile/memory/gpu_allocated_mb'] = self.memory['gpu_mem_allocated_mb'][-1]
            summary['profile/memory/gpu_reserved_mb'] = self.memory['gpu_mem_reserved_mb'][-1]
            summary['profile/memory/gpu_peak_mb'] = self.memory['gpu_mem_peak_mb'][-1]

        # System
        if self.system['cpu_percent']:
            recent = self.system['cpu_percent'][-min(10, len(self.system['cpu_percent'])):]
            summary['profile/system/cpu_percent'] = np.mean(recent)

        if self.system['disk_read_mb']:
            recent = self.system['disk_read_mb'][-min(10, len(self.system['disk_read_mb'])):]
            summary['profile/system/disk_read_mb_per_step'] = np.mean(recent)

        return summary

    def save_to_json(self, filepath: str):
        """Save profiling data to JSON file."""
        data = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'timings': {k: v for k, v in self.timings.items()},
            'memory': {k: v for k, v in self.memory.items()},
            'system': {k: v for k, v in self.system.items()},
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Profiling data saved to {filepath}")


@contextmanager
def profile_data_loading(profiler: Optional[BottleneckProfiler] = None):
    """
    Context manager specifically for profiling data loading operations.
    Breaks down into disk I/O and transform components.

    Example:
        with profile_data_loading(profiler) as ctx:
            with ctx.disk_io():
                data = joblib.load(path)
            with ctx.transform():
                data = transform(data)
    """
    class DataLoadingContext:
        def __init__(self, prof):
            self.prof = prof

        @contextmanager
        def disk_io(self):
            if self.prof:
                with self.prof.profile_section('disk_io'):
                    yield
            else:
                yield

        @contextmanager
        def transform(self):
            if self.prof:
                with self.prof.profile_section('data_transform'):
                    yield
            else:
                yield

    ctx = DataLoadingContext(profiler)

    if profiler:
        with profiler.profile_section('data_loading_total'):
            yield ctx
    else:
        yield ctx


class GPUTransferProfiler:
    """
    Specialized profiler for measuring PCIe transfer speeds between CPU and GPU.
    """

    @staticmethod
    def benchmark_pcie_bandwidth(size_mb: float = 100, num_iterations: int = 10):
        """
        Benchmark PCIe bandwidth for CPU->GPU and GPU->CPU transfers.

        Args:
            size_mb: Size of data to transfer in MB
            num_iterations: Number of iterations to average

        Returns:
            Dict with transfer speeds in MB/s
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping PCIe benchmark")
            return {}

        print(f"\n{'='*80}")
        print("PCIe BANDWIDTH BENCHMARK")
        print(f"{'='*80}")

        # Create test tensor
        num_elements = int((size_mb * 1024 * 1024) / 4)  # float32 = 4 bytes
        cpu_tensor = torch.randn(num_elements)

        # CPU -> GPU
        cpu_to_gpu_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            gpu_tensor = cpu_tensor.cuda()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cpu_to_gpu_times.append(elapsed)

        cpu_to_gpu_speed = size_mb / np.mean(cpu_to_gpu_times)

        # GPU -> CPU
        gpu_to_cpu_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cpu_result = gpu_tensor.cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            gpu_to_cpu_times.append(elapsed)

        gpu_to_cpu_speed = size_mb / np.mean(gpu_to_cpu_times)

        print(f"CPU -> GPU: {cpu_to_gpu_speed:8.1f} MB/s ({np.mean(cpu_to_gpu_times)*1000:.2f} ms for {size_mb:.0f} MB)")
        print(f"GPU -> CPU: {gpu_to_cpu_speed:8.1f} MB/s ({np.mean(gpu_to_cpu_times)*1000:.2f} ms for {size_mb:.0f} MB)")
        print(f"{'='*80}\n")

        return {
            'cpu_to_gpu_mb_per_sec': cpu_to_gpu_speed,
            'gpu_to_cpu_mb_per_sec': gpu_to_cpu_speed,
        }


# Global profiler instance (can be accessed from anywhere)
_global_profiler: Optional[BottleneckProfiler] = None


def get_global_profiler() -> Optional[BottleneckProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def set_global_profiler(profiler: BottleneckProfiler):
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler
