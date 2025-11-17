"""
PyTorch Lightning Callback for profiling training bottlenecks.
"""

import lightning.pytorch as pl
from src.profiling import BottleneckProfiler, set_global_profiler


class ProfilingCallback(pl.Callback):
    """
    Lightning callback to profile training and identify bottlenecks.

    This works with any model without requiring modifications to training_step.
    """

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.profiler = None

    def on_train_start(self, trainer, pl_module):
        """Create profiler when training starts."""
        # Only create on rank 0 in DDP mode, or always in single GPU
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass

        self.profiler = BottleneckProfiler(enabled=True, log_interval=self.log_interval)
        set_global_profiler(self.profiler)
        print("[ProfilingCallback] Profiler initialized")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record start of batch processing."""
        if self.profiler:
            # Data is already loaded at this point, so we missed disk I/O
            # But we can still profile GPU operations
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Call profiler.step() after each batch."""
        if self.profiler:
            self.profiler.step()

    def on_train_epoch_end(self, trainer, pl_module):
        """Print profiling summary at epoch end."""
        if self.profiler:
            self.profiler.epoch_end()

    def on_train_end(self, trainer, pl_module):
        """Print final profiling summary."""
        if self.profiler:
            print("\n" + "="*80)
            print("FINAL PROFILING SUMMARY")
            print("="*80)
            self.profiler.print_summary(detailed=True)
