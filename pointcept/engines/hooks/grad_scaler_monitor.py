"""
GradScaler Monitor Hook

Monitors the AMP GradScaler state during training to detect gradient overflow events.
Useful for diagnosing training instability when using mixed precision.

Author: Claude Code
"""

import torch
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class GradScalerMonitor(HookBase):
    """
    Hook to monitor GradScaler state during AMP training.

    Tracks:
    - scale: Current loss scale factor
    - growth_tracker: Steps since last overflow (resets on overflow)
    - overflow_count: Total overflow events detected during training

    Logs warnings when overflow is detected (scale decreases).

    Args:
        log_frequency (int): How often to log scaler stats in steps (default: 10)
        prefix (str): Prefix for logging keys (default: "grad_scaler")
        warn_on_overflow (bool): Log warning when overflow detected (default: True)
        warn_on_low_scale (float): Log warning if scale drops below this value (default: 1.0)
    """

    def __init__(
        self,
        log_frequency=10,
        prefix="grad_scaler",
        warn_on_overflow=True,
        warn_on_low_scale=1.0,
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.warn_on_overflow = warn_on_overflow
        self.warn_on_low_scale = warn_on_low_scale

        # Track state across steps
        self.step_count = 0
        self.prev_scale = None
        self.overflow_count = 0
        self.min_scale_seen = float('inf')
        self.max_scale_seen = 0.0

    def before_train(self):
        """Initialize monitoring."""
        if not self.trainer.cfg.enable_amp:
            self.trainer.logger.info(
                f"[{self.prefix}] AMP is disabled, GradScalerMonitor will be inactive"
            )
            return

        # Get initial scale
        if hasattr(self.trainer, 'scaler') and self.trainer.scaler is not None:
            self.prev_scale = self.trainer.scaler.get_scale()
            self.min_scale_seen = self.prev_scale
            self.max_scale_seen = self.prev_scale
            self.trainer.logger.info(
                f"[{self.prefix}] Monitoring GradScaler (initial scale={self.prev_scale:.0f}, "
                f"log_frequency={self.log_frequency})"
            )
        else:
            self.trainer.logger.warning(
                f"[{self.prefix}] No scaler found on trainer, monitoring disabled"
            )

    def after_step(self):
        """Check scaler state after each optimization step."""
        if not self.trainer.cfg.enable_amp:
            return

        if not hasattr(self.trainer, 'scaler') or self.trainer.scaler is None:
            return

        self.step_count += 1
        scaler = self.trainer.scaler

        # Get current state
        current_scale = scaler.get_scale()
        growth_tracker = scaler._growth_tracker

        # Update min/max tracking
        self.min_scale_seen = min(self.min_scale_seen, current_scale)
        self.max_scale_seen = max(self.max_scale_seen, current_scale)

        # Detect overflow (scale decreased)
        overflow_detected = False
        if self.prev_scale is not None and current_scale < self.prev_scale:
            self.overflow_count += 1
            overflow_detected = True

            if self.warn_on_overflow:
                global_step = self.trainer.comm_info.get("iter", self.step_count)
                self.trainer.logger.warning(
                    f"[{self.prefix}] Gradient overflow detected at step {global_step}! "
                    f"Scale: {self.prev_scale:.0f} -> {current_scale:.0f} "
                    f"(total overflows: {self.overflow_count})"
                )

        # Warn on critically low scale
        if current_scale < self.warn_on_low_scale and self.prev_scale is not None:
            if self.prev_scale >= self.warn_on_low_scale:  # Only warn on transition
                self.trainer.logger.error(
                    f"[{self.prefix}] CRITICAL: Scale dropped below {self.warn_on_low_scale}! "
                    f"Current scale: {current_scale:.6f}. Training may be unstable."
                )

        self.prev_scale = current_scale

        # Log every N steps
        if self.step_count % self.log_frequency == 0:
            self._log_metrics(current_scale, growth_tracker, overflow_detected)

    def _log_metrics(self, scale, growth_tracker, overflow_detected):
        """Log metrics to console, tensorboard, and wandb."""
        global_step = self.trainer.comm_info.get("iter", self.step_count)

        # Only log from main process
        if not comm.is_main_process():
            return

        metrics = {
            "scale": scale,
            "scale_log2": torch.tensor(scale).log2().item() if scale > 0 else 0,
            "growth_tracker": growth_tracker,
            "overflow_count": self.overflow_count,
            "min_scale_seen": self.min_scale_seen,
            "max_scale_seen": self.max_scale_seen,
        }

        # Console logging (brief)
        self.trainer.logger.info(
            f"[{self.prefix}] step={global_step} scale={scale:.0f} (2^{metrics['scale_log2']:.1f}) "
            f"growth_tracker={growth_tracker}/2000 overflows={self.overflow_count}"
        )

        # TensorBoard logging
        if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
            for metric_name, metric_value in metrics.items():
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/{metric_name}",
                    metric_value,
                    global_step
                )

        # WandB logging
        if getattr(self.trainer.cfg, 'enable_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb_metrics = {
                        f"{self.prefix}/{metric_name}": metric_value
                        for metric_name, metric_value in metrics.items()
                    }
                    # Add overflow as an event marker
                    if overflow_detected:
                        wandb_metrics[f"{self.prefix}/overflow_event"] = 1
                    else:
                        wandb_metrics[f"{self.prefix}/overflow_event"] = 0

                    wandb.log(wandb_metrics, step=global_step)
            except ImportError:
                pass

    def after_train(self):
        """Log final summary."""
        if not self.trainer.cfg.enable_amp:
            return

        if comm.is_main_process():
            self.trainer.logger.info(
                f"[{self.prefix}] Training complete. "
                f"Total overflow events: {self.overflow_count}, "
                f"Scale range: [{self.min_scale_seen:.0f}, {self.max_scale_seen:.0f}]"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"log_frequency={self.log_frequency}, "
            f"prefix='{self.prefix}', "
            f"warn_on_overflow={self.warn_on_overflow}, "
            f"warn_on_low_scale={self.warn_on_low_scale})"
        )
