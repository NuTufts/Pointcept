"""
Adam Optimizer State Monitor Hook

Monitors Adam/AdamW optimizer state during training to detect instability.
Tracks momentum buffers, variance estimates, and effective update magnitudes.

Useful for diagnosing:
- Gradient explosion (growing exp_avg_sq)
- Momentum accumulation issues (growing exp_avg)
- Layer-specific instability in transformers

Author: Claude Code
"""

import torch
import numpy as np
import pointcept.utils.comm as comm

from collections import defaultdict
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class AdamStateMonitor(HookBase):
    """
    Hook to monitor Adam/AdamW optimizer state during training.

    Tracks:
    - exp_avg: First moment (momentum) statistics
    - exp_avg_sq: Second moment (variance) statistics
    - update_ratio: |exp_avg| / sqrt(exp_avg_sq) - the normalized update
    - param_update_ratio: How much parameters change relative to their magnitude

    Can group statistics by layer type (embedding, attention, mlp, etc.)

    Args:
        log_frequency (int): How often to log stats in steps (default: 100)
        prefix (str): Prefix for logging keys (default: "adam_state")
        track_layers (bool): Track per-layer-type statistics (default: True)
        track_histograms (bool): Log histograms to wandb (default: False, expensive)
        histogram_frequency (int): How often to log histograms if enabled (default: 500)
        layer_patterns (dict): Patterns to identify layer types. Keys are layer names,
            values are list of substrings to match. Default groups: embedding, attention,
            mlp, norm, head.
    """

    def __init__(
        self,
        log_frequency=100,
        prefix="adam_state",
        track_layers=True,
        track_histograms=False,
        histogram_frequency=500,
        layer_patterns=None,
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.track_layers = track_layers
        self.track_histograms = track_histograms
        self.histogram_frequency = histogram_frequency

        # Default layer patterns for transformer architectures
        self.layer_patterns = layer_patterns or {
            "embedding": ["embed", "stem", "patch"],
            "attention": ["attn", "attention", "qkv", "proj"],
            "mlp": ["mlp", "ffn", "fc1", "fc2", "linear"],
            "norm": ["norm", "ln", "layer_norm", "batch_norm"],
            "head": ["head", "cls", "seg_head", "predictor"],
        }

        self.step_count = 0
        self.param_name_map = {}  # Maps param index to name
        self.param_layer_type = {}  # Maps param index to layer type

    def before_train(self):
        """Build parameter name mapping."""
        self.trainer.logger.info(
            f"[{self.prefix}] Monitoring Adam state (log_frequency={self.log_frequency})"
        )

        # Build mapping from parameter to name
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module

        param_to_name = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_to_name[id(param)] = name

        # Map optimizer param indices to names and layer types
        for group_idx, group in enumerate(self.trainer.optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                global_idx = (group_idx, param_idx)
                param_id = id(param)

                if param_id in param_to_name:
                    name = param_to_name[param_id]
                    self.param_name_map[global_idx] = name
                    self.param_layer_type[global_idx] = self._get_layer_type(name)

        # Log layer type distribution
        if self.track_layers:
            type_counts = defaultdict(int)
            for layer_type in self.param_layer_type.values():
                type_counts[layer_type] += 1
            self.trainer.logger.info(
                f"[{self.prefix}] Parameter layer types: {dict(type_counts)}"
            )

    def _get_layer_type(self, param_name):
        """Determine layer type from parameter name."""
        param_name_lower = param_name.lower()
        for layer_type, patterns in self.layer_patterns.items():
            for pattern in patterns:
                if pattern in param_name_lower:
                    return layer_type
        return "other"

    def after_step(self):
        """Analyze optimizer state after each step."""
        self.step_count += 1

        # Regular stats logging
        if self.step_count % self.log_frequency == 0:
            self._compute_and_log_stats()

        # Histogram logging (less frequent, more expensive)
        if self.track_histograms and self.step_count % self.histogram_frequency == 0:
            self._log_histograms()

    def _compute_and_log_stats(self):
        """Compute and log optimizer state statistics."""
        if not comm.is_main_process():
            return

        optimizer = self.trainer.optimizer
        global_step = self.trainer.comm_info.get("iter", self.step_count)

        # Collect statistics
        global_stats = {
            "exp_avg_abs_mean": [],
            "exp_avg_abs_max": [],
            "exp_avg_sq_mean": [],
            "exp_avg_sq_max": [],
            "update_ratio_mean": [],  # |exp_avg| / sqrt(exp_avg_sq + eps)
            "update_ratio_max": [],
            "param_norm": [],
        }

        layer_stats = defaultdict(lambda: defaultdict(list))

        eps = optimizer.defaults.get('eps', 1e-8)

        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                global_idx = (group_idx, param_idx)
                state = optimizer.state[param]

                if len(state) == 0:  # No state yet
                    continue

                # Get Adam buffers
                exp_avg = state.get('exp_avg')
                exp_avg_sq = state.get('exp_avg_sq')

                if exp_avg is None or exp_avg_sq is None:
                    continue

                with torch.no_grad():
                    # Compute statistics
                    exp_avg_abs = exp_avg.abs()
                    exp_avg_abs_mean = exp_avg_abs.mean().item()
                    exp_avg_abs_max = exp_avg_abs.max().item()

                    exp_avg_sq_mean = exp_avg_sq.mean().item()
                    exp_avg_sq_max = exp_avg_sq.max().item()

                    # Update ratio: this is what Adam actually uses for direction
                    # update = exp_avg / (sqrt(exp_avg_sq) + eps)
                    update_ratio = exp_avg_abs / (exp_avg_sq.sqrt() + eps)
                    update_ratio_mean = update_ratio.mean().item()
                    update_ratio_max = update_ratio.max().item()

                    param_norm = param.data.norm().item()

                # Global stats
                global_stats["exp_avg_abs_mean"].append(exp_avg_abs_mean)
                global_stats["exp_avg_abs_max"].append(exp_avg_abs_max)
                global_stats["exp_avg_sq_mean"].append(exp_avg_sq_mean)
                global_stats["exp_avg_sq_max"].append(exp_avg_sq_max)
                global_stats["update_ratio_mean"].append(update_ratio_mean)
                global_stats["update_ratio_max"].append(update_ratio_max)
                global_stats["param_norm"].append(param_norm)

                # Per-layer-type stats
                if self.track_layers and global_idx in self.param_layer_type:
                    layer_type = self.param_layer_type[global_idx]
                    layer_stats[layer_type]["exp_avg_abs_mean"].append(exp_avg_abs_mean)
                    layer_stats[layer_type]["exp_avg_sq_mean"].append(exp_avg_sq_mean)
                    layer_stats[layer_type]["update_ratio_mean"].append(update_ratio_mean)
                    layer_stats[layer_type]["update_ratio_max"].append(update_ratio_max)

        # Aggregate global stats
        metrics = {}
        if global_stats["exp_avg_abs_mean"]:
            metrics[f"{self.prefix}/exp_avg_abs_mean"] = np.mean(global_stats["exp_avg_abs_mean"])
            metrics[f"{self.prefix}/exp_avg_abs_max"] = np.max(global_stats["exp_avg_abs_max"])
            metrics[f"{self.prefix}/exp_avg_sq_mean"] = np.mean(global_stats["exp_avg_sq_mean"])
            metrics[f"{self.prefix}/exp_avg_sq_max"] = np.max(global_stats["exp_avg_sq_max"])
            metrics[f"{self.prefix}/update_ratio_mean"] = np.mean(global_stats["update_ratio_mean"])
            metrics[f"{self.prefix}/update_ratio_max"] = np.max(global_stats["update_ratio_max"])
            metrics[f"{self.prefix}/param_norm_mean"] = np.mean(global_stats["param_norm"])

        # Aggregate per-layer stats
        if self.track_layers:
            for layer_type, stats in layer_stats.items():
                if stats["exp_avg_sq_mean"]:
                    metrics[f"{self.prefix}/{layer_type}/exp_avg_sq_mean"] = np.mean(stats["exp_avg_sq_mean"])
                    metrics[f"{self.prefix}/{layer_type}/update_ratio_mean"] = np.mean(stats["update_ratio_mean"])
                    metrics[f"{self.prefix}/{layer_type}/update_ratio_max"] = np.max(stats["update_ratio_max"])

        # Console logging
        self.trainer.logger.info(
            f"[{self.prefix}] step={global_step} "
            f"exp_avg_abs=(mean={metrics.get(f'{self.prefix}/exp_avg_abs_mean', 0):.2e}, "
            f"max={metrics.get(f'{self.prefix}/exp_avg_abs_max', 0):.2e}) "
            f"exp_avg_sq=(mean={metrics.get(f'{self.prefix}/exp_avg_sq_mean', 0):.2e}, "
            f"max={metrics.get(f'{self.prefix}/exp_avg_sq_max', 0):.2e}) "
            f"update_ratio_max={metrics.get(f'{self.prefix}/update_ratio_max', 0):.2e}"
        )

        # TensorBoard logging
        if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
            for metric_name, metric_value in metrics.items():
                self.trainer.writer.add_scalar(metric_name, metric_value, global_step)

        # WandB logging
        if getattr(self.trainer.cfg, 'enable_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=global_step)
            except ImportError:
                pass

    def _log_histograms(self):
        """Log histograms of optimizer state to wandb."""
        if not comm.is_main_process():
            return

        if not getattr(self.trainer.cfg, 'enable_wandb', False):
            return

        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        optimizer = self.trainer.optimizer
        global_step = self.trainer.comm_info.get("iter", self.step_count)

        # Collect all exp_avg and exp_avg_sq values
        all_exp_avg = []
        all_exp_avg_sq = []
        layer_exp_avg_sq = defaultdict(list)

        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                global_idx = (group_idx, param_idx)
                state = optimizer.state[param]

                if len(state) == 0:
                    continue

                exp_avg = state.get('exp_avg')
                exp_avg_sq = state.get('exp_avg_sq')

                if exp_avg is not None:
                    # Sample to avoid memory issues
                    flat = exp_avg.flatten()
                    if len(flat) > 10000:
                        indices = torch.randperm(len(flat))[:10000]
                        flat = flat[indices]
                    all_exp_avg.extend(flat.cpu().numpy().tolist())

                if exp_avg_sq is not None:
                    flat = exp_avg_sq.flatten()
                    if len(flat) > 10000:
                        indices = torch.randperm(len(flat))[:10000]
                        flat = flat[indices]
                    all_exp_avg_sq.extend(flat.cpu().numpy().tolist())

                    # Per-layer histograms
                    if self.track_layers and global_idx in self.param_layer_type:
                        layer_type = self.param_layer_type[global_idx]
                        layer_exp_avg_sq[layer_type].extend(flat.cpu().numpy().tolist())

        # Log histograms
        histograms = {}

        if all_exp_avg:
            histograms[f"{self.prefix}/exp_avg_histogram"] = wandb.Histogram(
                np.array(all_exp_avg), num_bins=64
            )

        if all_exp_avg_sq:
            # Use log scale for exp_avg_sq (spans many orders of magnitude)
            log_exp_avg_sq = np.log10(np.array(all_exp_avg_sq) + 1e-10)
            histograms[f"{self.prefix}/exp_avg_sq_log_histogram"] = wandb.Histogram(
                log_exp_avg_sq, num_bins=64
            )

        # Per-layer histograms
        for layer_type, values in layer_exp_avg_sq.items():
            if values:
                log_values = np.log10(np.array(values) + 1e-10)
                histograms[f"{self.prefix}/{layer_type}/exp_avg_sq_log_histogram"] = wandb.Histogram(
                    log_values, num_bins=64
                )

        if histograms:
            wandb.log(histograms, step=global_step)
            self.trainer.logger.info(
                f"[{self.prefix}] Logged {len(histograms)} histograms at step {global_step}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"log_frequency={self.log_frequency}, "
            f"prefix='{self.prefix}', "
            f"track_layers={self.track_layers}, "
            f"track_histograms={self.track_histograms})"
        )
