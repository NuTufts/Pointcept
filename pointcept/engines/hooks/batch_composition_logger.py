"""
BatchCompositionLogger (metric M5 of the Phase 0.5 plan, see
lartpc/pretraining_studies/phase0_phase05_implementation_plan.md).

Logs the per-batch truth-class point fractions seen during pretraining, so
hypotheses like "the Sinkhorn heads mostly see muon points" become measured
numbers. Requires the truth labels to reach the collated batch: for Sonata
multi-view pretraining add "segment" to MultiViewGenerator's view_keys and
"global_segment" to the Collect keys (the dataset must run with
data_only=False so labels are loaded).
"""
import numpy as np
import torch

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class BatchCompositionLogger(HookBase):
    """
    Args:
        log_frequency: log every N steps.
        segment_key: key in the collated input_dict holding per-point labels.
            "global_segment" for Sonata multi-view batches, "segment" for
            standard (supervised / probe) batches.
        prefix: logging key prefix.
        running_window: also log a running mean over this many logged batches.
    """

    def __init__(self, log_frequency=50, segment_key="global_segment",
                 prefix="batch_comp", running_window=20):
        self.log_frequency = log_frequency
        self.segment_key = segment_key
        self.prefix = prefix
        self.running_window = running_window
        self._recent = []
        self._warned = False

    def after_step(self):
        curr_iter = (
            self.trainer.comm_info["iter"]
            + len(self.trainer.train_loader) * self.trainer.epoch
        )
        if (curr_iter + 1) % self.log_frequency != 0:
            return
        input_dict = self.trainer.comm_info.get("input_dict", None)
        if input_dict is None or self.segment_key not in input_dict:
            if not self._warned:
                self.trainer.logger.warning(
                    f"[BatchCompositionLogger] key '{self.segment_key}' not in "
                    f"batch — check view_keys/Collect config. Hook disabled."
                )
                self._warned = True
            return

        segment = input_dict[self.segment_key]
        if isinstance(segment, torch.Tensor):
            segment = segment.detach().cpu().numpy()
        segment = segment.reshape(-1)
        n_total = segment.shape[0]
        if n_total == 0:
            return

        num_classes = self.trainer.cfg.data.num_classes
        names = list(self.trainer.cfg.data.names)
        counts = np.bincount(
            segment[(segment >= 0) & (segment < num_classes)].astype(np.int64),
            minlength=num_classes,
        )
        fractions = {names[i]: counts[i] / n_total for i in range(num_classes)}
        fractions["unlabeled_or_other"] = 1.0 - counts.sum() / n_total

        self._recent.append(fractions)
        if len(self._recent) > self.running_window:
            self._recent.pop(0)

        if self.trainer.writer is not None:
            for name, frac in fractions.items():
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/{name}", frac, curr_iter + 1
                )
        if getattr(self.trainer.cfg, "enable_wandb", False):
            import wandb
            if wandb.run is not None:
                metrics = {f"{self.prefix}/{name}": frac
                           for name, frac in fractions.items()}
                metrics["Iter"] = curr_iter + 1
                for name in fractions:
                    metrics[f"{self.prefix}_run{self.running_window}/{name}"] = (
                        float(np.mean([r[name] for r in self._recent]))
                    )
                wandb.log(metrics, step=wandb.run.step)
