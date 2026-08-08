"""LArFormerParticleEvaluator — validation-metric hook for the Stage-3
particle segmenter.

Specialization of `LArFormerSlicerEvaluator` for Stage 3:

  - Default class taxonomy: 7-class
    {e, gamma, mu, pi, p, other, (unused), no_object}
    (matches `configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py`).
  - Drops the nu-specific metrics (`nu_recall` / `nu_purity` / `nu_mIoU`)
    that lose meaning when the leading class isn't "nu vs cosmic".
  - Default `best_metric = "mask_iou_mean"` — matched-pair IoU averaged
    over all classes.
  - **Active origin head metric**: per matched (q, k), reports the L2
    distance ‖ pred_origin[q] − gt_origin[k] ‖ in detector cm (i.e.
    coord_norm error × `coord_scale`). Reported as
        val/origin_l2_cm_mean / _median / _p25
        val/origin_l2_cm_<class>     (per particle class)
    Plugs in via the base evaluator's `_on_event_processed` hook
    (added in `evaluator.py`).

Registers as `@HOOKS.register_module()` on import. Configs trigger the
registration via:

    from pointcept.models.LArFormer import particle_evaluator as _ppev_mod
    del _ppev_mod
"""
from typing import List, Optional, Sequence

import numpy as np
import torch

from pointcept.engines.hooks.builder import HOOKS

from .evaluator import LArFormerSlicerEvaluator


DEFAULT_PARTICLE_CLASS_NAMES = (
    "e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object",
)


@HOOKS.register_module()
class LArFormerParticleEvaluator(LArFormerSlicerEvaluator):
    """Stage-3 evaluator: per-particle-class metrics + origin head L2."""

    def __init__(
        self,
        eval_freq: int = 0,
        best_metric: str = "mask_iou_mean",
        class_names: Optional[Sequence[str]] = None,
        empty_cache: bool = True,
        log_per_event: bool = False,
        report_origin_error: bool = True,
        coord_scale: float = 179.55,
        # Intra-epoch val-LOSS probe (overfitting watch) — implemented in the
        # parent LArFormerSlicerEvaluator; forwarded here so Stage-3 configs
        # can enable it. 0 = off (default, preserves existing behavior).
        probe_freq: int = 0,
        probe_max_events: int = 128,
    ):
        super().__init__(
            eval_freq=eval_freq,
            best_metric=best_metric,
            class_names=(list(class_names) if class_names is not None
                         else list(DEFAULT_PARTICLE_CLASS_NAMES)),
            nu_class_id=0,         # carried for parent compat; nu_*
                                   # metrics are stripped in _aggregate.
            empty_cache=empty_cache,
            log_per_event=log_per_event,
            probe_freq=probe_freq,
            probe_max_events=probe_max_events,
        )
        self.report_origin_error = bool(report_origin_error)
        self.coord_scale = float(coord_scale)

    # ------------------------------------------------------------------
    # Parent hooks
    # ------------------------------------------------------------------

    def _init_extra_state(self) -> None:
        # All matched pairs across the val epoch, in detector cm.
        self._origin_l2_cm: List[float] = []
        # Per-class buckets keyed by GT class name.
        self._origin_l2_per_class: dict = {
            name: [] for name in self.class_names
        }

    def _on_event_processed(self, *, ev_pred: dict, eval_loss: dict,
                            q_idx, k_idx,
                            no_object_class_id: int,
                            input_dict=None, event_in_batch=None) -> None:
        del no_object_class_id   # base-hook signature parity; unused here.
        if not self.report_origin_error:
            return
        if len(q_idx) == 0:
            return
        pred_origin = ev_pred.get("origin")
        if pred_origin is None:
            return
        gt_origin = eval_loss.get("gt_origin")
        if gt_origin is None:
            # Origin head disabled for this event's GT — skip silently.
            return
        gt_classes = eval_loss["gt_classes"]
        device = pred_origin.device
        q_t = torch.as_tensor(q_idx, dtype=torch.long, device=device)
        k_t = torch.as_tensor(k_idx, dtype=torch.long, device=device)

        # L2 in coord_norm units, then scale to detector cm.
        diff = (pred_origin[q_t] - gt_origin[k_t]).float()
        l2_norm = diff.norm(dim=-1)
        l2_cm = (l2_norm * self.coord_scale).cpu().numpy()

        cls_per_pair = gt_classes[k_t].cpu().numpy()
        for p, err_cm in enumerate(l2_cm):
            self._origin_l2_cm.append(float(err_cm))
            gc = int(cls_per_pair[p])
            if 0 <= gc < len(self.class_names):
                self._origin_l2_per_class[self.class_names[gc]].append(
                    float(err_cm))

    # ------------------------------------------------------------------
    # Aggregation: drop nu-specific keys, add origin metrics
    # ------------------------------------------------------------------

    def _aggregate(self, **k) -> dict:
        scalars = super()._aggregate(**k)
        for key in ("val/nu_recall", "val/nu_purity", "val/nu_mIoU",
                    "val/deghost_keep_frac_mean"):
            scalars.pop(key, None)

        if self.report_origin_error and self._origin_l2_cm:
            arr = np.asarray(self._origin_l2_cm, dtype=np.float64)
            scalars["val/origin_l2_cm_mean"] = float(arr.mean())
            scalars["val/origin_l2_cm_median"] = float(np.median(arr))
            scalars["val/origin_l2_cm_p25"] = float(np.percentile(arr, 25))
            scalars["val/origin_l2_cm_p75"] = float(np.percentile(arr, 75))
            for name, vals in self._origin_l2_per_class.items():
                if vals:
                    scalars[f"val/origin_l2_cm_{name}"] = float(
                        np.mean(vals))
        return scalars

    # ------------------------------------------------------------------
    # Log line: drop nu-specific bits, add origin metric
    # ------------------------------------------------------------------

    def _log_and_publish(self, scalars: dict) -> None:
        lines = [
            f"Val: loss {scalars.get('val/loss', float('nan')):.4f} "
            f"| cls_acc {scalars['val/cls_accuracy']:.4f} "
            f"| iou_mean {scalars['val/mask_iou_mean']:.4f} "
            f"(med {scalars['val/mask_iou_median']:.4f}, "
            f"p25 {scalars['val/mask_iou_p25']:.4f}) "
            f"| matched_frac {scalars['val/matched_fraction']:.3f} "
            f"| n_active_q_avg {scalars['val/n_active_queries_mean']:.1f}"
        ]
        if "val/origin_l2_cm_mean" in scalars:
            lines.append(
                f"Val origin err (cm): mean "
                f"{scalars['val/origin_l2_cm_mean']:.2f}  "
                f"median {scalars['val/origin_l2_cm_median']:.2f}  "
                f"p25 {scalars['val/origin_l2_cm_p25']:.2f}  "
                f"p75 {scalars['val/origin_l2_cm_p75']:.2f}"
            )

        per_class_iou_parts = []
        per_class_origin_parts = []
        for name in self.class_names:
            iou_key = f"val/mask_iou_{name}"
            ori_key = f"val/origin_l2_cm_{name}"
            if iou_key in scalars and not np.isnan(scalars[iou_key]):
                per_class_iou_parts.append(
                    f"{name}={scalars[iou_key]:.3f}"
                )
            if ori_key in scalars and not np.isnan(scalars[ori_key]):
                per_class_origin_parts.append(
                    f"{name}={scalars[ori_key]:.1f}"
                )
        if per_class_iou_parts:
            lines.append("Val per-class IoU: " + " ".join(per_class_iou_parts))
        if per_class_origin_parts:
            lines.append("Val per-class origin err (cm): "
                         + " ".join(per_class_origin_parts))

        for ln in lines:
            self.trainer.logger.info(ln)

        # Publish to TensorBoard + (optionally) wandb. Mirrors the slicer
        # evaluator's pattern so val/* lands in the same dashboard group
        # without epoch-step alignment surprises.
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            for k, v in scalars.items():
                if isinstance(v, float) and v != v:  # NaN
                    continue
                try:
                    self.trainer.writer.add_scalar(k, float(v),
                                                   current_epoch)
                except Exception:
                    pass
            if getattr(self.trainer.cfg, "enable_wandb", False):
                try:
                    import wandb
                    payload = {"Epoch": current_epoch}
                    for k, v in scalars.items():
                        if isinstance(v, float) and v != v:  # NaN
                            continue
                        payload[k] = float(v)
                    wandb.log(payload, step=wandb.run.step)
                except ImportError:
                    pass

        # Park val/loss so LR plateau schedulers / other epoch-level hooks
        # can read it from comm_info.
        self.trainer.comm_info["val_loss"] = float(
            scalars.get("val/loss", float("nan"))
        )

        best = scalars.get(f"val/{self.best_metric}",
                           scalars.get(self.best_metric))
        if best is None:
            self.trainer.logger.warning(
                f"best_metric '{self.best_metric}' not found in val "
                f"scalars; checkpoint saver won't update."
            )
        else:
            self.trainer.comm_info["current_metric_value"] = float(best)
            self.trainer.comm_info["current_metric_name"] = self.best_metric

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End LArFormerParticle Evaluation <<<<<<<<<<<<<<<<"
        )
