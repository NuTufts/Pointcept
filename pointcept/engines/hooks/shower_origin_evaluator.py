"""
Shower Origin Evaluator Hook

Validation metrics for the K=1 shower origin prediction model:
- Origin Recall / Non-origin Specificity (score threshold metrics)
- Peak Distance from score-head argmax to score target (= trunk_start_coord
  under the new scheme, or origin_coord under the legacy one).
- Origin Pred Error: 3D distance from the new regression head's predicted
  origin to the ground-truth origin_coord, split by routing scenario:
    * anchor-bound: origin_coord == score_target (e.g. type=1 outside-TPC,
      or type=0 with unobservable vertex). Regression head is expected to
      reproduce the score-target / trunk-start.
    * vertex-target: origin_coord != score_target (e.g. type=0 with visible
      pi0 vertex). Regression head must learn the displacement.
- Origin Type Classification Accuracy (per-class and overall)

Follows SemSegEvaluator patterns for logging and CheckpointSaver integration.
"""

import numpy as np
import wandb
import torch

import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS

# Origin type class names (must match dataset encoding)
# Types 0-2 are truth-based, types 3-4 are reco-specific
ORIGIN_CLASS_NAMES = ["inside", "outside", "on_track", "ghost", "true_track"]


@HOOKS.register_module()
class ShowerOriginEvaluator(HookBase):
    """
    Evaluator hook for ShowerOriginPredictorV3.

    Computes per-epoch or per-step validation metrics:
      - val/loss: average loss
      - val/origin_recall: recall among GT-positive points
      - val/nonorigin_specificity: specificity among GT-negative points
      - val/mean_peak_distance: mean Euclidean distance from predicted peak to GT origin
      - val/median_peak_distance: median of above
      - val/cls_accuracy: overall origin-type classification accuracy
      - val/cls_{name}_acc: per-class accuracy

    Best metric for CheckpointSaver: origin_recall (higher = better).

    Args:
        score_threshold: Threshold for GT and predicted positive (default: 0.05)
        eval_freq: 0 = evaluate after epoch; >0 = every N global steps
    """

    def __init__(self, score_threshold=0.05, eval_freq=0):
        self.score_threshold = score_threshold
        self.eval_freq = eval_freq

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_step(self):
        if self.trainer.cfg.evaluate and self.eval_freq > 0:
            iter_per_epoch = len(self.trainer.train_loader)
            global_iter = (
                self.trainer.comm_info["iter"]
                + iter_per_epoch * self.trainer.epoch
            )
            if (global_iter + 1) % self.eval_freq == 0:
                self.eval()

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.eval_freq == 0:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Shower Origin Evaluation >>>>>>>>>>>>>>>>"
        )

        # Reset histories to prevent accumulation across evals
        for key in (
            "val_loss",
            "val_tp",
            "val_fp",
            "val_tn",
            "val_fn",
            "val_cls_correct",
            "val_cls_total",
        ):
            if key in self.trainer.storage._history:
                self.trainer.storage.reset_history(key)
        for name in ORIGIN_CLASS_NAMES:
            for suffix in ("_correct", "_total"):
                key = f"val_cls_{name}{suffix}"
                if key in self.trainer.storage._history:
                    self.trainer.storage.reset_history(key)

        torch.cuda.empty_cache()
        self.trainer.model.eval()

        model = getattr(self.trainer.model, "module", self.trainer.model)
        gaussian_sigma = model.gaussian_sigma
        score_target_key = getattr(model, "score_target_key", "origin_coord")
        all_peak_distances = []  # peak vs score_target
        all_origin_errors = []   # pred_origin vs origin_coord
        all_origin_errors_anchor = []   # subset where origin == score_target
        all_origin_errors_vertex = []   # subset where origin != score_target

        for i, input_dict in enumerate(self.trainer.val_loader):
            # Move tensors to GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            # Handle both single-sample and batched outputs
            scores_list = output_dict["origin_scores"]
            coords_list = output_dict["all_coords"]
            cls_out = output_dict["origin_class"]

            if not isinstance(scores_list, list):
                scores_list = [scores_list]
                coords_list = [coords_list]
                cls_out = [cls_out]
            else:
                cls_out = [cls_out[j] for j in range(cls_out.shape[0])]

            # Parse batched origin_coord, score_target_coord, origin_type,
            # and (when present) origin_pred_coord.
            origin_coord_raw = input_dict["origin_coord"]
            origin_type_raw = input_dict.get("origin_type", None)
            score_target_raw = input_dict.get(score_target_key, None)
            if score_target_raw is None:
                # Fallback: score target == origin_coord (legacy V3 behavior)
                score_target_raw = origin_coord_raw

            origin_pred_raw = output_dict.get("origin_pred_coord", None)

            B_sample = len(scores_list)

            def _to_b3(t):
                if t is None:
                    return None
                if t.dim() == 1 and B_sample > 1:
                    return t.view(B_sample, 3)
                if t.dim() == 1:
                    return t.unsqueeze(0)
                return t

            origin_coords = _to_b3(origin_coord_raw)
            score_targets = _to_b3(score_target_raw)
            origin_preds = _to_b3(origin_pred_raw)

            for j in range(B_sample):
                pred_scores = scores_list[j]
                coords = coords_list[j]
                origin_coord = origin_coords[j]
                score_target = score_targets[j]
                origin_pred = origin_preds[j] if origin_preds is not None else None

                # Determine origin type for this sample
                gt_cls = None
                gt_cls_val = -1
                if origin_type_raw is not None:
                    if origin_type_raw.dim() == 0:
                        gt_cls = origin_type_raw
                    elif B_sample == 1 and origin_type_raw.shape[0] == 1:
                        gt_cls = origin_type_raw[0]
                    else:
                        gt_cls = origin_type_raw[j]
                    gt_cls_val = gt_cls.long().item()

                # --- Score and peak metrics ---
                # Skip for ghost (3) and true_track (4) — origin_coord is
                # placeholder zeros, so Gaussian target is meaningless.
                # Also skip for unknown (-1).
                has_valid_origin = (0 <= gt_cls_val < 3)

                tp = fp = tn = fn = 0
                peak_distance = None
                origin_error = None
                anchor_bound = None  # True when origin == score_target
                if has_valid_origin:
                    # Recompute GT Gaussian at output resolution. The peak now
                    # sits at score_target (= trunk_start_coord under the new
                    # scheme), not at origin_coord.
                    gt_dist = torch.norm(
                        coords - score_target.unsqueeze(0), dim=-1
                    )
                    gt_scores = torch.exp(
                        -0.5 * (gt_dist / gaussian_sigma) ** 2
                    )

                    gt_pos = gt_scores > self.score_threshold
                    gt_neg = ~gt_pos
                    pred_pos = pred_scores > self.score_threshold
                    pred_neg = ~pred_pos

                    tp = (pred_pos & gt_pos).sum().item()
                    fp = (pred_pos & gt_neg).sum().item()
                    tn = (pred_neg & gt_neg).sum().item()
                    fn = (pred_neg & gt_pos).sum().item()

                    # Peak localization vs score target
                    peak_idx = pred_scores.argmax()
                    peak_coord = coords[peak_idx]
                    peak_distance = torch.norm(
                        peak_coord - score_target
                    ).item()

                    # Origin regression error vs ground-truth origin_coord
                    if origin_pred is not None:
                        origin_error = torch.norm(
                            origin_pred - origin_coord
                        ).item()
                        # "anchor-bound" when origin_coord == score_target,
                        # i.e. regression target degenerates to the trunk start
                        # (type=1 outside, or type=0 unobservable vertex).
                        anchor_bound = bool(
                            torch.allclose(origin_coord, score_target,
                                            atol=1e-5, rtol=0.0)
                        )

                # --- Classification ---
                cls_correct = 0
                cls_total = 0
                if gt_cls_val >= 0:
                    pred_cls = cls_out[j]
                    pred_cls_val = (
                        pred_cls.item() if pred_cls.dim() == 0
                        else pred_cls[0].item()
                    )
                    cls_correct = int(pred_cls_val == gt_cls_val)
                    cls_total = 1

                    if gt_cls_val < len(ORIGIN_CLASS_NAMES):
                        name = ORIGIN_CLASS_NAMES[gt_cls_val]
                        self.trainer.storage.put_scalar(
                            f"val_cls_{name}_correct", cls_correct
                        )
                        self.trainer.storage.put_scalar(
                            f"val_cls_{name}_total", 1
                        )

                # Accumulate
                if j == 0:
                    self.trainer.storage.put_scalar("val_loss", loss.item())
                if has_valid_origin:
                    self.trainer.storage.put_scalar("val_tp", tp)
                    self.trainer.storage.put_scalar("val_fp", fp)
                    self.trainer.storage.put_scalar("val_tn", tn)
                    self.trainer.storage.put_scalar("val_fn", fn)
                    all_peak_distances.append(peak_distance)
                    if origin_error is not None:
                        all_origin_errors.append(origin_error)
                        if anchor_bound:
                            all_origin_errors_anchor.append(origin_error)
                        else:
                            all_origin_errors_vertex.append(origin_error)
                self.trainer.storage.put_scalar("val_cls_correct", cls_correct)
                self.trainer.storage.put_scalar("val_cls_total", cls_total)

            self.trainer.logger.info(
                "Val: [{iter}/{max_iter}] "
                "Loss {loss:.4f} "
                "Samples {B}".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.val_loader),
                    loss=loss.item(),
                    B=B_sample,
                )
            )

        # --- Aggregate metrics ---
        loss_avg = self.trainer.storage.history("val_loss").avg

        total_tp = self.trainer.storage.history("val_tp").total
        total_fp = self.trainer.storage.history("val_fp").total
        total_tn = self.trainer.storage.history("val_tn").total
        total_fn = self.trainer.storage.history("val_fn").total

        origin_recall = total_tp / max(total_tp + total_fn, 1)
        nonorigin_specificity = total_tn / max(total_tn + total_fp, 1)

        # Peak distances from local list
        mean_peak_distance = (
            float(np.mean(all_peak_distances)) if all_peak_distances else 0.0
        )
        median_peak_distance = (
            float(np.median(all_peak_distances)) if all_peak_distances else 0.0
        )

        def _mean_med(arr):
            if not arr:
                return 0.0, 0.0
            return float(np.mean(arr)), float(np.median(arr))

        mean_orig_err, median_orig_err = _mean_med(all_origin_errors)
        mean_orig_err_anchor, median_orig_err_anchor = _mean_med(all_origin_errors_anchor)
        mean_orig_err_vertex, median_orig_err_vertex = _mean_med(all_origin_errors_vertex)

        # Classification accuracy
        total_cls_correct = self.trainer.storage.history("val_cls_correct").total
        total_cls_total = self.trainer.storage.history("val_cls_total").total
        cls_accuracy = total_cls_correct / max(total_cls_total, 1)

        # Per-class classification accuracy
        per_class_acc = {}
        for name in ORIGIN_CLASS_NAMES:
            correct_key = f"val_cls_{name}_correct"
            total_key = f"val_cls_{name}_total"
            if correct_key in self.trainer.storage._history:
                c = self.trainer.storage.history(correct_key).total
                t = self.trainer.storage.history(total_key).total
                per_class_acc[name] = c / max(t, 1)
            else:
                per_class_acc[name] = float("nan")

        # --- Log results ---
        self.trainer.logger.info(
            "Val result: Loss {loss:.4f} | "
            "OriginRecall {recall:.4f} | "
            "Specificity {spec:.4f} | "
            "MeanPeakDist {mpd:.4f} | "
            "MedianPeakDist {medpd:.4f} | "
            "ClsAcc {cls:.4f}".format(
                loss=loss_avg,
                recall=origin_recall,
                spec=nonorigin_specificity,
                mpd=mean_peak_distance,
                medpd=median_peak_distance,
                cls=cls_accuracy,
            )
        )
        if all_origin_errors:
            self.trainer.logger.info(
                "  OriginPredErr: mean={oe:.4f} median={moe:.4f} "
                "(anchor-bound: mean={aem:.4f}/n={an}, "
                "vertex-target: mean={vem:.4f}/n={vn})".format(
                    oe=mean_orig_err, moe=median_orig_err,
                    aem=mean_orig_err_anchor, an=len(all_origin_errors_anchor),
                    vem=mean_orig_err_vertex, vn=len(all_origin_errors_vertex),
                )
            )
        for name, acc in per_class_acc.items():
            if not np.isnan(acc):
                self.trainer.logger.info(
                    "  cls_{name}_acc: {acc:.4f}".format(name=name, acc=acc)
                )
            else:
                self.trainer.logger.info(
                    "  cls_{name}_acc: N/A (no samples)".format(name=name)
                )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(
                "val/origin_recall", origin_recall, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/nonorigin_specificity", nonorigin_specificity, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mean_peak_distance", mean_peak_distance, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/median_peak_distance", median_peak_distance, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/cls_accuracy", cls_accuracy, current_epoch
            )
            if all_origin_errors:
                self.trainer.writer.add_scalar(
                    "val/origin_pred_error_mean", mean_orig_err, current_epoch
                )
                self.trainer.writer.add_scalar(
                    "val/origin_pred_error_median", median_orig_err, current_epoch
                )
                if all_origin_errors_anchor:
                    self.trainer.writer.add_scalar(
                        "val/origin_pred_error_anchor_bound_mean",
                        mean_orig_err_anchor, current_epoch,
                    )
                if all_origin_errors_vertex:
                    self.trainer.writer.add_scalar(
                        "val/origin_pred_error_vertex_target_mean",
                        mean_orig_err_vertex, current_epoch,
                    )
            for name, acc in per_class_acc.items():
                if not np.isnan(acc):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{name}_acc", acc, current_epoch
                    )

            if self.trainer.cfg.enable_wandb:
                log_dict = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/origin_recall": origin_recall,
                    "val/nonorigin_specificity": nonorigin_specificity,
                    "val/mean_peak_distance": mean_peak_distance,
                    "val/median_peak_distance": median_peak_distance,
                    "val/cls_accuracy": cls_accuracy,
                }
                if all_origin_errors:
                    log_dict["val/origin_pred_error_mean"] = mean_orig_err
                    log_dict["val/origin_pred_error_median"] = median_orig_err
                if all_origin_errors_anchor:
                    log_dict["val/origin_pred_error_anchor_bound_mean"] = (
                        mean_orig_err_anchor
                    )
                if all_origin_errors_vertex:
                    log_dict["val/origin_pred_error_vertex_target_mean"] = (
                        mean_orig_err_vertex
                    )
                for name, acc in per_class_acc.items():
                    if not np.isnan(acc):
                        log_dict[f"val/cls_{name}_acc"] = acc
                wandb.log(log_dict, step=wandb.run.step)

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<<< End Shower Origin Evaluation <<<<<<<<<<<<<<<<<"
        )

        # Save for CheckpointSaver (higher origin_recall = better)
        self.trainer.comm_info["current_metric_value"] = origin_recall
        self.trainer.comm_info["current_metric_name"] = "origin_recall"

        # Restore train mode when evaluating mid-epoch (step-based eval)
        if self.eval_freq > 0:
            self.trainer.model.train()

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                "origin_recall", self.trainer.best_metric_value
            )
        )
