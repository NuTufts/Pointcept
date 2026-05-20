"""
LArFormerSlicerEvaluator — validation-metric hook for the cascaded slicer.

Computes per-epoch val metrics for `CascadedSlicer` (and standalone
`LArFormer` with Mask2Former-style queries — the class is generic over
"models whose eval-mode forward produces per-event predictions with an
`eval_loss` field"). Per epoch:

    val/loss + val/loss_*           mean total + per-component slicer losses
    val/cls_accuracy                matched-pair argmax class accuracy
    val/cls_{name}_acc              per-class breakdown
    val/mask_iou_mean / median / p25  spacepoint-mask IoU per matched pair
    val/mask_iou_{name}             per-class mean IoU
    val/nu_recall                   per-event: did a query match the nu GT
    val/nu_purity                   of queries argmax-predicting nu, what
                                     fraction overlap a nu GT instance
    val/matched_fraction            n_matched / n_gt
    val/n_active_queries_mean       queries predicting non-no_object
    val/per_level_cls_acc_{name}    per-token cls accuracy where declared
    val/deghost_keep_frac_mean      CascadedSlicer cascade-health monitor
                                     (silently 0 for standalone LArFormer)

Best-checkpoint metric (default): `nu_mIoU` for slicer configs that name
class 0 = nu. Override via the hook's `best_metric` arg.

The hook assumes the model's eval-mode forward attaches `eval_loss` to
each per-event prediction dict when `gt_instances_per_event` is in the
data_dict (LArFormer.forward does this; CascadedSlicer passes through).
This avoids the double-forward pattern the legacy ShowerClusteringEvaluator
uses — important for the cascade where each forward is two backbones.

Registers as `@HOOKS.register_module()` on import. Configs trigger the
registration via:

    from pointcept.models.LArFormer import evaluator as _eval_module
    del _eval_module
"""

from typing import List, Optional, Sequence

import numpy as np
import torch

from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase


@HOOKS.register_module()
class LArFormerSlicerEvaluator(HookBase):
    """Validation metrics hook for LArFormer / CascadedSlicer models.

    Args:
        eval_freq:       0 = run after each epoch (default). >0 = run every
                          N global iterations.
        best_metric:     metric name (without `val/`) the CheckpointSaver
                          tracks for "best so far". Default `nu_mIoU` —
                          name of the matched-pair IoU mean restricted to
                          the nu-class slot.
        class_names:     list of class names indexed by the query-class
                          slots. Last slot is implicitly no_object and is
                          excluded from per-class metrics. Default
                          ['nu', 'cosmic'] (slicer convention from
                          larformer-slicer-v0.py).
        nu_class_id:     index of the "nu" class in `class_names`. Default
                          0. Used for nu_recall / nu_purity / nu_mIoU.
        empty_cache:     torch.cuda.empty_cache() before evaluating.
        log_per_event:   verbose per-event log (debug).
    """

    def __init__(
        self,
        eval_freq: int = 0,
        best_metric: str = "nu_mIoU",
        class_names: Optional[Sequence[str]] = None,
        nu_class_id: int = 0,
        empty_cache: bool = True,
        log_per_event: bool = False,
    ):
        self.eval_freq = int(eval_freq)
        self.best_metric = str(best_metric)
        self.class_names = (list(class_names) if class_names is not None
                            else ["nu", "cosmic"])
        self.nu_class_id = int(nu_class_id)
        self.empty_cache = bool(empty_cache)
        self.log_per_event = bool(log_per_event)

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def before_train(self):
        if (self.trainer.writer is not None
                and getattr(self.trainer.cfg, "enable_wandb", False)):
            try:
                import wandb
                wandb.define_metric("val/*", step_metric="Epoch")
            except ImportError:
                self.trainer.logger.warning(
                    "wandb requested but not importable; skipping define_metric"
                )

    def after_step(self):
        if (self.trainer.cfg.evaluate
                and self.eval_freq > 0
                and self.trainer.val_loader is not None):
            iter_per_epoch = len(self.trainer.train_loader)
            global_iter = (
                self.trainer.comm_info["iter"]
                + iter_per_epoch * self.trainer.epoch
            )
            if (global_iter + 1) % self.eval_freq == 0:
                self.eval()

    def after_epoch(self):
        if (self.trainer.cfg.evaluate
                and self.eval_freq == 0
                and self.trainer.val_loader is not None):
            self.eval()

    # ------------------------------------------------------------------
    # Per-pair metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_iou_per_pair(
        pred_logits: torch.Tensor,
        gt_truth_indices: List[torch.Tensor],
        q_idx: np.ndarray,
        k_idx: np.ndarray,
    ) -> torch.Tensor:
        """Spacepoint-scale IoU per matched (q, k) pair.

        Binarize prediction at logit > 0 (sigmoid > 0.5). GT mask is built
        from the per-instance truth_indices list. Output is (P,) on the
        same device as pred_logits.
        """
        device = pred_logits.device
        P = len(q_idx)
        if P == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)
        n_sp = pred_logits.shape[1]
        ious = torch.zeros(P, dtype=torch.float32, device=device)
        pred_bool = pred_logits > 0.0
        for p in range(P):
            q = int(q_idx[p]); k = int(k_idx[p])
            gt_idx = gt_truth_indices[k].to(device)
            gt_mask = torch.zeros(n_sp, dtype=torch.bool, device=device)
            if gt_idx.numel() > 0:
                gt_mask[gt_idx] = True
            pm = pred_bool[q]
            inter = (pm & gt_mask).sum().item()
            union = (pm | gt_mask).sum().item()
            ious[p] = (inter / union) if union > 0 else 0.0
        return ious

    # ------------------------------------------------------------------
    # Main eval loop
    # ------------------------------------------------------------------

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start LArFormerSlicer Evaluation >>>>>>>>>>>>>>>>"
        )
        if self.empty_cache:
            torch.cuda.empty_cache()

        self.trainer.model.eval()
        model = getattr(self.trainer.model, "module", self.trainer.model)
        no_object_class_id = self._infer_no_object_id(model)

        # Bookkeeping
        loss_acc: dict = {}
        n_loss_events = 0
        all_ious: List[float] = []
        per_class_ious = {name: [] for name in self.class_names}
        cls_correct = 0
        cls_total = 0
        per_class_correct = {name: 0 for name in self.class_names}
        per_class_total = {name: 0 for name in self.class_names}
        n_active_per_event: List[int] = []
        n_matched_total = 0
        n_gt_total = 0
        nu_recall_events = []          # 1 if any matched pair has gt=nu
        nu_purity_num = 0              # queries argmax-nu AND overlap nu GT
        nu_purity_den = 0              # queries argmax-nu (regardless)
        per_level_cls_correct: dict = {}
        per_level_cls_total: dict = {}
        deghost_keep_fracs: List[float] = []

        with torch.no_grad():
            for batch_i, input_dict in enumerate(self.trainer.val_loader):
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                # Run model in eval mode; the eval-with-GT branch attaches
                # `eval_loss` to each per-event prediction dict.
                output = self.trainer.model(input_dict)
                preds = output.get("predictions", [])
                if isinstance(output.get("deghost_keep_frac"), (int, float)):
                    deghost_keep_fracs.append(float(output["deghost_keep_frac"]))
                elif isinstance(output.get("deghost_keep_frac"), torch.Tensor):
                    deghost_keep_fracs.append(
                        float(output["deghost_keep_frac"].item())
                    )

                for ev_pred in preds:
                    eval_loss = ev_pred.get("eval_loss")
                    if eval_loss is None:
                        # Eval-with-GT was disabled (no gt_instances in
                        # this event); skip metric computation for it.
                        continue
                    n_loss_events += 1
                    for k, v in eval_loss.items():
                        if (isinstance(v, torch.Tensor) and v.dim() == 0
                                and k not in (
                                    "n_matched", "n_gt_instances",
                                )):
                            # Carry per-component losses (cls, mask_primary,
                            # dice_primary, aux_mask_*, cls_*, origin, total).
                            loss_acc[k] = loss_acc.get(k, 0.0) + float(v.item())

                    q_idx = eval_loss["q_idx"]
                    k_idx = eval_loss["k_idx"]
                    gt_classes = eval_loss["gt_classes"]
                    gt_truth_indices = eval_loss["gt_truth_indices"]
                    K = int(gt_classes.shape[0])
                    n_matched_total += len(q_idx)
                    n_gt_total += K

                    cls_logits = ev_pred.get("class_logits")
                    sp_mask = ev_pred.get("mask_logits", {}).get("spacepoint")
                    if cls_logits is None or sp_mask is None:
                        continue

                    device = cls_logits.device
                    q_t = (torch.as_tensor(q_idx, dtype=torch.long, device=device)
                           if len(q_idx) > 0 else None)
                    k_t = (torch.as_tensor(k_idx, dtype=torch.long, device=device)
                           if len(k_idx) > 0 else None)
                    pred_classes_all = cls_logits.argmax(dim=-1)

                    # Matched-pair class accuracy + per-class breakdown
                    nu_in_event = False
                    if q_t is not None:
                        pred_class = pred_classes_all[q_t]
                        gt_class = gt_classes[k_t]
                        eq = (pred_class == gt_class)
                        cls_correct += int(eq.sum().item())
                        cls_total += int(eq.numel())
                        for p_i in range(eq.numel()):
                            gc = int(gt_class[p_i].item())
                            if gc == self.nu_class_id:
                                nu_in_event = True
                            name = (self.class_names[gc]
                                    if 0 <= gc < len(self.class_names)
                                    else None)
                            if name is not None:
                                per_class_total[name] += 1
                                if bool(eq[p_i].item()):
                                    per_class_correct[name] += 1
                    nu_recall_events.append(1 if nu_in_event else 0)

                    # Per-pair mask IoU (binarized at logit > 0)
                    if q_t is not None:
                        ious = self._mask_iou_per_pair(
                            sp_mask, gt_truth_indices, q_idx, k_idx,
                        )
                        ious_list = ious.cpu().tolist()
                        all_ious.extend(ious_list)
                        for p_i in range(len(q_idx)):
                            gc = int(gt_classes[int(k_idx[p_i])].item())
                            name = (self.class_names[gc]
                                    if 0 <= gc < len(self.class_names)
                                    else None)
                            if name is not None:
                                per_class_ious[name].append(ious_list[p_i])

                    # Nu purity: queries predicting nu, intersect with any nu GT
                    nu_query_mask = (pred_classes_all == self.nu_class_id)
                    n_nu_q = int(nu_query_mask.sum().item())
                    nu_purity_den += n_nu_q
                    if n_nu_q > 0 and K > 0:
                        # Build a flat nu-GT mask: union of truth_indices of
                        # all GT instances whose class is nu.
                        n_sp = sp_mask.shape[1]
                        nu_gt_mask = torch.zeros(n_sp, dtype=torch.bool,
                                                 device=device)
                        for kk in range(K):
                            if int(gt_classes[kk].item()) == self.nu_class_id:
                                ti = gt_truth_indices[kk].to(device)
                                if ti.numel() > 0:
                                    nu_gt_mask[ti] = True
                        # A predicting-nu query "hits" the nu GT if any of
                        # its predicted SPs (logit>0) overlap nu_gt_mask.
                        nu_q_indices = nu_query_mask.nonzero(as_tuple=False).squeeze(-1)
                        for q in nu_q_indices.cpu().tolist():
                            pm = sp_mask[q] > 0.0
                            if (pm & nu_gt_mask).any():
                                nu_purity_num += 1

                    # Active queries (not no_object)
                    n_active_per_event.append(
                        int((pred_classes_all != no_object_class_id).sum().item())
                    )

                    # Per-level cls accuracy (e.g. voxel_10cm cls head)
                    per_level_cls = ev_pred.get("per_level_cls", {})
                    per_level_gt = eval_loss.get("per_level_gt_mask")  # not target
                    # We need cls_target — recover from the loss's helper
                    # by reading `per_level_gt` keys; but the per_level_gt
                    # we stashed in eval_loss is the mask dict, not cls.
                    # Skip per-level cls metrics for now (cheap to add
                    # later when the target dict is also surfaced).
                    _ = per_level_cls; _ = per_level_gt  # explicit unused

                    if self.log_per_event:
                        self.trainer.logger.info(
                            f"  Val ev {batch_i}/{n_loss_events}: K={K} "
                            f"matched={len(q_idx)} "
                            f"cls_acc={cls_correct/max(cls_total,1):.3f} "
                            f"n_nu_q={n_nu_q}"
                        )

                self.trainer.logger.info(
                    f"Val: [{batch_i+1}/{len(self.trainer.val_loader)}] "
                    f"events={n_loss_events} "
                    f"loss={loss_acc.get('total', 0.0)/max(n_loss_events,1):.4f}"
                )

        scalars = self._aggregate(
            loss_acc=loss_acc, n_loss_events=n_loss_events,
            all_ious=all_ious, per_class_ious=per_class_ious,
            cls_correct=cls_correct, cls_total=cls_total,
            per_class_correct=per_class_correct,
            per_class_total=per_class_total,
            n_matched_total=n_matched_total, n_gt_total=n_gt_total,
            nu_recall_events=nu_recall_events,
            nu_purity_num=nu_purity_num, nu_purity_den=nu_purity_den,
            n_active_per_event=n_active_per_event,
            deghost_keep_fracs=deghost_keep_fracs,
        )
        self._log_and_publish(scalars)

    def _infer_no_object_id(self, model) -> int:
        """Resolve no_object class id from the model's loss config.

        Works for both standalone LArFormer (model.loss_fn) and
        CascadedSlicer (model.slicer.loss_fn).
        """
        if hasattr(model, "loss_fn"):
            return int(model.loss_fn.no_object_class_id)
        if hasattr(model, "slicer") and hasattr(model.slicer, "loss_fn"):
            return int(model.slicer.loss_fn.no_object_class_id)
        # Fallback: last class slot
        return len(self.class_names)

    # ------------------------------------------------------------------
    # Aggregation + logging
    # ------------------------------------------------------------------

    def _aggregate(self, **k) -> dict:
        def meanmed(arr):
            if not arr:
                return float("nan"), float("nan"), float("nan")
            return (float(np.mean(arr)),
                    float(np.median(arr)),
                    float(np.percentile(arr, 25)))

        loss_avg = {kk: v / max(k["n_loss_events"], 1)
                    for kk, v in k["loss_acc"].items()}

        iou_mean, iou_median, iou_p25 = meanmed(k["all_ious"])
        per_class_iou_mean = {
            name: (float(np.mean(arr)) if arr else float("nan"))
            for name, arr in k["per_class_ious"].items()
        }
        cls_acc = k["cls_correct"] / max(k["cls_total"], 1)
        per_class_acc = {
            name: k["per_class_correct"][name] / max(k["per_class_total"][name], 1)
            for name in self.class_names
        }
        matched_fraction = k["n_matched_total"] / max(k["n_gt_total"], 1)
        n_active_mean = (float(np.mean(k["n_active_per_event"]))
                         if k["n_active_per_event"] else 0.0)
        nu_recall = (float(np.mean(k["nu_recall_events"]))
                     if k["nu_recall_events"] else float("nan"))
        nu_purity = (k["nu_purity_num"] / k["nu_purity_den"]
                     if k["nu_purity_den"] > 0 else float("nan"))
        deghost_keep_mean = (float(np.mean(k["deghost_keep_fracs"]))
                             if k["deghost_keep_fracs"] else float("nan"))

        scalars = {
            "val/cls_accuracy": cls_acc,
            "val/mask_iou_mean": iou_mean,
            "val/mask_iou_median": iou_median,
            "val/mask_iou_p25": iou_p25,
            "val/matched_fraction": matched_fraction,
            "val/n_active_queries_mean": n_active_mean,
            "val/nu_recall": nu_recall,
            "val/nu_purity": nu_purity,
            "val/deghost_keep_frac_mean": deghost_keep_mean,
        }
        for kk, v in loss_avg.items():
            scalars[f"val/loss_{kk}" if kk != "total" else "val/loss"] = v
        for name in self.class_names:
            if k["per_class_total"][name] > 0:
                scalars[f"val/cls_{name}_acc"] = per_class_acc[name]
                scalars[f"val/mask_iou_{name}"] = per_class_iou_mean[name]
        # Convenience alias: nu_mIoU = mask_iou_nu (so best_metric
        # `nu_mIoU` works without users digging through metric names).
        if f"val/mask_iou_{self.class_names[self.nu_class_id]}" in scalars:
            scalars["val/nu_mIoU"] = scalars[
                f"val/mask_iou_{self.class_names[self.nu_class_id]}"
            ]
        return scalars

    def _log_and_publish(self, scalars: dict) -> None:
        # stdout summary
        lines = [
            f"Val: loss {scalars.get('val/loss', float('nan')):.4f} "
            f"| cls_acc {scalars['val/cls_accuracy']:.4f} "
            f"| iou_mean {scalars['val/mask_iou_mean']:.4f} "
            f"(med {scalars['val/mask_iou_median']:.4f}, "
            f"p25 {scalars['val/mask_iou_p25']:.4f}) "
            f"| matched_frac {scalars['val/matched_fraction']:.3f} "
            f"| n_active_q_avg {scalars['val/n_active_queries_mean']:.1f} "
            f"| nu_recall {scalars['val/nu_recall']:.3f} "
            f"| nu_purity {scalars['val/nu_purity']:.3f} "
            f"| keep_frac {scalars['val/deghost_keep_frac_mean']:.3f}"
        ]
        for name in self.class_names:
            acc_k = f"val/cls_{name}_acc"
            iou_k = f"val/mask_iou_{name}"
            if acc_k in scalars and iou_k in scalars:
                lines.append(f"  cls={name}: acc {scalars[acc_k]:.4f}  "
                             f"iou {scalars[iou_k]:.4f}")
        for line in lines:
            self.trainer.logger.info(line)

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            for k, v in scalars.items():
                if v == v:  # skip NaN
                    self.trainer.writer.add_scalar(k, v, current_epoch)
            if getattr(self.trainer.cfg, "enable_wandb", False):
                try:
                    import wandb
                    payload = {"Epoch": current_epoch}
                    for k, v in scalars.items():
                        if v == v:
                            payload[k] = v
                    wandb.log(payload, step=wandb.run.step)
                except ImportError:
                    pass

        # Park val/loss for epoch-level hooks (e.g. LR plateau schedulers)
        self.trainer.comm_info["val_loss"] = float(
            scalars.get("val/loss", float("nan"))
        )

        # Best-checkpoint metric. Accept either "nu_mIoU" or "val/nu_mIoU".
        best = scalars.get(f"val/{self.best_metric}",
                           scalars.get(self.best_metric))
        if best is None:
            self.trainer.logger.warning(
                f"best_metric '{self.best_metric}' not found in val scalars; "
                f"CheckpointSaver won't track best model"
            )
        else:
            self.trainer.comm_info["current_metric_value"] = float(best)
            self.trainer.comm_info["current_metric_name"] = self.best_metric

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End LArFormerSlicer Evaluation <<<<<<<<<<<<<<<<"
        )
