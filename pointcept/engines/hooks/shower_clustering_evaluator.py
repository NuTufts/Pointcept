"""
Shower-Clustering Evaluator Hook

Validation metrics for the Mask2Former-style shower clustering model
(Phase 8.b of the design — see pointcept/docs/reference/shower_clustering_design.md).

Computes per-epoch (or per-N-step) val metrics:

    val/loss                          mean total loss across val events
    val/loss_*                        per-component losses (cls, mask, dice,
                                       origin, aux_mask_voxel, aux_mask_fragment)

    val/cls_accuracy                  matched pairs only — (q,k) where the
                                       argmax-predicted class equals GT class
    val/cls_{name}_acc                per-class accuracy on matched pairs

    val/mask_iou_mean                 spacepoint-scale mask IoU per matched
    val/mask_iou_median                pair, aggregated over events
    val/mask_iou_p25                  bottom-quartile (long-tail diagnostic)
    val/mask_iou_per_class_{name}     mean IoU split by GT class

    val/origin_error_cm_mean          |pred_origin - gt_origin|_2 in cm
    val/origin_error_cm_median         (matched pairs only, denormalized)

    val/matched_fraction              n_matched / n_gt across all events
    val/n_active_queries_mean         per-event count of queries whose argmax
                                       class != no_object (non-bg predictions)

Best-checkpoint metric: `mask_iou_mean` (higher is better).

The hook reuses the model's `loss_fn` with `return_matching=True` to recover
the Hungarian assignment without redoing the match. Per-pair mask IoU is
computed on the FULL spacepoint mask (binarized at logit > 0), not on the
sampled subset used for the loss — this gives an unbiased estimate of
per-shower mask quality.

Wandb is logged only when `cfg.enable_wandb` is True. Otherwise tensorboard
+ stdout only. Same gating as ShowerOriginEvaluator.
"""

from typing import Optional

import numpy as np
import torch

import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS

# Class names must match ShowerClusteringDataset / loss class indices.
ORIGIN_CLASS_NAMES = ["inside", "outside", "on_track", "ghost", "true_track"]


@HOOKS.register_module()
class ShowerClusteringEvaluator(HookBase):
    """Evaluator hook for ShowerClusteringMask2Former.

    Args:
        eval_freq: 0 = run after each epoch (default). >0 = run every N
                   global iterations.
        best_metric: which val metric drives `current_metric_value` for
                     the CheckpointSaver. Default `mask_iou_mean`.
        empty_cache: torch.cuda.empty_cache() before evaluating to make
                     room for the val forward + matched IoU buffers.
        log_per_event: log progress per val event (verbose; useful for
                       debugging the eval pipeline).
    """

    def __init__(
        self,
        eval_freq: int = 0,
        best_metric: str = "mask_iou_mean",
        empty_cache: bool = True,
        log_per_event: bool = False,
    ):
        self.eval_freq = int(eval_freq)
        self.best_metric = str(best_metric)
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
    # Per-event metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_iou_per_pair(
        pred_logits: torch.Tensor,   # (Q, N) raw logits
        gt_indices_per_inst,         # list[K] of LongTensors of GT spacepoints
        q_idx: np.ndarray,            # matched query indices
        k_idx: np.ndarray,            # matched GT instance indices
        n_spacepoints: int,
    ) -> torch.Tensor:
        """Compute per-matched-pair spacepoint-level mask IoU.

        Args:
            pred_logits: (Q, N) raw mask logits at the spacepoint scale.
            gt_indices_per_inst: list of K LongTensors, the GT spacepoint
                indices per instance.
            q_idx, k_idx: matched (q, k) pairs from Hungarian.
            n_spacepoints: N.
        Returns:
            (P,) FloatTensor — IoU per matched pair, P = len(q_idx).
        """
        device = pred_logits.device
        P = len(q_idx)
        if P == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)
        ious = torch.zeros(P, dtype=torch.float32, device=device)
        # Predicted mask: logit > 0 (sigmoid > 0.5)
        pred_bool = pred_logits > 0.0  # (Q, N) bool
        for p in range(P):
            q = int(q_idx[p])
            k = int(k_idx[p])
            gt_idx = gt_indices_per_inst[k]
            # GT mask as a sparse bool (N,)
            gt_mask = torch.zeros(n_spacepoints, dtype=torch.bool, device=device)
            gt_mask[gt_idx] = True
            pm = pred_bool[q]
            inter = (pm & gt_mask).sum().item()
            union = (pm | gt_mask).sum().item()
            if union == 0:
                ious[p] = 0.0
            else:
                ious[p] = inter / union
        return ious

    @staticmethod
    def _origin_error_cm(
        origin_pred_norm: torch.Tensor,   # (Q, 3) normalized
        origin_gt_norm: torch.Tensor,     # (K, 3) normalized
        q_idx: np.ndarray,
        k_idx: np.ndarray,
        coord_scale: float,
    ) -> torch.Tensor:
        """L2 distance per matched pair in detector cm."""
        if len(q_idx) == 0:
            return origin_pred_norm.new_zeros(0)
        q_t = torch.as_tensor(q_idx, dtype=torch.long,
                              device=origin_pred_norm.device)
        k_t = torch.as_tensor(k_idx, dtype=torch.long,
                              device=origin_gt_norm.device)
        diff_norm = origin_pred_norm[q_t] - origin_gt_norm[k_t]
        diff_cm = diff_norm * coord_scale
        return diff_cm.norm(dim=-1)

    @staticmethod
    def _per_event_slices(batch: dict):
        """Same logic as model._per_event_slices, used to recover per-event
        bookkeeping inside the evaluator (since model.forward in eval mode
        returns predictions only, but the evaluator needs the per-event GT
        and voxel/spacepoint counts)."""
        offsets = batch["offset"].detach().cpu().tolist()
        voxel_offsets = batch["voxel_offset"].detach().cpu().tolist()
        events = []
        prev_o = 0
        prev_v = 0
        for ei in range(len(offsets)):
            n_sp = offsets[ei] - prev_o
            n_vox = voxel_offsets[ei] - prev_v
            events.append({
                "n_sp": n_sp,
                "n_vox": n_vox,
                "sp_slice": slice(prev_o, offsets[ei]),
                "vox_slice": slice(prev_v, voxel_offsets[ei]),
                "voxel_id_offset": prev_v,
                "fragment_indices": batch["fragment_indices_per_event"][ei],
                "fragment_trackid": batch["fragment_trackid_per_event"][ei],
                "gt_instances": batch["gt_instances_per_event"][ei],
            })
            prev_o = offsets[ei]
            prev_v = voxel_offsets[ei]
        return events

    # ------------------------------------------------------------------
    # eval() main loop
    # ------------------------------------------------------------------

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Shower-Clustering Evaluation >>>>>>>>>>>>>>>>"
        )
        if self.empty_cache:
            torch.cuda.empty_cache()
        self.trainer.model.eval()
        model = getattr(self.trainer.model, "module", self.trainer.model)
        # coord_scale: the model stores voxel_size_norm = voxel_size_cm /
        # coord_scale, but not coord_scale itself. Recover from cfg (cfg
        # is set in the same config that builds the model, so this is
        # always consistent with the model's normalization).
        coord_scale = float(getattr(self.trainer.cfg, "coord_scale", 179.55))

        # Bookkeeping for aggregation
        loss_components_acc: dict = {}
        n_loss_steps = 0
        all_ious = []
        per_class_ious = {name: [] for name in ORIGIN_CLASS_NAMES}
        all_origin_errors = []
        cls_correct = 0
        cls_total = 0
        per_class_correct = {name: 0 for name in ORIGIN_CLASS_NAMES}
        per_class_total = {name: 0 for name in ORIGIN_CLASS_NAMES}
        n_active_queries_per_event = []
        n_matched_total = 0
        n_gt_total = 0

        no_object_class_id = int(model.loss_fn.no_object_class_id)

        with torch.no_grad():
            for i, input_dict in enumerate(self.trainer.val_loader):
                # Move tensors to GPU (Pointcept's loop only runs run_step
                # which we don't go through here; so we replicate the move).
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                # Forward in eval mode → predictions per event
                # Temporarily put model in train mode to reuse loss_fn
                # via the SAME forward path as the model's training call,
                # so the per-event slicing & device moves match.
                self.trainer.model.train()
                output = self.trainer.model(input_dict)
                self.trainer.model.eval()
                # output["loss"] etc. — same scalars as during training
                for k, v in output.items():
                    if k == "loss" or k.startswith("loss_"):
                        loss_components_acc[k] = (
                            loss_components_acc.get(k, 0.0) + v.item()
                        )
                n_loss_steps += 1

                # Re-run forward with grad disabled to get the per-event
                # decoder predictions for matching + per-pair metrics.
                # (Easier than threading a "give me predictions too" path
                # through the training-mode forward.)
                events = self._per_event_slices(input_dict)
                sp_feat_all = model.encode_event(input_dict)
                device = sp_feat_all.device

                for ei, e in enumerate(events):
                    sp = e["sp_slice"]
                    vx = e["vox_slice"]
                    sp_feat = sp_feat_all[sp]
                    coord_norm = input_dict["coord_norm"][sp]
                    strength = input_dict["feat"][sp, 3:6]
                    voxel_id_local = (
                        input_dict["voxel_id"][sp] - e["voxel_id_offset"]
                    )
                    voxel_keys = input_dict["voxel_keys"][vx]
                    fragment_indices = [
                        idx.to(device, non_blocking=True)
                        for idx in e["fragment_indices"]
                    ]
                    fragment_trackid = e["fragment_trackid"].to(
                        device, non_blocking=True
                    )

                    tokens = model.tokenizer(
                        sp_feat=sp_feat,
                        coord_norm=coord_norm,
                        strength=strength,
                        voxel_id=voxel_id_local,
                        voxel_keys=voxel_keys,
                        n_voxels=e["n_vox"],
                        fragment_indices=fragment_indices,
                    )
                    decoder_out = model.decoder(
                        voxel_tokens=tokens["voxel_tokens"],
                        voxel_coords=tokens["voxel_coords"],
                        fragment_tokens=tokens["fragment_tokens"],
                        fragment_coords=tokens["fragment_coords"],
                        spacepoint_tokens=tokens["spacepoint_tokens"],
                        spacepoint_coords=tokens["spacepoint_coords"],
                    )

                    # Use loss_fn's matcher to get the assignment for this
                    # event (with return_matching=True, it also gives GT
                    # tensors back, saving us from rebuilding them).
                    loss_dict = model.loss_fn(
                        decoder_output=decoder_out,
                        gt_instances=e["gt_instances"],
                        voxel_id=voxel_id_local,
                        n_voxels=e["n_vox"],
                        n_spacepoints=e["n_sp"],
                        fragment_trackid=fragment_trackid,
                        return_matching=True,
                    )
                    q_idx = loss_dict["q_idx"]
                    k_idx = loss_dict["k_idx"]
                    gt_classes = loss_dict["gt_classes"]
                    gt_origin = loss_dict["gt_origin"]
                    gt_truth_indices = loss_dict["gt_truth_indices"]
                    K_event = gt_classes.shape[0]

                    n_matched_total += len(q_idx)
                    n_gt_total += K_event

                    final = decoder_out["final"]
                    cls_logits = final["class_logits"]      # (Q, C)
                    origin_pred = final["origin"]            # (Q, 3)
                    sp_mask = final["mask_logits"]["spacepoint"]  # (Q, N)

                    # Class accuracy on matched pairs
                    if len(q_idx) > 0:
                        q_t = torch.as_tensor(q_idx, dtype=torch.long,
                                              device=device)
                        k_t = torch.as_tensor(k_idx, dtype=torch.long,
                                              device=device)
                        pred_class = cls_logits[q_t].argmax(dim=-1)  # (P,)
                        gt_class = gt_classes[k_t]
                        eq = (pred_class == gt_class)
                        cls_correct += int(eq.sum().item())
                        cls_total += int(eq.numel())
                        for p in range(eq.numel()):
                            gc = int(gt_class[p].item())
                            name = ORIGIN_CLASS_NAMES[gc] if 0 <= gc < len(ORIGIN_CLASS_NAMES) else None
                            if name is not None:
                                per_class_total[name] += 1
                                if bool(eq[p].item()):
                                    per_class_correct[name] += 1

                    # Mask IoU per matched pair (full spacepoint scale)
                    if len(q_idx) > 0:
                        ious = self._mask_iou_per_pair(
                            pred_logits=sp_mask,
                            gt_indices_per_inst=gt_truth_indices,
                            q_idx=q_idx, k_idx=k_idx,
                            n_spacepoints=e["n_sp"],
                        )
                        all_ious.extend(ious.cpu().tolist())
                        # Per class
                        for p in range(len(q_idx)):
                            gc = int(gt_classes[int(k_idx[p])].item())
                            name = ORIGIN_CLASS_NAMES[gc] if 0 <= gc < len(ORIGIN_CLASS_NAMES) else None
                            if name is not None:
                                per_class_ious[name].append(float(ious[p].item()))

                    # Origin error in cm
                    if len(q_idx) > 0:
                        errs = self._origin_error_cm(
                            origin_pred_norm=origin_pred,
                            origin_gt_norm=gt_origin,
                            q_idx=q_idx, k_idx=k_idx,
                            coord_scale=coord_scale,
                        )
                        all_origin_errors.extend(errs.cpu().tolist())

                    # Active-queries: queries whose argmax class is not no_object
                    pred_classes_all = cls_logits.argmax(dim=-1)  # (Q,)
                    n_active = int((pred_classes_all != no_object_class_id).sum().item())
                    n_active_queries_per_event.append(n_active)

                    if self.log_per_event:
                        ev_iou = float(np.mean(all_ious[-len(q_idx):])) \
                            if len(q_idx) > 0 else float("nan")
                        self.trainer.logger.info(
                            f"  Val event {i}.{ei}: K={K_event} matched={len(q_idx)} "
                            f"cls_acc={cls_correct/max(cls_total,1):.3f} "
                            f"ev_iou={ev_iou:.3f}"
                        )

                self.trainer.logger.info(
                    f"Val: [{i+1}/{len(self.trainer.val_loader)}] "
                    f"events_so_far={n_loss_steps} "
                    f"loss={loss_components_acc.get('loss', 0.0)/n_loss_steps:.4f}"
                )

        # ---- aggregate -----------------------------------------------------
        # Loss averages
        loss_avg = {k: v / max(n_loss_steps, 1)
                    for k, v in loss_components_acc.items()}

        def _meanmed(arr):
            if not arr:
                return float("nan"), float("nan"), float("nan")
            return (float(np.mean(arr)),
                    float(np.median(arr)),
                    float(np.percentile(arr, 25)))

        iou_mean, iou_median, iou_p25 = _meanmed(all_ious)
        oe_mean, oe_median, _ = _meanmed(all_origin_errors)
        cls_acc = cls_correct / max(cls_total, 1)
        per_class_acc = {
            name: per_class_correct[name] / max(per_class_total[name], 1)
            for name in ORIGIN_CLASS_NAMES
        }
        per_class_iou_mean = {}
        for name in ORIGIN_CLASS_NAMES:
            arr = per_class_ious[name]
            per_class_iou_mean[name] = (float(np.mean(arr))
                                        if arr else float("nan"))
        matched_fraction = n_matched_total / max(n_gt_total, 1)
        n_active_mean = (float(np.mean(n_active_queries_per_event))
                         if n_active_queries_per_event else 0.0)

        # ---- log ----------------------------------------------------------
        log_lines = [
            f"Val result: loss {loss_avg.get('loss', 0.0):.4f} "
            f"| cls_acc {cls_acc:.4f} "
            f"| iou_mean {iou_mean:.4f} (med {iou_median:.4f}, p25 {iou_p25:.4f}) "
            f"| origin_err_cm mean {oe_mean:.2f} med {oe_median:.2f} "
            f"| matched {n_matched_total}/{n_gt_total} ({matched_fraction:.3f}) "
            f"| n_active_q_avg {n_active_mean:.1f}"
        ]
        for name in ORIGIN_CLASS_NAMES:
            n_t = per_class_total[name]
            if n_t > 0:
                log_lines.append(
                    f"  cls={name}: acc {per_class_acc[name]:.4f} "
                    f"({per_class_correct[name]}/{n_t})  "
                    f"iou_mean {per_class_iou_mean[name]:.4f} "
                    f"(n={len(per_class_ious[name])})"
                )
        for line in log_lines:
            self.trainer.logger.info(line)

        # ---- write to tensorboard / wandb --------------------------------
        scalars = {
            "val/cls_accuracy": cls_acc,
            "val/mask_iou_mean": iou_mean,
            "val/mask_iou_median": iou_median,
            "val/mask_iou_p25": iou_p25,
            "val/origin_error_cm_mean": oe_mean,
            "val/origin_error_cm_median": oe_median,
            "val/matched_fraction": matched_fraction,
            "val/n_active_queries_mean": n_active_mean,
        }
        for k, v in loss_avg.items():
            scalars[f"val/{k}"] = v
        for name in ORIGIN_CLASS_NAMES:
            if per_class_total[name] > 0:
                scalars[f"val/cls_{name}_acc"] = per_class_acc[name]
                scalars[f"val/mask_iou_{name}"] = per_class_iou_mean[name]

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            for k, v in scalars.items():
                if v == v:  # skip NaN
                    self.trainer.writer.add_scalar(k, v, current_epoch)
            if getattr(self.trainer.cfg, "enable_wandb", False):
                try:
                    import wandb
                    wandb_payload = {"Epoch": current_epoch}
                    for k, v in scalars.items():
                        if v == v:
                            wandb_payload[k] = v
                    wandb.log(wandb_payload, step=wandb.run.step)
                except ImportError:
                    pass

        # Park val/loss in comm_info so epoch-level hooks (e.g.
        # LREpochScheduler driving FlatWithDecayLR's plateau trigger) can
        # read it after this evaluator runs. NaN-safe: pass through as-is
        # so the consumer can decide whether to skip the update.
        self.trainer.comm_info["val_loss"] = float(loss_avg.get("loss", float("nan")))

        # ---- best-checkpoint metric --------------------------------------
        best_value = scalars.get(f"val/{self.best_metric}", None)
        if best_value is None:
            # tolerate metric name without 'val/' prefix
            best_value = scalars.get(self.best_metric, None)
        if best_value is None:
            self.trainer.logger.warning(
                f"best_metric '{self.best_metric}' not found in val scalars; "
                "CheckpointSaver won't track best model"
            )
        else:
            self.trainer.comm_info["current_metric_value"] = float(best_value)
            self.trainer.comm_info["current_metric_name"] = self.best_metric

        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End Shower-Clustering Evaluation <<<<<<<<<<<<<<<<"
        )
