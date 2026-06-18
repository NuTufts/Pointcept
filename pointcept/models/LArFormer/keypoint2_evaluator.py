"""LArFormerLevelKeypointEvaluator — val metrics for the attempt-2 Phase-1
slice-level dense keypoint model (`level_keypoint_heads`).

Standalone hook (does NOT subclass the particle/slicer evaluator, whose eval
loop assumes query class/mask/origin outputs this model doesn't have). It runs
its own val pass over `pred["level_kp"][name]` and logs:

  OBJECT head (e.g. ptv3_dec2, the mixed-query source):
    val/object_ap            average precision of the per-token soft-object
                             classifier (GT = pooled max(kpscores[:,kp_types])
                             over a token's member SPs, binarized at
                             target_pos_thresh). Threshold-free headline.
    val/object_P, val/object_R   precision/recall at `score_thresh`.
    val/object_pos_frac      fraction of tokens that are GT-object (prevalence).

  NU-VERTEX head (spacepoint level):
    val/nu_vertex_res_cm_{median,mean}   ||pred - GT|| * coord_scale, where the
                             pred vertex is the score-weighted centroid of SPs
                             above `nu_decode_thresh` and GT is the type-0
                             mckeypoint (same recentered coord_norm frame, so a
                             plain scaled distance is exact — see the dataset's
                             recenter of mckeypoints alongside coord_norm).
    val/nu_vertex_recall     fraction of GT-bearing events whose decoded vertex
                             lands within `nu_recall_cm`.

`best_metric` (default val/object_ap) is published for CheckpointSaver.

Registers on import; configs trigger via:
    from pointcept.models.LArFormer import keypoint2_evaluator as _m
    del _m
"""
from typing import Sequence

import numpy as np
import torch

from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase

from lartpc_data_prep.keypoint_labels import KPTYPE_NU_VERTEX
from .keypoint2_particle import KP_CLS_START, KP_CLS_END


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (no sklearn dep). scores (M,)
    float, labels (M,) bool/0-1. Returns 0 if no positives."""
    n_pos = int(labels.sum())
    if n_pos == 0 or scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    lab = labels[order].astype(np.float64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1.0 - lab)
    precision = tp / np.clip(tp + fp, 1.0, None)
    recall = tp / float(n_pos)
    # AP = sum over thresholds of (R_i - R_{i-1}) * P_i
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


@HOOKS.register_module()
class LArFormerLevelKeypointEvaluator(HookBase):

    def __init__(
        self,
        object_head: str = "object",
        nu_vertex_head: str = "nu_vertex",
        object_kp_types: Sequence[int] = (1, 2, 3, 4, 5),
        target_pos_thresh: float = 0.5,
        score_thresh: float = 0.5,
        nu_decode_thresh: float = 0.3,
        nu_recall_cm: float = 3.0,
        coord_scale: float = 179.55,
        best_metric: str = "val/object_ap",
        # per-particle keypoint thresholds (cm); 1cm is the precision target.
        pkp_thresholds_cm: Sequence[float] = (1.0, 3.0, 10.0),
        eval_freq: int = 0,
        empty_cache: bool = True,
    ):
        self.object_head = str(object_head)
        self.nu_vertex_head = str(nu_vertex_head)
        self.object_kp_types = [int(t) for t in object_kp_types]
        self.target_pos_thresh = float(target_pos_thresh)
        self.score_thresh = float(score_thresh)
        self.nu_decode_thresh = float(nu_decode_thresh)
        self.nu_recall_cm = float(nu_recall_cm)
        self.coord_scale = float(coord_scale)
        self.best_metric = str(best_metric)
        self.pkp_thresholds_cm = [float(t) for t in pkp_thresholds_cm]
        self.eval_freq = int(eval_freq)
        self.empty_cache = bool(empty_cache)

    # ------------------------------------------------------------------

    def after_step(self):
        if (self.trainer.cfg.evaluate and self.eval_freq > 0
                and self.trainer.val_loader is not None):
            iters = len(self.trainer.train_loader)
            g = self.trainer.comm_info["iter"] + iters * self.trainer.epoch
            if (g + 1) % self.eval_freq == 0:
                self.eval()

    def after_epoch(self):
        if (self.trainer.cfg.evaluate and self.eval_freq == 0
                and self.trainer.val_loader is not None):
            self.eval()

    # ------------------------------------------------------------------

    @staticmethod
    def _pool_max_np(per_sp, sp_to_level_id, n_tokens):
        tok = np.zeros(int(n_tokens), dtype=np.float64)
        if per_sp.size == 0 or n_tokens == 0:
            return tok
        valid = sp_to_level_id >= 0
        if valid.any():
            np.maximum.at(tok, sp_to_level_id[valid].astype(np.int64),
                          per_sp[valid].astype(np.float64))
        return tok

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start LArFormer LevelKeypoint Eval >>>>>>>>>>>>>>>>")
        if self.empty_cache:
            torch.cuda.empty_cache()
        self.trainer.model.eval()

        obj_scores, obj_labels = [], []
        nu_res_cm = []
        nu_n_gt_events = 0
        nu_n_recovered = 0
        pkp_start_err, pkp_end_err = [], []   # per-particle localization (cm)

        with torch.no_grad():
            for input_dict in self.trainer.val_loader:
                for k in input_dict:
                    if isinstance(input_dict[k], torch.Tensor):
                        input_dict[k] = input_dict[k].cuda(non_blocking=True)
                preds = self.trainer.model(input_dict).get("predictions", [])
                offset = input_dict["offset"]
                kpscores = input_dict.get("kpscores")
                pos_list = input_dict.get("mckeypoints_pos_norm_per_event")
                typ_list = input_dict.get("mckeypoints_type_per_event")
                gti_list = input_dict.get("gt_instances_per_event")

                for ei, ev in enumerate(preds):
                    lk = ev.get("level_kp", {})
                    prev = int(offset[ei - 1].item()) if ei > 0 else 0
                    cur = int(offset[ei].item())

                    # ---- OBJECT head: token classification AP + P/R ----
                    oh = lk.get(self.object_head)
                    if oh is not None and kpscores is not None:
                        score = oh["score"].detach().cpu().numpy()
                        s2l = oh["sp_to_level_id"].detach().cpu().numpy()
                        kps_ev = kpscores[prev:cur][:, self.object_kp_types]
                        per_sp = kps_ev.amax(dim=1).detach().cpu().numpy()
                        tgt = self._pool_max_np(per_sp, s2l, score.shape[0])
                        obj_scores.append(score)
                        obj_labels.append(tgt > self.target_pos_thresh)

                    # ---- NU-VERTEX head: decode + resolution ----
                    nh = lk.get(self.nu_vertex_head)
                    if (nh is not None and pos_list is not None
                            and typ_list is not None and ei < len(pos_list)):
                        gt_pos = self._np(pos_list[ei])
                        gt_typ = self._np(typ_list[ei]).astype(np.int64)
                        gt_nu = gt_pos[gt_typ == KPTYPE_NU_VERTEX]
                        if gt_nu.shape[0] > 0:
                            nu_n_gt_events += 1
                            sc = nh["score"].detach().cpu().numpy()
                            cn = nh["coords"].detach().cpu().numpy()
                            sel = sc > self.nu_decode_thresh
                            if sel.any():
                                w = sc[sel]
                                pred_nv = (cn[sel] * w[:, None]).sum(0) / w.sum()
                            else:
                                pred_nv = cn[int(sc.argmax())]
                            d = (np.linalg.norm(gt_nu - pred_nv, axis=1).min()
                                 * self.coord_scale)
                            nu_res_cm.append(d)
                            if d < self.nu_recall_cm:
                                nu_n_recovered += 1

                    # ---- PER-PARTICLE keypoints: start/end localization ----
                    pkp = ev.get("particle_kp")
                    if pkp and gti_list is not None and ei < len(gti_list):
                        insts = gti_list[ei]
                        for r in pkp:
                            inst = insts[r["inst_idx"]]
                            prob = r["class_logits"].softmax(-1).cpu().numpy()
                            pos = r["pos"].cpu().numpy()
                            # START = visible start keypoint (matches the loss
                            # target); fall back to origin if none tagged.
                            o = (inst["start_coord_norm"]
                                 if inst.get("has_start")
                                 else inst.get("origin_coord_norm"))
                            if o is not None:
                                qi = int(prob[:, KP_CLS_START].argmax())
                                d = (np.linalg.norm(pos[qi] - self._np(o))
                                     * self.coord_scale)
                                pkp_start_err.append(d)
                            # END (tracks only)
                            if inst.get("has_end") and \
                                    inst.get("end_coord_norm") is not None:
                                qi = int(prob[:, KP_CLS_END].argmax())
                                d = (np.linalg.norm(
                                    pos[qi] - self._np(inst["end_coord_norm"]))
                                     * self.coord_scale)
                                pkp_end_err.append(d)

        scalars = {}
        if obj_scores:
            s = np.concatenate(obj_scores)
            y = np.concatenate(obj_labels)
            pred_pos = s > self.score_thresh
            tp = int((pred_pos & y).sum())
            fp = int((pred_pos & ~y).sum())
            fn = int((~pred_pos & y).sum())
            scalars["val/object_ap"] = _average_precision(s, y)
            scalars["val/object_P"] = tp / max(tp + fp, 1)
            scalars["val/object_R"] = tp / max(tp + fn, 1)
            scalars["val/object_pos_frac"] = float(y.mean())
        if nu_res_cm:
            r = np.asarray(nu_res_cm, dtype=np.float64)
            scalars["val/nu_vertex_res_cm_median"] = float(np.median(r))
            scalars["val/nu_vertex_res_cm_mean"] = float(r.mean())
            scalars["val/nu_vertex_recall"] = nu_n_recovered / max(nu_n_gt_events, 1)
        for name, errs in (("start", pkp_start_err), ("end", pkp_end_err)):
            if errs:
                e = np.asarray(errs, dtype=np.float64)
                scalars[f"val/pkp_{name}_err_cm_median"] = float(np.median(e))
                scalars[f"val/pkp_{name}_err_cm_mean"] = float(e.mean())
                for thr in self.pkp_thresholds_cm:
                    scalars[f"val/pkp_{name}_R{thr:g}"] = float((e < thr).mean())

        self._log_and_publish(scalars)

    # ------------------------------------------------------------------

    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def _log_and_publish(self, scalars: dict):
        log = self.trainer.logger
        if "val/object_ap" in scalars:
            log.info(
                "Val object head: AP %.3f  P/R@%.2f %.3f/%.3f  pos_frac %.3f"
                % (scalars["val/object_ap"], self.score_thresh,
                   scalars["val/object_P"], scalars["val/object_R"],
                   scalars["val/object_pos_frac"]))
        if "val/nu_vertex_res_cm_median" in scalars:
            log.info(
                "Val nu-vertex: res(cm) median %.2f mean %.2f  recall@%.0fcm %.3f"
                % (scalars["val/nu_vertex_res_cm_median"],
                   scalars["val/nu_vertex_res_cm_mean"], self.nu_recall_cm,
                   scalars["val/nu_vertex_recall"]))
        thr_lo = min(self.pkp_thresholds_cm)
        for name in ("start", "end"):
            mk = f"val/pkp_{name}_err_cm_median"
            if mk in scalars:
                log.info(
                    "Val particle %s: err(cm) median %.2f mean %.2f  R@%gcm %.3f"
                    % (name, scalars[mk], scalars[f"val/pkp_{name}_err_cm_mean"],
                       thr_lo, scalars[f"val/pkp_{name}_R{thr_lo:g}"]))
        epoch1 = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            for k, v in scalars.items():
                self.trainer.writer.add_scalar(k, v, epoch1)
            if getattr(self.trainer.cfg, "enable_wandb", False):
                try:
                    import wandb
                    wandb.log({**scalars, "Epoch": epoch1})
                except ImportError:
                    pass
        if self.best_metric in scalars:
            self.trainer.comm_info["current_metric_value"] = float(
                scalars[self.best_metric])
            self.trainer.comm_info["current_metric_name"] = self.best_metric
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End LArFormer LevelKeypoint Eval <<<<<<<<<<<<<<<<")
