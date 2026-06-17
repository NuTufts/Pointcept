"""LArFormerKeypointEvaluator — val-metric hook for the Phase-2 query keypoint
head (per-particle start/end), layered on the particle evaluator.

Inherits everything `LArFormerParticleEvaluator` logs (mask IoU, cls accuracy,
per-pair origin/start L2 in cm — and, automatically via the eval_loss
accumulation, `val/loss_kpq_start_err_cm` / `val/loss_kpq_end_err_cm`), and
ADDS the PPN-style per-type recall/precision within distance thresholds:

    val/kp_R{thr}_{track_start,track_end,shower}   recall   @ thr cm
    val/kp_P{thr}_{...}                            precision @ thr cm

It decodes the query head's per-event start/end points (`decode_query_points`
→ `query_points_to_typed`) and matches them to the event's `mckeypoints` GT,
restricted to the "reconstructable" set within `gt_acceptance_cm` of an SP
(the nu slice can't find cosmic keypoints with no nearby points). The nu
vertex / michel / delta are NOT scored here — they come from the dense head;
this hook covers the query head's particle-associated keypoints only.

Frame note: this is recenter-aware — acceptance uses `denorm(coord_norm)` so
pred / GT / acceptance share the (recentered) frame the model trained in.

Registers on import; configs trigger it via:
    from pointcept.models.LArFormer import keypoint_particle_evaluator as _m
    del _m
"""
from typing import Optional, Sequence

import numpy as np

from pointcept.engines.hooks.builder import HOOKS

from .particle_evaluator import LArFormerParticleEvaluator
from .keypoint_eval import (
    decode_query_points, query_points_to_typed, gt_keypoints_by_type,
    accumulate_metrics, denorm, DEFAULT_TRACK_CLASS_IDS, DEFAULT_QUERY_TYPES,
    decode_dense_votes, decode_nu_vertex, reconcile_keypoints,
    dedup_query_effective_argmax,
)
from lartpc_data_prep.keypoint_labels import (
    KEYPOINT_TYPE_NAMES, KPTYPE_NU_VERTEX,
)


@HOOKS.register_module()
class LArFormerKeypointEvaluator(LArFormerParticleEvaluator):
    """Particle evaluator + PPN-style per-type keypoint recall/precision."""

    def __init__(
        self,
        eval_freq: int = 0,
        best_metric: str = "mask_iou_mean",
        class_names: Optional[Sequence[str]] = None,
        empty_cache: bool = True,
        log_per_event: bool = False,
        report_origin_error: bool = True,
        coord_scale: float = 179.55,
        coord_center: Sequence[float] = (125.0, 0.0, 518.0),
        # 1 cm is the precision TARGET (GT proximity sigma is 3 cm); 3/10 cm
        # give coarse/loose context. val/kp_R1_* is the headline to watch.
        thresholds_cm: Sequence[float] = (1.0, 3.0, 10.0),
        class_prob_floor: float = 0.3,
        end_prob_floor: float = 0.5,
        gt_acceptance_cm: float = 5.0,
        dense_score_thresh: float = 0.3,
        dense_cluster_radius_cm: float = 3.0,
        dedup_iou_threshold: float = 0.6,
    ):
        super().__init__(
            eval_freq=eval_freq, best_metric=best_metric,
            class_names=class_names, empty_cache=empty_cache,
            log_per_event=log_per_event,
            report_origin_error=report_origin_error, coord_scale=coord_scale,
        )
        self.coord_center = np.asarray(coord_center, dtype=np.float32)
        self.thresholds_cm = [float(t) for t in thresholds_cm]
        self.class_prob_floor = float(class_prob_floor)
        self.end_prob_floor = float(end_prob_floor)
        self.gt_acceptance_cm = float(gt_acceptance_cm)
        self.dense_score_thresh = float(dense_score_thresh)
        self.dense_cluster_radius_cm = float(dense_cluster_radius_cm)
        self.dedup_iou_threshold = float(dedup_iou_threshold)

    # ------------------------------------------------------------------

    def _init_extra_state(self) -> None:
        super()._init_extra_state()
        self._kp_events_pred = []
        self._kp_events_gt = []
        # nu-vertex (dense-head) bookkeeping — populated only when the model
        # carries a dense keypoint head (ev_pred has `dense_kpscores`).
        self._kp_has_dense = False
        self._kp_vertex_res = []          # per-event |pred_vertex - GT| (cm)

    def _on_event_processed(self, *, ev_pred, eval_loss, q_idx, k_idx,
                            no_object_class_id,
                            input_dict=None, event_in_batch=None):
        # Parent records per-pair start (origin) L2 error.
        super()._on_event_processed(
            ev_pred=ev_pred, eval_loss=eval_loss, q_idx=q_idx, k_idx=k_idx,
            no_object_class_id=no_object_class_id)

        if input_dict is None or event_in_batch is None:
            return
        if "mckeypoints_pos_norm_per_event" not in input_dict:
            return  # dataset emitted no keypoints — skip silently.
        cls_logits = ev_pred.get("class_logits")
        origin = ev_pred.get("origin")
        if cls_logits is None or origin is None:
            return
        ei = int(event_in_batch)
        pos_list = input_dict["mckeypoints_pos_norm_per_event"]
        typ_list = input_dict["mckeypoints_type_per_event"]
        if ei >= len(pos_list):
            return

        center, scale = self.coord_center, self.coord_scale
        # Dedup duplicate queries (mask-IoU NMS — SAME inference.dedup_queries
        # as the particle masks) so co-extensive duplicate queries don't emit
        # duplicate keypoint FPs. Needs the spacepoint mask logits.
        eff = None
        sp_mask = ev_pred.get("mask_logits", {}).get("spacepoint")
        if sp_mask is not None and self.dedup_iou_threshold > 0:
            eff = dedup_query_effective_argmax(
                cls_logits, sp_mask,
                no_object_class_id=int(no_object_class_id),
                class_prob_floor=self.class_prob_floor,
                iou_threshold=self.dedup_iou_threshold)
        qpts = decode_query_points(
            cls_logits.detach().cpu().numpy(),
            origin.detach().cpu().numpy(), center, scale,
            end_norm=(ev_pred["kp_end"].detach().cpu().numpy()
                      if "kp_end" in ev_pred else None),
            end_logit=(ev_pred["kp_end_logit"].detach().cpu().numpy()
                       if "kp_end_logit" in ev_pred else None),
            no_object_class_id=int(no_object_class_id),
            class_prob_floor=self.class_prob_floor,
            end_prob_floor=self.end_prob_floor,
            track_class_ids=set(DEFAULT_TRACK_CLASS_IDS),
            effective_argmax=eff)
        typed = query_points_to_typed(qpts)

        # Event SP coords for acceptance (recentered frame → denorm(coord_norm)).
        offset = input_dict["offset"]
        prev = int(offset[ei - 1].item()) if ei > 0 else 0
        cur = int(offset[ei].item())
        coord_norm_ev = input_dict["coord_norm"][prev:cur].detach().cpu().numpy()
        coord_cm = denorm(coord_norm_ev, center, scale)

        def _np(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        gt = gt_keypoints_by_type(
            _np(pos_list[ei]), _np(typ_list[ei]), center, scale,
            coord_cm=coord_cm,
            acceptance_cm=(self.gt_acceptance_cm if self.gt_acceptance_cm > 0
                           else None))

        # Dense head (nu vertex + michel/delta): decode the per-SP score+offset
        # field, reconcile with the query points (query owns track/shower;
        # dense owns vertex/michel/delta), and score the chosen vertex.
        if "dense_kpscores" in ev_pred:
            self._kp_has_dense = True
            dense_decoded = decode_dense_votes(
                _np(ev_pred["dense_coord_norm"]),
                _np(ev_pred["dense_kpscores"]),
                (_np(ev_pred["dense_kpoffsets"])
                 if "dense_kpoffsets" in ev_pred else None),
                center, scale, score_thresh=self.dense_score_thresh,
                cluster_radius_cm=self.dense_cluster_radius_cm)
            typed = reconcile_keypoints(dense_decoded, typed)
            nv = decode_nu_vertex(dense_decoded)
            gtv = gt.get(KPTYPE_NU_VERTEX)
            if nv is not None and gtv is not None and gtv.shape[0] > 0:
                self._kp_vertex_res.append(float(
                    np.linalg.norm(gtv - nv["pos_cm"], axis=1).min()))

        self._kp_events_pred.append(typed)
        self._kp_events_gt.append(gt)

    # ------------------------------------------------------------------

    def _aggregate(self, **k) -> dict:
        scalars = super()._aggregate(**k)
        if self._kp_events_pred:
            m = accumulate_metrics(
                self._kp_events_pred, self._kp_events_gt, self.thresholds_cm)
            # Query types always; add the nu vertex when a dense head fed it.
            types = list(DEFAULT_QUERY_TYPES)
            if self._kp_has_dense:
                types = [KPTYPE_NU_VERTEX] + types
            for t in types:
                name = KEYPOINT_TYPE_NAMES[t]
                for thr in self.thresholds_cm:
                    scalars[f"val/kp_R{thr:g}_{name}"] = m[t][thr]["recall"]
                    scalars[f"val/kp_P{thr:g}_{name}"] = m[t][thr]["precision"]
        if self._kp_vertex_res:
            vr = np.asarray(self._kp_vertex_res, dtype=np.float64)
            scalars["val/nu_vertex_res_cm_median"] = float(np.median(vr))
            scalars["val/nu_vertex_res_cm_mean"] = float(vr.mean())
        return scalars

    def _log_and_publish(self, scalars: dict) -> None:
        # Keypoint summary: show the TIGHT precision target (min threshold,
        # = 1 cm) and a LOOSE context threshold (max, = 10 cm) side by side, so
        # 1 cm recall — the number that matters — is front and center while
        # training. All thresholds also publish as val/kp_{R,P}{thr}_* scalars.
        # Parent publishes the full scalar dict + sets the best-checkpoint metric.
        thr_lo = min(self.thresholds_cm)
        thr_hi = max(self.thresholds_cm)
        # Lead with nu_vertex when the dense head is present, then the query
        # types — so the slice-level vertex is visible alongside start/end.
        types = (([KPTYPE_NU_VERTEX] if self._kp_has_dense else [])
                 + list(DEFAULT_QUERY_TYPES))
        parts = []
        for t in types:
            name = KEYPOINT_TYPE_NAMES[t]
            rk_lo, pk_lo = f"val/kp_R{thr_lo:g}_{name}", f"val/kp_P{thr_lo:g}_{name}"
            rk_hi, pk_hi = f"val/kp_R{thr_hi:g}_{name}", f"val/kp_P{thr_hi:g}_{name}"
            if rk_lo in scalars:
                parts.append(
                    f"{name} {scalars[pk_lo]:.2f}/{scalars[rk_lo]:.2f}"
                    f"|{scalars[pk_hi]:.2f}/{scalars[rk_hi]:.2f}")
        if parts:
            self.trainer.logger.info(
                f"Val keypoints P/R (@{thr_lo:g}cm | @{thr_hi:g}cm): "
                + "  ".join(parts))
        if "val/nu_vertex_res_cm_median" in scalars:
            self.trainer.logger.info(
                f"Val nu-vertex resolution (cm): "
                f"median {scalars['val/nu_vertex_res_cm_median']:.2f}  "
                f"mean {scalars['val/nu_vertex_res_cm_mean']:.2f}")
        super()._log_and_publish(scalars)
