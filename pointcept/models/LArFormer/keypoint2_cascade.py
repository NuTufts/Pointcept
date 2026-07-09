"""CascadedKeypoint (attempt-2 Phase 3) — wraps the frozen Stage-3 cascade
(`CascadedParticleSegmenter`, which itself contains the Stage-2 slicer) and runs
the attempt-2 keypoint model on the nu slice it produces.

Mirrors how Stage-3 wraps Stage-2: build the inner model, freeze it, pin it to
eval, override `train()` to keep it frozen, and run it under `no_grad` in
forward. The cascade is configured with `expose_filtered_batch=True` so its eval
output carries the recentered nu-slice batch (`ps_batch`) the keypoint model's
own backbone runs on, plus the per-event particle predictions.

Particle source for the per-particle keypoint decoder:
  - "predicted" (production): each predicted particle mask (deduped via the SAME
    `inference.dedup_queries` as the masks) becomes a pseudo-instance whose
    `truth_indices` are its spacepoints. INFERENCE: pseudo-instances with no GT
    → predictions only. The per-particle decoder predicts in the cascade's
    recentered coord_norm; decode to cm downstream via the recovered affine.
  - "gt": use the cascade's filtered GT instances directly.

FRAME CAVEAT (training through the cascade): the cascade recenters ONLY
`coord_norm` (not GT origin/end or keypoint positions), because Stage-3 *trains
on the stage-12 cache* — which recenters everything together — and only *infers*
through the cascade. So this wrapper is correct for INFERENCE; training the
keypoint model THROUGH the cascade (predicted-mask matching with IoU-attached GT
keypoints) needs the matched GT recentered by the same per-event centroid first
(a follow-up — extend the cascade recenter to also recenter gt_instances, or
expose the centroid). For now: TRAIN the keypoint model standalone on the cache
(Phase 1/2, where the dataset recenters consistently); the Phase-2b denoising
path already provides imperfect-mask robustness. The IoU-match branch below is
left in place for when that recentering is wired.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model

from .keypoint_eval import dedup_query_effective_argmax


def predicted_masks_to_instances(
    class_logits: torch.Tensor,        # (Q, C)
    sp_mask_logits: torch.Tensor,      # (Q, N)
    no_object_class_id: int,
    class_prob_floor: float = 0.3,
    dedup_iou_threshold: float = 0.6,
    mask_thresh: float = 0.0,
    min_points: int = 3,
    gt_instances: Optional[List[dict]] = None,
    iou_match_thresh: float = 0.3,
    loose_fallback: bool = False,
    loose_class_id: Optional[int] = None,
) -> List[dict]:
    """Convert one event's predicted particle masks into keypoint-decoder
    instances. Deduped + confidence-floored via the SAME path as the particle
    masks (no drift). When `gt_instances` is given, each predicted instance is
    IoU-matched to a GT instance and inherits its origin/end keypoints (training
    on predicted masks). Returns a list of dicts with at least `truth_indices`
    (event-local SP indices, long) + `pred_class`.

    `loose_fallback`: if the normal (above-`class_prob_floor`) decode yields NOTHING
    for the event, return the SINGLE best below-floor query (with >= `min_points`),
    even though it's below the floor. Ranking:
      - `loose_class_id` given -> highest P(class=loose_class_id): targeted search,
        e.g. loose_class_id=gamma surfaces the most photon-like query.
      - `loose_class_id` None    -> highest object-ness 1 - P(no_object) (any class).
    The instance is flagged `loose_pass=True` (+ `loose_conf` = the ranking score
    used) so a consumer can tell it was recovered loosely. `pred_class` is still the
    query's best non-no_object class argmax. Normal above-threshold behaviour is
    unchanged."""
    if class_logits.shape[0] == 0 or sp_mask_logits.shape[1] == 0:
        return []
    eff = dedup_query_effective_argmax(
        class_logits, sp_mask_logits, no_object_class_id=int(no_object_class_id),
        class_prob_floor=class_prob_floor, iou_threshold=dedup_iou_threshold)
    N = int(sp_mask_logits.shape[1])
    dev = sp_mask_logits.device
    pred_bool = sp_mask_logits > mask_thresh                    # (Q, N)

    gt_masks = None
    if gt_instances:
        gt_masks = []
        for g in gt_instances:
            ti = g.get("truth_indices")
            m = torch.zeros(N, dtype=torch.bool, device=dev)
            if ti is not None:
                t = torch.as_tensor(ti, device=dev, dtype=torch.long).reshape(-1)
                t = t[(t >= 0) & (t < N)]
                if t.numel() > 0:
                    m[t] = True
            gt_masks.append(m)

    def _attach_gt(inst, q):
        if not gt_masks:
            return
        pm = pred_bool[q]
        best_iou, best_j = 0.0, -1
        for j, gm in enumerate(gt_masks):
            inter = int((pm & gm).sum().item())
            if inter == 0:
                continue
            union = int((pm | gm).sum().item())
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_match_thresh:
            g = gt_instances[best_j]
            for k in ("origin_coord_norm", "start_coord_norm", "has_start",
                      "end_coord_norm", "has_end", "primary_trackid"):
                if k in g:
                    inst[k] = g[k]
            if "truth_indices" in g:      # GT point set (for the GT viz)
                inst["gt_truth_indices"] = g["truth_indices"]
            inst["match_iou"] = best_iou

    out: List[dict] = []
    all_probs = class_logits.softmax(dim=-1)                    # (Q, C)
    for q in range(eff.shape[0]):
        if int(eff[q]) == int(no_object_class_id):
            continue
        idx = pred_bool[q].nonzero(as_tuple=False).reshape(-1)
        if int(idx.numel()) < min_points:
            continue
        inst = {"truth_indices": idx.long(),
                "primary_trackid": -1,
                "pred_class": int(eff[q]),
                "class_probs": all_probs[q].detach().cpu()}
        _attach_gt(inst, q)
        out.append(inst)

    if not out and loose_fallback:
        probs = all_probs                                       # (Q, C)
        obj_probs = probs.clone()
        obj_probs[:, int(no_object_class_id)] = -1.0
        obj_cls = obj_probs.argmax(dim=-1)                      # best object class
        if (loose_class_id is not None
                and 0 <= int(loose_class_id) < probs.shape[1]
                and int(loose_class_id) != int(no_object_class_id)):
            rank = probs[:, int(loose_class_id)]                # targeted: P(class)
        else:
            rank = 1.0 - probs[:, int(no_object_class_id)]      # object-ness
        for q in torch.argsort(rank, descending=True).tolist():
            idx = pred_bool[q].nonzero(as_tuple=False).reshape(-1)
            if int(idx.numel()) < min_points:
                continue
            inst = {"truth_indices": idx.long(), "primary_trackid": -1,
                    "pred_class": int(obj_cls[q]),
                    "class_probs": all_probs[q].detach().cpu(),
                    "loose_pass": True, "loose_conf": float(rank[q])}
            _attach_gt(inst, q)
            out.append(inst)
            break
    return out


@MODELS.register_module()
class CascadedKeypoint(nn.Module):
    """Frozen Stage-3 cascade + attempt-2 keypoint model on the nu slice."""

    def __init__(
        self,
        cascade: dict,
        keypoint_model: dict,
        cascade_weight: Optional[str] = None,
        particle_source: str = "predicted",     # "predicted" | "gt"
        no_object_class_id: Optional[int] = None,
        class_prob_floor: float = 0.3,
        dedup_iou_threshold: float = 0.6,
        mask_thresh: float = 0.0,
        min_points: int = 3,
        iou_match_thresh: float = 0.3,
        loose_fallback: bool = False,
        loose_class_id: Optional[int] = 1,   # gamma; None -> rank by object-ness
    ):
        super().__init__()
        cascade = dict(cascade)
        cascade.setdefault("expose_filtered_batch", True)
        self.cascade = build_model(cascade)
        if cascade_weight is not None:
            self._load_cascade_weight(cascade_weight)
        # Freeze the WHOLE cascade (slicer + particle segmenter).
        for p in self.cascade.parameters():
            p.requires_grad = False
        self.cascade.eval()

        self.keypoint_model = build_model(keypoint_model)
        self.particle_source = str(particle_source)
        if no_object_class_id is None:
            no_object_class_id = int(keypoint_model.get("num_classes", 8)) - 1
        self.no_object_class_id = int(no_object_class_id)
        self.class_prob_floor = float(class_prob_floor)
        self.dedup_iou_threshold = float(dedup_iou_threshold)
        self.mask_thresh = float(mask_thresh)
        self.min_points = int(min_points)
        self.iou_match_thresh = float(iou_match_thresh)
        self.loose_fallback = bool(loose_fallback)
        self.loose_class_id = (int(loose_class_id)
                               if loose_class_id is not None else None)

    def _load_cascade_weight(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items()}
        missing, unexpected = self.cascade.load_state_dict(sd, strict=False)
        print(f"[CascadedKeypoint] loaded cascade weight {path}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.cascade.eval()    # keep the frozen cascade in eval always
        return self

    def forward(self, data_dict: dict) -> dict:
        with torch.no_grad():
            cout = self.cascade(data_dict)
        if "ps_batch" not in cout:
            # cascade produced no nu slice (all SPs filtered) — nothing to do.
            return {"loss": data_dict["coord_norm"].new_zeros(())} \
                if self.training else {"predictions": []}

        ps_batch = cout["ps_batch"]
        if self.particle_source == "predicted":
            preds = cout.get("predictions", [])
            gti = ps_batch.get("gt_instances_per_event")
            new_gti: List[List[dict]] = []
            for ei, ev in enumerate(preds):
                cl = ev.get("class_logits")
                sm = ev.get("mask_logits", {}).get("spacepoint")
                if cl is None or sm is None:
                    new_gti.append([])
                    continue
                gt = gti[ei] if (gti is not None and ei < len(gti)) else None
                new_gti.append(predicted_masks_to_instances(
                    cl, sm, self.no_object_class_id,
                    class_prob_floor=self.class_prob_floor,
                    dedup_iou_threshold=self.dedup_iou_threshold,
                    mask_thresh=self.mask_thresh, min_points=self.min_points,
                    gt_instances=gt, iou_match_thresh=self.iou_match_thresh,
                    loose_fallback=getattr(self, "loose_fallback", False),
                    loose_class_id=getattr(self, "loose_class_id", 1)))
            ps_batch = dict(ps_batch)
            # Feed PREDICTED instances to the per-particle pass via a dedicated
            # key (NOT gt_instances_per_event) so the keypoint model's
            # eval-with-GT diagnostic path isn't handed pseudo-instances that
            # lack GT fields like origin_type. Drop real gt_instances at
            # inference so that path stays off.
            ps_batch["particle_instances_per_event"] = new_gti
            ps_batch.pop("gt_instances_per_event", None)
        elif self.particle_source == "gt":
            # Diagnostic: feed the decoder the GT particle masks (= training-time
            # input). The cascade strips gt_instances before the slicer, so
            # reconstruct them in the SLICE frame by grouping the surviving
            # per-SP trackid (the same grouping the viz GT uses).
            trk = ps_batch.get("trackid")
            offs = ps_batch["offset"]
            new_gti = []
            prev = 0
            for ei in range(int(offs.shape[0])):
                cur = int(offs[ei].item())
                insts = []
                if trk is not None:
                    tev = trk[prev:cur]
                    for tv in torch.unique(tev):
                        t = int(tv)
                        if t < 0:
                            continue
                        idx = (tev == t).nonzero(as_tuple=False).reshape(-1)
                        if int(idx.numel()) >= self.min_points:
                            insts.append({"truth_indices": idx,
                                          "primary_trackid": t, "pred_class": -1})
                new_gti.append(insts)
                prev = cur
            ps_batch = dict(ps_batch)
            ps_batch["particle_instances_per_event"] = new_gti
            ps_batch.pop("gt_instances_per_event", None)

        out = self.keypoint_model(ps_batch)
        if not self.training and isinstance(out, dict) and "predictions" in out:
            # Carry the nu-slice geometry so a decoder can recover the per-event
            # affine (recentered coord_norm → detector cm) for the keypoints.
            out["ps_coord"] = ps_batch.get("coord")
            out["ps_coord_norm"] = ps_batch.get("coord_norm")
            out["ps_offset"] = ps_batch.get("offset")
            # Per-particle instances actually fed to the keypoint decoder (the
            # predicted masks' point sets + their IoU-matched GT, or the GT
            # instances under particle_source="gt") — for the visualizer.
            out["ps_particle_instances"] = ps_batch.get(
                "particle_instances_per_event",
                ps_batch.get("gt_instances_per_event"))
            # Per-SP GT trackid (survives the deghost/nu-keep subsetting) — the
            # visualizer matches a predicted particle to GT by majority trackid
            # vote (the cascade strips gt_instances before the slicer, so IoU vs
            # gt_instances isn't available here; trackid is more robust anyway).
            if "trackid" in ps_batch:
                out["ps_trackid"] = ps_batch["trackid"]
            # GT keypoints (mckeypoints) — nu-vertex (type 0) + per-trackid
            # start/end. Present when the dataset emitted them (emit_keypoints).
            for k in ("mckeypoints_pos_norm_per_event",
                      "mckeypoints_type_per_event",
                      "mckeypoints_trackid_per_event"):
                if k in ps_batch:
                    out[f"ps_{k}"] = ps_batch[k]
            for k in ("run", "subrun", "event", "name"):
                if k in ps_batch:
                    out[f"ps_{k}"] = ps_batch[k]
        return out
