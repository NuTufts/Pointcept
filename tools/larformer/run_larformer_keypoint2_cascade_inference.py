"""End-to-end inference for the attempt-2 keypoint model (CascadedKeypoint).

Runs the frozen Stage-3 cascade (deghoster + slicer + particle segmenter) on raw
merged_h5 events, then the attempt-2 keypoint model on the nu slice, and writes
per-event keypoints (per-particle start/end + the dense nu vertex) in DETECTOR
CM to an H5 per event.

Driven by a SINGLE config (configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py):
  --config       : the full-cascade keypoint config. It assembles the
                   `CascadedKeypoint` model (cascade + keypoint model), the raw
                   merged_h5 test dataset, and the two checkpoint paths
                   (`particle_weights` = Stage-3 segmenter, `keypoint_weights` =
                   keypoint model). The deghoster/slicer/sonata weights load
                   inside the cascade portion of the config.
  --input-list   : text file of raw merged_h5 paths (e.g.
                   inputlists/..._10event_test.txt)

The two model checkpoints come from the config (`particle_weights`,
`keypoint_weights`); --particle-weights / --keypoint-weights still override them.

Decode: per particle, the highest-prob START and END queries give start/end;
positions (recentered coord_norm) → cm via the per-event affine recovered from
(ps_coord, ps_coord_norm). The dense nu-vertex head is decoded as the
score-weighted centroid of spacepoints above --nu-thresh.
"""
import argparse
import os

import numpy as np
import torch

import pointcept.datasets  # noqa: F401
import pointcept.models.LArFormer  # noqa: F401  (registers CascadedKeypoint)
from pointcept.utils.config import Config
from pointcept.datasets import build_dataset, larformer_collate
from pointcept.models.builder import build_model
from pointcept.models.LArFormer.keypoint_eval import _recover_affine
from pointcept.models.LArFormer.keypoint2_particle import (
    KP_CLS_START, KP_CLS_END)
from tools.larformer.run_larformer_stage3_inference import (
    _load_weights_into, set_deterministic, reseed_per_event)
from lartpc.flashmatch.flash_predict import (
    predict_many_slices_pe, select_charge_y_with_uv_fallback_np)
from lartpc.flashmatch.flash_chi2 import neyman_chi2, drift_correct, oob_mask


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


_NAN3 = np.full(3, np.nan, np.float32)


_NU_SID = -5               # full-event slice_id value for the nu slice (cosmic
                           # slices use their query index >= 0; -2 ghost, -1
                           # kept-unclustered, -3 pre-filtered)
_KP_START_TYPES = (1, 3)   # track_start, shower
_KP_END_TYPE = 2           # track_end
_KP_NU_TYPE = 0            # nu_vertex


def _mode_trackid(trk):
    """Majority trackid over a particle's points, ignoring -1. -1 if none."""
    trk = trk[trk >= 0]
    if trk.size == 0:
        return -1
    vals, cnt = np.unique(trk, return_counts=True)
    return int(vals[int(cnt.argmax())])


def _decode_event(ev, inst_list, gt, coord_cm, coord_norm, nu_thresh,
                  coord_center, coord_scale, save_score_maps=False,
                  head_kp_types=None):
    """Rich per-event decode for the visualizer. Returns the nu-slice coords
    (cm), one record per predicted particle (its point indices, class, predicted
    start/end in cm, and — matched by majority per-point GT trackid — the GT
    particle's point indices + GT start/end), and the predicted + GT nu vertex.

    GT matching uses per-SP `trackid` (gt["trackid"]) because the cascade strips
    gt_instances before the slicer (no IoU vs gt_instances available). GT
    start/end come from the SAME per-instance source the LOSS used — the raw
    sample's `gt_instances` `origin_coord_norm` (start = particle birth/origin)
    and `end_coord_norm` (track_end), keyed by trackid in gt["inst_map"] — NOT
    the raw mckeypoint track_start/shower (which is the CONVERSION point for
    photons, far from the origin the model was trained to predict). GT nu-vertex
    still comes from mckeypoints (gt["mckp_*"], type 0).

    TWO frames are in play and MUST be denormalized differently:
      * PREDICTED keypoints (the model's `pos`) are in the cascade's RECENTERED
        coord_norm → use the per-event affine recovered from (coord_cm,
        coord_norm) [`to_cm`].
      * GT keypoints (mckeypoints_pos_norm) are in the dataset's FIXED
        normalization (coord_center/coord_scale; NOT recentered — the cascade
        recenters only coord_norm) → use `fixed_to_cm`. Using the recovered
        affine on them offsets the GT keypoints by the slice-centroid shift.
      GT particle spacepoints come straight from coord_cm (already absolute), so
      with `fixed_to_cm` the GT keypoints land on the GT points.
    """
    scale, center = _recover_affine(coord_cm, coord_norm)
    if scale is None:
        scale = np.ones(3, np.float32)
        center = np.zeros(3, np.float32)
    coord_center = np.asarray(coord_center, np.float32)
    coord_scale = float(coord_scale)

    def to_cm(p):       # recentered coord_norm -> cm (predicted keypoints)
        return (np.asarray(p, np.float32) * scale + center).astype(np.float32)

    def fixed_to_cm(p):  # fixed-normalized -> cm (GT mckeypoints)
        return (np.asarray(p, np.float32) * coord_scale
                + coord_center).astype(np.float32)

    trk = gt.get("trackid")           # (N_slice,) or None
    if trk is not None:
        trk = _np(trk).reshape(-1)
    mck_pos = None if gt.get("mckp_pos") is None else _np(gt["mckp_pos"])
    mck_typ = (None if gt.get("mckp_type") is None
               else _np(gt["mckp_type"]).astype(np.int64).reshape(-1))
    mck_trk = (None if gt.get("mckp_trackid") is None
               else _np(gt["mckp_trackid"]).astype(np.int64).reshape(-1))
    have_mck = mck_pos is not None and mck_typ is not None

    inst_map = gt.get("inst_map") or {}    # {trackid: {origin,end,has_end}}

    def gt_start_for(track_id):
        """Per-instance VISIBLE start (= the loss start target) for track_id."""
        o = inst_map.get(int(track_id), {}).get("start")
        return fixed_to_cm(_np(o)) if o is not None else _NAN3.copy()

    def gt_end_for(track_id):
        """Per-instance end_coord_norm (= the loss end target, track_end), cm."""
        r = inst_map.get(int(track_id))
        if r is None or not r.get("has_end") or r.get("end") is None:
            return _NAN3.copy()
        return fixed_to_cm(_np(r["end"]))

    particles = []
    for p in ev.get("particle_kp", []):
        prob = p["class_logits"].softmax(-1).cpu().numpy()
        pos = p["pos"].cpu().numpy()
        si = int(prob[:, KP_CLS_START].argmax())
        ei = int(prob[:, KP_CLS_END].argmax())
        end_cm = (to_cm(pos[ei]) if prob[ei, KP_CLS_END] > prob[ei, -1]
                  else _NAN3.copy())
        inst = inst_list[int(p["inst_idx"])] if inst_list is not None else {}
        pt_idx = _np(inst.get("truth_indices",
                              np.zeros(0))).astype(np.int32).reshape(-1)
        rec = {
            "point_idx": pt_idx,
            "cls": int(p.get("pred_class", -1)),
            "start_cm": to_cm(pos[si]),
            "end_cm": end_cm,
            "has_match": False, "iou": 0.0, "gt_trackid": -1,
            "gt_point_idx": np.zeros(0, np.int32),
            "gt_start_cm": _NAN3.copy(), "gt_end_cm": _NAN3.copy(),
            # loose fallback: this instance passed only via the below-threshold
            # single-object recovery (highest object confidence).
            "loose_pass": bool(inst.get("loose_pass", False)),
            "loose_conf": float(inst.get("loose_conf", np.nan)),
        }
        if trk is not None and pt_idx.size:
            t = _mode_trackid(trk[pt_idx])
            if t >= 0:
                gt_pts = np.nonzero(trk == t)[0].astype(np.int32)
                if gt_pts.size:
                    inter = np.intersect1d(pt_idx, gt_pts).size
                    union = np.union1d(pt_idx, gt_pts).size
                    rec.update(
                        has_match=True, gt_trackid=t, gt_point_idx=gt_pts,
                        iou=float(inter / max(union, 1)),
                        gt_start_cm=gt_start_for(t),
                        gt_end_cm=gt_end_for(t))
        particles.append(rec)

    # predicted nu vertex (dense spacepoint head)
    nu = _NAN3.copy()
    nh = ev.get("level_kp", {}).get("nu_vertex")
    if nh is not None:
        sc = nh["score"].cpu().numpy()
        cn = nh["coords"].cpu().numpy()
        sel = sc > nu_thresh
        nv_norm = ((cn[sel] * sc[sel, None]).sum(0) / sc[sel].sum()
                   if sel.any() else cn[int(sc.argmax())])
        nu = to_cm(nv_norm)

    # GT nu vertex (mckeypoints type 0), if available
    gt_nu = _NAN3.copy()
    if have_mck:
        sel0 = mck_typ == _KP_NU_TYPE
        if sel0.any():
            gt_nu = fixed_to_cm(mck_pos[sel0][0])

    out = {
        "slice_coord_cm": np.asarray(coord_cm, np.float32),
        "particles": particles,
        "nu_vertex_cm": nu.astype(np.float32),
        "gt_nu_vertex_cm": gt_nu.astype(np.float32),
    }

    # --- optional: dense head score maps (object + nu_vertex) for diagnostics --
    # Save every per-level keypoint head's sigmoid score + its token coords in
    # cm (recentered coord_norm -> cm via the same per-event affine), plus the
    # raw GT mckeypoints (fixed normalization -> cm), so the visualizer can show
    # whether the heads put high scores at the GT keypoints. See
    # tools/viz/visualize_keypoint2_cascade_dash.py.
    if save_score_maps:
        head_kp_types = head_kp_types or {}
        smaps = {}
        for name, head in (ev.get("level_kp", {}) or {}).items():
            if head is None or head.get("coords") is None:
                continue
            smaps[name] = {
                "coords_cm": to_cm(_np(head["coords"])),
                "score": _np(head["score"]).reshape(-1).astype(np.float32),
                "level": str(head.get("level", "")),
                "kp_types": np.asarray(head_kp_types.get(name, []), np.int32),
            }
        out["score_maps"] = smaps
        # All GT keypoints (every mckeypoint type), for type-filtered overlay.
        if have_mck:
            out["gt_keypoints"] = {
                "pos_cm": fixed_to_cm(mck_pos).reshape(-1, 3),
                "type": mck_typ.astype(np.int32).reshape(-1),
                "trackid": (mck_trk.astype(np.int32).reshape(-1)
                            if mck_trk is not None
                            else np.full(mck_typ.shape, -1, np.int32)),
            }
        else:
            out["gt_keypoints"] = {
                "pos_cm": np.zeros((0, 3), np.float32),
                "type": np.zeros(0, np.int32),
                "trackid": np.zeros(0, np.int32),
            }
    return out


def _write_event_h5(path, dec, attrs, flash_tbl=None):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        if flash_tbl is not None:
            _write_flash_groups(f, flash_tbl)
        f.attrs["n_particles"] = len(dec["particles"])
        f.attrs["has_gt"] = bool(
            any(p["has_match"] for p in dec["particles"])
            or np.isfinite(dec["gt_nu_vertex_cm"]).all())
        f.create_dataset("slice/coord_cm", data=dec["slice_coord_cm"],
                         compression="gzip")
        f.create_dataset("nu_vertex_cm", data=dec["nu_vertex_cm"])
        f.create_dataset("gt_nu_vertex_cm", data=dec["gt_nu_vertex_cm"])
        for i, p in enumerate(dec["particles"]):
            g = f.create_group(f"particle/{i}")
            g.attrs["cls"] = p["cls"]
            g.attrs["has_match"] = p["has_match"]
            g.attrs["iou"] = p["iou"]
            g.attrs["gt_trackid"] = p["gt_trackid"]
            g.attrs["loose_pass"] = bool(p.get("loose_pass", False))
            g.attrs["loose_conf"] = float(p.get("loose_conf", np.nan))
            g.create_dataset("point_idx", data=p["point_idx"],
                             compression="gzip")
            g.create_dataset("start_cm", data=p["start_cm"])
            g.create_dataset("end_cm", data=p["end_cm"])
            g.create_dataset("gt_point_idx", data=p["gt_point_idx"],
                             compression="gzip")
            g.create_dataset("gt_start_cm", data=p["gt_start_cm"])
            g.create_dataset("gt_end_cm", data=p["gt_end_cm"])

        # --- optional: dense head score maps + GT keypoints (--save-score-maps) -
        smaps = dec.get("score_maps")
        if smaps is not None:
            f.attrs["has_score_maps"] = True
            for name, sm in smaps.items():
                g = f.create_group(f"score_maps/{name}")
                g.attrs["level"] = sm["level"]
                g.attrs["kp_types"] = sm["kp_types"]
                g.create_dataset("coords_cm", data=sm["coords_cm"],
                                 compression="gzip")
                g.create_dataset("score", data=sm["score"], compression="gzip")
            gk = dec.get("gt_keypoints", {})
            gg = f.create_group("gt_keypoints")
            gg.create_dataset("pos_cm", data=gk.get(
                "pos_cm", np.zeros((0, 3), np.float32)), compression="gzip")
            gg.create_dataset("type", data=gk.get(
                "type", np.zeros(0, np.int32)), compression="gzip")
            gg.create_dataset("trackid", data=gk.get(
                "trackid", np.zeros(0, np.int32)), compression="gzip")


def _compute_slice_ids(sl_out, batch, nu_class_id, mask_prob_threshold,
                       spacepoint_level="spacepoint"):
    """Per-spacepoint slice_id over the FULL event cloud (single-event batch).

    From a separate CascadedSlicer forward (eval, deterministic tau). The deghoster
    gives P(real) over the full cloud; the slicer's query masks partition the kept
    points into the nu slice (nu-class queries, unioned) and cosmic slices (each
    other non-no_object query). slice_id:
      -2 ghost (deghoster dropped it), -1 kept but in no slice,
      _NU_SID (-5) nu slice, q (>= 0) cosmic slice = slicer query index q.
    Returns (coord_cm[M,3], slice_id[M], p_real[M]) numpy, or None.
    """
    import math
    coord = batch.get("coord")
    if coord is None:
        return None
    coord_np = coord.detach().cpu().numpy().astype(np.float32)
    M = coord_np.shape[0]
    p_real = sl_out.get("deghost_p_real")
    if p_real is None:                       # deghoster emptied the event -> ghost
        return coord_np, np.full(M, -2, np.int64), np.zeros(M, np.float32)
    p_real_np = p_real.detach().cpu().numpy().astype(np.float32)
    tau = float(sl_out.get("deghost_tau", 0.5))
    keep = p_real_np > tau
    sid = np.full(M, -2, np.int64)
    sid[keep] = -1                           # kept but unclustered (default)
    preds = sl_out.get("predictions", [])
    n_filt = int(keep.sum())
    if preds and n_filt > 0:
        pred = preds[0]
        cls = pred.get("class_logits")
        ml = (pred.get("mask_logits") or {}).get(spacepoint_level)
        if cls is not None and ml is not None and cls.shape[0] > 0 \
                and ml.shape[1] == n_filt:
            tau_logit = math.log(mask_prob_threshold / (1.0 - mask_prob_threshold))
            argmax = cls.argmax(dim=-1).detach().cpu().numpy()
            C = int(cls.shape[1])
            ml_np = ml.detach().cpu().numpy()
            filt_sid = np.full(n_filt, -1, np.int64)
            # cosmic slice id = the slicer QUERY INDEX q (so it matches the
            # per-slice keypoint2 labels 'cosmicQQ' from _enumerate_slices).
            for q in range(len(argmax)):
                if argmax[q] == nu_class_id or argmax[q] == C - 1:
                    continue
                filt_sid[ml_np[q] > tau_logit] = q
            nu_qs = [q for q in range(len(argmax)) if argmax[q] == nu_class_id]
            if nu_qs:                        # nu overrides cosmic
                filt_sid[(ml_np[nu_qs] > tau_logit).any(axis=0)] = _NU_SID
            sid[keep] = filt_sid
    return coord_np, sid, p_real_np


def _slicer_input(batch):
    """Batch minus gt_instances, matching what CascadedParticleSegmenter feeds the
    slicer -- so our SEPARATE slicer calls don't trigger the slicer's eval GT
    matcher (which can hit an event-specific CUDA device-side assert)."""
    return {k: v for k, v in batch.items()
            if k not in ("gt_instances_per_event", "n_gt_instances")}


def _enumerate_slices(slicer_predictions, n_sp_per_event, nu_class_id,
                      mask_prob_threshold, spacepoint_level="spacepoint",
                      min_points=20):
    """List the slices to process: [(label, spec), ...] -- the nu slice
    ('nu','nu') plus each cosmic-class query with >= min_points ('cosmicQQ',
    ('q', qi)). `spec` is resolved to a keep mask INSIDE the model's own forward
    (build_forced_keep_mask), so the mask always matches that forward's filtered
    batch -- this enumeration only needs the (stable) query slots + classes, not
    exact per-forward point counts. Single-event (B == 1).
    """
    import math
    out = []
    n_sp = [int(x) for x in n_sp_per_event.detach().cpu().tolist()]
    if not slicer_predictions or not n_sp:
        return out
    n_filt = n_sp[0]
    pred = slicer_predictions[0]
    cls = pred.get("class_logits")
    ml = (pred.get("mask_logits") or {}).get(spacepoint_level)
    if cls is None or ml is None or cls.shape[0] == 0 or ml.shape[1] != n_filt:
        return out
    tau_logit = math.log(mask_prob_threshold / (1.0 - mask_prob_threshold))
    argmax = cls.argmax(dim=-1)
    C = int(cls.shape[1])
    above = ml > tau_logit
    nu_qs = (argmax == nu_class_id).nonzero(as_tuple=False).reshape(-1)
    if nu_qs.numel() and int(above[nu_qs].any(dim=0).sum().item()) >= min_points:
        out.append(("nu", "nu"))
    for q in range(len(argmax)):
        a = int(argmax[q].item())
        if a == nu_class_id or a == C - 1:
            continue
        if int(above[q].sum().item()) >= min_points:
            out.append((f"cosmic{q:02d}", ("q", q)))
    return out


def _write_slice_ids_h5(path, coord_cm, slice_id, p_real, attrs):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        g = f.create_group("full_slice")
        g.create_dataset("coord_cm", data=coord_cm, compression="gzip")
        g.create_dataset("slice_id", data=slice_id, compression="gzip")
        g.create_dataset("deghost_p_real", data=p_real, compression="gzip")
        f.attrs["n_ghost"] = int((slice_id == -2).sum())
        f.attrs["n_unclustered"] = int((slice_id == -1).sum())
        f.attrs["n_nu"] = int((slice_id == _NU_SID).sum())
        f.attrs["n_cosmic"] = int((slice_id >= 0).sum())
        f.attrs["n_cosmic_slices"] = int(np.unique(slice_id[slice_id >= 0]).size)


def _rng_snapshot():
    """Capture torch (CPU + all CUDA) + numpy RNG state. The backbones'
    serialization shuffle consumes global RNG on every forward, so any work
    between two forwards that touches the RNG shifts the later forward's
    draws; snapshot/restore keeps the stream passes bit-compatible with the
    pre-stream single-forward flow."""
    st = {"torch": torch.get_rng_state(), "np": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["np"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def _vox_keys(pos):
    """1 cm voxel keys, one bytes key per row (matches the single_photon study's
    voxel hash)."""
    k = np.round(np.asarray(pos, np.float64)).astype(np.int32)
    return [row.tobytes() for row in k]


def _load_msp_flash_charge(msp_path):
    """Observed flashes + 1 cm voxel->charge map from the input merged_h5.

    Returns (flashes, vox2charge):
      flashes: dict(pe (Nf,32), producer_id, total_pe, time_us[, flash_index])
               or None if the file has no `flashes` group;
      vox2charge: {voxel bytes key: mean charge} from triplet_data
               (Y-plane ADC with U/V fallback), or None if unavailable.
    """
    import h5py
    flashes, vox2charge = None, None
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        if "flashes" in e:
            fl = e["flashes"]
            flashes = {
                "pe": fl["pe"][()].astype(np.float32),
                "producer_id": fl["producer_id"][()].astype(np.int32),
                "total_pe": fl["total_pe"][()].astype(np.float32),
                "time_us": fl["time_us"][()].astype(np.float32),
            }
            if "flash_index" in fl:
                flashes["flash_index"] = fl["flash_index"][()].astype(np.int32)
        td = e.get("triplet_data")
        if td is not None and "pos" in td and "pixval" in td:
            pos = td["pos"][()].astype(np.float32)
            charge = select_charge_y_with_uv_fallback_np(td["pixval"][()])
            acc = {}
            for k, q in zip(_vox_keys(pos), charge):
                acc.setdefault(k, []).append(float(q))
            vox2charge = {k: float(np.mean(v)) for k, v in acc.items()}
    return flashes, vox2charge


def _flash_slice_table(coord_cm, sid, p_nu_per_query, nu_qs, flashes,
                       vox2charge, args):
    """Per-slice flash-match table over the full-event slicer partition.

    Rows: the nu union (query == _NU_SID) if present, plus every cosmic-class
    query with >= --slice-min-points spacepoints. For each row: predicted PE
    per PMT (PhotonLib, drift-corrected by the in-time beam flash t0), Neyman
    chi2 vs the observed beam flash, OOB fraction, and 1-based chi2 rank among
    rows passing --flash-oob-max (rank 0 = not ranked). Also carries the
    slicer's P(nu) per row and the per-query nu-score subtable.

    Returns dict for _write_event_h5, plus 'best_query' (the rank-1 row's
    query value, _NU_SID for the nu union) or None.
    """
    # ---- observed in-time beam flash (producer 0, max total PE) -------------
    obs = None
    if flashes is not None and flashes["pe"].size:
        beam = np.where(flashes["producer_id"] == 0)[0]
        if beam.size:
            bi = int(beam[int(np.argmax(flashes["total_pe"][beam]))])
            obs = {
                "pe": flashes["pe"][bi],
                "time_us": float(flashes["time_us"][bi]),
                "total_pe": float(flashes["total_pe"][bi]),
                "producer_id": 0,
                "flash_index": int(flashes.get(
                    "flash_index", np.arange(len(flashes["pe"])))[bi]),
            }

    # ---- slice rows ----------------------------------------------------------
    rows = []                       # (label, query, point_index_array)
    if (sid == _NU_SID).any():
        rows.append(("nu", _NU_SID, np.nonzero(sid == _NU_SID)[0]))
    for q in np.unique(sid[sid >= 0]):
        pts = np.nonzero(sid == q)[0]
        if pts.size >= args.slice_min_points:
            rows.append((f"cosmic{int(q):02d}", int(q), pts))
    S = len(rows)
    tbl = {
        "label": [r[0] for r in rows],
        "query": np.asarray([r[1] for r in rows], np.int32),
        "n_points": np.asarray([r[2].size for r in rows], np.int32),
        "pred_pe": np.full((S, 32), np.nan, np.float32),
        "chi2": np.full(S, np.nan, np.float32),
        "oob_frac": np.full(S, np.nan, np.float32),
        "chi2_rank": np.zeros(S, np.int32),
        "p_nu": np.asarray(
            [(max((p_nu_per_query[q] for q in nu_qs), default=np.nan)
              if r[1] == _NU_SID else p_nu_per_query.get(r[1], np.nan))
             for r in rows], np.float32),
        "flashes": flashes, "observed": obs, "best_query": None,
        "nu_queries": {
            "query": np.asarray(sorted(nu_qs), np.int32),
            "p_nu": np.asarray([p_nu_per_query.get(q, np.nan)
                                for q in sorted(nu_qs)], np.float32),
        },
        "params": {"gamma_beam": args.gamma_beam, "f_sys": args.flash_f_sys,
                   "eps": args.flash_eps, "oob_max": args.flash_oob_max},
    }
    if obs is None or vox2charge is None or S == 0:
        return tbl

    # ---- predicted PE + chi2 + OOB per row -----------------------------------
    sel = np.concatenate([r[2] for r in rows])
    cid = np.concatenate([np.full(r[2].size, i, np.int64)
                          for i, r in enumerate(rows)])
    spos = coord_cm[sel].astype(np.float32)
    scharge = np.asarray([vox2charge.get(k, 0.0) for k in _vox_keys(spos)],
                         np.float32)
    tbl["pred_pe"] = predict_many_slices_pe(
        spos, scharge, cid, S, obs["time_us"], producer_id=0,
        gamma_by_producer=(args.gamma_beam, args.gamma_beam),
        photonlib_cache=args.photonlib).astype(np.float32)
    for i, r in enumerate(rows):
        pos_i = coord_cm[r[2]]
        tbl["oob_frac"][i] = float(
            oob_mask(drift_correct(pos_i, obs["time_us"])).mean())
        tbl["chi2"][i] = float(neyman_chi2(
            tbl["pred_pe"][i], obs["pe"],
            f_sys=args.flash_f_sys, eps=args.flash_eps))
    ok = (tbl["oob_frac"] <= args.flash_oob_max) & np.isfinite(tbl["chi2"])
    order = np.argsort(np.where(ok, tbl["chi2"], np.inf), kind="stable")
    rank = 1
    for j in order:
        if ok[j]:
            tbl["chi2_rank"][j] = rank
            rank += 1
    if ok.any():
        tbl["best_query"] = int(tbl["query"][order[0]])
    return tbl


def _write_flash_groups(f, tbl):
    """Write flash/ + slices/ groups (schema in lartpc/larformer_reco/README)."""
    import h5py
    fg = f.create_group("flash")
    for k, v in tbl["params"].items():
        fg.attrs[k] = v
    obs = tbl.get("observed")
    fg.attrs["has_beam_flash"] = obs is not None
    if obs is not None:
        fg.create_dataset("observed_pe", data=obs["pe"].astype(np.float32))
        for k in ("time_us", "total_pe", "producer_id", "flash_index"):
            fg.attrs[k] = obs[k]
    fl = tbl.get("flashes")
    if fl is not None:
        ag = fg.create_group("all")
        for k in ("pe", "producer_id", "total_pe", "time_us"):
            ag.create_dataset(k, data=fl[k], compression="gzip")
    sg = f.create_group("slices")
    sg.create_dataset("label", data=np.array(
        [s.encode() for s in tbl["label"]]))
    for k in ("query", "n_points", "pred_pe", "chi2", "oob_frac",
              "chi2_rank", "p_nu"):
        sg.create_dataset(k, data=tbl[k])
    nq = tbl["nu_queries"]
    g = f.create_group("slices/nu_queries")
    g.create_dataset("query", data=nq["query"])
    g.create_dataset("p_nu", data=nq["p_nu"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="single full-cascade keypoint config "
                         "(configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py)")
    ap.add_argument("--input-list", required=True)
    ap.add_argument("--output-dir", default="kp2_cascade_out")
    ap.add_argument("--particle-weights", default=None,
                    help="override cfg.particle_weights (Stage-3 segmenter ckpt)")
    ap.add_argument("--keypoint-weights", default=None,
                    help="override cfg.keypoint_weights (keypoint model ckpt)")
    ap.add_argument("--n-events", type=int, default=-1)
    ap.add_argument("--start-event", type=int, default=0,
                    help="Dataset index (in the SORTED file list) to start at. "
                         "Combine with --n-events to process a contiguous shard "
                         "[start-event, start-event+n-events). Output filenames "
                         "carry the dataset index, so shards never collide.")
    ap.add_argument("--nu-thresh", type=float, default=None,
                    help="override cfg.nu_thresh (dense nu-vertex decode thresh)")
    ap.add_argument("--save-score-maps", dest="save_score_maps",
                    action="store_true", default=None,
                    help="Also write the dense head score maps (object @ dec2 + "
                         "nu_vertex @ spacepoint) and the raw GT mckeypoints, so "
                         "tools/viz/visualize_keypoint2_cascade_dash.py can overlay "
                         "predicted scores on the GT keypoints. Default off "
                         "(falls back to cfg.save_score_maps).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--particle-source", choices=["predicted", "gt"],
                    default=None,
                    help="Point sets fed to the per-particle decoder: 'predicted'"
                         " (Stage-3 masks, production) or 'gt' (GT instance masks,"
                         " = the training-time input — use to DIAGNOSE the "
                         "train/inference gap: if start error drops to ~val with "
                         "'gt', the gap is GT-mask vs predicted-mask).")
    ap.add_argument("--with-gt", dest="with_gt", action="store_true",
                    default=True,
                    help="Emit MC keypoints so the output carries GT (matched "
                         "particle + GT start/end/nu-vertex) for the visualizer. "
                         "Default ON (sim).")
    ap.add_argument("--no-gt", dest="with_gt", action="store_false",
                    help="Disable GT emission (real data, no MC truth).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Bit-exact inference on a fixed GPU/driver/lib stack "
                         "(TF32 off, deterministic algorithms, seeded, "
                         "CUBLAS_WORKSPACE_CONFIG). See "
                         "docs/reference/LArFormer_Reproducibility.md. ~1.3-2x slower.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-slice-ids", action="store_true",
                    help="Also write a per-event sidecar sliceid_event{i}.h5 with "
                         "the FULL-event per-spacepoint slice_id (ghost / nu-slice "
                         "/ cosmic-slice-k / kept-unclustered) + deghoster P(real), "
                         "from a separate slicer forward. Written for EVERY event, "
                         "including those with no nu slice.")
    ap.add_argument("--slice-ids-only", action="store_true",
                    help="Only dump the slice-id sidecars (skip the Stage-3 "
                         "segmenter + keypoint model and the keypoint2 output). "
                         "Fast; does not need particle/keypoint weights.")
    ap.add_argument("--slice-id-dir", default=None,
                    help="dir for the sliceid_event{i}.h5 sidecars "
                         "(default: --output-dir)")
    ap.add_argument("--all-slices", action="store_true",
                    help="Run Stage-3 (segmenter + keypoint) SEPARATELY on EVERY "
                         "slice (nu + each cosmic slice), not just the nu union. "
                         "Writes keypoint2_event{i}_{slice}_{ei}.h5 per slice. Very "
                         "expensive (one full Stage-3 pass per slice); for small "
                         "study samples only.")
    ap.add_argument("--slice-min-points", type=int, default=20,
                    help="skip slices with fewer post-deghost points (--all-slices)")
    ap.add_argument("--loose-fallback", action="store_true",
                    help="If a slice yields NO above-threshold segmenter instance, "
                         "recover the single best below-threshold query, flagged "
                         "loose_pass in the output. Ranked by P(--loose-class-id) "
                         "(default gamma). Higher recall for a targeted single-object "
                         "search.")
    ap.add_argument("--loose-class-id", type=int, default=1,
                    help="Class the loose fallback ranks below-threshold queries by "
                         "(default 1=gamma). Use -1 to rank by generic object-ness "
                         "(1-P(no_object)) instead.")
    ap.add_argument("--no-flash", action="store_true",
                    help="Disable the flash-match machinery entirely: no flash/ "
                         "+ slices/ groups in the output and no flashmatch "
                         "stream. Use when the input has no flashes group or "
                         "the photonlib cache is unavailable.")
    ap.add_argument("--no-flashmatch-stream", action="store_true",
                    help="Still write the flash/ + slices/ chi2 table, but do "
                         "NOT run Stage-3 on the best-chi2 slice (no "
                         "stream='flashmatch' output files).")
    ap.add_argument("--gamma-beam", type=float, default=5.25,
                    help="PhotonLib PE scale for beam flashes (calibrated on "
                         "the slicer gamma tune; see single_photon study).")
    ap.add_argument("--flash-f-sys", type=float, default=0.10,
                    help="fractional systematic on observed PE in the Neyman "
                         "chi2 variance")
    ap.add_argument("--flash-eps", type=float, default=1.0,
                    help="absolute variance floor (dark PMTs) in the chi2")
    ap.add_argument("--flash-oob-max", type=float, default=0.05,
                    help="max fraction of a slice's points outside the TPC "
                         "(post drift-correction) for it to be chi2-ranked")
    ap.add_argument("--photonlib", default=None,
                    help="photon-library npz cache (default: "
                         "lartpc/flashmatch/data/photonlib_v6_70kV.npz)")
    ap.add_argument("--max-spacepoints", type=int, default=None,
                    help="Override cfg.data.test.max_spacepoints (random "
                         "subsample before deghoster+slicer). The config "
                         "defaults this to 500000 to match the training cache "
                         "(only very noisy events hit it); lower it only to "
                         "diagnose density effects. See "
                         "lartpc/larformer_reco/README.md.")
    args = ap.parse_args()

    # MUST run before the model is built / any cuBLAS call (sets env + flags).
    if args.deterministic:
        set_deterministic(args.seed)
        print(">>> deterministic mode ON")

    cfg = Config.fromfile(args.config)
    # Fixed normalization used by the dataset for mckeypoints (GT keypoints) —
    # denormalize GT keypoints with THESE, not the recentered slice affine.
    coord_center = np.asarray(cfg.get("coord_center", (125.0, 0.0, 518.0)),
                              np.float32)
    coord_scale = float(cfg.get("coord_scale", 179.55))
    nu_thresh = (args.nu_thresh if args.nu_thresh is not None
                 else float(cfg.get("nu_thresh", 0.3)))
    save_score_maps = (args.save_score_maps if args.save_score_maps is not None
                       else bool(cfg.get("save_score_maps", False)))

    # The config already assembles the full CascadedKeypoint model dict.
    model_cfg = dict(cfg.model)
    # Map each dense head -> the GT keypoint types it targets (object: track
    # start/end/shower/..., nu_vertex: [0]); saved with the score maps so the
    # visualizer can filter the GT-keypoint overlay per head.
    head_kp_types = {
        hc["name"]: list(hc.get("kp_types", []))
        for hc in (model_cfg.get("keypoint_model", {})
                   .get("level_keypoint_heads", []) or [])}
    if save_score_maps:
        print(f">>> saving dense head score maps for: "
              f"{sorted(head_kp_types) or 'all heads'}")
    if args.particle_source is not None:
        model_cfg["particle_source"] = args.particle_source
    print(f">>> building CascadedKeypoint "
          f"(particle_source={model_cfg.get('particle_source')}) ...")
    model = build_model(model_cfg)

    # slice-ids-only uses just the (cascade-config-loaded) slicer, so the Stage-3
    # segmenter + keypoint weights aren't needed.
    if not args.slice_ids_only:
        particle_weights = args.particle_weights or cfg.get("particle_weights")
        keypoint_weights = args.keypoint_weights or cfg.get("keypoint_weights")
        assert particle_weights, ("no Stage-3 particle weights: set "
                                  "cfg.particle_weights or --particle-weights")
        assert keypoint_weights, ("no keypoint weights: set cfg.keypoint_weights "
                                  "or --keypoint-weights")
        _load_weights_into(model.cascade.particle_segmenter, particle_weights)
        _load_weights_into(model.keypoint_model, keypoint_weights)
    model = model.to(args.device).eval()
    if args.loose_fallback:
        model.loose_fallback = True    # read by CascadedKeypoint.forward decode
        model.loose_class_id = (None if int(args.loose_class_id) < 0
                                else int(args.loose_class_id))
        _lc = ("object-ness" if model.loose_class_id is None
               else f"P(class={model.loose_class_id})")
        print(f">>> loose single-object fallback ON (rank by {_lc})")

    save_slice_ids = args.save_slice_ids or args.slice_ids_only
    sid_dir = args.slice_id_dir or args.output_dir
    if save_slice_ids:
        os.makedirs(sid_dir, exist_ok=True)
        print(f">>> writing slice-id sidecars -> {sid_dir}"
              f"{'  (slice-ids only)' if args.slice_ids_only else ''}")

    ds_cfg = dict(cfg.data.test)
    ds_cfg["data_root"] = "/"
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    if args.max_spacepoints is not None:
        ds_cfg["max_spacepoints"] = int(args.max_spacepoints)
        print(f">>> max_spacepoints override = {args.max_spacepoints}")
    if args.with_gt and not args.slice_ids_only:
        # Surface MC keypoints so the output carries GT (matched particle +
        # GT start/end/nu-vertex) for the side-by-side visualizer. Sim only;
        # use --no-gt on real data (no MC truth).
        #
        # slice-ids-only forces this OFF: it needs only the slicer's query
        # predictions + deghoster P(real), which don't depend on GT. Emitting
        # gt_instances makes the slicer's LArFormer eval path run its GT matcher
        # (loss/matcher), which can hit an event-specific CUDA device-side assert
        # and is pure waste here.
        ds_cfg["emit_keypoints"] = True
        ds_cfg["gt_source"] = "particle"
    else:
        # Real data: no MC truth — turn GT emission off (config defaults it on).
        ds_cfg["emit_keypoints"] = False
        ds_cfg["gt_source"] = "deghost"
    ds = build_dataset(ds_cfg)
    # Contiguous shard [start, end) over the SORTED file list. --n-events < 0
    # means "to the end of the list" from start; both bounds are clamped so an
    # over-provisioned final shard simply processes fewer events.
    start = min(max(0, args.start_event), len(ds))
    end = len(ds) if args.n_events < 0 else min(start + args.n_events, len(ds))
    n = max(0, end - start)
    os.makedirs(args.output_dir, exist_ok=True)
    # The dataset SORTS its file list (get_data_list -> sorted), so ds[i] is the
    # i-th SORTED file, not the i-th input line — processing order is canonical
    # regardless of input order. Label outputs by the actual processed file.
    real_files = list(getattr(ds, "data_list", []))
    print(f">>> {n} events (indices [{start}, {end}) of {len(ds)})")

    for i in range(start, end):
        # Per-event order invariance (shared helper; see its docstring).
        if args.deterministic:
            reseed_per_event(args.seed)
        batch = larformer_collate([ds[i]])
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        # --- slice-ids-only: sidecar from a separate slicer forward, no Stage-3.
        # (In the unified path below, the sidecar is written from the SAME slicer
        # forward the streams use, so slice ids always match the stream labels.)
        if args.slice_ids_only:
            with torch.no_grad():
                sl_out = model.cascade.cascaded_slicer(_slicer_input(batch))
            sids = _compute_slice_ids(
                sl_out, batch, model.cascade.nu_class_id,
                model.cascade.mask_prob_threshold,
                model.cascade.spacepoint_level)
            if sids is not None:
                sattrs = {"src_file": (os.path.basename(real_files[i])
                                       if i < len(real_files) else "")}
                _write_slice_ids_h5(
                    os.path.join(sid_dir, f"sliceid_event{i:05d}.h5"),
                    *sids, sattrs)
            print(f"  [{i}] slice-ids -> sliceid_event{i:05d}.h5")
            continue
        # Decode Stage-3 output + write one keypoint2 H5 per nu-slice prediction.
        # `tag_prefix` labels the slice for --all-slices (e.g. 'nu_', 'cosmic07_').
        # `attrs_extra` adds stream labels; `flash_tbl` adds flash/ + slices/.
        def decode_and_write(out, tag_prefix="", attrs_extra=None,
                             flash_tbl=None):
            preds = out.get("predictions", [])
            if not preds:
                return 0
            offset = out["ps_offset"]
            coord = out["ps_coord"].cpu().numpy()
            coord_norm = out["ps_coord_norm"].cpu().numpy()
            trackid = (out["ps_trackid"].cpu().numpy()
                       if out.get("ps_trackid") is not None else None)
            inst_all = out.get("ps_particle_instances")
            mck_pos = out.get("ps_mckeypoints_pos_norm_per_event")
            mck_typ = out.get("ps_mckeypoints_type_per_event")
            mck_trk = out.get("ps_mckeypoints_trackid_per_event")
            raw_gti = batch.get("gt_instances_per_event")

            def _ev(lst, ei):
                return lst[ei] if lst is not None and ei < len(lst) else None

            def _inst_map(insts):
                m = {}
                for g in insts or []:
                    t = int(g.get("primary_trackid", -1))
                    if t < 0:
                        continue
                    start = (g["start_coord_norm"] if g.get("has_start")
                             else g.get("origin_coord_norm"))
                    m[t] = dict(start=start, end=g.get("end_coord_norm"),
                                has_end=bool(g.get("has_end", False)))
                return m

            for ei, ev in enumerate(preds):
                a = int(offset[ei - 1].item()) if ei > 0 else 0
                b = int(offset[ei].item())
                gt = dict(
                    trackid=(trackid[a:b] if trackid is not None else None),
                    mckp_pos=_ev(mck_pos, ei), mckp_type=_ev(mck_typ, ei),
                    mckp_trackid=_ev(mck_trk, ei),
                    inst_map=_inst_map(_ev(raw_gti, ei)))
                dec = _decode_event(
                    ev, _ev(inst_all, ei), gt,
                    coord[a:b], coord_norm[a:b], nu_thresh,
                    coord_center, coord_scale,
                    save_score_maps=save_score_maps, head_kp_types=head_kp_types)
                attrs = {"src_file": (os.path.basename(real_files[i])
                                      if i < len(real_files) else "")}
                if tag_prefix:
                    attrs["slice_label"] = tag_prefix.rstrip("_")
                if attrs_extra:
                    attrs.update(attrs_extra)
                for k in ("run", "subrun", "event"):
                    v = out.get(f"ps_{k}")
                    if v is not None:
                        attrs[k] = (int(v[ei]) if hasattr(v, "__getitem__")
                                    else int(v))
                outp = os.path.join(
                    args.output_dir,
                    f"keypoint2_event{i:05d}_{tag_prefix}{ei}.h5")
                _write_event_h5(outp, dec, attrs, flash_tbl=flash_tbl)
                n_match = sum(1 for p in dec["particles"] if p["has_match"])
                print(f"  [{i}.{tag_prefix}{ei}] "
                      f"n_particles={len(dec['particles'])} "
                      f"(matched_gt={n_match}) -> {os.path.basename(outp)}")
            return len(preds)

        # ---- --all-slices: run Stage-3 on EVERY slice separately -------------
        if args.all_slices:
            with torch.no_grad():
                sl_out = model.cascade.cascaded_slicer(_slicer_input(batch))
            # Write the full-event slice_id sidecar from THIS same slicer forward
            # so its cosmic ids (= query indices) match the per-slice labels below.
            os.makedirs(sid_dir, exist_ok=True)
            sids = _compute_slice_ids(
                sl_out, batch, model.cascade.nu_class_id,
                model.cascade.mask_prob_threshold, model.cascade.spacepoint_level)
            if sids is not None:
                _write_slice_ids_h5(
                    os.path.join(sid_dir, f"sliceid_event{i:05d}.h5"), *sids,
                    {"src_file": (os.path.basename(real_files[i])
                                  if i < len(real_files) else "")})
            fb = sl_out.get("filtered_batch")
            slices = (_enumerate_slices(
                sl_out.get("predictions", []), fb["n_spacepoints"],
                model.cascade.nu_class_id, model.cascade.mask_prob_threshold,
                model.cascade.spacepoint_level, min_points=args.slice_min_points)
                if fb is not None else [])
            print(f"  [{i}] {len(slices)} slices: "
                  f"{', '.join(l for l, _ in slices)}")
            # Reuse THIS slicer forward for every per-slice Stage-3 pass so each
            # slice (and the sidecar above) refers to the same physical clusters.
            model.cascade._force_slicer_out = sl_out
            try:
                for label, spec in slices:
                    model.cascade._force_slice = spec
                    try:
                        with torch.no_grad():
                            out = model(batch)
                    finally:
                        model.cascade._force_slice = None
                    decode_and_write(out, tag_prefix=f"{label}_")
            finally:
                model.cascade._force_slicer_out = None
            continue

        # ---- unified stream path (default) ------------------------------------
        # One slicer forward feeds everything for the event: the slice-id
        # sidecar, the flash-match slice table, and up to two Stage-3+keypoint
        # passes — the nu union (stream='nu') and, when the best-chi2 slice is
        # NOT the nu union, the flash-match slice (stream='flashmatch', with the
        # loose single-object fallback enabled for that pass).
        with torch.no_grad():
            sl_out = model.cascade.cascaded_slicer(_slicer_input(batch))
        # RNG state right after the (deghost+slicer) forward == what Stage-3 saw
        # in the pre-stream flow; restored before each stream's Stage-3 pass so
        # the sidecar/flash-table work in between cannot shift the draws.
        rng_post_slicer = _rng_snapshot()
        sids = _compute_slice_ids(
            sl_out, batch, model.cascade.nu_class_id,
            model.cascade.mask_prob_threshold, model.cascade.spacepoint_level)
        if save_slice_ids and sids is not None:
            _write_slice_ids_h5(
                os.path.join(sid_dir, f"sliceid_event{i:05d}.h5"), *sids,
                {"src_file": (os.path.basename(real_files[i])
                              if i < len(real_files) else "")})

        flash_tbl = None
        if not args.no_flash and sids is not None:
            try:
                p_nu_per_query, nu_qs = {}, []
                preds_sl = sl_out.get("predictions") or []
                if preds_sl:
                    cls = preds_sl[0].get("class_logits")
                    if cls is not None and cls.shape[0]:
                        probs = torch.softmax(
                            cls.detach().float(), dim=-1).cpu().numpy()
                        nu_id = model.cascade.nu_class_id
                        p_nu_per_query = {
                            q: float(probs[q, nu_id])
                            for q in range(len(probs))}
                        am = cls.argmax(dim=-1).detach().cpu().numpy()
                        nu_qs = [q for q in range(len(am)) if am[q] == nu_id]
                flashes, vox2charge = _load_msp_flash_charge(real_files[i])
                flash_tbl = _flash_slice_table(
                    sids[0], sids[1], p_nu_per_query, nu_qs,
                    flashes, vox2charge, args)
            except Exception as ex:
                print(f"  [warn] flash table failed for event {i}: {ex}")

        best_q = flash_tbl.get("best_query") if flash_tbl else None
        nu_stream = ("nu,flashmatch" if best_q == _NU_SID else "nu")
        wrote = 0
        model.cascade._force_slicer_out = sl_out
        try:
            # nu stream: the cascade's default nu-union path on this slicer out.
            _rng_restore(rng_post_slicer)
            with torch.no_grad():
                out = model(batch)
            wrote += decode_and_write(out, attrs_extra={"stream": nu_stream},
                                      flash_tbl=flash_tbl)
            # flashmatch stream: best-chi2 slice, if it is not the nu union.
            if (best_q is not None and best_q != _NU_SID
                    and not args.no_flashmatch_stream):
                old_lf = getattr(model, "loose_fallback", False)
                old_lc = getattr(model, "loose_class_id", None)
                model.loose_fallback = True
                model.loose_class_id = (None if int(args.loose_class_id) < 0
                                        else int(args.loose_class_id))
                model.cascade._force_slice = ("q", int(best_q))
                try:
                    # same post-slicer state for the fm pass, so its output does
                    # not depend on whether a nu pass ran before it.
                    _rng_restore(rng_post_slicer)
                    with torch.no_grad():
                        out = model(batch)
                finally:
                    model.cascade._force_slice = None
                    model.loose_fallback = old_lf
                    model.loose_class_id = old_lc
                row = int(np.nonzero(flash_tbl["query"] == best_q)[0][0])
                wrote += decode_and_write(
                    out, tag_prefix="fm_",
                    attrs_extra={
                        "stream": "flashmatch",
                        "slice_label": f"cosmic{int(best_q):02d}",
                        "flash_chi2": float(flash_tbl["chi2"][row]),
                    },
                    flash_tbl=flash_tbl)
        finally:
            model.cascade._force_slicer_out = None
        if wrote == 0:
            print(f"  [{i}] no nu slice"
                  + ("" if best_q is None else " / empty flash-match pass")
                  + " — skipped")
    print("DONE")


if __name__ == "__main__":
    main()
