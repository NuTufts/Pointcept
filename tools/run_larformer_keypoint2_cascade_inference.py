"""End-to-end inference for the attempt-2 keypoint model (CascadedKeypoint).

Runs the frozen Stage-3 cascade (deghoster + slicer + particle segmenter) on raw
merged_h5 events, then the attempt-2 keypoint model on the nu slice, and writes
per-event keypoints (per-particle start/end + the dense nu vertex) in DETECTOR
CM to an H5 per event.

Driven by a SINGLE config (configs/lartpc/larformer-keypoint2-fullcascade.py):
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
from tools.run_larformer_stage3_inference import (
    _load_weights_into, set_deterministic, reseed_per_event)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


_NAN3 = np.full(3, np.nan, np.float32)


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
    # tools/visualize_keypoint2_cascade_dash.py.
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


def _write_event_h5(path, dec, attrs):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="single full-cascade keypoint config "
                         "(configs/lartpc/larformer-keypoint2-fullcascade.py)")
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
                         "tools/visualize_keypoint2_cascade_dash.py can overlay "
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
                         "docs/LArFormer_Reproducibility.md. ~1.3-2x slower.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-spacepoints", type=int, default=None,
                    help="Override cfg.data.test.max_spacepoints (random "
                         "subsample before deghoster+slicer). The config "
                         "defaults this to 500000 to match the training cache "
                         "(only very noisy events hit it); lower it only to "
                         "diagnose density effects. See "
                         "lartpc_data_prep/larformer_keypoint_v2/README.md.")
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

    particle_weights = args.particle_weights or cfg.get("particle_weights")
    keypoint_weights = args.keypoint_weights or cfg.get("keypoint_weights")
    assert particle_weights, ("no Stage-3 particle weights: set "
                              "cfg.particle_weights or --particle-weights")
    assert keypoint_weights, ("no keypoint weights: set cfg.keypoint_weights "
                              "or --keypoint-weights")
    _load_weights_into(model.cascade.particle_segmenter, particle_weights)
    _load_weights_into(model.keypoint_model, keypoint_weights)
    model = model.to(args.device).eval()

    ds_cfg = dict(cfg.data.test)
    ds_cfg["data_root"] = "/"
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    if args.max_spacepoints is not None:
        ds_cfg["max_spacepoints"] = int(args.max_spacepoints)
        print(f">>> max_spacepoints override = {args.max_spacepoints}")
    if args.with_gt:
        # Surface MC keypoints so the output carries GT (matched particle +
        # GT start/end/nu-vertex) for the side-by-side visualizer. Sim only;
        # use --no-gt on real data (no MC truth).
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
        with torch.no_grad():
            out = model(batch)
        preds = out.get("predictions", [])
        if not preds:
            print(f"  [{i}] no nu slice — skipped")
            continue
        offset = out["ps_offset"]
        coord = out["ps_coord"].cpu().numpy()
        coord_norm = out["ps_coord_norm"].cpu().numpy()
        trackid = (out["ps_trackid"].cpu().numpy()
                   if out.get("ps_trackid") is not None else None)
        inst_all = out.get("ps_particle_instances")
        mck_pos = out.get("ps_mckeypoints_pos_norm_per_event")
        mck_typ = out.get("ps_mckeypoints_type_per_event")
        mck_trk = out.get("ps_mckeypoints_trackid_per_event")

        def _ev(lst, ei):
            return lst[ei] if lst is not None and ei < len(lst) else None

        # Per-trackid GT start(origin)/end from the RAW sample's gt_instances —
        # the SAME source the loss used (the cascade strips them internally, but
        # the batch dict we hold still carries them). Keyed by primary_trackid.
        raw_gti = batch.get("gt_instances_per_event")

        def _inst_map(insts):
            m = {}
            for g in insts or []:
                t = int(g.get("primary_trackid", -1))
                if t < 0:
                    continue
                # VISIBLE start (track_start/shower kp) = the loss target;
                # fall back to origin if no start keypoint was tagged.
                start = (g["start_coord_norm"] if g.get("has_start")
                         else g.get("origin_coord_norm"))
                m[t] = dict(start=start,
                            end=g.get("end_coord_norm"),
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
            for k in ("run", "subrun", "event"):
                v = out.get(f"ps_{k}")
                if v is not None:
                    attrs[k] = int(v[ei]) if hasattr(v, "__getitem__") else int(v)
            outp = os.path.join(args.output_dir,
                                f"keypoint2_event{i:05d}_{ei}.h5")
            _write_event_h5(outp, dec, attrs)
            n_match = sum(1 for p in dec["particles"] if p["has_match"])
            print(f"  [{i}.{ei}] n_particles={len(dec['particles'])} "
                  f"(matched_gt={n_match}) "
                  f"nu_vertex_cm={dec['nu_vertex_cm'].round(1).tolist()} -> {outp}")
    print("DONE")


if __name__ == "__main__":
    main()
