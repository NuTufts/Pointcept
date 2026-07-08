"""Production driver: run the nu-interaction reco (vertices + tracks + showers +
4-momenta) over a shard of the keypoint2-cascade output, writing results to one
H5 per shard.

The reco is CPU-only. Two input lists:
  --keypoint2-list : the cascade output files (this is what gets sharded)
  --merged-sp-list : the parent merged_sp files (for charge/truth); each keypoint2
                     file names its parent via the `src_file` attr, resolved here
                     by basename against this list.
Shard by contiguous slice of the keypoint2 list: --start / --n (final shard is
clamped to the list length), matching the cascade-inference convention.

    python run_nu_reco.py --keypoint2-list KP.txt --merged-sp-list MSP.txt \
        --output-dir OUT --start 0 --n 500

Output: OUT/nu_reco_shard{START:07d}.h5, one group `event_{gidx}` per event with
the reconstructed interactions and a flat per-particle 4-momentum table (schema
in _write_event).
"""
import os
import sys
import argparse
from types import SimpleNamespace

import numpy as np
import h5py

# the reco lives in trajfit/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
from lartpc.larformer_reco.trajfit import trajfit_io as tio  # noqa: E402
from lartpc.larformer_reco.utils import read_list  # noqa: E402                                             # noqa: E402
from lartpc.larformer_reco.trajfit.nu_interaction import (vertex_candidates, build_tracks,         # noqa: E402
                            reco_interactions, read_nu_vertices)
from lartpc.larformer_reco.trajfit.range_momentum import RangeMomentum                            # noqa: E402
from lartpc.larformer_reco.trajfit.particle_momentum import assign_momenta, load_shower_calib, _entry  # noqa: E402

_METHOD = {"range": 0, "calo": 1}


def reco_args(a):
    """Tuned reco parameters as a namespace for the nu_interaction functions."""
    return SimpleNamespace(
        no_snap=False, snap_radius=a.snap_radius, snap_support_radius=5.0,
        d_vertex=12.0, d_perp=4.0, front_tol=1.5, merge_radius=3.0,
        shower_mode=a.shower_mode, shower_d_impact=a.shower_d_impact,
        shower_cos_min=a.shower_cos_min, shower_d_gap=60.0,
        shower_gap_touch=getattr(a, "shower_gap_touch", 3.0),
        shower_seed_score=getattr(a, "shower_seed_score", 0.5),
        max_interactions=a.max_interactions, reco_exclude_radius=10.0,
        reco_min_unassoc_points=30)


def build_msp_map(path):
    m = {}
    with open(path) as f:
        for line in f:
            p = line.strip()
            if p:
                m[os.path.basename(p)] = p
    return m


def reco_one(kp_path, msp_path, rmom, calib, a):
    """Run the full reco on one event. Returns interactions or None."""
    msp_dir = os.path.dirname(msp_path) if msp_path else None
    recs = tio.load_instances(kp_path, msp_dir, tracks_only=True,
                              min_points=a.min_points)
    cands = vertex_candidates(kp_path)
    if not cands:                       # need a nu-vertex candidate to root on
        return None, None
    args = reco_args(a)
    # recs (tracks) may be EMPTY for shower-only (CC-nu_e) events; build_tracks
    # returns [] and reco_interactions then seeds a shower-only interaction.
    tracks = build_tracks(recs, seg_cm=3.0, kink_tol=3.0, eps=1.2,
                          max_gap_live=20.0)
    all_recs = tio.load_instances(kp_path, msp_dir, tracks_only=False, min_points=1)
    shower_recs = [r for r in all_recs if r.pred_cls in tio.SHOWER_CLASSES
                   and np.all(np.isfinite(r.pred_start))
                   and r.n_points >= a.shower_min_points]
    if not tracks and not shower_recs:
        return None, all_recs
    interactions, _lo_t, _lo_s = reco_interactions(cands, tracks, shower_recs, args)
    if not interactions:
        return None, all_recs
    entry, fh = _entry(kp_path, msp_dir)
    try:
        assign_momenta(interactions, entry, rmom, calib)
    finally:
        if fh is not None:
            fh.close()
    return interactions, all_recs


def _write_event(g, interactions, all_recs, kp_path):
    """Per-particle 4-momentum table + geometry: the fitted track polylines
    (`part_poly_cm` concatenated + `part_npoly` counts), per-particle start point
    (`part_start_cm`) and attach-vertex index (`part_vtx`), and the full vertex
    tree (`vtx_pos_cm`/`vtx_interaction`/`vtx_depth`/`vtx_parent_track`, primary +
    secondary), so a downstream viewer can draw the interaction on the real
    spacepoints without re-running the reco. `vertices_cm` (primary per
    interaction) is kept for backward compatibility."""
    ke_by = {r.inst_idx: (r.energy_mev, int(r.gt_trackid)) for r in (all_recs or [])}
    with h5py.File(kp_path, "r") as f:
        for k in ("src_file", "run", "subrun", "event",
                  "stream", "slice_label", "flash_chi2"):
            if k in f.attrs:
                g.attrs[k] = f.attrs[k]
        if "stream" not in f.attrs:
            g.attrs["stream"] = "nu"       # pre-stream keypoint2 files
    _, gt_nu = read_nu_vertices(kp_path)
    g.attrs["gt_nu_vertex_cm"] = gt_nu.astype(np.float32)
    g.attrs["n_interactions"] = len(interactions)
    g.create_dataset("vertices_cm",
                     data=np.array([I["vertex"] for I in interactions], np.float32))

    # --- full vertex tree (primary + secondary), flattened across interactions --
    vtx_pos, vtx_int, vtx_depth, vtx_parent = [], [], [], []
    vtx_gidx = {}                     # (interaction_idx, local_vertex_id) -> global
    for ii, I in enumerate(interactions):
        for v in I["res"]["vertices"]:
            vtx_gidx[(ii, int(v["id"]))] = len(vtx_pos)
            vtx_pos.append(np.asarray(v["pos"], np.float32))
            vtx_int.append(ii)
            vtx_depth.append(int(v.get("depth", 0)))
            pt = v.get("parent_track")
            vtx_parent.append(int(pt) if pt is not None else -1)
    g.create_dataset("vtx_pos_cm",
                     data=(np.asarray(vtx_pos, np.float32).reshape(-1, 3)
                           if vtx_pos else np.zeros((0, 3), np.float32)))
    g.create_dataset("vtx_interaction", data=np.asarray(vtx_int, np.int64))
    g.create_dataset("vtx_depth", data=np.asarray(vtx_depth, np.int64))
    g.create_dataset("vtx_parent_track", data=np.asarray(vtx_parent, np.int64))

    rows = dict(interaction=[], kind=[], pred_class=[], energy=[], length=[],
                charge=[], method=[], gt_trackid=[], true_ke=[], vtx=[])
    mom, fourvec, direction, start_cm, polys = [], [], [], [], []
    for ii, I in enumerate(interactions):
        objs = ([("track", T) for T in I["tracks"]]
                + [("shower", s) for s in I["showers"] if s["attached"]])
        for kind, o in objs:
            m = o.get("mom") or {}
            inst = o["inst_idx"] if kind == "track" else o["inst"]
            tke, gtid = ke_by.get(inst, (float("nan"), -1))
            rows["interaction"].append(I["iter"])
            rows["kind"].append(0 if kind == "track" else 1)
            rows["pred_class"].append(int(o["cls"]))
            rows["energy"].append(float(m.get("energy", np.nan)))
            rows["length"].append(float(o["length"]) if kind == "track" else np.nan)
            rows["charge"].append(float(m.get("charge_comb", np.nan)))
            rows["method"].append(_METHOD.get(m.get("momentum_method"), -1))
            rows["gt_trackid"].append(gtid)
            rows["true_ke"].append(float(tke))
            mom.append(np.asarray(m.get("momentum", [np.nan] * 3), np.float32))
            fourvec.append(np.asarray(m.get("fourvec", [np.nan] * 4), np.float32))
            direction.append(np.asarray(m.get("direction", [np.nan] * 3), np.float32))
            if kind == "track":
                poly = np.asarray(o.get("poly", np.zeros((0, 3))),
                                  np.float32).reshape(-1, 3)
                att = o.get("attach") or {}
                ei = int(att.get("end", 0))
                if ei == 1 and len(poly) >= 2:        # orient poly to start at vtx
                    poly = poly[::-1].copy()
                st = (poly[0] if len(poly) else np.full(3, np.nan, np.float32))
                av = att.get("vertex")
                gv = vtx_gidx.get((ii, int(av)), -1) if av is not None else -1
            else:                                     # shower: trunk start + dir
                tk = o.get("trunk")
                st = np.asarray(getattr(tk, "start", None)
                                if tk is not None else o.get("cp_pos"),
                                np.float32) if (tk is not None or o.get("cp_pos")
                                                is not None) else np.full(3, np.nan,
                                                                          np.float32)
                poly = np.zeros((0, 3), np.float32)   # drawn from start + direction
                cpid = o.get("cp_id") or ""
                gv = -1
                if isinstance(cpid, str) and cpid.startswith("v"):
                    try:
                        gv = vtx_gidx.get((ii, int(cpid[1:])), -1)
                    except ValueError:
                        gv = -1
            rows["vtx"].append(int(gv))
            start_cm.append(st)
            polys.append(poly)
    n = len(rows["kind"])
    g.attrs["n_particles"] = n
    for k, v in rows.items():
        g.create_dataset(f"part_{k}", data=np.asarray(v))
    g.create_dataset("part_momentum", data=np.asarray(mom).reshape(n, 3))
    g.create_dataset("part_fourvec", data=np.asarray(fourvec).reshape(n, 4))
    g.create_dataset("part_direction", data=np.asarray(direction).reshape(n, 3))
    g.create_dataset("part_start_cm",
                     data=(np.asarray(start_cm, np.float32).reshape(n, 3)
                           if n else np.zeros((0, 3), np.float32)))
    npoly = np.asarray([len(p) for p in polys], np.int64)
    g.create_dataset("part_npoly", data=npoly)
    g.create_dataset("part_poly_cm",
                     data=(np.concatenate(polys, 0).astype(np.float32)
                           if polys and npoly.sum() > 0
                           else np.zeros((0, 3), np.float32)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoint2-list", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1, help="-1 = to end of list")
    # reco knobs (tuned defaults)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--shower-min-points", type=int, default=20)
    ap.add_argument("--shower-mode", default="greedy")
    ap.add_argument("--shower-d-impact", type=float, default=15.0)
    ap.add_argument("--shower-cos-min", type=float, default=0.80)
    ap.add_argument("--shower-gap-touch", type=float, default=3.0,
                    help="attach a shower regardless of back-pointing cosine when "
                         "its trunk-start is within this many cm of the vertex "
                         "(cosine is ill-conditioned at ~zero gap)")
    ap.add_argument("--shower-seed-score", type=float, default=0.5,
                    help="min nu-vertex candidate score to seed a SHOWER-ONLY "
                         "interaction (CC-nu_e with no tracks) when leftover "
                         "showers point back to an unused candidate")
    ap.add_argument("--snap-radius", type=float, default=30.0)
    ap.add_argument("--max-interactions", type=int, default=3)
    args = ap.parse_args()

    kp_list = read_list(args.keypoint2_list)
    end = len(kp_list) if args.n < 0 else min(args.start + args.n, len(kp_list))
    shard = list(range(args.start, min(end, len(kp_list))))
    msp_map = build_msp_map(args.merged_sp_list)
    rmom = RangeMomentum()
    calib = load_shower_calib()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"nu_reco_shard{args.start:07d}.h5")
    print(f">>> shard start={args.start} n={len(shard)} -> {out_path} | "
          f"calib={calib} | {len(msp_map)} merged_sp in map", flush=True)

    n_ok = n_skip = n_err = 0
    with h5py.File(out_path, "w") as fout:
        fout.attrs["shard_start"] = args.start
        fout.attrs["n_requested"] = len(shard)
        for gidx in shard:
            kp = kp_list[gidx]
            try:
                with h5py.File(kp, "r") as f:
                    src = f.attrs.get("src_file", "")
                src = src.decode() if isinstance(src, bytes) else src
                msp = msp_map.get(os.path.basename(src))
                interactions, all_recs = reco_one(kp, msp, rmom, calib, args)
                if interactions is None:
                    n_skip += 1
                    continue
                _write_event(fout.create_group(f"event_{gidx:07d}"),
                             interactions, all_recs, kp)
                n_ok += 1
                if n_ok % 100 == 0:
                    print(f"    {n_ok} done ({gidx})", flush=True)
            except Exception as e:                       # keep the shard alive
                n_err += 1
                print(f"    [ERR] {os.path.basename(kp)}: "
                      f"{type(e).__name__}: {e}", flush=True)
        fout.attrs["n_reco"] = n_ok
        fout.attrs["n_skip"] = n_skip
        fout.attrs["n_err"] = n_err
    print(f">>> DONE: {n_ok} reco, {n_skip} skipped (no slice/interaction), "
          f"{n_err} errors -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
