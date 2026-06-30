"""Parameter scan for shower->nu-vertex attachment cuts (d_impact, cos_min).

Builds each event's interaction once, precomputes the trunk + connection geometry
for every (shower, connection-point) pair, then sweeps the cut grid against the
provenance truth (shower primary iff true origin within 5 cm of the true vertex).
Reports precision/recall/F1 per grid point so a working point can be chosen.

Run INSIDE the container:
    ./run_in_local_pointcept_container.sh python \
        lartpc_data_prep/larformer_keypoint_v2/trajfit_dev/scan_shower_attach.py
"""
import os
import glob
import argparse
from types import SimpleNamespace

import numpy as np

import trajfit_io as tio
import nu_interaction as ni
from shower_trunk import trunk_vertex_biased
from shower_connect import connection_geometry
from shower_truth import load_shower_fragments, shower_is_primary

# track-side params (match nu_interaction defaults); the scan varies only the
# shower-attachment cuts.
TRACK_ARGS = SimpleNamespace(no_snap=False, snap_radius=30.0, d_vertex=12.0,
                             d_perp=4.0, front_tol=1.5, merge_radius=3.0)
NUVTX_RADIUS = 5.0      # a connection point this close to the nu vtx counts as it
D_GAP = 60.0           # fixed generous gap cut (rarely binds; primaries gap<~35cm)


def build_cache(files, msp):
    """Per event -> list of showers, each {prim, cps:[{dist,is_nuvtx,impact,
    cosine,gap}]} sorted by distance-to-nu-vtx. Trunk geometry is cut-independent
    so it's computed once here."""
    cache, n_ev, n_show = [], 0, 0
    for fp in files:
        recs = tio.load_instances(fp, msp, tracks_only=True, min_points=20)
        if not recs:
            continue
        cands = ni.vertex_candidates(fp)
        if not cands:
            continue
        tracks = ni.build_tracks(recs, seg_cm=3.0, kink_tol=3.0, eps=1.2,
                                 max_gap_live=20.0)
        res = ni.best_over_candidates(cands, tracks, TRACK_ARGS)["res"]
        nuv0 = res["vertices"][0]["pos"]
        conn = [dict(pos=v["pos"]) for v in res["vertices"]]
        for T in tracks:
            if T["attached"]:
                conn += [dict(pos=kp) for kp in T.get("kinks", [])]
        for c in conn:
            c["dist"] = float(np.linalg.norm(c["pos"] - nuv0))
        conn.sort(key=lambda c: c["dist"])

        gt_nu = ni.read_nu_vertices(fp)[1]
        frag = load_shower_fragments(fp, msp)
        all_recs = tio.load_instances(fp, msp, tracks_only=False, min_points=1)
        ev = []
        for r in all_recs:
            if (r.pred_cls not in tio.SHOWER_CLASSES
                    or not np.all(np.isfinite(r.pred_start)) or r.n_points < 20):
                continue
            prim, _, _ = shower_is_primary(r.gt_start, frag, gt_nu)
            cps = []
            for c in conn:
                tk = trunk_vertex_biased(r.points, c["pos"])
                g = connection_geometry(tk.start, tk.direction, c["pos"])
                cps.append(dict(dist=c["dist"], is_nuvtx=c["dist"] <= NUVTX_RADIUS,
                                impact=g["impact"], cosine=g["cosine"], gap=g["gap"]))
            ev.append(dict(prim=prim, cps=cps))
            n_show += 1
        cache.append(ev)
        n_ev += 1
    print(f">>> cached {n_ev} events, {n_show} showers")
    return cache


def evaluate(cache, d_impact, cos_min, d_gap=D_GAP):
    """Greedy (closest-first) nu-vertex attachment P/R/F1 at one cut point."""
    tp = fp = fn = tn = 0
    for ev in cache:
        for sh in ev:
            if sh["prim"] is None:
                continue
            at_nuvtx = False
            for c in sh["cps"]:                     # sorted closest-first
                if (c["impact"] <= d_impact and c["cosine"] >= cos_min
                        and c["gap"] <= d_gap):
                    at_nuvtx = c["is_nuvtx"]
                    break
            if at_nuvtx and sh["prim"]:
                tp += 1
            elif at_nuvtx:
                fp += 1
            elif sh["prim"]:
                fn += 1
            else:
                tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data", "bnb_pi0_valdata")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--d-impact", default="5,8,10,12,15,20,30")
    ap.add_argument("--cos-min", default="0.80,0.85,0.90,0.93,0.95,0.97")
    args = ap.parse_args()
    di = [float(x) for x in args.d_impact.split(",")]
    cm = [float(x) for x in args.cos_min.split(",")]
    msp = (args.merged_sp_dir if os.path.isdir(args.merged_sp_dir) else None)

    cache = build_cache(sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5"))),
                        msp)

    # recall matrix (precision in parens) over the d_impact x cos_min grid
    print(f"\nRECALL  (precision)   rows=d_impact[cm]  cols=cos_min   d_gap={D_GAP}")
    print("d_imp\\cos | " + "  ".join(f"{c:>11.2f}" for c in cm))
    rows = []
    for d in di:
        cells = []
        for c in cm:
            m = evaluate(cache, d, c)
            cells.append(f"{m['rec']:.2f}({m['prec']:.2f})")
            rows.append((d, c, m))
        print(f"{d:8.0f} | " + "  ".join(f"{x:>11}" for x in cells))

    # ranked working points
    print("\nWorking points:")
    cur = evaluate(cache, 10.0, 0.90)
    print(f"  current (d_imp=10, cos=0.90): P={cur['prec']:.2f} R={cur['rec']:.2f} "
          f"F1={cur['f1']:.2f}")
    best_f1 = max(rows, key=lambda r: r[2]["f1"])
    print(f"  best F1: d_imp={best_f1[0]:.0f} cos={best_f1[1]:.2f} -> "
          f"P={best_f1[2]['prec']:.2f} R={best_f1[2]['rec']:.2f} F1={best_f1[2]['f1']:.2f}")
    for pmin in (1.00, 0.95, 0.90):
        ok = [r for r in rows if r[2]["prec"] >= pmin]
        if ok:
            b = max(ok, key=lambda r: r[2]["rec"])
            print(f"  max recall @ precision>={pmin:.2f}: d_imp={b[0]:.0f} "
                  f"cos={b[1]:.2f} -> P={b[2]['prec']:.2f} R={b[2]['rec']:.2f}")


if __name__ == "__main__":
    main()
