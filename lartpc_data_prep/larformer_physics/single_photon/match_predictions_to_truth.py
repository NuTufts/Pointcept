"""
Pass-2 efficiency: match cascade-predicted photon showers to detectable true photons.

For each event we have:
  - TRUTH (merged H5 entry_0): GT shower fragments grouped by trackid = per photon.
    Each true photon's spacepoints are triplet_data/pos[fragment indices] (cm).
    Detectable if its trunk spacepoint count nSP_trunk >= --nsp-threshold
    (proxy for >=20 MeV in a single cluster; older mcc9 sim has no true edep).
  - PREDICTION (stage3pred H5): active post-dedup queries with class==1 (photon).
    Each predicted photon's spacepoints are stage3/coord[pred_query==q] (cm).

Both point sets are in detector cm, so we voxelize at --voxel cm and match true
photons to predicted photon queries by occupied-voxel IoU (greedy, descending IoU,
above --iou-min). Then:
    TP = detectable true photon matched to a predicted photon query
    FN = detectable true photon with no match            (missed)
    FP = predicted photon query matched to no true photon (fake)
Efficiency = TP / (TP+FN), reported overall and binned by true photon energy
(the turn-on curve). Purity = TP / (TP+FP) over predicted photons.

Run inside the pointcept container (needs h5py):
  python3 match_predictions_to_truth.py --pred-dir workdir/larformer_h5 \
      --merged-dir workdir/larformer_h5 --nsp-threshold 80 --voxel 3 --iou-min 0.1 \
      --out workdir/photon_match.csv
"""
import argparse
import glob
import os
import h5py
import numpy as np

PHOTON_PDG = 22
PHOTON_CLASS = 1
NU_ORIGIN = 1


def voxset(pts, vox):
    """set of occupied integer voxel tuples for points (N,3) in cm."""
    if len(pts) == 0:
        return set()
    v = np.floor(pts / vox).astype(np.int64)
    return set(map(tuple, v))


def true_photons(mh5, vox):
    """All nu-origin true photons from mc_particle_tree (the full denominator —
    not just the depositing ones), each annotated with its GT-fragment voxels /
    nSP_trunk (0 / empty if it left no visible fragment)."""
    e = mh5["entry_0"]
    mt = e["mc_particle_tree"]
    mt_tid = mt["trackid"][:]; mt_pid = mt["pid"][:]
    mt_e = mt["energy_mev"][:]; mt_org = mt["origin"][:]
    nuvtx = (mt["nu_vertices"][:][0] if mt["nu_vertices"].shape[0] > 0
             else np.array([np.nan] * 3))

    # GT shower-fragment info keyed by photon trackid
    frag = {}   # trackid -> (voxels, nsp_trunk, nsp_total, originpt)
    if "shower_fragments" in e and int(e["shower_fragments"].attrs.get("num_fragments", 0)) > 0:
        sf = e["shower_fragments"]
        pid = sf["pid"][:]; tid = sf["trackid"][:]; istrunk = sf["istrunk"][:]
        cnt = sf["pointindices_counts"][:]; flat = sf["pointindices_flat"][:]
        originpt = sf["originpt"][:]
        offs = np.concatenate([[0], np.cumsum(cnt)])
        pos = e["triplet_data"]["pos"][:]
        for t in np.unique(tid[pid == PHOTON_PDG]):
            fr = np.where((tid == t) & (pid == PHOTON_PDG))[0]
            idx = np.unique(np.concatenate([flat[offs[i]:offs[i + 1]] for i in fr]))
            trunk = fr[istrunk[fr] == 1]
            op = originpt[trunk[0]] if len(trunk) else originpt[fr[0]]
            frag[int(t)] = (voxset(pos[idx], vox),
                            int(cnt[trunk].sum()) if len(trunk) else 0,
                            int(cnt[fr].sum()), np.asarray(op, dtype=float))

    out = []
    for i in range(len(mt_tid)):
        if mt_pid[i] != PHOTON_PDG or mt_org[i] != NU_ORIGIN:
            continue
        vx, nsp_trunk, nsp_total, op = frag.get(int(mt_tid[i]), (set(), 0, 0, None))
        dist = (float(np.linalg.norm(op - nuvtx))
                if (op is not None and np.isfinite(nuvtx).all()) else -1.0)
        out.append(dict(trackid=int(mt_tid[i]), voxels=vx, nsp_trunk=nsp_trunk,
                        nsp_total=nsp_total, trueE=float(mt_e[i]), origin=NU_ORIGIN,
                        dist=dist))
    return out


def pred_photons(ph5, vox):
    """list of dicts: per active predicted photon query — voxels."""
    if "stage3" not in ph5 or "stage3_queries" not in ph5:
        return []                       # dropped event
    s3 = ph5["stage3"]; q = ph5["stage3_queries"]
    if "class_argmax" not in q:
        return []
    co = s3["coord"][:]; pq = s3["pred_query"][:]
    ca = q["class_argmax"][:]; act = q["is_active_postdedup"][:].astype(bool)
    out = []
    for qi in np.where(act & (ca == PHOTON_CLASS))[0]:
        pts = co[pq == qi]
        out.append(dict(query=int(qi), voxels=voxset(pts, vox), npts=len(pts)))
    return out


def iou(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="workdir/larformer_h5")
    ap.add_argument("--merged-dir", default="workdir/larformer_h5")
    ap.add_argument("--nsp-threshold", type=int, default=80,
                    help="detectable if nSP_trunk >= this (proxy for >=20 MeV)")
    ap.add_argument("--voxel", type=float, default=3.0, help="voxel size cm")
    ap.add_argument("--iou-min", type=float, default=0.1,
                    help="min voxel-IoU to call a true<->pred match")
    ap.add_argument("--out", default="workdir/photon_match.csv")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "**", "stage3pred_*.h5"),
                             recursive=True))
    print("stage3pred files:", len(preds))

    rows = []           # per true photon
    n_fp = 0
    fp_rows = []
    for pf in preds:
        base = os.path.basename(pf)[len("stage3pred_"):]
        merged = None
        for c in glob.glob(os.path.join(args.merged_dir, "**", base), recursive=True):
            merged = c; break
        if merged is None:
            print("  WARN no merged H5 for", base); continue
        with h5py.File(pf, "r") as ph5, h5py.File(merged, "r") as mh5:
            run = int(mh5["entry_0"].attrs.get("run", -1))
            sub = int(mh5["entry_0"].attrs.get("subrun", -1))
            evt = int(mh5["entry_0"].attrs.get("event", -1))
            T = true_photons(mh5, args.voxel)
            P = pred_photons(ph5, args.voxel)

        # IoU matrix, greedy match (descending IoU)
        pairs = []
        for ti, t in enumerate(T):
            for pi, p in enumerate(P):
                v = iou(t["voxels"], p["voxels"])
                if v >= args.iou_min:
                    pairs.append((v, ti, pi))
        pairs.sort(reverse=True)
        t_match = {}; p_match = {}
        for v, ti, pi in pairs:
            if ti in t_match or pi in p_match:
                continue
            t_match[ti] = (pi, v); p_match[pi] = ti

        for ti, t in enumerate(T):
            det = (t["origin"] == NU_ORIGIN) and (t["nsp_trunk"] >= args.nsp_threshold)
            matched = ti in t_match
            best_iou = t_match[ti][1] if matched else 0.0
            mq = P[t_match[ti][0]]["query"] if matched else -1
            rows.append((run, sub, evt, t["trackid"], t["trueE"], t["nsp_trunk"],
                         int(t["origin"] == NU_ORIGIN), int(det), int(matched),
                         best_iou, mq, t["dist"]))
        for pi, p in enumerate(P):
            if pi not in p_match:
                n_fp += 1
                fp_rows.append((run, sub, evt, p["query"], p["npts"]))

    # write per-photon csv
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("run,subrun,event,trackid,trueE_mev,nSP_trunk,nu_origin,"
                 "detectable,matched,best_iou,matched_query,dist_to_vtx_cm\n")
        for r in rows:
            fh.write("%d,%d,%d,%d,%.3f,%d,%d,%d,%d,%.4f,%d,%.3f\n" % r)

    # ---- summary ----
    allp = rows                                          # all nu-origin true photons
    det = [r for r in rows if r[7] == 1]                 # detectable (nSP_trunk>=thr)
    tp = sum(1 for r in det if r[8] == 1)
    fn = sum(1 for r in det if r[8] == 0)
    eff = tp / float(tp + fn) if (tp + fn) else 0.0
    pur = tp / float(tp + n_fp) if (tp + n_fp) else 0.0
    BINS = [(0, 20), (20, 40), (40, 80), (80, 150), (150, 400), (400, 1e9)]
    print("=" * 70)
    print("nsp_threshold=%d  voxel=%.1fcm  iou_min=%.2f"
          % (args.nsp_threshold, args.voxel, args.iou_min))
    print("all nu-origin true photons : %d" % len(allp))
    print("detectable (nSP_trunk>=%d) : %d" % (args.nsp_threshold, len(det)))
    print("  TP (matched)             : %d" % tp)
    print("  FN (missed)              : %d" % fn)
    print("  FP (fake photon preds)   : %d" % n_fp)
    print("EFFICIENCY (detectable) TP/(TP+FN) : %.3f" % eff)
    print("PURITY                  TP/(TP+FP) : %.3f" % pur)
    print("-" * 70)

    Ea = np.array([r[4] for r in allp]); Ma = np.array([r[8] for r in allp])
    Da = np.array([r[7] for r in allp])
    print("%-14s %6s %10s %14s %14s" % ("E bin (MeV)", "Nall", "detect-frac",
                                        "eff|all (turn-on)", "eff|detectable"))
    for lo, hi in BINS:
        m = (Ea >= lo) & (Ea < hi)
        if not m.sum():
            continue
        md = m & (Da == 1)
        eff_all = Ma[m].mean()
        eff_det = Ma[md].mean() if md.sum() else float("nan")
        print("%5.0f-%-8.0f %6d %10.2f %8d/%-5d %.2f %5d/%-5d %.2f"
              % (lo, hi, m.sum(), Da[m].mean(),
                 Ma[m].sum(), m.sum(), eff_all,
                 Ma[md].sum(), md.sum(), eff_det))
    print("-" * 70)
    print("eff|all   = efficiency over ALL true photons in the bin (the turn-on)")
    print("eff|detec = efficiency over only detectable photons (nSP_trunk>=thr)")

    # ---- efficiency vs distance from nu vertex (detectable photons) ----
    Dd = np.array([r[11] for r in det]); Md = np.array([r[8] for r in det])
    valid = Dd >= 0
    print("-" * 70)
    print("efficiency vs photon distance from nu vertex (detectable photons):")
    print("%-16s %6s %10s" % ("dist (cm)", "N", "eff"))
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]:
        m = valid & (Dd >= lo) & (Dd < hi)
        if m.sum():
            print("  %4.0f-%-8.0f %6d %10.2f" % (lo, hi, m.sum(), Md[m].mean()))
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
