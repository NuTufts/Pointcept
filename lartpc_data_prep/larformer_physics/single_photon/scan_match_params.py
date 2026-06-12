"""
Sensitivity scan of the photon match parameters (voxel size, IoU threshold,
nSP detectability threshold) for match_predictions_to_truth.py.

Loads each event's raw truth-photon and predicted-photon point sets ONCE, then
sweeps the grid in memory:
  - voxel cm        : affects the occupied-voxel sets (re-voxelized per value)
  - iou_min         : affects which true<->pred pairs count as a match
  - nsp_threshold   : affects the detectable denominator only (cheap inner loop)

Reports efficiency = TP/(TP+FN) over detectable photons, and purity =
matched_queries / active_photon_queries (nSP-independent), so we can see how stable
the headline numbers are.

Run inside the pointcept container:
  python3 scan_match_params.py --pred-dir workdir/larformer_h5 --merged-dir workdir/larformer_h5
"""
import argparse
import glob
import os
import h5py
import numpy as np

from match_predictions_to_truth import voxset, iou, PHOTON_PDG, PHOTON_CLASS, NU_ORIGIN

VOXELS = [2.0, 3.0, 5.0]
IOU_MINS = [0.05, 0.10, 0.20, 0.30]
NSP_THRS = [50, 80, 120, 200]
BASE = (3.0, 0.10, 80)


def load_event(pred_file, merged_file):
    """raw points (cm) per true nu photon and per active predicted photon query."""
    with h5py.File(merged_file, "r") as mh5:
        e = mh5["entry_0"]
        mt = e["mc_particle_tree"]
        mt_tid = mt["trackid"][:]; mt_pid = mt["pid"][:]
        mt_e = mt["energy_mev"][:]; mt_org = mt["origin"][:]
        frag = {}
        if "shower_fragments" in e and int(e["shower_fragments"].attrs.get("num_fragments", 0)) > 0:
            sf = e["shower_fragments"]
            pid = sf["pid"][:]; tid = sf["trackid"][:]; istrunk = sf["istrunk"][:]
            cnt = sf["pointindices_counts"][:]; flat = sf["pointindices_flat"][:]
            offs = np.concatenate([[0], np.cumsum(cnt)])
            pos = e["triplet_data"]["pos"][:]
            for t in np.unique(tid[pid == PHOTON_PDG]):
                fr = np.where((tid == t) & (pid == PHOTON_PDG))[0]
                idx = np.unique(np.concatenate([flat[offs[i]:offs[i + 1]] for i in fr]))
                trunk = fr[istrunk[fr] == 1]
                frag[int(t)] = (pos[idx], int(cnt[trunk].sum()) if len(trunk) else 0)
        truth = []
        for i in range(len(mt_tid)):
            if mt_pid[i] != PHOTON_PDG or mt_org[i] != NU_ORIGIN:
                continue
            pts, nsp = frag.get(int(mt_tid[i]), (np.zeros((0, 3)), 0))
            truth.append(dict(pts=pts, nsp_trunk=nsp, trueE=float(mt_e[i])))
    with h5py.File(pred_file, "r") as ph5:
        pred = []
        if "stage3" in ph5 and "stage3_queries" in ph5 and "class_argmax" in ph5["stage3_queries"]:
            co = ph5["stage3"]["coord"][:]; pq = ph5["stage3"]["pred_query"][:]
            ca = ph5["stage3_queries"]["class_argmax"][:]
            act = ph5["stage3_queries"]["is_active_postdedup"][:].astype(bool)
            for qi in np.where(act & (ca == PHOTON_CLASS))[0]:
                pred.append(co[pq == qi])
    return truth, pred


def evaluate(events, vox, iou_min, nsp):
    """events: list of (truth list, pred list) with RAW pts. returns eff, purity, counts."""
    det = tp = fn = fp = matched_q = total_q = 0
    for truth, pred in events:
        tvox = [(voxset(t["pts"], vox), t["nsp_trunk"]) for t in truth]
        pvox = [voxset(p, vox) for p in pred]
        total_q += len(pvox)
        pairs = []
        for ti, (tv, _) in enumerate(tvox):
            for pi, pv in enumerate(pvox):
                v = iou(tv, pv)
                if v >= iou_min:
                    pairs.append((v, ti, pi))
        pairs.sort(reverse=True)
        tmatch = {}; pmatch = {}
        for v, ti, pi in pairs:
            if ti in tmatch or pi in pmatch:
                continue
            tmatch[ti] = pi; pmatch[pi] = ti
        matched_q += len(pmatch)
        fp += len(pvox) - len(pmatch)
        for ti, (_, nsp_trunk) in enumerate(tvox):
            if nsp_trunk >= nsp:
                det += 1
                if ti in tmatch:
                    tp += 1
                else:
                    fn += 1
    eff = tp / float(tp + fn) if (tp + fn) else float("nan")
    pur = matched_q / float(total_q) if total_q else float("nan")
    return dict(det=det, tp=tp, fn=fn, fp=fp, eff=eff, pur=pur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="workdir/larformer_h5")
    ap.add_argument("--merged-dir", default="workdir/larformer_h5")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "**", "stage3pred_*.h5"),
                             recursive=True))
    events = []
    for pf in preds:
        base = os.path.basename(pf)[len("stage3pred_"):]
        m = None
        for c in glob.glob(os.path.join(args.merged_dir, "**", base), recursive=True):
            m = c; break
        if m is None:
            continue
        events.append(load_event(pf, m))
    print("events loaded:", len(events))

    # purity depends only on (voxel, iou_min); efficiency also on nsp.
    print("\n=== EFFICIENCY  TP/(TP+FN)  (rows=iou_min, cols=nSP thr) ===")
    for vox in VOXELS:
        print("\nvoxel = %.1f cm" % vox)
        print("  iou\\nSP   " + "".join("%8d" % n for n in NSP_THRS))
        for im in IOU_MINS:
            cells = []
            for nsp in NSP_THRS:
                r = evaluate(events, vox, im, nsp)
                cells.append("%8.3f" % r["eff"])
            print("  %6.2f   %s" % (im, "".join(cells)))

    print("\n=== PURITY  matched_q/active_q  (nSP-independent; rows=iou_min) ===")
    print("  iou\\vox  " + "".join("%9.1f" % v for v in VOXELS))
    for im in IOU_MINS:
        cells = []
        for vox in VOXELS:
            r = evaluate(events, vox, im, BASE[2])
            cells.append("%9.3f" % r["pur"])
        print("  %6.2f  %s" % (im, "".join(cells)))

    b = evaluate(events, *BASE)
    print("\nbaseline (voxel=%.1f, iou_min=%.2f, nSP=%d): eff=%.3f purity=%.3f "
          "(det=%d TP=%d FN=%d FP=%d)"
          % (BASE[0], BASE[1], BASE[2], b["eff"], b["pur"], b["det"], b["tp"], b["fn"], b["fp"]))


if __name__ == "__main__":
    main()
