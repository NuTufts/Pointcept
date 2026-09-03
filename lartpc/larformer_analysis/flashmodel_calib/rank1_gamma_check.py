"""Flash-match tuning check for a NEW cascade chain (pre-campaign gate).

Answers, on ~1500 BNB-nu overlay events run through the candidate chain with
flash matching ON and --save-slice-ids:

  1. gamma tuning: median sum(obs)/sum(pred) over live PMTs on the TRUE-nu
     slice (pred_pe is stored at the baked gamma, so ratio ~1 = tuned);
  2. rank-1 rate of the truth-matched, nu-LABELED slice in flash chi2
     (old-chain satfix reference: 72.1% "nu slice is rank-1");
  3. for events whose true-nu slice was COSMIC-labeled by the slicer: the
     rate at which that slice is still rank-1 in min chi2 (the fm-stream
     rescue path);
  4. a post-hoc gamma sweep (rescaling stored pred_pe, Neyman chi2
     f_sys=0.10 eps=1.0, oob<=0.05 gate) -> rank-1 rate vs gamma, to see if
     retuning gamma would improve the ranking WITHOUT a GPU re-run.

True-nu slice = argmax-IoU slice vs the GT-nu point set (merged_sp
mc_particle_tree origin==1 trackids, exact-coordinate row match), from the
sliceid sidecar (slice_id: -5 nu union, q>=0 cosmic slice = query index,
matching slices/query in the kp2 table).

    PYTHONPATH=./ python3 .../rank1_gamma_check.py \
        --kp2-dir <cascade out> --sliceid-dir <same or sidecar dir> \
        --merged-sp-list <first-1500 list> [--iou-min 0.2] [--dead 15]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
try:
    from lartpc.flashmatch.saturation import find_saturated
except Exception:
    find_saturated = None

F_SYS, EPS, OOB_MAX = 0.10, 1.0, 0.05


def _neyman(obs, pred, live):
    o, p = obs[live], pred[live]
    return float(np.sum((o - p) ** 2 / (o + (F_SYS * o) ** 2 + EPS)))


def _gt_nu_rows(msp_path):
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        mt = e["mc_particle_tree"]
        nu_tids = set(np.asarray(mt["trackid"][()], np.int64)
                      [np.asarray(mt["origin"][()], np.int64) == 1].tolist())
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        ttid = np.asarray(td["trackid"][()], np.int64)
    isnu = np.fromiter((int(t) in nu_tids for t in ttid), bool, len(ttid))
    return {pos[i].tobytes() for i in np.nonzero(isnu)[0]}, int(isnu.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kp2-dir", required=True)
    ap.add_argument("--sliceid-dir", default=None)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--iou-min", type=float, default=0.2)
    ap.add_argument("--dead", default="15",
                    help="comma-sep dead opdets (run3: 15; run1: '')")
    args = ap.parse_args()
    sid_dir = args.sliceid_dir or args.kp2_dir
    dead = [int(x) for x in args.dead.split(",") if x.strip()]

    msp = [l.strip() for l in open(args.merged_sp_list) if l.strip()]
    kp2 = {}
    for p in glob.glob(os.path.join(args.kp2_dir, "**", "keypoint2_event*_0.h5"),
                       recursive=True):
        m = re.search(r"event(\d+)(_fm)?_0\.h5$", os.path.basename(p))
        if m and not m.group(2):
            kp2[int(m.group(1))] = p

    n = dict(ev=0, no_kp2=0, no_flash=0, no_gtnu=0, no_match=0)
    ratios = []
    # per-event record: (true-nu slice label is nu?, stored rank==1?,
    #                    per-gamma rank1 bools, iou)
    rec = []
    G = np.linspace(0.5, 1.5, 21)
    for i in range(len(msp)):
        n["ev"] += 1
        if i not in kp2:
            n["no_kp2"] += 1
            continue
        sp = os.path.join(sid_dir, f"sliceid_event{i:05d}.h5")
        if not os.path.exists(sp):
            n["no_kp2"] += 1
            continue
        try:
            gt_set, n_gt = _gt_nu_rows(msp[i])
        except Exception:
            n["no_gtnu"] += 1
            continue
        if n_gt < 20:
            n["no_gtnu"] += 1
            continue
        with h5py.File(kp2[i], "r") as fk, h5py.File(sp, "r") as fs:
            if "slices" not in fk or "flash" not in fk \
                    or "observed_pe" not in fk["flash"]:
                n["no_flash"] += 1
                continue
            S = fk["slices"]
            q = S["query"][()]
            lab = np.array([x.decode() for x in S["label"][()]])
            chi2 = S["chi2"][()].astype(np.float64)
            rank = S["chi2_rank"][()]
            pred = S["pred_pe"][()].astype(np.float64)
            oob = S["oob_frac"][()].astype(np.float64)
            obs = fk["flash"]["observed_pe"][()].astype(np.float64)
            coord = fs["full_slice"]["coord_cm"][()].astype(np.float32)
            sid = fs["full_slice"]["slice_id"][()]
        if not np.isfinite(obs).any() or obs.sum() <= 0:
            n["no_flash"] += 1
            continue
        # per-point GT-nu membership by exact coord match
        in_gt = np.fromiter((coord[j].tobytes() in gt_set
                             for j in range(len(coord))), bool, len(coord))
        # slice membership: row r of the slices table
        best_iou, best_r = 0.0, -1
        for r in range(len(q)):
            m = (sid == -5) if lab[r] == "nu" else (sid == q[r])
            ns = int(m.sum())
            if ns == 0:
                continue
            inter = int((m & in_gt).sum())
            iou = inter / (ns + n_gt - inter)
            if iou > best_iou:
                best_iou, best_r = iou, r
        if best_r < 0 or best_iou < args.iou_min:
            n["no_match"] += 1
            continue
        # gamma ratio on the true-nu slice (live = not dead, not sat-hole)
        live = np.ones(32, bool)
        live[dead] = False
        if find_saturated is not None:
            try:
                live[list(find_saturated(obs))] = False
            except Exception:
                pass
        p = pred[best_r]
        if p[live].sum() > 0:
            ratios.append(float(obs[live].sum() / p[live].sum()))
        # gamma sweep: rank of true-nu slice among oob-gated slices
        gate = oob <= OOB_MAX
        gate[best_r] = gate[best_r]      # true-nu slice subject to same gate
        r1g = []
        for g in G:
            if not gate[best_r]:
                r1g.append(False)
                continue
            c = np.array([_neyman(obs, g * pred[r], live) if gate[r]
                          else np.inf for r in range(len(q))])
            r1g.append(bool(np.argmin(c) == best_r))
        rec.append((lab[best_r] == "nu", int(rank[best_r]) == 1,
                    np.array(r1g), best_iou))

    print(f">>> events {n['ev']} | no kp2/sidecar {n['no_kp2']} | "
          f"no flash {n['no_flash']} | no GT-nu {n['no_gtnu']} | "
          f"no IoU>={args.iou_min} match {n['no_match']} | "
          f"USED {len(rec)}")
    if not rec:
        return
    isnu = np.array([r[0] for r in rec])
    r1 = np.array([r[1] for r in rec])
    sweep = np.stack([r[2] for r in rec])
    iou = np.array([r[3] for r in rec])
    ratios = np.array(ratios)
    print(f"\n== gamma tuning (true-nu slice, live PMTs) ==")
    print(f"  obs/pred ratio: median {np.median(ratios):.3f} | "
          f"mean {ratios.mean():.3f} | p10/p90 "
          f"{np.percentile(ratios,10):.3f}/{np.percentile(ratios,90):.3f} "
          f"(1.0 = baked gamma correct)")
    print(f"\n== rank-1 rates (stored, as-deployed chi2/rank) ==")
    print(f"  true-nu slice median IoU: {np.median(iou):.3f}")
    print(f"  slicer labeled true-nu slice 'nu': {isnu.mean():.3f} "
          f"({int(isnu.sum())}/{len(isnu)})")
    print(f"  nu-labeled & true-nu -> rank-1  : {r1[isnu].mean():.3f} "
          f"({int(r1[isnu].sum())}/{int(isnu.sum())})   [ref old chain 0.721]")
    if (~isnu).sum():
        print(f"  COSMIC-labeled but true-nu -> rank-1: {r1[~isnu].mean():.3f} "
              f"({int(r1[~isnu].sum())}/{int((~isnu).sum())})  [fm rescue rate]")
    print(f"  overall true-nu slice rank-1    : {r1.mean():.3f}")
    print(f"\n== gamma sweep (recomputed chi2, rank-1 rate of true-nu slice) ==")
    for gi, g in enumerate(np.linspace(0.5, 1.5, 21)):
        print(f"  g={g:4.2f}: {sweep[:,gi].mean():.3f}")


if __name__ == "__main__":
    main()
