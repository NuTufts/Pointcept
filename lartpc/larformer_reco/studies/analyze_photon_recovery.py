"""Charge-based photon-RECOVERY aggregate for the single-visible-photon sample,
measured against the per-slice ``--all-slices`` segmenter outputs.

Complements ``analyze_photon_slices.py`` (which asks only WHERE the photon's charge
sits by slice category).  This script also asks how well the downstream particle
segmenter -- run separately on EVERY slice -- actually RECONSTRUCTS the photon, and
what the P(gamma) loose fallback adds.

For each event's visible photon (nu-origin gamma with the most charge):

  1. WHERE is its charge?  De-double-count once over the photon's true triplet
     spacepoints (calo.dedup_charge, Y-else-mean(U,V)), then partition that charge
     across the cascade categories using the full-event slice_id sidecar:
       nu  /  cosmic  /  unclustered(-1)  /  ghost(-2)  /  deghosted-or-prefiltered
     (the residual = photon charge with no matching survivor in the sidecar).

  2. How well is it RECONSTRUCTED?  Join every per-slice reco instance's points to
     the photon by coordinate (KDTree, exact subset match), and score each instance
     by charge RECALL = (photon charge it covers) / (total photon charge).  Recovery
     is thus overlap-based, NOT the strict IoU>0.3 gt-match (which is far too
     stringent for a shower split across slices).  Tracks the best instance overall,
     the best STANDARD-pass instance, and the best gamma-classed instance, so the
     loose fallback's contribution can be isolated.

IMPORTANT: ``--all-slices`` output files are named by the dataset's SORTED index,
NOT input-list order.  This script maps each merged_sp to its outputs via the
sidecar ``src_file`` attribute -- do not assume line i <-> event{i:05d}.

    python analyze_photon_recovery.py \
        --browse-list inputlists/merged_sp_valdata_singlephoton.txt \
        --all-slices-dir output/all_slices_singlephoton

Reproducibility: run the inference with --deterministic on a conforming (A100/L40S)
node, else these numbers churn run-to-run (docs/reference/LArFormer_Reproducibility.md).
Pure h5py / numpy / scipy.
"""
import os
import sys
import glob
import json
import argparse
from collections import Counter

import numpy as np
import h5py
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
import select_single_photon_events as S          # noqa: E402  visible_particles/read_list
from lartpc.larformer_reco.trajfit.calo import dedup_charge         # noqa: E402

CLASS = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
GAMMA_CLS = 1
NU_SID = -5
MATCH_CM = 0.1     # slice/photon coords are an exact subset -> tiny tolerance


def load_a_vis():
    try:
        c = json.load(open(os.path.join(_HERE, "calib", "shower_calib.json")))
        return float(c.get("gamma", 0.0253)) or 0.0253
    except Exception:
        return 0.0253


def build_base2idx(all_slices_dir):
    """basename(src_file) -> sorted event index, from the sidecars."""
    m = {}
    for sf in glob.glob(os.path.join(all_slices_dir, "sliceid_event*.h5")):
        idx = int(os.path.basename(sf)[len("sliceid_event"):-len(".h5")])
        with h5py.File(sf) as h:
            m[str(h.attrs.get("src_file", ""))] = idx
    return m


def analyze(i, msp, D, a_vis):
    vis, _ = S.visible_particles(msp, a_vis)
    gam = [v for v in vis if v[0] == "gamma"]
    if not gam:
        return None
    pt = int(max(gam, key=lambda v: v[3])[1])          # (species,tid,ke,evis,parent)
    with h5py.File(msp) as h:
        td = h["entry_0/triplet_data"]
        pos = np.asarray(td["pos"][()], np.float64)
        sel = (np.asarray(td["trackid"][()], np.int64) == pt)
        if sel.sum() == 0:
            return None
        Ppos = pos[sel]
        _, q_comb = dedup_charge(np.asarray(td["pixval"][()])[sel],
                                 np.asarray(td["tick"][()])[sel],
                                 np.asarray(td["uwire"][()])[sel],
                                 np.asarray(td["vwire"][()])[sel],
                                 np.asarray(td["ywire"][()])[sel])
    Qpt = float(q_comb.sum())
    if Qpt <= 0:
        return None
    tree = cKDTree(Ppos)

    # ---- 1. charge-location breakdown from the sidecar (all survivors) --------
    loc = dict(nu=0.0, cosmic=0.0, unclust=0.0, ghost=0.0)
    sf = os.path.join(D, f"sliceid_event{i:05d}.h5")
    if os.path.exists(sf):
        with h5py.File(sf) as h:
            fc = np.asarray(h["full_slice/coord_cm"][()], np.float64)
            sid = np.asarray(h["full_slice/slice_id"][()], np.int64)
        d, pk = tree.query(fc, k=1)
        hit = d < MATCH_CM
        psid = np.full(len(Ppos), -99, np.int64)       # -99 = no survivor (deghosted)
        psid[pk[hit]] = sid[hit]
        loc["nu"] = float(q_comb[psid == NU_SID].sum()) / Qpt
        loc["cosmic"] = float(q_comb[psid >= 0].sum()) / Qpt
        loc["unclust"] = float(q_comb[psid == -1].sum()) / Qpt
        loc["ghost"] = float(q_comb[psid == -2].sum()) / Qpt
    loc["deghosted"] = max(0.0, 1.0 - sum(loc.values()))

    # ---- 2. segmenter recovery: best instance charge-recall of the photon ----
    best = dict(recall=0.0, cls=None, loose=False, is_nu=None, conf=np.nan)
    best_std = 0.0
    best_gamma = 0.0
    for f in sorted(glob.glob(os.path.join(D, f"keypoint2_event{i:05d}_*_0.h5"))):
        lbl = os.path.basename(f).split(f"event{i:05d}_")[1].rsplit("_0.h5", 1)[0]
        is_nu = (lbl == "nu")
        with h5py.File(f) as h:
            sc = np.asarray(h["slice/coord_cm"][()], np.float64)
            if len(sc) == 0:
                continue
            d, pk = tree.query(sc, k=1)                 # slice pt -> nearest photon pt
            hit = d < MATCH_CM
            for j in range(int(h.attrs.get("n_particles", 0))):
                g = h[f"particle/{j}"]
                pidx = np.asarray(g["point_idx"][()], np.int64)
                m = hit[pidx]
                if not m.any():
                    continue
                covered = np.unique(pk[pidx[m]])         # photon pts this instance owns
                rec = float(q_comb[covered].sum()) / Qpt
                cls = int(g.attrs["cls"])
                loose = bool(g.attrs.get("loose_pass", False))
                if rec > best["recall"]:
                    best = dict(recall=rec, cls=cls, loose=loose, is_nu=is_nu,
                                conf=float(g.attrs.get("loose_conf", np.nan)))
                if not loose:
                    best_std = max(best_std, rec)
                if cls == GAMMA_CLS:
                    best_gamma = max(best_gamma, rec)
    return dict(i=i, pt=pt, Qpt=Qpt, evis=Qpt * a_vis, loc=loc,
                best=best, best_std=best_std, best_gamma=best_gamma)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--browse-list",
                    default=os.path.join(_HERE, "inputlists",
                                         "merged_sp_valdata_singlephoton.txt"),
                    help="merged_sp list of the single-photon sample")
    ap.add_argument("--all-slices-dir",
                    default=os.path.join(_HERE, "output", "all_slices_singlephoton"),
                    help="dir with keypoint2_event*_<slice>_0.h5 + sliceid_event*.h5")
    args = ap.parse_args()

    a_vis = load_a_vis()
    msps = S.read_list(args.browse_list)
    base2idx = build_base2idx(args.all_slices_dir)
    print(f">>> {len(msps)} merged_sp; {len(base2idx)} sidecars in "
          f"{args.all_slices_dir}   a_vis(gamma)={a_vis:.4f}")

    R = []
    for msp in msps:
        base = os.path.basename(msp)
        if base not in base2idx:
            R.append(dict(err=f"no output for {base}"))
            continue
        try:
            r = analyze(base2idx[base], msp, args.all_slices_dir, a_vis)
        except Exception as e:
            r = dict(err=repr(e))
        if r:
            R.append(r)
    ok = [r for r in R if "err" not in r]
    errs = [r["err"] for r in R if "err" in r]
    print(f"analyzed {len(ok)}/{len(msps)} events   errors={errs[:3]}"
          f"{' ...' if len(errs) > 3 else ''}")
    if not ok:
        return

    def Lk(k):
        return np.array([r["loc"][k] for r in ok])

    print("\n=== WHERE IS THE PHOTON'S VISIBLE CHARGE (mean fraction / event) ===")
    for k in ("nu", "cosmic", "unclust", "ghost", "deghosted"):
        print(f"  {k:10s}: {Lk(k).mean():6.1%}   "
              f"(>=50% in {int((Lk(k) >= 0.5).sum())} events)")
    print(f"  events with >=20% photon charge in a COSMIC slice: "
          f"{int((Lk('cosmic') >= 0.2).sum())}")

    rec = np.array([r["best"]["recall"] for r in ok])
    print("\n=== SEGMENTER RECOVERY (best instance, charge recall of photon) ===")
    for thr in (0.1, 0.3, 0.5, 0.7):
        print(f"  best-instance recall >= {thr:.0%}: {int((rec >= thr).sum())}/{len(ok)}")
    print(f"  mean {rec.mean():.1%}   median {np.median(rec):.1%}")

    dec = [r for r in ok if r["best"]["recall"] >= 0.3]
    cc = Counter(CLASS[r["best"]["cls"]] for r in dec if r["best"]["cls"] is not None)
    sl = Counter("nu" if r["best"]["is_nu"] else "cosmic" for r in dec)
    loose_ct = sum(1 for r in dec if r["best"]["loose"])
    print(f"\n  of {len(dec)} events with >=30% recovery:")
    print(f"    best-instance class : {dict(cc)}")
    print(f"    best-instance slice : {dict(sl)}")
    print(f"    best via LOOSE pass : {loose_ct}")

    bstd = np.array([r["best_std"] for r in ok])
    bg = np.array([r["best_gamma"] for r in ok])
    print("\n=== LOOSE FALLBACK IMPACT (best recall: standard-only vs +loose) ===")
    for thr in (0.3, 0.5):
        a, b = int((bstd >= thr).sum()), int((rec >= thr).sum())
        print(f"  events >= {thr:.0%} recall:  standard-only={a}   +loose={b}   "
              f"rescued={b - a}")
    print(f"  events with a GAMMA-classed instance >=30% recall: "
          f"{int((bg >= 0.3).sum())}/{len(ok)}")


if __name__ == "__main__":
    main()
