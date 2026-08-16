"""Where does the pi0 photons' charge go? Old vs new chain, per true photon.

For every true photon of the CC-1pi0 signal pair (the 174 pilot signal events,
truth defined exactly as in sbnd_cc1pi0.py), computes the DE-DOUBLE-COUNTED
(dedup_charge over the full event triplet set, q_comb) charge-weighted fate of
the photon's spacepoints through each chain's kp2 output:

  slicer level   : fraction of the photon's charge inside the predicted nu
                   slice (slice/coord_cm)          -> "is the slicer missing it?"
  segmenter level: within-slice charge split by the PREDICTED CLASS of the
                   particle instance each SP landed in (gamma = correct;
                   e/mu/pi/p/other = mis-ID), plus "in slice, unclustered"
                   and "not in slice"              -> "is the segmenter missing it?"

Photon SPs = triplet_data rows with trackid == the photon's TID (the repo's
direct trackid labeling, no descendant walk); the pair's TIDs come from the
same orphan-photon-pair test the selection uses. kp2 slice coords are an exact
float32 subset of triplet pos (verified per event; falls back to a 0.05 cm
KD-match if the exact join underperforms).

Outputs (to --out-dir): per-photon records npz, summary txt, and three PNGs
(slice-completeness distribution, per-photon gamma-fraction distribution,
stacked mean charge-fate bars).

    PYTHONPATH=./ python3 pi0_photon_charge_flow.py \
        --ntuple .../dlgen2_pilot_old_bnbnu_pred.root \
        --pilot-list .../merged_sp_mcc9_bnbnu_satfix_pilot10k.txt \
        --old-kp2-list .../keypoint2_out_bnbnu_satfix_pilot10k_nu.txt \
        --new-kp2-list .../keypoint2_out_kp2v2_bnbnu_pilot10k_nu.txt \
        --out-dir .../pilot_ntuples/photon_charge_flow
"""
import argparse
import os
import re
import sys

import numpy as np
import h5py
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pi0mass_peak"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
from sbnd_cc1pi0 import (_BR, _BR_MC, sbnd_truth_cat, A_GAMMA,   # noqa: E402
                         EVIS_MIN)
from lartpc.larformer_reco.trajfit.calo import dedup_charge      # noqa: E402

# fate buckets, stack order (bottom -> top). First 6 = predicted class of the
# instance the charge landed in (repo class ids 0..5); then the two non-class
# fates. Colors follow the repo's particle-type convention; the non-class
# buckets are grays with hatch + direct labels as the secondary encoding.
BUCKETS = ["gamma", "e", "mu", "pi", "p", "other", "unclustered", "not in slice"]
CLS_TO_BUCKET = {1: 0, 0: 1, 2: 2, 3: 3, 4: 4, 5: 5}   # cls id -> bucket idx
COLORS = ["#ff7f0e", "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
          "#8c564b", "#d9d9d9", "#4d4d4d"]
HATCH = [None, None, None, None, None, None, "//", "xx"]
# hatch inherits edgecolor: dark hatch on the light bucket, white on the dark
EDGEC = ["white"] * 6 + ["#7a7a7a", "white"]


def photon_pair_tids(a, i):
    """TIDs + visible energies of the signal pair (same orphan-pair test as
    sbnd_cc1pi0._pi0_detectable, but returning identities)."""
    pdg = np.asarray(a["trueSimPartPDG"][i])
    proc = np.asarray(a["trueSimPartProcess"][i])
    mid = np.asarray(a["trueSimPartMID"][i])
    tid = np.asarray(a["trueSimPartTID"][i])
    tset = set(tid.tolist())
    q = np.asarray(a["trueSimPartPixelSumQ"][i])
    ph = (np.abs(pdg) == 22) & (proc == 1)
    orphan = ph & np.asarray([int(m) not in tset for m in mid], bool)
    if not orphan.any():
        return None
    mids, cnt = np.unique(mid[orphan], return_counts=True)
    for m, c in zip(mids, cnt):
        if c != 2:
            continue
        sel = orphan & (mid == m)
        evis = A_GAMMA * np.clip(q[sel], 0, None)
        if np.all(evis > EVIS_MIN):
            return tid[sel].tolist(), evis.tolist()
    return None


def build_kp2_index(list_path):
    """event index -> kp2 path (from keypoint2_event{i:05d}_... names)."""
    out = {}
    for line in open(list_path):
        p = line.strip()
        m = re.search(r"keypoint2_event0*(\d+)_", os.path.basename(p))
        if p and m:
            out[int(m.group(1))] = p
    return out


GRID_CM = 0.25   # LArFormerDataset DEFAULT_BACKBONE_GRID_SIZE_CM: the dataset
                 # dedups the event cloud to ONE representative SP per 0.25 cm
                 # cell (original coords kept), and slice/coord_cm + instance
                 # point_idx live on that deduped set. Fate is therefore a
                 # CELL property: every triplet SP inherits its cell's fate.


def _cell_keys(pos):
    """(N,) bytes keys of the 0.25 cm grid cells (absolute floor, so the same
    cell hashes identically for the full triplet set and the kp2 reps)."""
    g = np.floor(np.asarray(pos, np.float64) / GRID_CM).astype(np.int64)
    g = np.ascontiguousarray(g)
    return [g[j].tobytes() for j in range(len(g))]


def read_kp2(kp_path):
    """(slice rep coords, per-instance (cls, rep idx)) from a kp2 file."""
    with h5py.File(kp_path, "r") as f:
        sc = np.ascontiguousarray(f["slice/coord_cm"][()], np.float32)
        insts = []
        if "particle" in f:
            for k in f["particle"]:
                g = f[f"particle/{k}"]
                insts.append((int(g.attrs["cls"]),
                              g["point_idx"][()].astype(np.int64)))
    return sc, insts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True,
                    help="any bnbnu pilot ntuple (truth is cell-independent)")
    ap.add_argument("--pilot-list", required=True)
    ap.add_argument("--old-kp2-list", required=True)
    ap.add_argument("--new-kp2-list", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    msp = [l.strip() for l in open(args.pilot_list) if l.strip()]
    kp2 = {"old": build_kp2_index(args.old_kp2_list),
           "new": build_kp2_index(args.new_kp2_list)}

    t = uproot.open(args.ntuple)["EventTree"]
    br = list(_BR) + list(_BR_MC)
    a = t.arrays([b for b in br if b in set(t.keys())])
    n = len(a["run"])
    assert n == len(msp), f"ntuple rows {n} != pilot list {len(msp)}"
    cat = np.array([sbnd_truth_cat(a, j)[0] for j in range(n)])
    sig_idx = np.nonzero(cat == 0)[0]
    print(f">>> {len(sig_idx)} signal events; universe {n}")

    rec = dict(event=[], tid=[], evis=[], is_lead=[], q_total=[],
               frac=dict((c, {b: [] for b in BUCKETS}) for c in ("old", "new")),
               slice_match=dict(old=[], new=[]))
    n_nofile = dict(old=0, new=0)
    n_badmatch = 0

    for ev in sig_idx:
        pair = photon_pair_tids(a, ev)
        if pair is None:
            continue
        tids, evis = pair
        with h5py.File(msp[ev], "r") as f:
            td = f["entry_0/triplet_data"]
            tpos = np.ascontiguousarray(td["pos"][()], np.float32)
            trackid = td["trackid"][()].astype(np.int64)
            _, q_comb = dedup_charge(td["pixval"][()], td["tick"][()],
                                     td["uwire"][()], td["vwire"][()],
                                     td["ywire"][()])
        tkeys = _cell_keys(tpos)

        # per-chain fate per triplet SP = fate of its 0.25 cm cell's rep
        fate = {}
        for chain in ("old", "new"):
            sp_fate = np.full(len(tpos), 7, np.int8)          # not in slice
            path = kp2[chain].get(int(ev))
            if path is None:
                n_nofile[chain] += 1
            else:
                sc, insts = read_kp2(path)
                skeys = _cell_keys(sc)
                cell_fate = {k: 6 for k in skeys}             # unclustered
                for cls, pidx in insts:                       # first-wins
                    b = CLS_TO_BUCKET.get(cls, 5)
                    for j in pidx:
                        if j < len(skeys) and cell_fate.get(skeys[j]) == 6:
                            cell_fate[skeys[j]] = b
                sp_fate = np.array([cell_fate.get(k, 7) for k in tkeys],
                                   np.int8)
                # diagnostic: fraction of slice reps whose cell holds >=1
                # triplet SP (should be ~1; <1 => coordinate mismatch)
                tset = set(tkeys)
                rec["slice_match"][chain].append(
                    float(np.mean([k in tset for k in skeys]))
                    if skeys else 1.0)
                if rec["slice_match"][chain][-1] < 0.99:
                    n_badmatch += 1
            fate[chain] = sp_fate

        lead = int(np.argmax(evis))
        for k, (tid, ev_g) in enumerate(zip(tids, evis)):
            m = trackid == int(tid)
            qt = float(q_comb[m].sum())
            if qt <= 0:
                continue
            rec["event"].append(int(ev)); rec["tid"].append(int(tid))
            rec["evis"].append(float(ev_g)); rec["is_lead"].append(k == lead)
            rec["q_total"].append(qt)
            for chain in ("old", "new"):
                for bi, b in enumerate(BUCKETS):
                    rec["frac"][chain][b].append(
                        float(q_comb[m & (fate[chain] == bi)].sum()) / qt)

    nph = len(rec["q_total"])
    q = np.asarray(rec["q_total"])
    print(f">>> {nph} photons  | kp2 missing: old {n_nofile['old']} "
          f"new {n_nofile['new']} events | slice-coord exact-match "
          f"old {np.mean(rec['slice_match']['old']):.4f} "
          f"new {np.mean(rec['slice_match']['new']):.4f} "
          f"| events with match<0.99: {n_badmatch}")

    F = {c: {b: np.asarray(rec["frac"][c][b]) for b in BUCKETS}
         for c in ("old", "new")}
    inslice = {c: 1.0 - F[c]["not in slice"] for c in ("old", "new")}
    lines = [f"pi0 photon charge flow — {nph} photons from {len(sig_idx)} "
             f"signal events (dedup q_comb; charge-weighted means)",
             f"kp2 file missing (no nu slice): old {n_nofile['old']}  "
             f"new {n_nofile['new']} events", ""]
    lines.append(f"{'':14s}{'OLD':>10s}{'NEW':>10s}   (charge-weighted mean "
                 "fraction of each photon's charge)")
    lines.append(f"{'in nu slice':14s}"
                 f"{np.average(inslice['old'], weights=q):10.3f}"
                 f"{np.average(inslice['new'], weights=q):10.3f}")
    for b in BUCKETS:
        lines.append(f"{b:14s}{np.average(F['old'][b], weights=q):10.3f}"
                     f"{np.average(F['new'][b], weights=q):10.3f}")
    for thr, lab in ((0.5, "0.5"), (0.2, "0.2")):
        lines.append("")
        for c in ("old", "new"):
            nlow_s = int((inslice[c] < thr).sum())
            nlow_g = int((F[c]["gamma"] < thr).sum())
            lines.append(f"photons with <{lab} of charge in slice ({c}): "
                         f"{nlow_s:3d}/{nph}   in gamma-classed instance: "
                         f"{nlow_g:3d}/{nph}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    np.savez(os.path.join(args.out_dir, "photon_records.npz"),
             event=rec["event"], tid=rec["tid"], evis=rec["evis"],
             is_lead=rec["is_lead"], q_total=q,
             **{f"{c}_{b.replace(' ', '_')}": F[c][b]
                for c in ("old", "new") for b in BUCKETS})

    # ---- fig 1: slice-level completeness distribution -----------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, 1, 26)
    for c, col in (("old", "#5c5c5c"), ("new", "#1f77b4")):
        wm = np.average(inslice[c], weights=q)
        ax.hist(np.clip(inslice[c], 0, 1 - 1e-9), bins=bins, weights=q,
                histtype="step", lw=2, color=col,
                label=f"{c} slicer  (q-wgt mean {wm:.3f})")
    ax.set_xlabel("fraction of photon charge in predicted nu slice (dedup)")
    ax.set_ylabel("summed photon charge / bin")
    ax.set_title(f"CC-1pi0 signal photons ({nph}): slicer-level completeness")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "slice_completeness.png"), dpi=140)

    # ---- fig 2: gamma-classed fraction distribution -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c, col in (("old", "#5c5c5c"), ("new", "#1f77b4")):
        wm = np.average(F[c]["gamma"], weights=q)
        ax.hist(np.clip(F[c]["gamma"], 0, 1 - 1e-9), bins=bins, weights=q,
                histtype="step", lw=2, color=col,
                label=f"{c} chain  (q-wgt mean {wm:.3f})")
    ax.set_xlabel("fraction of photon charge in gamma-classed instances (dedup)")
    ax.set_ylabel("summed photon charge / bin")
    ax.set_title("segmenter-level: charge correctly labeled gamma")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "gamma_fraction.png"), dpi=140)

    # ---- fig 3: stacked mean charge fate ------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    xs = [0, 1]
    bottoms = np.zeros(2)
    for bi, b in enumerate(BUCKETS):
        vals = np.array([np.average(F["old"][b], weights=q),
                         np.average(F["new"][b], weights=q)])
        ax.bar(xs, vals, bottom=bottoms, width=0.55, color=COLORS[bi],
               hatch=HATCH[bi], edgecolor=EDGEC[bi], linewidth=1.5, label=b)
        for x, v, bo in zip(xs, vals, bottoms):
            if v >= 0.03:
                ax.text(x, bo + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if bi in (2, 4, 5, 7) else "#1a1a1a")
        bottoms += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(["old chain", "new chain"])
    ax.set_ylabel("charge-weighted mean fraction of photon charge")
    ax.set_ylim(0, 1.02)
    ax.set_title("fate of CC-1pi0 signal-photon charge (dedup, per true photon)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="instance class / fate")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "charge_fate_stack.png"), dpi=140)
    print(f">>> wrote plots + summary to {args.out_dir}")


if __name__ == "__main__":
    main()
