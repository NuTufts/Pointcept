"""Offline evaluation of a dead-band / hysteresis at the deghoster keep-cut,
using capture_cascade_tensors.py captures from two runs (e.g. two GPUs).

The deghoster keep decision `keep = P(real) > tau` flips for points whose
P(real) sits within the ~1e-6 numerical noise of tau. A dead-band decides
ambiguous points (|P(real) - tau| <= eps) by a STABLE per-point key (a spatial
hash of the quantized coordinate, identical across runs/GPUs) instead of by the
noisy P(real). Confident points keep deciding by P(real).

This script measures, WITHOUT touching the model:
  - baseline keep-flips between the two runs (eps=0),
  - residual keep-flips under the dead-band (parity tie-break) vs eps,
  - residual keep-flips under a simpler "drop-band" rule (keep = P>tau+eps),
  - the per-event fraction of spacepoints inside the band, mean +/- std.

Stage-1 P(real) is INDEX-aligned (same input, same order across runs), so this
is the clean comparison.

  python deadband_offline.py <capdirA> <capdirB> [--tau 0.5] [--eps 0.01 0.02 ...]
"""
import argparse
import glob
import os

import numpy as np


def _spatial_parity(coord, q=0.1):
    """Stable per-point keep/drop key: parity of a spatial hash of the coord
    quantized to `q` cm. Pure function of the point -> identical across runs,
    GPUs, list-membership. Returns bool array (True = keep when ambiguous)."""
    g = np.round(coord / q).astype(np.int64)
    h = (g[:, 0] * 73856093) ^ (g[:, 1] * 19349663) ^ (g[:, 2] * 83492791)
    return (h & 1).astype(bool)


def _load(d):
    return {os.path.basename(p): p for p in glob.glob(os.path.join(d, "capture_*.npz"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirA"); ap.add_argument("dirB")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--qhash", type=float, default=0.1, help="coord quant (cm) for the hash")
    ap.add_argument("--bias-eps", type=float, default=0.1,
                    help="eps at which to check the parity for detector-position bias")
    args = ap.parse_args()
    if args.bias_eps not in args.eps:
        args.eps = sorted(set(args.eps) | {args.bias_eps})

    A, B = _load(args.dirA), _load(args.dirB)
    common = sorted(set(A) & set(B))
    print(f"dirA={args.dirA}\ndirB={args.dirB}\ncommon events: {len(common)}  tau={args.tau}\n")
    if not common:
        raise SystemExit("no common captures")

    # accumulators per eps
    base_flips = base_tot = 0
    db_flips = {e: 0 for e in args.eps}
    drop_flips = {e: 0 for e in args.eps}
    inband_frac = {e: [] for e in args.eps}     # per-event fraction in band
    pos_mismatch = 0
    # position-bias check: keep-fraction of in-band parity binned in x/y/z
    NB = 8
    AX = {"x": (0.0, 256.0, 0), "y": (-116.5, 116.5, 1), "z": (0.0, 1036.0, 2)}
    bias_cnt = {k: np.zeros(NB) for k in AX}     # total in-band points per bin
    bias_keep = {k: np.zeros(NB) for k in AX}    # of which parity==keep

    for n in common:
        a = np.load(A[n], allow_pickle=True); b = np.load(B[n], allow_pickle=True)
        if "s1_p_real" not in a.files or "s1_p_real" not in b.files:
            continue
        pa, pb = a["s1_p_real"].astype(np.float64), b["s1_p_real"].astype(np.float64)
        if pa.shape != pb.shape:
            continue
        ca = a["s1_pos"].astype(np.float32)
        if not np.array_equal(ca, b["s1_pos"].astype(np.float32)):
            pos_mismatch += 1
        par = _spatial_parity(ca, args.qhash)
        tau = args.tau

        # baseline
        ka0, kb0 = pa > tau, pb > tau
        base_flips += int((ka0 != kb0).sum()); base_tot += len(pa)

        for e in args.eps:
            # per-event in-band fraction (using run A's P(real) — what production sees)
            amb_a = np.abs(pa - tau) <= e
            inband_frac[e].append(float(amb_a.mean()))
            if e == 0.0:
                db_flips[e] += int((ka0 != kb0).sum())
                drop_flips[e] += int((ka0 != kb0).sum())
                continue
            # dead-band parity: confident by P, ambiguous by stable parity
            ka = pa > tau; kb = pb > tau
            amb_b = np.abs(pb - tau) <= e
            ka[amb_a] = par[amb_a]; kb[amb_b] = par[amb_b]
            db_flips[e] += int((ka != kb).sum())
            # drop-band: keep only confident-above
            drop_flips[e] += int(((pa > tau + e) != (pb > tau + e)).sum())

        # position-bias bins for the reference eps (run-A band + parity)
        amb_ref = np.abs(pa - tau) <= args.bias_eps
        if amb_ref.any():
            cr = ca[amb_ref]; pr = par[amb_ref]
            for k, (lo, hi, j) in AX.items():
                bi = np.clip(((cr[:, j] - lo) / (hi - lo) * NB).astype(int), 0, NB - 1)
                np.add.at(bias_cnt[k], bi, 1.0)
                np.add.at(bias_keep[k], bi, pr.astype(float))

    def pct(x): return 100.0 * x / base_tot if base_tot else float("nan")
    print(f"input-coord mismatches: {pos_mismatch} (should be 0)")
    print(f"total Stage-1 SPs compared: {base_tot}")
    print(f"BASELINE keep-flips (eps=0): {base_flips}  ({pct(base_flips):.4f}%)\n")
    print(f"{'eps':>6} | {'dead-band(parity)':>22} | {'drop-band(P>tau+eps)':>22} | "
          f"{'in-band frac/event':>22}")
    print("-" * 82)
    for e in args.eps:
        fr = np.array(inband_frac[e])
        print(f"{e:>6.3g} | {db_flips[e]:>8} flips ({pct(db_flips[e]):>6.4f}%) | "
              f"{drop_flips[e]:>8} flips ({pct(drop_flips[e]):>6.4f}%) | "
              f"mean={fr.mean()*100:>6.3f}% std={fr.std()*100:>6.3f}%")
    print("-" * 82)
    print("dead-band(parity): confident points by P(real); |P-tau|<=eps by stable coord-parity.")
    print("Lower flips => more reproducible. in-band frac = fraction of an event's SPs decided")
    print("by the stable key (the physics perturbation vs the FP32 baseline).")

    # ---- position-bias diagnostic (the user's concern) ----
    print(f"\nPOSITION-BIAS CHECK (eps={args.bias_eps}): in-band parity keep-fraction per "
          f"detector bin (flat ~50% = no position bias)")
    for k in ("x", "y", "z"):
        c, kp = bias_cnt[k], bias_keep[k]
        frac = np.where(c > 0, 100.0 * kp / np.maximum(c, 1), np.nan)
        tot = c.sum()
        overall = 100.0 * kp.sum() / tot if tot else float("nan")
        spread = np.nanmax(frac) - np.nanmin(frac)
        print(f"  {k}: overall keep={overall:5.2f}%  per-bin=[" +
              " ".join(f"{f:4.1f}" for f in frac) + f"]  spread={spread:4.1f}%")
    print("  (spread within a few % of statistical noise = parity is position-neutral)")


if __name__ == "__main__":
    main()
