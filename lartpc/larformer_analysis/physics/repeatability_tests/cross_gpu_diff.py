"""Diff two capture-dirs from capture_cascade_tensors.py (same events, two GPUs)
and attribute the run-to-run variability stage-by-stage.

Answers:
  - Stage 1 (input identical across GPUs): how much does P(real) diverge, how
    many keep-decisions flip, and how knife-edge are those flips (margin to τ)?
    This is the CLEAN, un-confounded source measurement.
  - Stage 2/3 (coordinate-aligned): among spacepoints that survive on BOTH GPUs,
    how many slice / particle-class assignments flip? How many points are
    gained/lost purely because Stage-1 keep flipped (divergence propagation)?
  - Margin analysis: what fraction of flips sit within ε of their threshold —
    i.e. how much a dead-band / hysteresis / higher-precision decision could
    recover. (The headline number for evaluating workarounds offline.)

  python cross_gpu_diff.py <captureDirA> <captureDirB> [--vox 0.3] [--csv out.csv]
"""
import argparse
import glob
import os

import numpy as np


def _load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "capture_*.npz")):
        out[os.path.basename(p)] = p
    return out


def _voxkeys(pos, vox):
    """Per-row hashable keys from quantized coords. `vox` sets the rounding
    (cm). Surviving-point coords are byte-identical across runs/GPUs (the
    deghoster selects points, it does not recompute coords), so this matches
    co-surviving points exactly; `vox` only guards float-repr noise. Keys are
    the quantized-int rows' bytes — collision-free regardless of coord range."""
    q = np.round(pos / vox).astype(np.int64)
    return [row.tobytes() for row in q]


def _match(posA, posB, vox):
    """Return (idxA, idxB) of points sharing a coord key (1-1, first match)."""
    ka, kb = _voxkeys(posA, vox), _voxkeys(posB, vox)
    bmap = {}
    for j, k in enumerate(kb):
        bmap.setdefault(k, j)
    ia, ib = [], []
    used = set()
    for i, k in enumerate(ka):
        j = bmap.get(k)
        if j is not None and j not in used:
            ia.append(i); ib.append(j); used.add(j)
    return np.asarray(ia, np.int64), np.asarray(ib, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirA"); ap.add_argument("dirB")
    ap.add_argument("--vox", type=float, default=0.01,
                    help="coord quantization (cm) for Stage 2/3 alignment. Coords "
                         "are byte-identical across runs/GPUs, so this only needs "
                         "to keep distinct points apart, not absorb float noise.")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1],
                    help="margin thresholds for the knife-edge analysis")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    A, B = _load(args.dirA), _load(args.dirB)
    common = sorted(set(A) & set(B))
    print(f"dirA={args.dirA}\ndirB={args.dirB}\ncommon events: {len(common)}\n")
    if not common:
        raise SystemExit("no common capture files")

    # accumulators
    s1_in_total = s1_keepflip = 0
    s1_dpr_max = []                       # per-event max |Δ p_real|
    s1_flip_margins = []                  # |p_real - τ| for flipped input SPs
    s1_pos_mismatch = 0                   # events where input coords are NOT identical
    s2_common = s2_qflip = s2_cflip = s2_only = 0
    s3_common = s3_cflip = s3_only = 0
    drop_flip = 0
    rows = []

    for n in common:
        a = np.load(A[n], allow_pickle=True); b = np.load(B[n], allow_pickle=True)
        da, db = int(a.get("dropped", 0)), int(b.get("dropped", 0))
        if da != db:
            drop_flip += 1

        # ---- Stage 1: index-aligned (input identical) ----
        row = dict(event=n.replace("capture_", "").replace(".npz", ""))
        if "s1_p_real" in a.files and "s1_p_real" in b.files:
            pa, pb = a["s1_p_real"], b["s1_p_real"]
            posa, posb = a["s1_pos"], b["s1_pos"]
            if pa.shape == pb.shape:
                if posa.shape == posb.shape and not np.array_equal(posa, posb):
                    s1_pos_mismatch += 1
                d = np.abs(pa - pb)
                s1_dpr_max.append(float(d.max()) if d.size else 0.0)
                ka, kb = a["s1_keep"], b["s1_keep"]
                flips = ka != kb
                nfl = int(flips.sum())
                s1_in_total += len(pa); s1_keepflip += nfl
                tau = float(a["tau"])
                if nfl:
                    # margin = how far the (averaged) P(real) sat from τ at a flip
                    m = np.abs(0.5 * (pa[flips] + pb[flips]) - tau)
                    s1_flip_margins.append(m)
                row.update(n_in=len(pa), dpr_max=float(d.max()) if d.size else 0.0,
                           keepflip=nfl)

        # ---- Stage 2: coord-aligned slice assignment ----
        if "s2_pos" in a.files and "s2_pos" in b.files and len(a["s2_pos"]) and len(b["s2_pos"]):
            ia, ib = _match(a["s2_pos"], b["s2_pos"], args.vox)
            s2_common += len(ia)
            s2_only += (len(a["s2_pos"]) - len(ia)) + (len(b["s2_pos"]) - len(ib))
            if len(ia):
                qf = int((a["s2_pred_query"][ia] != b["s2_pred_query"][ib]).sum())
                cf = int((a["s2_pred_class"][ia] != b["s2_pred_class"][ib]).sum())
                s2_qflip += qf; s2_cflip += cf
                row.update(s2_common=len(ia), s2_qflip=qf, s2_cflip=cf)

        # ---- Stage 3: coord-aligned particle class ----
        if "s3_pos" in a.files and "s3_pos" in b.files and len(a["s3_pos"]) and len(b["s3_pos"]):
            ia, ib = _match(a["s3_pos"], b["s3_pos"], args.vox)
            s3_common += len(ia)
            s3_only += (len(a["s3_pos"]) - len(ia)) + (len(b["s3_pos"]) - len(ib))
            if len(ia):
                cf = int((a["s3_class"][ia] != b["s3_class"][ib]).sum())
                s3_cflip += cf
                row.update(s3_common=len(ia), s3_cflip=cf)
        rows.append(row)

    margins = np.concatenate(s1_flip_margins) if s1_flip_margins else np.zeros(0)

    def pct(x, d): return (100.0 * x / d) if d else float("nan")
    print("=" * 70)
    print("STAGE 1  (deghoster P(real); input identical -> PURE GPU divergence)")
    print(f"  events with non-identical input coords : {s1_pos_mismatch} (should be 0)")
    if s1_dpr_max:
        a = np.array(s1_dpr_max)
        print(f"  |Δ P(real)| per-event max: median={np.median(a):.2e} "
              f"p95={np.percentile(a,95):.2e} worst={a.max():.2e}")
    print(f"  keep-mask FLIPS: {s1_keepflip} / {s1_in_total} SPs ({pct(s1_keepflip,s1_in_total):.4f}%)")
    if margins.size:
        print(f"  flip margin |P(real)-τ|: median={np.median(margins):.3f} "
              f"min={margins.min():.4f}")
        for e in args.eps:
            within = int((margins < e).sum())
            print(f"    flips within ε={e:<5g} of τ : {within}/{margins.size} "
                  f"({pct(within,margins.size):.1f}%)  -> recoverable by a {e:g} dead-band")
    print("-" * 70)
    print("STAGE 2  (slicer; spacepoints surviving on BOTH GPUs, coord-matched)")
    print(f"  matched SPs: {s2_common}   |  gained/lost (keep-flip propagation): {s2_only}")
    print(f"  slice-query flips: {s2_qflip} ({pct(s2_qflip,s2_common):.4f}%)   "
          f"slicer-class flips: {s2_cflip} ({pct(s2_cflip,s2_common):.4f}%)")
    print("-" * 70)
    print("STAGE 3  (particle segmenter; coord-matched)")
    print(f"  matched SPs: {s3_common}   |  gained/lost: {s3_only}")
    print(f"  particle-class flips: {s3_cflip} ({pct(s3_cflip,s3_common):.4f}%)")
    print("-" * 70)
    print(f"EVENT drop-flag flips: {drop_flip} / {len(common)} ({pct(drop_flip,len(common)):.1f}%)")
    print("=" * 70)

    if args.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote per-event detail: {args.csv}")


if __name__ == "__main__":
    main()
