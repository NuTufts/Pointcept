"""How often does the slicer's `max_spacepoints` cap bite?

The cap in LArFormerDataset (pointcept/datasets/larformer.py, step 2c) acts on
`n_keep` = the spacepoint count AFTER two steps applied in `_load_event`:
  1. lm_score filter:  keep = lm_score >= tau   (tau=0 with the slicer config's
     lm_score_val_threshold=0 + aug 0/0, so a no-op here -- keeps all SPs)
  2. backbone dedup:   unique voxels at backbone_grid_size_cm (0.25 cm)
When n_keep > max_spacepoints the loader RANDOMLY subsamples down to the cap,
i.e. it thins the event BEFORE the deghoster/backbone ever sees it. Sparse soft
photons in big (cosmic-heavy) events are exactly what gets decimated.

This script replicates just those two steps (h5py + numpy, no torch) over a
random sample of the train list and reports the fraction of events that exceed
a range of candidate caps -- so we can pick a cap and record its bite rate.

    python3 cap_study_spacepoints.py \
        --list .../h5list_mcall_lantern_train.txt --n 4000 --seed 0
"""
import argparse
import os
import random

import numpy as np
import h5py

GRID_CM = 0.25          # DEFAULT_BACKBONE_GRID_SIZE_CM
LM_TAU = 0.0            # slicer config: lm_score_val_threshold=0, aug 0/0
CAPS = [80_000, 100_000, 150_000, 200_000, 250_000,
        300_000, 400_000, 500_000, 750_000]


def n_keep_for_event(path):
    """Post-lm-filter, post-0.25cm-dedup spacepoint count = what the cap sees."""
    with h5py.File(path, "r") as f:
        td = f["entry_0"]["triplet_data"]
        pos = td["pos"][:].astype(np.float32)
        lm = td["lm_score"][:].astype(np.float32)
    keep = lm >= LM_TAU
    pos_k = pos[keep]
    if pos_k.shape[0] == 0:
        return 0
    g = np.floor(pos_k / GRID_CM).astype(np.int64)
    g -= g.min(axis=0)
    # unique rows == number of occupied 0.25 cm voxels
    uniq = np.unique(g, axis=0)
    return int(uniq.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="h5 list file (one path/line)")
    ap.add_argument("--n", type=int, default=4000, help="random sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional .npz of per-event counts")
    # Emit a list of the biggest events (paths with count > --dump-over), up to
    # --dump-max, sorted descending. Used to build a worst-case memory smoke set.
    ap.add_argument("--dump-over", type=int, default=None)
    ap.add_argument("--dump-max", type=int, default=40)
    ap.add_argument("--dump-list", default=None)
    args = ap.parse_args()

    with open(args.list) as f:
        paths = [ln.strip() for ln in f
                 if ln.strip() and not ln.startswith("#")]
    random.seed(args.seed)
    if args.n < len(paths):
        paths = random.sample(paths, args.n)
    print(f">>> sampling {len(paths)} events from {args.list}")

    counts = []
    kept_paths = []
    missing = 0
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            missing += 1
            continue
        try:
            c = n_keep_for_event(p)
            counts.append(c)
            kept_paths.append(p)
        except Exception as e:                       # noqa: BLE001
            missing += 1
            if missing <= 5:
                print(f"    skip {p}: {e}")
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(paths)} ...")
    counts = np.asarray(counts, dtype=np.int64)
    N = counts.size
    print(f">>> got {N} events ({missing} missing/failed)")
    if N == 0:
        return

    pct = np.percentile(counts, [50, 90, 95, 99, 99.9, 100])
    print("\n=== post-dedup spacepoint count (what the cap compares against) ===")
    print(f"    mean {counts.mean():.0f}   median {pct[0]:.0f}")
    for q, v in zip([50, 90, 95, 99, 99.9, 100], pct):
        print(f"    p{q:<5} {v:>10.0f}")

    print("\n=== fraction of events EXCEEDING each candidate cap ===")
    print(f"    {'cap':>10}  {'% cut':>7}  {'lost SP among cut evts (mean frac)':>34}")
    for c in CAPS:
        over = counts > c
        f_over = 100.0 * over.mean()
        # among events that get cut, mean fraction of SPs randomly discarded
        if over.any():
            lost = 1.0 - c / counts[over]
            lost_str = f"{100 * lost.mean():.1f}% (max {100 * lost.max():.1f}%)"
        else:
            lost_str = "-"
        print(f"    {c:>10}  {f_over:>6.2f}%  {lost_str:>34}")

    if args.out:
        np.savez(args.out, counts=counts, caps=np.array(CAPS))
        print(f"\n>>> wrote {args.out}")

    if args.dump_over is not None and args.dump_list:
        order = np.argsort(counts)[::-1]
        big = [(kept_paths[j], int(counts[j])) for j in order
               if counts[j] > args.dump_over][:args.dump_max]
        with open(args.dump_list, "w") as fh:
            for pth, _c in big:
                fh.write(pth + "\n")
        print(f"\n>>> wrote {len(big)} big-event paths (> {args.dump_over}) "
              f"to {args.dump_list}"
              + (f"  [{big[0][1]}..{big[-1][1]} SP]" if big else ""))


if __name__ == "__main__":
    main()
