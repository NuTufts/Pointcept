"""Verify a stage-1+2 cache after a (re)build.

Five phases; exits non-zero if any FAIL:

  1. COVERAGE  — h5 + .skipped counts per split vs --expect-<split>;
                 coverage = (h5 + skipped) / expected must be >= --min-coverage.
  2. INTEGRITY — open EVERY h5 (attrs + one dataset shape) in a process
                 pool; truncated/corrupt files (e.g. from preempted writer
                 tasks) are listed to <cache>/bad_files_<split>.txt.
                 Delete those files and re-run the (idempotent) build array
                 to repair, then re-verify.
  3. CONTENT   — random sample of --sample files: required entry_0 datasets
                 and groups present; attrs deghost_tau == --expect-deghost-tau,
                 tau_loose_floor == --expect-tau-loose-floor, n_in_cache > 0.
  4. DATASET   — LArFormerStage12CacheDataset per split: len(), then
                 --dataset-sample random __getitem__ reads through the full
                 loading pipeline; aggregates n_spacepoints / n_gt_instances
                 stats and checks coords are finite.
  5. AUGMENT   — reports whether per-SP `particle_class_id` is present
                 (required by the stage-3 voxel_4cm soft-presence cls head;
                 if absent, run augment_stage12_cache_particle_class_id.py
                 before the stage-3 retrain).

Usage:
    python3 tools/larformer/verify_stage12_cache.py \
        --cache-root <root> --splits train,val \
        --expect-train 410000 --expect-val 10000 \
        [--sample 500] [--dataset-sample 200] [--workers 8]
        [--quick]   # counts + small samples only (skip full integrity scan)
"""

import argparse
import os
import random
import sys
from glob import glob
from multiprocessing import Pool

import h5py
import numpy as np

REQUIRED_DATASETS = (
    "coord", "coord_norm", "deghost_p_real", "feat", "hasmatch",
    "lm_score", "origin_label", "pid", "slice_id", "source_mask",
    "ssnet_label",
)
# particle_instances is only written when the event has GT particle
# instances (n_particle_instances > 0) — checked conditionally below.
REQUIRED_GROUPS = ("slicer",)


def _check_openable(path):
    """Cheap integrity probe: open, touch attrs + one dataset's shape."""
    try:
        with h5py.File(path, "r") as h:
            g = h["entry_0"]
            _ = dict(h.attrs)
            _ = g["coord"].shape
        return None
    except Exception as ex:
        return (path, repr(ex)[:200])


def phase_coverage(root, split, expected, min_coverage):
    h5s = glob(os.path.join(root, split, "**", "*.h5"), recursive=True)
    skipped = glob(os.path.join(root, split, "**", "*.skipped"),
                   recursive=True)
    n_h5, n_skip = len(h5s), len(skipped)
    cov = (n_h5 + n_skip) / expected if expected else float("nan")
    ok = (expected is None) or (cov >= min_coverage)
    print(f"[coverage] {split}: h5={n_h5}  skipped={n_skip}  "
          f"expected={expected}  coverage={cov:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, h5s


def phase_integrity(root, split, h5s, workers):
    print(f"[integrity] {split}: scanning {len(h5s)} files "
          f"with {workers} workers ...")
    bad = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_check_openable, h5s,
                                                    chunksize=256)):
            if res is not None:
                bad.append(res)
            if (i + 1) % 50000 == 0:
                print(f"  ...{i + 1}/{len(h5s)} scanned, {len(bad)} bad")
    if bad:
        out = os.path.join(root, f"bad_files_{split}.txt")
        with open(out, "w") as f:
            for p, e in bad:
                f.write(f"{p}\t{e}\n")
        print(f"[integrity] {split}: FAIL — {len(bad)} corrupt/truncated "
              f"files -> {out}")
        print("            Delete them and re-run the build array "
              "(idempotent) to repair.")
        return False
    print(f"[integrity] {split}: PASS — 0 corrupt files")
    return True


def phase_content(split, h5s, n_sample, exp_tau, exp_floor):
    sample = random.sample(h5s, min(n_sample, len(h5s)))
    n_bad = 0
    n_cache_vals, keepfrac_vals = [], []
    for p in sample:
        try:
            with h5py.File(p, "r") as h:
                g = h["entry_0"]
                a = h.attrs
                missing = [k for k in REQUIRED_DATASETS if k not in g]
                missing += [k for k in REQUIRED_GROUPS if k not in g]
                if int(a.get("n_particle_instances", 0)) > 0 \
                        and "particle_instances" not in g:
                    missing.append("particle_instances")
                bad = bool(missing)
                bad |= abs(float(a["deghost_tau"]) - exp_tau) > 1e-6
                bad |= abs(float(a["tau_loose_floor"]) - exp_floor) > 1e-6
                bad |= int(a["n_in_cache"]) <= 0
                if bad:
                    n_bad += 1
                    if n_bad <= 3:
                        print(f"  [content] bad file {os.path.basename(p)}: "
                              f"missing={missing} tau={a.get('deghost_tau')}")
                else:
                    n_cache_vals.append(int(a["n_in_cache"]))
                    keepfrac_vals.append(float(a["deghost_keep_frac"]))
        except Exception as ex:
            n_bad += 1
            print(f"  [content] unreadable {os.path.basename(p)}: {ex}")
    ok = n_bad == 0
    if n_cache_vals:
        print(f"[content] {split}: sampled {len(sample)}  bad={n_bad}  "
              f"n_in_cache med={int(np.median(n_cache_vals))}  "
              f"keep_frac mean={np.mean(keepfrac_vals):.3f}  "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        print(f"[content] {split}: sampled {len(sample)}, all bad — FAIL")
    return ok


def phase_dataset(root, split, n_sample):
    from pointcept.datasets import build_dataset
    ds = build_dataset(dict(
        type="LArFormerStage12CacheDataset", split=split,
        data_root=os.path.join(root, split),
        source_set_filter="stage2_pass",
        recenter_to_centroid=True, min_spacepoints=20, loop=1))
    n = len(ds)
    print(f"[dataset] {split}: LArFormerStage12CacheDataset len={n}")
    idxs = random.sample(range(n), min(n_sample, n))
    n_sp, n_gt, n_fail = [], [], 0
    for i in idxs:
        try:
            s = ds[i]
            c = s["coord"]
            assert np.isfinite(np.asarray(c)).all(), "non-finite coords"
            n_sp.append(len(c))
            n_gt.append(int(s.get("n_gt_instances", -1)))
        except Exception as ex:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [dataset] read fail idx {i}: {ex}")
    ok = n_fail == 0 and len(n_sp) > 0
    if n_sp:
        print(f"[dataset] {split}: read {len(idxs)} samples, fails={n_fail}; "
              f"n_sp med={int(np.median(n_sp))} "
              f"n_gt_inst med={int(np.median(n_gt))}  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok


def phase_augment(h5s):
    p = random.choice(h5s)
    with h5py.File(p, "r") as h:
        present = "particle_class_id" in h["entry_0"]
    print(f"[augment] particle_class_id present: {present}"
          + ("" if present else
         "  -> run augment_stage12_cache_particle_class_id.py before stage-3"))
    return present  # informational; not a pass/fail gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument("--expect-val", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.999)
    ap.add_argument("--expect-deghost-tau", type=float, default=0.2)
    ap.add_argument("--expect-tau-loose-floor", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--dataset-sample", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true",
                    help="skip the full integrity scan")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    random.seed(args.seed)

    all_ok = True
    for split in args.splits.split(","):
        expected = getattr(args, f"expect_{split}", None)
        ok_cov, h5s = phase_coverage(args.cache_root, split, expected,
                                     args.min_coverage)
        all_ok &= ok_cov
        if not h5s:
            all_ok = False
            continue
        if not args.quick:
            all_ok &= phase_integrity(args.cache_root, split, h5s,
                                      args.workers)
        all_ok &= phase_content(split, h5s, args.sample,
                                args.expect_deghost_tau,
                                args.expect_tau_loose_floor)
        all_ok &= phase_dataset(args.cache_root, split, args.dataset_sample)
        phase_augment(h5s)

    print("\n" + ("CACHE VERIFICATION PASSED" if all_ok
                  else "CACHE VERIFICATION FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
