#!/usr/bin/env python3
"""
Remap the Isambard train/val/diagnostic file lists onto Tufts source paths.

The Isambard lists (made by make_filelists.py) carry squashfs-mount paths of the
form

    /data/inter_NNNN/ub_on_tufts/hdf5/<TAIL>

where <TAIL> = v3_larmatch/<sample>/.../<file>.h5. The same physical files live
on the Tufts cluster under a different prefix

    /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/<TAIL>

so the two are joined on the path tail after "/hdf5/". This script writes, for
each Isambard list, a *_tufts.txt list of the matching Tufts source paths, in the
ORIGINAL line order, and records counts + sha256 in filelist_stats_tufts.txt.

Matching (see requirement notes in the task):
  - Primary key: the tail after "/hdf5/". Uniqueness is checked in BOTH the
    source list and every input list before it is relied on.
  - If any entry fails to match on tail (directory layouts diverge), the script
    falls back to basename matching -- but only after verifying that basenames
    are globally unique in the source list; otherwise it ABORTS with a report.

Nothing is silently dropped: any unmatched Isambard entries are written to a
<output>.missing file and counted in the summary. diag1k must match 1000/1000.

Run inside the pointcept container so the h5py spot-check can open real files:

  apptainer exec -B /cluster:/cluster \
      /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif \
      python3 lartpc/filelists/remap_filelists_tufts.py
"""
import argparse
import collections
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HDF5_MARKER = "/hdf5/"

# (Isambard input list, Tufts output list). Order preserved within each.
LIST_PAIRS = [
    ("h5list_v3_mc_only_train.txt", "h5list_v3_mc_only_train_tufts.txt"),
    ("h5list_v3_mc_only_val.txt", "h5list_v3_mc_only_val_tufts.txt"),
    ("h5list_v3_mc_diag1k.txt", "h5list_v3_mc_diag1k_tufts.txt"),
]

REQUIRED_KEYS = ("pos", "hasmatch", "ssnet_label")
TRIPLET_GROUP = "/entry_0/triplet_data"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def tail_after_hdf5(p):
    i = p.find(HDF5_MARKER)
    return p[i + len(HDF5_MARKER):] if i >= 0 else None


def report_dupes(label, keys):
    """Return (is_unique, n_dupe_keys) and print a few offenders."""
    counts = collections.Counter(k for k in keys if k is not None)
    dupes = {k: c for k, c in counts.items() if c > 1}
    n_nokey = sum(k is None for k in keys)
    if n_nokey:
        print(f"  [{label}] {n_nokey} entries have no '{HDF5_MARKER}' segment")
    if dupes:
        print(f"  [{label}] NOT unique: {len(dupes)} duplicated keys, e.g.:")
        for k in list(dupes)[:5]:
            print(f"      {dupes[k]}x  {k}")
    return (not dupes and not n_nokey), len(dupes)


def build_source_index(src_paths):
    """Build tail->path and basename->path maps; report uniqueness of each."""
    tails = [tail_after_hdf5(p) for p in src_paths]
    bases = [os.path.basename(p) for p in src_paths]

    print("Source list uniqueness:")
    tail_unique, _ = report_dupes("src tail", tails)
    base_unique, _ = report_dupes("src basename", bases)
    print(f"  src tail unique={tail_unique}  src basename unique={base_unique}")

    tail_map = {}
    for t, p in zip(tails, src_paths):
        if t is not None:
            tail_map.setdefault(t, p)
    base_map = {}
    for b, p in zip(bases, src_paths):
        base_map.setdefault(b, p)
    return tail_map, base_map, tail_unique, base_unique


def try_match(lines, key_fn, key_map):
    """Return (matched_paths_in_order, missing_lines) for the chosen key."""
    matched, missing = [], []
    for ln in lines:
        k = key_fn(ln)
        hit = key_map.get(k) if k is not None else None
        if hit is None:
            missing.append(ln)
        else:
            matched.append(hit)
    return matched, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join(
        HERE, "h5list_mcall_lantern_validated.txt"),
        help="Tufts source-file list to map onto")
    ap.add_argument("--indir", default=HERE, help="dir holding the Isambard lists")
    ap.add_argument("--outdir", default=HERE, help="dir to write *_tufts.txt lists")
    ap.add_argument("--n-spotcheck", type=int, default=10,
                    help="matched files to open with h5py (0 disables)")
    args = ap.parse_args()

    src_paths = read_list(args.source)
    print(f"Source: {args.source}  ({len(src_paths)} paths)\n")
    tail_map, base_map, src_tail_unique, src_base_unique = build_source_index(src_paths)

    if not src_tail_unique and not src_base_unique:
        sys.exit("ABORT: source list has neither unique tails nor unique basenames "
                 "-- cannot establish a reliable join key.")

    # --- match each list, preferring the tail key, falling back to basename ---
    print("\nMatching lists:")
    results = {}  # out_name -> (matched_paths, missing_lines, key_mode)
    for in_name, out_name in LIST_PAIRS:
        in_path = os.path.join(args.indir, in_name)
        lines = read_list(in_path)
        tails = [tail_after_hdf5(p) for p in lines]
        list_tail_unique, _ = report_dupes(f"{in_name} tail", tails)

        key_mode = "tail"
        matched, missing = ([], list(lines))
        if src_tail_unique:
            matched, missing = try_match(lines, tail_after_hdf5, tail_map)

        if missing:
            # Directory layouts diverged (or source tails weren't usable):
            # fall back to basename ONLY if source basenames are globally unique.
            if not src_tail_unique:
                print(f"  [{in_name}] tail key unusable (source tails not unique)")
            else:
                print(f"  [{in_name}] {len(missing)} unmatched on tail; "
                      f"trying basename fallback")
            if not src_base_unique:
                sys.exit(f"ABORT: {len(missing)} entries in {in_name} did not match "
                         f"on tail and source basenames are NOT globally unique -- "
                         f"refusing to guess. Inspect directory-layout divergence.")
            b_matched, b_missing = try_match(
                lines, lambda p: os.path.basename(p), base_map)
            # Basename fallback is all-or-nothing per list for a clean provenance.
            if len(b_missing) < len(missing):
                matched, missing, key_mode = b_matched, b_missing, "basename"

        results[out_name] = (matched, missing, key_mode)
        n = len(lines)
        print(f"  [{in_name}] key={key_mode}  matched={len(matched)}/{n}  "
              f"missing={len(missing)}")

    # --- write outputs + .missing files ---
    print("\nWriting outputs:")
    written = []
    for in_name, out_name in LIST_PAIRS:
        matched, missing, key_mode = results[out_name]
        out_path = os.path.join(args.outdir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(matched) + ("\n" if matched else ""))
        written.append((out_name, out_path, len(matched), key_mode))
        print(f"  wrote {out_name}: {len(matched)} paths (key={key_mode})")
        if missing:
            miss_path = out_path + ".missing"
            with open(miss_path, "w") as f:
                f.write("\n".join(missing) + "\n")
            print(f"  !! {len(missing)} MISSING -> {miss_path} (NOT dropped silently)")

    # --- sanity: output count == input count ---
    print("\nSanity checks:")
    ok = True
    for in_name, out_name in LIST_PAIRS:
        n_in = len(read_list(os.path.join(args.indir, in_name)))
        matched, missing, _ = results[out_name]
        same = (len(matched) + len(missing)) == n_in
        print(f"  count {out_name}: in={n_in} matched={len(matched)} "
              f"missing={len(missing)} accounted={'OK' if same else 'FAIL'}")
        ok = ok and same
        if out_name.endswith("diag1k_tufts.txt") and len(missing) != 0:
            print("  !! diag1k did NOT match 1000/1000 -- this is a hard requirement")
            ok = False

    # --- sanity: pairwise disjoint outputs ---
    sets = {out: set(results[out][0]) for _, out in LIST_PAIRS}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            status = "OK" if not inter else f"FAIL ({len(inter)} shared)"
            print(f"  disjoint {names[i]} vs {names[j]}: {status}")
            if inter:
                ok = False
                for p in list(inter)[:3]:
                    print(f"      shared: {p}")

    # --- sanity: spot-open matched files with h5py ---
    if args.n_spotcheck > 0:
        ok = ok and spotcheck(results, args.n_spotcheck)
    else:
        print("  h5py spot-check skipped (--n-spotcheck 0)")

    # --- record stats ---
    stats_path = os.path.join(args.outdir, "filelist_stats_tufts.txt")
    lines_out = [
        f"source: {args.source}",
        f"source_sha256: {sha256_of(args.source)}",
        f"source_n: {len(src_paths)}",
        f"src_tail_unique: {src_tail_unique}  src_basename_unique: {src_base_unique}",
    ]
    for out_name, out_path, n, key_mode in written:
        _, missing, _ = results[out_name]
        lines_out.append(
            f"{out_name}: n={n} missing={len(missing)} key={key_mode} "
            f"sha256={sha256_of(out_path)}")
    with open(stats_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    print(f"\nWrote {stats_path}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** CHECKS FAILED -- review above ***"))
    sys.exit(0 if ok else 1)


def spotcheck(results, n):
    try:
        import h5py
    except ImportError:
        print("  h5py NOT available -- spot-check REQUIRED but cannot run.")
        print("  Re-run inside the pointcept container (see module docstring).")
        return False
    import numpy as np  # noqa: F401  (kept parallel to check_truth_keys.py deps)

    # Spread the sample across all three lists.
    targets = []
    for _, out_name in LIST_PAIRS:
        matched = results[out_name][0]
        if not matched:
            continue
        k = min(max(1, n // len(LIST_PAIRS)), len(matched))
        idx = [round(i * (len(matched) - 1) / max(k - 1, 1)) for i in range(k)]
        for j in sorted(set(idx)):
            targets.append((out_name, matched[j]))

    print(f"  h5py spot-check: opening {len(targets)} matched files")
    all_ok = True
    for out_name, path in targets:
        try:
            with h5py.File(path, "r") as f:
                td = f[TRIPLET_GROUP]
                missing = [k for k in REQUIRED_KEYS if k not in td]
                assert not missing, f"missing {missing}"
            print(f"    OK  [{out_name}] {os.path.basename(path)}")
        except Exception as e:
            all_ok = False
            print(f"    BAD [{out_name}] {path}: {e}")
    print(f"  spot-check: {'OK' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    main()
