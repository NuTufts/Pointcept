"""Move a flat merged_sp directory into a 2-level fileno tree, to relieve the
shared-filesystem metadata hotspot of a single 100k-1M-file directory.

Tree layout (deterministic, filename-derived -> cheap, no file reads):
    <dir>/<fileno//1000 : %03d>/<(fileno%1000)//bucket : %02d>/<file>.h5
default bucket=25 -> ~25 filenos/leaf x ~15 entries ~= 375 files/leaf.

`mv` is a metadata-only RENAME within the same filesystem (no data copy). The
move is safe for the reco chain because downstream lists are rebuilt with the
(fileno,entry) stable sort (list_merged_sp.py) and the cascade is re-run, so no
index<->cascade linkage survives the move to be scrambled.

Dry-run first (default) -> reports the per-leaf plan without moving. Pass
--execute to actually rename. --sleep-every/--sleep rate-limit the metadata
burst. Verifies every file has a parseable fileno and lands exactly once.

    python3 reorg_merged_sp.py --dir <flat merged_sp dir> [--execute]
"""
import argparse
import os
import re
import time
from collections import Counter

_FILENO = re.compile(r"fileno(\d+)_entry\d+\.h5$")


def leaf(fileno, bucket):
    return os.path.join("%03d" % (fileno // 1000),
                        "%02d" % ((fileno % 1000) // bucket))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="flat merged_sp directory")
    ap.add_argument("--bucket", type=int, default=25, help="filenos per leaf")
    ap.add_argument("--execute", action="store_true",
                    help="actually rename (default: dry-run report only)")
    ap.add_argument("--sleep-every", type=int, default=5000)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds to pause every --sleep-every moves")
    args = ap.parse_args()

    per_leaf = Counter()
    n_bad = n_already = n_moved = 0
    t0 = time.time()
    # Materialize the full flat-file list FIRST (only immediate .h5 files, not
    # already-treed subdirs), so moving files into subdirs can't perturb the
    # scandir iterator (behavior under concurrent modification is FS-dependent).
    names = [e.name for e in os.scandir(args.dir)
             if e.name.endswith(".h5") and e.is_file()]
    n_files = len(names)
    print(f">>> scanned {n_files} flat .h5 files ({time.time()-t0:.0f}s)",
          flush=True)
    for name in names:
        m = _FILENO.search(name)
        if not m:
            n_bad += 1
            continue
        sub = leaf(int(m.group(1)), args.bucket)
        per_leaf[sub] += 1
        if not args.execute:
            if per_leaf[sub] == 1 and len(per_leaf) <= 3:
                print(f"    e.g. {name} -> {sub}/")
            continue
        dst_dir = os.path.join(args.dir, sub)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            n_already += 1
            continue
        os.rename(os.path.join(args.dir, name), dst)
        n_moved += 1
        if args.sleep and n_moved % args.sleep_every == 0:
            print(f"    moved {n_moved} ({time.time()-t0:.0f}s)", flush=True)
            time.sleep(args.sleep)

    sizes = list(per_leaf.values())
    print(f">>> dir {args.dir}")
    print(f">>> {n_files} .h5 files | {n_bad} unparseable | "
          f"{len(per_leaf)} leaves | files/leaf min/med/max = "
          f"{min(sizes) if sizes else 0}/"
          f"{sorted(sizes)[len(sizes)//2] if sizes else 0}/"
          f"{max(sizes) if sizes else 0}")
    over = sum(1 for s in sizes if s >= 1000)
    print(f">>> leaves with >=1000 files: {over} "
          f"(reduce --bucket if >0)")
    if args.execute:
        print(f">>> MOVED {n_moved} ({n_already} already present) in "
              f"{time.time()-t0:.0f}s")
    else:
        print(">>> DRY RUN (no files moved); pass --execute to move")


if __name__ == "__main__":
    main()
