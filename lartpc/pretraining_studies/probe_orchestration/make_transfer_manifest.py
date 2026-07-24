#!/usr/bin/env python3
"""
Generate the preservation-transfer manifest (RUN ON ISAMBARD while the
project disk is still accessible).

Walks the run directories for the preservation set — the same files
sync_from_isambard.sh pulls — and records relpath,size_bytes so
check_transfer.py can verify a local copy against ground truth. Excludes
smoke/ and the redundant per-epoch epoch_*.pth checkpoints.

Usage:
  python3 make_transfer_manifest.py [--repo /projects/u6jo/work/pointcept]
  # then: git add + commit the manifest CSV

Output: transfer_manifest_isambard.csv in this directory.
"""
import argparse
import csv
import os

RUN_ROOTS = ["sonata/p05", "sonata/p1a", "sonata/p5a", "sonata/p5b", "sonata/p5e"]
EXTRA_FILES = ["exp/registry.csv"]


def keep(relpath):
    name = os.path.basename(relpath)
    if "/smoke/" in relpath:
        return False
    if name.startswith("epoch_") and name.endswith(".pth"):
        return False
    if name in ("train.log", "config.py"):
        return True
    if name.startswith("events.out.tfevents."):
        return True
    if name == "model_last.pth":
        return True
    if "/snapshot/" in relpath and name.endswith(".pth"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/projects/u6jo/work/pointcept")
    args = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "transfer_manifest_isambard.csv")
    rows = []
    for root in RUN_ROOTS:
        top = os.path.join(args.repo, root)
        if not os.path.isdir(top):
            continue
        for dirpath, _, files in os.walk(top):
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, args.repo)
                if keep(rel):
                    rows.append((rel, os.path.getsize(full)))
    for rel in EXTRA_FILES:
        full = os.path.join(args.repo, rel)
        if os.path.isfile(full):
            rows.append((rel, os.path.getsize(full)))
    rows.sort()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "size_bytes"])
        w.writerows(rows)
    total = sum(s for _, s in rows)
    n_runs = len({r.split("/")[2] for r, _ in rows if r.startswith("sonata/")})
    print(f"manifest: {len(rows)} files, {total/1e9:.1f} GB, {n_runs} runs -> {out}")


if __name__ == "__main__":
    main()
