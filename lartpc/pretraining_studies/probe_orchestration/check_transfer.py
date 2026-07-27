#!/usr/bin/env python3
"""
Verify the preservation transfer against the committed manifest
(RUN AT TUFTS after sync_from_isambard.sh).

Compares every file in transfer_manifest_isambard.csv against the local
repo copy: missing files and size mismatches (truncated transfers) are
reported grouped by run. Exit code 0 only when everything matches.

Usage:
  python3 check_transfer.py [--repo $LOCAL_REPO] [--only model_last]
    --only model_last : check just the irreplaceable resume states
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def run_of(relpath):
    parts = relpath.split("/")
    return parts[2] if relpath.startswith("sonata/") and len(parts) > 2 else "(top-level)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get(
        "LOCAL_REPO",
        "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept"))
    ap.add_argument("--only", default=None,
                    help="substring filter, e.g. 'model_last' or 'snapshot'")
    args = ap.parse_args()
    manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "transfer_manifest_isambard.csv")
    with open(manifest) as f:
        rows = [(r["relpath"], int(r["size_bytes"])) for r in csv.DictReader(f)]
    if args.only:
        rows = [r for r in rows if args.only in r[0]]

    missing = defaultdict(list)
    mismatched = defaultdict(list)
    ok_files, ok_bytes, want_bytes = 0, 0, sum(s for _, s in rows)
    # The sync intentionally renames the Isambard registry to avoid
    # clobbering the Tufts-local one; accept the alias.
    ALIASES = {"exp/registry.csv": "exp/registry_isambard.csv"}
    for rel, size in rows:
        local = os.path.join(args.repo, rel)
        if not os.path.isfile(local) and rel in ALIASES:
            local = os.path.join(args.repo, ALIASES[rel])
        if not os.path.isfile(local):
            missing[run_of(rel)].append(rel)
        elif os.path.getsize(local) != size:
            mismatched[run_of(rel)].append(
                f"{rel} (have {os.path.getsize(local)}, want {size})")
        else:
            ok_files += 1
            ok_bytes += size

    print(f"manifest: {len(rows)} files / {want_bytes/1e9:.1f} GB"
          + (f" (filter: {args.only})" if args.only else ""))
    print(f"present+intact: {ok_files} files / {ok_bytes/1e9:.1f} GB "
          f"({100*ok_bytes/max(want_bytes,1):.1f}% by size)")
    if not missing and not mismatched:
        print("TRANSFER COMPLETE — every manifest file present with matching size")
        return
    if missing:
        print(f"\nMISSING ({sum(len(v) for v in missing.values())} files):")
        for run in sorted(missing):
            print(f"  {run}: {len(missing[run])} files")
            for rel in missing[run][:4]:
                print(f"    - {rel}")
            if len(missing[run]) > 4:
                print(f"    ... and {len(missing[run]) - 4} more")
    if mismatched:
        print(f"\nSIZE MISMATCH / truncated ({sum(len(v) for v in mismatched.values())}):")
        for run in sorted(mismatched):
            for line in mismatched[run]:
                print(f"    - {line}")
    print("\nRerun ./sync_from_isambard.sh to fetch the gaps (idempotent), "
          "then rerun this check.")
    sys.exit(1)


if __name__ == "__main__":
    main()
