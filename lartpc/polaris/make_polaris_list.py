"""Build the bnb5e19 inference input list on Polaris in the PRODUCTION order.

The Tufts production list (176,302 events; 34 events of the 176,336 files were
excluded by the earlier list surgery) is shipped as basenames in
bnb5e19_production_basenames.txt.gz. This script maps each basename to its
path under the mounted squashfs tree and writes the list in the same order,
so event index i (the cascade's file/gidx linkage) is identical to Tufts and
the two productions can be compared event by event.

    python3 lartpc/polaris/make_polaris_list.py \
        --merged-sp /data/bnb5e19/merged_sp --out lartpc/larformer_reco/inputlists/merged_sp_bnb5e19_polaris.txt
"""
import argparse
import gzip
import os
import re
import sys

_KEY = re.compile(r"fileno(\d+)_entry(\d+)\.h5$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--merged-sp", required=True, help="mounted merged_sp root (NNN/NN/*.h5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--basenames", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "bnb5e19_production_basenames.txt.gz"))
    args = ap.parse_args()
    found = {}
    for root, _d, files in os.walk(args.merged_sp):
        for f in files:
            if f.endswith(".h5"):
                found[f] = os.path.join(root, f)
    print(f">>> {len(found)} h5 files under {args.merged_sp}")
    with gzip.open(args.basenames, "rt") as fh:
        want = [l.strip() for l in fh if l.strip()]
    missing = [b for b in want if b not in found]
    if missing:
        print(f"ERROR: {len(missing)} production events not found, e.g. {missing[:3]}")
        sys.exit(2)
    with open(args.out, "w") as fo:
        for b in want:
            fo.write(found[b] + "\n")
    extra = len(found) - len(want)
    print(f">>> wrote {len(want)} paths in production order -> {args.out} "
          f"({extra} files present but not in the production list, as expected: 34)")


if __name__ == "__main__":
    main()
