"""Translate an OLD flat merged_sp list to the post-reorg fileno-tree paths,
PRESERVING LINE ORDER.

The tree reorg (reorg_merged_sp.py) moved merged_sp files into
<dir>/<fileno//1000 : %03d>/<(fileno%1000)//bucket : %02d>/, which invalidates
the flat paths in lists built before the move. Existing ntuples were exported
against those old lists, so their entry index i <-> line i mapping must be kept
-- hence a same-order translation rather than a fresh listing.

    python3 retree_list.py --in <old_flat_list> --out <tree_list> [--bucket 25]
"""
import argparse
import os
import re

_FILENO = re.compile(r"fileno(\d+)_entry\d+\.h5$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", type=int, default=25)
    ap.add_argument("--check", type=int, default=200,
                    help="how many translated paths to sample-verify on disk")
    args = ap.parse_args()

    out, n_bad = [], 0
    for line in open(args.inp):
        p = line.strip()
        if not p:
            continue
        base = os.path.basename(p)
        m = _FILENO.search(base)
        if not m:
            n_bad += 1
            out.append(p)               # leave untouched
            continue
        fn = int(m.group(1))
        root = os.path.dirname(p)
        # if already treed (…/NNN/NN/file.h5) strip the two leaf levels
        if re.fullmatch(r"\d{2,3}", os.path.basename(root)):
            root = os.path.dirname(os.path.dirname(root))
        out.append(os.path.join(root, "%03d" % (fn // 1000),
                                "%02d" % ((fn % 1000) // args.bucket), base))
    with open(args.out, "w") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
    # sample-verify (a full existence check over ~1e5 paths is slow on NFS)
    step = max(1, len(out) // args.check)
    chk = out[::step][:args.check]
    miss = [p for p in chk if not os.path.exists(p)]
    print(f">>> {len(out)} lines -> {args.out}"
          + (f"  [{n_bad} unparseable kept as-is]" if n_bad else ""))
    print(f">>> sampled {len(chk)} paths: {len(chk)-len(miss)} exist, "
          f"{len(miss)} missing" + (f"  e.g. {miss[0]}" if miss else "  (OK)"))


if __name__ == "__main__":
    main()
