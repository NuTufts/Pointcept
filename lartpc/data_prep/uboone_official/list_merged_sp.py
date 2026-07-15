"""Build a merged_sp file list sorted by (fileno, entry) -- STABLE across flat
vs tree directory layouts.

The cascade indexes its per-event outputs by list position (keypoint2_event{i}
where i = position in the sorted merged_sp list), and nu_reco/export resolve by
that same order. Sorting by full PATH (the old `find | sort`) reorders when
files move flat->tree, which scrambles the index<->cascade linkage. Sorting by
the (fileno, entry) key parsed from the BASENAME is layout-invariant, so the
list is identical whether merged_sp is flat or in a fileno tree.

    python3 list_merged_sp.py --dir <merged_sp dir> --out <list.txt>
"""
import argparse
import os
import re

_KEY = re.compile(r"fileno(\d+)_entry(\d+)\.h5$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None,
                    help="optional file of basename substrings to drop "
                         "(one per line; e.g. noise-veto excludes)")
    args = ap.parse_args()

    excl = []
    if args.exclude and os.path.exists(args.exclude):
        excl = [l.strip() for l in open(args.exclude) if l.strip()]

    rows = []
    n_bad = 0
    for root, _dirs, files in os.walk(args.dir):
        for name in files:
            if not name.endswith(".h5"):
                continue
            if excl and any(x in name for x in excl):
                continue
            m = _KEY.search(name)
            if not m:
                n_bad += 1
                continue
            rows.append(((int(m.group(1)), int(m.group(2))),
                         os.path.join(root, name)))
    rows.sort(key=lambda r: r[0])
    with open(args.out, "w") as f:
        f.write("\n".join(p for _, p in rows) + ("\n" if rows else ""))
    print(f">>> {len(rows)} files (sorted by fileno,entry) -> {args.out}"
          + (f"  [{n_bad} unparseable skipped]" if n_bad else ""))


if __name__ == "__main__":
    main()
