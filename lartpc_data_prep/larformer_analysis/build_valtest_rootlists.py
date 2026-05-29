"""Stage 0: build per-config ROOT inputlists + manifests for the val+test set.

Reads the val+test h5 list (lines = absolute paths to
`merged_<TAG>_filenoNNNNN_entryNNNNNN.h5`), groups entries by TAG, looks up
the source dlmerged ROOT file for each fileno from each TAG's production
inputlist, and emits:

  <output_dir>/rootlists/<TAG>.txt      — one ROOT path per line, deduped
                                          (drives Stage 1: flashinfo)
  <output_dir>/manifest/<TAG>.csv       — one row per (fileno, entry):
                                          tag, fileno, entry, root_path,
                                          merged_h5, flashinfo_h5
  <output_dir>/summary.txt              — per-TAG counts

The manifest CSVs drive Stage 3 (the per-event analysis); the rootlists
drive the flashinfo regen.

Filename conventions (from the production pipeline):
  ROOT source:  /.../<TAG_root>/NNN/NNN/dlmerged_*_filenoNNNNNN.root
                (6-digit fileno; TAG_root may have a name typo like
                 "coriska" vs "corsika" — DOES NOT have to equal TAG)
  merged H5:    /.../<TAG>/merged_h5/NNN/NNN/
                  merged_<TAG>_filenoNNNNN_entryNNNNNN.h5
                (5-digit fileno; TAG here is the canonical / corrected
                 spelling used by the pointcept-side pipeline)

The fileno is the only join key — it matches between the ROOT filename
and the merged H5 filename modulo zero-padding width.

Usage:
  python build_valtest_rootlists.py \\
      --h5-list   /path/to/h5list_mcall_lantern_valtest.txt \\
      --conf-dir  $POINTCEPT/lartpc_data_prep/lantern_scripts/lantern_configs \\
      --output    $POINTCEPT/lartpc_data_prep/larformer_analysis/valtest \\
      [--flashinfo-root  /custom/flashinfo/parent]

The conf-dir is scanned for *.conf files; from each we read TAG and
INPUTLIST. The same {TAG} should match the merged-h5 filename's TAG.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

_MERGED_H5_RE = re.compile(
    r"^merged_(?P<tag>.+?)_fileno(?P<fileno>\d+)_entry(?P<entry>\d+)\.h5$"
)
_ROOT_FILENO_RE = re.compile(r"_fileno(\d+)\.root$")


def parse_merged_h5(path: str):
    """Return (tag, fileno, entry) or None if name doesn't match."""
    base = os.path.basename(path)
    m = _MERGED_H5_RE.match(base)
    if not m:
        return None
    return (m["tag"], int(m["fileno"]), int(m["entry"]))


def parse_root_fileno(path: str):
    """Return fileno (int) or None if name doesn't have a _filenoXXXXXX.root."""
    base = os.path.basename(path)
    m = _ROOT_FILENO_RE.search(base)
    if not m:
        return None
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Dataset-config (.conf) reader — pulls TAG + INPUTLIST from a bash conf
# ---------------------------------------------------------------------------

_TAG_RE       = re.compile(r"^\s*TAG\s*=\s*(.+?)\s*$")
_INPUTLIST_RE = re.compile(r"^\s*INPUTLIST\s*=\s*(.+?)\s*$")


def read_conf(conf_path: str):
    """Return (tag, inputlist_path) for a lantern_configs/*.conf, or None."""
    tag = None
    inp = None
    with open(conf_path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0]
            mt = _TAG_RE.match(line)
            if mt:
                tag = mt.group(1).strip()
            mi = _INPUTLIST_RE.match(line)
            if mi:
                inp = mi.group(1).strip()
    if tag is None or inp is None:
        return None
    return (tag, inp)


def discover_configs(conf_dir: str):
    """Walk conf_dir for *.conf; return {tag: inputlist_path}."""
    out = {}
    for name in sorted(os.listdir(conf_dir)):
        if not name.endswith(".conf"):
            continue
        info = read_conf(os.path.join(conf_dir, name))
        if info is None:
            continue
        tag, inp = info
        if tag in out and out[tag] != inp:
            sys.stderr.write(
                f"[warn] duplicate TAG {tag!r} across configs; "
                f"keeping {out[tag]!r}, ignoring {inp!r}\n"
            )
            continue
        out[tag] = inp
    return out


def load_fileno_to_root(inputlist_path: str):
    """Read inputlist, return {fileno (int) : root_path (str)}."""
    out = {}
    with open(inputlist_path, "r") as f:
        for raw in f:
            p = raw.strip()
            if not p or p.startswith("#"):
                continue
            fn = parse_root_fileno(p)
            if fn is None:
                continue
            if fn in out and out[fn] != p:
                sys.stderr.write(
                    f"[warn] duplicate fileno {fn} in {inputlist_path}: "
                    f"keeping {out[fn]!r}, ignoring {p!r}\n"
                )
                continue
            out[fn] = p
    return out


# ---------------------------------------------------------------------------
# Flashinfo path derivation
# ---------------------------------------------------------------------------

def derive_flashinfo_path(merged_h5_path: str,
                          flashinfo_root_override: str = None) -> str:
    """Map a merged_h5 path to its sibling flashinfo path under the
    production layout `<base>/merged_h5/NNN/NNN/merged_*.h5` →
    `<base>/flashinfo_h5/NNN/NNN/flashinfo_*.h5`. When the merged_h5
    layout doesn't follow this convention, returns a sibling path with
    'merged' → 'flashinfo' substituted.

    `flashinfo_root_override` lets the caller redirect to a different
    parent directory (useful when regenerating flashinfo into a scratch
    space).
    """
    base = os.path.basename(merged_h5_path).replace("merged_", "flashinfo_", 1)
    dirn = os.path.dirname(merged_h5_path)

    if flashinfo_root_override is not None:
        # Preserve the trailing NNN/NNN hash dirs (last 2 components of dirn).
        tail2 = os.sep.join(dirn.rstrip(os.sep).split(os.sep)[-2:])
        return os.path.join(flashinfo_root_override, tail2, base)

    if "/merged_h5/" in dirn:
        dirn = dirn.replace("/merged_h5/", "/flashinfo_h5/")
        return os.path.join(dirn, base)
    return os.path.join(dirn, base)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--h5-list", required=True,
                    help="val+test h5 list (one merged_h5 path per line)")
    ap.add_argument("--conf-dir", required=True,
                    help="Directory containing per-TAG .conf files "
                         "(typically lartpc_data_prep/lantern_scripts/"
                         "lantern_configs/)")
    ap.add_argument("--output", required=True,
                    help="Output directory; writes rootlists/<TAG>.txt + "
                         "manifest/<TAG>.csv + summary.txt")
    ap.add_argument("--flashinfo-root", default=None,
                    help="Override the flashinfo parent dir in the manifest "
                         "(default: derive from merged_h5 path via "
                         "'/merged_h5/' → '/flashinfo_h5/')")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output, "rootlists"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "manifest"), exist_ok=True)

    # Build {tag: {fileno: root_path}} from the dataset configs.
    tag_to_inputlist = discover_configs(args.conf_dir)
    if not tag_to_inputlist:
        sys.exit(f"no .conf files found in {args.conf_dir!r}")
    print(f"Found {len(tag_to_inputlist)} dataset configs:")
    for t, p in sorted(tag_to_inputlist.items()):
        print(f"  {t:40s}  inputlist={p}")

    tag_fileno_to_root = {}
    for tag, inp in tag_to_inputlist.items():
        if not os.path.exists(inp):
            sys.stderr.write(
                f"[warn] inputlist for tag {tag!r} not found at {inp!r}; "
                f"skipping (its h5 entries will be reported as 'unmatched_root')\n"
            )
            tag_fileno_to_root[tag] = {}
            continue
        tag_fileno_to_root[tag] = load_fileno_to_root(inp)

    # Walk the val+test h5 list, group by tag.
    by_tag = defaultdict(list)         # tag -> list of (fileno, entry, h5_path)
    unmatched_tag = []                 # h5s whose tag has no .conf
    unmatched_root = []                # h5s whose tag has a .conf but no fileno match
    with open(args.h5_list, "r") as f:
        for raw in f:
            p = raw.strip()
            if not p or p.startswith("#"):
                continue
            parsed = parse_merged_h5(p)
            if parsed is None:
                sys.stderr.write(f"[warn] unparsable h5 name: {p}\n")
                continue
            tag, fileno, entry = parsed
            by_tag[tag].append((fileno, entry, p))

    # Build rootlists + manifests.
    summary_lines = []
    for tag in sorted(by_tag.keys()):
        events = by_tag[tag]
        if tag not in tag_fileno_to_root:
            sys.stderr.write(
                f"[warn] no .conf for tag {tag!r}; {len(events)} entries skipped\n"
            )
            unmatched_tag.extend(events)
            continue
        fn_to_root = tag_fileno_to_root[tag]
        if not fn_to_root:
            unmatched_root.extend([(tag, *e) for e in events])
            continue

        # Stable order: by fileno then entry.
        events.sort()

        # Dedup ROOT paths; track which filenos were resolved.
        roots_in_order = []
        seen_roots = set()
        n_resolved = 0
        manifest_rows = []
        n_unmatched_local = 0
        for fileno, entry, merged_p in events:
            root_p = fn_to_root.get(fileno)
            flashinfo_p = derive_flashinfo_path(
                merged_p, flashinfo_root_override=args.flashinfo_root,
            )
            if root_p is None:
                n_unmatched_local += 1
                unmatched_root.append((tag, fileno, entry, merged_p))
                root_p = ""
            else:
                n_resolved += 1
                if root_p not in seen_roots:
                    roots_in_order.append(root_p)
                    seen_roots.add(root_p)
            manifest_rows.append(dict(
                tag=tag, fileno=fileno, entry=entry,
                root_path=root_p,
                merged_h5=merged_p,
                flashinfo_h5=flashinfo_p,
            ))

        # rootlists/<TAG>.txt
        rl_path = os.path.join(args.output, "rootlists", f"{tag}.txt")
        with open(rl_path, "w") as f:
            for rp in roots_in_order:
                f.write(rp + "\n")

        # manifest/<TAG>.csv
        mf_path = os.path.join(args.output, "manifest", f"{tag}.csv")
        with open(mf_path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["tag", "fileno", "entry",
                               "root_path", "merged_h5", "flashinfo_h5"],
            )
            w.writeheader()
            w.writerows(manifest_rows)

        summary_lines.append(
            f"{tag:40s}  n_events={len(events):6d}  "
            f"n_roots={len(roots_in_order):5d}  "
            f"n_unresolved={n_unmatched_local}"
        )
        print(f"[ok]  {tag}: {len(events)} events → "
              f"{len(roots_in_order)} ROOTs ({n_unmatched_local} unresolved)")

    # summary.txt
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"val+test h5 list: {args.h5_list}\n")
        f.write(f"conf dir:         {args.conf_dir}\n")
        f.write(f"output dir:       {args.output}\n\n")
        f.write("Per-tag summary:\n")
        for line in summary_lines:
            f.write("  " + line + "\n")
        f.write(f"\nUnmatched (no .conf for tag): "
                f"{len(unmatched_tag)} entries\n")
        f.write(f"Unmatched (no ROOT for fileno): "
                f"{len(unmatched_root)} entries\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
