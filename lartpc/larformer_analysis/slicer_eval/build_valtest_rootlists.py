"""Stage 0: build per-TAG rerun-line files + manifests for the val+test set.

Reads the val+test h5 list (lines = absolute paths to
`merged_<TAG>_filenoNNNNN_entryNNNNNN.h5`), groups entries by TAG, and
emits a per-TAG list of original ROOT-inputlist line numbers needed to
re-run flashinfo (and any other lineno-keyed stage):

  <output_dir>/rerun_lines/<TAG>.txt    — one ORIGINAL lineno per line
                                          (= the fileno embedded in the
                                          merged-H5 name, which the
                                          production pipeline writes
                                          based on inputlist lineno).
                                          Feeds the existing wconfig
                                          RERUN_LINES_FILE mechanism.
  <output_dir>/manifest/<TAG>.csv       — one row per (fileno, entry):
                                          tag, fileno, entry, root_path,
                                          merged_h5, flashinfo_h5.
                                          Drives Stage 3 (per-event
                                          analysis). `root_path` filled
                                          in only when --conf-dir is
                                          provided AND the inputlist
                                          contains a ROOT with that
                                          fileno.
  <output_dir>/rootlists/<TAG>.txt      — (optional, only when --conf-dir
                                          is provided) deduped ROOT paths
                                          for the val+test filenos. Kept
                                          as a sanity-check artifact;
                                          NOT used as a flashinfo input
                                          (the truncate-inputlist approach
                                          breaks the merged-H5 fileno
                                          numbering).
  <output_dir>/summary.txt              — per-TAG counts.

Why rerun lines instead of a truncated inputlist:
  The merged H5's fileno is assigned from the inputlist lineno at
  production time, not parsed from the ROOT filename. A truncated
  inputlist re-numbers everything → the prep script picks the wrong
  merged H5 to pair with each ROOT entry. The rerun-list mechanism
  (RERUN_LINES_FILE in the wconfig) re-uses the original numbering by
  pointing at the same ORIGINAL inputlist + a list of which lineno
  positions to process.

Filename conventions (from the production pipeline):
  ROOT source:  /.../<TAG_root>/NNN/NNN/dlmerged_*_filenoNNNNNN.root
                (6-digit fileno IN THE FILENAME — but the merged-H5's
                 fileno is the LINENO of this ROOT in the production
                 inputlist, NOT this number; they happen to match for
                 sequentially-named files but the wconfig pipeline
                 doesn't enforce it.)
  merged H5:    /.../<TAG>/merged_h5/NNN/NNN/
                  merged_<TAG>_filenoNNNNN_entryNNNNNN.h5
                (5-digit fileno = original inputlist lineno).

Usage:
  python build_valtest_rootlists.py \\
      --h5-list   /path/to/h5list_mcall_lantern_valtest.txt \\
      --output    $POINTCEPT/lartpc/larformer_analysis/slicer_eval/valtest \\
      [--conf-dir $POINTCEPT/lartpc/data_prep/training_data/lantern_configs] \\
      [--flashinfo-root  /custom/flashinfo/parent]

Then on the cluster, point the wconfig at the rerun_lines file:
  RERUN_LINES_FILE=<output_dir>/rerun_lines/<TAG>.txt
  stride=1   # or whatever cadence the SLURM array uses
  OFFSET=0
  sbatch --array=0-$((N-1)) scripts/submit_flashinfo.sh configs/<TAG>.conf
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
    ap.add_argument("--output", required=True,
                    help="Output directory; writes rerun_lines/<TAG>.txt + "
                         "manifest/<TAG>.csv + summary.txt + "
                         "(optionally) rootlists/<TAG>.txt")
    ap.add_argument("--conf-dir", default=None,
                    help="Optional. Directory containing per-TAG .conf files "
                         "(lartpc/data_prep/training_data/lantern_configs/). "
                         "When provided, ROOT paths are looked up from each "
                         "TAG's inputlist and written to manifest + the "
                         "(sanity-check) rootlists/. Without it, those "
                         "columns are blank and only rerun_lines + manifest "
                         "(without root_path) are produced.")
    ap.add_argument("--flashinfo-root", default=None,
                    help="Override the flashinfo parent dir in the manifest "
                         "(default: derive from merged_h5 path via "
                         "'/merged_h5/' → '/flashinfo_h5/')")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output, "rerun_lines"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "manifest"), exist_ok=True)
    if args.conf_dir is not None:
        os.makedirs(os.path.join(args.output, "rootlists"), exist_ok=True)

    # Build {tag: {fileno: root_path}} from the dataset configs (optional).
    tag_fileno_to_root = {}
    if args.conf_dir is not None:
        tag_to_inputlist = discover_configs(args.conf_dir)
        if tag_to_inputlist:
            print(f"Found {len(tag_to_inputlist)} dataset configs:")
            for t, p in sorted(tag_to_inputlist.items()):
                print(f"  {t:40s}  inputlist={p}")
            for tag, inp in tag_to_inputlist.items():
                if not os.path.exists(inp):
                    sys.stderr.write(
                        f"[warn] inputlist for tag {tag!r} not found at "
                        f"{inp!r}; rootlists / manifest.root_path will be "
                        f"blank for this tag\n"
                    )
                    tag_fileno_to_root[tag] = {}
                    continue
                tag_fileno_to_root[tag] = load_fileno_to_root(inp)
        else:
            sys.stderr.write(
                f"[warn] --conf-dir provided but no .conf files found in "
                f"{args.conf_dir!r}; rerun_lines + manifest still written, "
                f"rootlists skipped\n"
            )

    # Walk the val+test h5 list, group by tag.
    by_tag = defaultdict(list)         # tag -> list of (fileno, entry, h5_path)
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

    # Build rerun_lines + manifest (+ rootlists if conf-dir provided).
    summary_lines = []
    total_unmatched_roots = 0
    for tag in sorted(by_tag.keys()):
        events = by_tag[tag]
        events.sort()                       # stable order: by fileno, then entry

        # Distinct filenos for this tag — one entry per unique fileno
        # goes into the rerun_lines file. The fileno from the merged-H5
        # name IS the original inputlist lineno (because production
        # named the H5 from the lineno at job time).
        unique_filenos = sorted({fileno for fileno, _, _ in events})

        # rerun_lines/<TAG>.txt — primary Stage-1 driver.
        rr_path = os.path.join(args.output, "rerun_lines", f"{tag}.txt")
        with open(rr_path, "w") as f:
            for fn in unique_filenos:
                f.write(f"{fn}\n")

        # Optional ROOT-path resolution + rootlists/<TAG>.txt.
        fn_to_root = tag_fileno_to_root.get(tag, {})
        roots_in_order = []
        seen_roots = set()
        n_unmatched_local = 0

        # manifest/<TAG>.csv (always written; root_path blank when not
        # resolvable).
        manifest_rows = []
        for fileno, entry, merged_p in events:
            root_p = fn_to_root.get(fileno, "") if fn_to_root else ""
            if fn_to_root:
                if root_p:
                    if root_p not in seen_roots:
                        roots_in_order.append(root_p)
                        seen_roots.add(root_p)
                else:
                    n_unmatched_local += 1
            flashinfo_p = derive_flashinfo_path(
                merged_p, flashinfo_root_override=args.flashinfo_root,
            )
            manifest_rows.append(dict(
                tag=tag, fileno=fileno, entry=entry,
                root_path=root_p,
                merged_h5=merged_p,
                flashinfo_h5=flashinfo_p,
            ))

        mf_path = os.path.join(args.output, "manifest", f"{tag}.csv")
        with open(mf_path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["tag", "fileno", "entry",
                               "root_path", "merged_h5", "flashinfo_h5"],
            )
            w.writeheader()
            w.writerows(manifest_rows)

        if args.conf_dir is not None and fn_to_root:
            rl_path = os.path.join(args.output, "rootlists", f"{tag}.txt")
            with open(rl_path, "w") as f:
                for rp in roots_in_order:
                    f.write(rp + "\n")

        total_unmatched_roots += n_unmatched_local
        summary_lines.append(
            f"{tag:40s}  n_events={len(events):6d}  "
            f"n_filenos={len(unique_filenos):5d}  "
            f"n_unresolved_roots={n_unmatched_local}"
        )
        print(f"[ok]  {tag}: {len(events)} events → "
              f"{len(unique_filenos)} unique filenos "
              f"({n_unmatched_local} ROOT paths unresolved)")

    # summary.txt
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"val+test h5 list: {args.h5_list}\n")
        f.write(f"conf dir:         {args.conf_dir}\n")
        f.write(f"output dir:       {args.output}\n\n")
        f.write("Stage-1 invocation pattern (per TAG):\n")
        f.write("  RERUN_LINES_FILE=<output_dir>/rerun_lines/<TAG>.txt\n")
        f.write("  stride=1  OFFSET=0\n")
        f.write("  sbatch --array=0-$((N_filenos-1)) "
                "scripts/submit_flashinfo.sh configs/<TAG>.conf\n\n")
        f.write("Per-tag summary:\n")
        for line in summary_lines:
            f.write("  " + line + "\n")
        f.write(f"\nTotal ROOT-path lookups that failed (ok if "
                f"--conf-dir omitted): {total_unmatched_roots}\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
