#!/usr/bin/env python3
"""
Backfill completion sentinels for lines whose merged_*.h5 outputs are
already complete.

Counts existing merged_${TAG}_fileno${ZFILENO}_entry*.h5 files in the
3-level hashed output dir, opens the corresponding input ROOT file with
larlite to learn the total entry count, and writes
  ${MERGEFILE_OUTPUT_DIR}/<lineno/1000>/<lineno/100>/${TAG}_fileno${ZFILENO}.complete
when the existing entry indices are exactly {0..total-1}.

Run inside the pointcept container (needs larlite). Wrapper:
  bash write_completion_sentinels.sh <config> [--report-only] [--lines a:b]
"""

import argparse
import os
import re
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputlist", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="MERGEFILE_OUTPUT_DIR (root of 3-level hash).")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lines", default=None,
                    help="Restrict to lines a:b (1-indexed, inclusive).")
    ap.add_argument("--report-only", action="store_true",
                    help="Print status; do not write sentinels.")
    ap.add_argument("--force", action="store_true",
                    help="Write sentinels for any line with at least one "
                         "existing entry, without verifying total entry "
                         "count from the input ROOT file. Use when input "
                         "files are unavailable but you know which lines "
                         "ran cleanly to completion.")
    return ap.parse_args()


def existing_entry_set(outfolder, tag, zfileno):
    pattern = re.compile(
        rf"^merged_{re.escape(tag)}_fileno{zfileno}_entry(\d+)\.h5$"
    )
    if not os.path.isdir(outfolder):
        return set()
    out = set()
    for fn in os.listdir(outfolder):
        m = pattern.match(fn)
        if m:
            out.add(int(m.group(1)))
    return out


def open_larlite(path):
    from larlite import larlite as ll
    io = ll.storage_manager(ll.storage_manager.kREAD)
    io.add_in_filename(path)
    io.set_verbosity(2)
    io.open()
    n = io.get_entries()
    io.close()
    return n


def main():
    args = parse_args()

    with open(args.inputlist) as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    if args.lines:
        a, b = args.lines.split(":")
        lo = int(a) if a else 1
        hi = int(b) if b else len(lines)
    else:
        lo, hi = 1, len(lines)

    n_already = 0
    n_written = 0
    n_partial = 0
    n_no_entries = 0
    n_missing_input = 0
    n_errors = 0

    for lineno in range(lo, min(hi, len(lines)) + 1):
        inputfile = lines[lineno - 1].strip()
        if not inputfile:
            continue

        zsub1 = f"{lineno // 1000:03d}"
        zsub2 = f"{lineno // 100:03d}"
        zfileno = f"{lineno:05d}"
        outfolder = os.path.join(args.output_dir, zsub1, zsub2)
        sentinel = os.path.join(
            outfolder, f"{args.tag}_fileno{zfileno}.complete"
        )

        if os.path.exists(sentinel):
            n_already += 1
            continue

        existing = existing_entry_set(outfolder, args.tag, zfileno)
        if not existing:
            n_no_entries += 1
            print(f"[{lineno:5d}] (no entries)        {os.path.basename(inputfile)}")
            continue

        max_existing = max(existing)
        contig_from_zero = (existing == set(range(max_existing + 1)))

        if args.force:
            is_complete = contig_from_zero
            n_total_str = "?"
        else:
            if not os.path.isfile(inputfile):
                n_missing_input += 1
                print(f"[{lineno:5d}] (input missing)     {inputfile}")
                continue
            try:
                n_total = open_larlite(inputfile)
            except Exception as e:
                n_errors += 1
                print(f"[{lineno:5d}] (entry-count err)   {os.path.basename(inputfile)}: {e}")
                continue
            is_complete = (len(existing) == n_total) and \
                          (existing == set(range(n_total)))
            n_total_str = str(n_total)

        status = "COMPLETE" if is_complete else "partial "
        print(f"[{lineno:5d}] {status} existing={len(existing):4d} "
              f"max={max_existing:4d} total={n_total_str:>5s}  "
              f"{os.path.basename(inputfile)}")

        if is_complete:
            if not args.report_only:
                os.makedirs(outfolder, exist_ok=True)
                open(sentinel, "w").close()
            n_written += 1
        else:
            n_partial += 1

    print()
    print("Summary:")
    print(f"  Sentinels already present: {n_already}")
    print(f"  Sentinels written now:     {n_written}"
          + (" (dry run)" if args.report_only else ""))
    print(f"  Partial (need resume):     {n_partial}")
    print(f"  Lines with 0 entries:      {n_no_entries}")
    print(f"  Input files missing:       {n_missing_input}")
    print(f"  Entry-count errors:        {n_errors}")


if __name__ == "__main__":
    main()
