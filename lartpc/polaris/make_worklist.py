#!/usr/bin/env python3
"""Build a text worklist (one task per line) for lartpc/polaris/node_launcher.sh.

Only the standard library is used, so it runs on the Polaris host (outside the
container). Modes:

  cascade  --list <merged_sp list> --nshards N [--max-events M]
           lines: "<shard> <start> <n>"  -- contiguous ranges, PER = ceil(len/N)
           (same arithmetic as larformer_reco/slurm/submit_inference_shard.sh);
           --max-events restricts the universe to the first M events (tests).
  range    --start S --n N --nshards K            lines: "<shard> <start> <n>"
           (K contiguous shards over [S, S+N); e.g. the 4 x 50-event throughput test)
  nu_reco  --list <kp2 list> --stream nu|fm --nshards N [--min-per-shard 50]
           lines: "<stream> <start> <n>"  (append both streams into one file)
  larpid   --nu-reco-dir <dir> --stream nu|fm     lines: "<stream> <nu_reco_shard path>"
  export   --list <merged_sp list> --nshards N [--min-per-shard 50]
           lines: "<shard> <start> <n>"  (shard <= 99: the ntuple naming is %02d)

--out <file>: written (append with --append). Prints the line count.
"""
import argparse
import glob
import math
import os
import sys


def count_lines(path):
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                n += 1
    return n


def contiguous(total, nshards, offset=0):
    """(shard, start, n) for nshards contiguous ranges over [offset, offset+total)."""
    if total <= 0:
        return []
    nshards = max(1, min(nshards, total))
    per = math.ceil(total / nshards)
    out = []
    for k in range(nshards):
        start = k * per
        if start >= total:
            break
        out.append((k, offset + start, min(per, total - start)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["cascade", "range", "nu_reco", "larpid", "export"])
    ap.add_argument("--list", help="input list (merged_sp or keypoint2)")
    ap.add_argument("--nshards", type=int, default=64)
    ap.add_argument("--max-events", type=int, default=None,
                    help="cascade: use only the first M events of the list")
    ap.add_argument("--start", type=int, default=0, help="range mode")
    ap.add_argument("--n", type=int, default=0, help="range mode")
    ap.add_argument("--stream", choices=["nu", "fm"], help="nu_reco / larpid")
    ap.add_argument("--nu-reco-dir", help="larpid mode")
    ap.add_argument("--min-per-shard", type=int, default=50,
                    help="nu_reco/export: do not make shards smaller than this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    lines = []
    if args.mode in ("cascade", "export"):
        if not args.list:
            sys.exit("--list required")
        total = count_lines(args.list)
        if args.max_events is not None:
            total = min(total, args.max_events)
        nsh = args.nshards
        if args.mode == "export":
            nsh = max(1, min(nsh, 99, math.ceil(total / max(1, args.min_per_shard))))
        for k, s, n in contiguous(total, nsh):
            lines.append(f"{k} {s} {n}")
        print(f">>> {args.mode}: {total} events -> {len(lines)} shards "
              f"(per shard <= {math.ceil(total / max(1, len(lines)))})")
    elif args.mode == "range":
        if args.n <= 0:
            sys.exit("--n > 0 required")
        for k, s, n in contiguous(args.n, args.nshards, offset=args.start):
            lines.append(f"{k} {s} {n}")
        print(f">>> range [{args.start}, {args.start + args.n}) -> {len(lines)} shards")
    elif args.mode == "nu_reco":
        if not (args.list and args.stream):
            sys.exit("--list and --stream required")
        total = count_lines(args.list)
        nsh = max(1, min(args.nshards, math.ceil(total / max(1, args.min_per_shard))))
        for _k, s, n in contiguous(total, nsh):
            lines.append(f"{args.stream} {s} {n}")
        print(f">>> nu_reco {args.stream}: {total} kp2 files -> {len(lines)} shards")
    elif args.mode == "larpid":
        if not (args.nu_reco_dir and args.stream):
            sys.exit("--nu-reco-dir and --stream required")
        shards = sorted(glob.glob(os.path.join(args.nu_reco_dir, "nu_reco_shard*.h5")))
        for p in shards:
            lines.append(f"{args.stream} {p}")
        print(f">>> larpid {args.stream}: {len(lines)} nu_reco shards in {args.nu_reco_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a" if args.append else "w") as fo:
        for line in lines:
            fo.write(line + "\n")
    print(f">>> {'appended' if args.append else 'wrote'} {len(lines)} lines -> {args.out}")


if __name__ == "__main__":
    main()
