"""
Pass-2 bridge between Stage A (convert, all events) and Stage B (cascade, capped).

Stage A converts EVERY event in each subsample file to a per-event H5. Most of
those are not our signal events, and the GPU cascade is expensive, so we run
Stage B only on a CAPPED set of the actual signal events.

This script scans the Stage-A H5 outputs, reads each H5's entry_0 (run,subrun,event)
attrs, keeps the ones present in signal_events.csv, caps the count, and writes the
input list that Stage B (run_larformer_stage3_inference.py --input-list) consumes.

Run inside the pointcept container (needs h5py):
  python3 select_cascade_events.py --h5-dir workdir/larformer_h5 \
      --signal-csv workdir/signal_events.csv --nevents 200 --out workdir/cascade_inputs.txt
"""
import argparse
import csv
import glob
import os
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default="workdir/larformer_h5",
                    help="dir holding Stage-A merged_*_entry*.h5 (searched recursively)")
    ap.add_argument("--signal-csv", default="workdir/signal_events.csv")
    ap.add_argument("--tag", default="sp_bnb_nu_overlay")
    ap.add_argument("--nevents", type=int, default=200,
                    help="cap on number of signal-event H5s for the cascade (-1 = all)")
    ap.add_argument("--out", default="workdir/cascade_inputs.txt")
    args = ap.parse_args()

    # signal (run,subrun,event) set
    signal = set()
    with open(args.signal_csv) as fh:
        for row in csv.DictReader(fh):
            signal.add((int(row["run"]), int(row["subrun"]), int(row["event"])))
    print("signal events in CSV:", len(signal))

    h5s = sorted(glob.glob(os.path.join(args.h5_dir, "**",
                                        "merged_%s_fileno*_entry*.h5" % args.tag),
                           recursive=True))
    print("Stage-A H5 files found:", len(h5s))

    chosen = []          # (path, (r,s,e))
    n_scanned = n_signal = 0
    for p in h5s:
        n_scanned += 1
        try:
            with h5py.File(p, "r") as f:
                a = f["entry_0"].attrs
                rse = (int(a["run"]), int(a["subrun"]), int(a["event"]))
        except Exception as ex:
            print("  WARN could not read %s: %s" % (p, ex))
            continue
        if rse in signal:
            n_signal += 1
            # absolute path: the cascade runs from a different CWD, and relative
            # paths would resolve to a bogus location (e.g. /workdir/...).
            chosen.append((os.path.abspath(p), rse))

    # determinism + capping
    chosen.sort(key=lambda x: x[1])
    if args.nevents >= 0:
        chosen = chosen[:args.nevents]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        for p, _ in chosen:
            fh.write(p + "\n")

    print("-" * 60)
    print("H5 scanned          : %d" % n_scanned)
    print("matched signal events: %d" % n_signal)
    print("written to cascade   : %d (cap=%d)" % (len(chosen), args.nevents))
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
