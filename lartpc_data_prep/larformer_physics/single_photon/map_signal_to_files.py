"""
Step 2 of the single-photon study: map selected signal events to the official
merged_dlreco files that contain them, so they can be fed to the LArFormer cascade.

WHY NOT fileid: the ntuple `fileid` is meant to be the line number into the
sample master filelist, but that filelist was reordered after the ntuples were
produced, so fileid no longer points to the right file (verified: fileid=0 events
are run 16934, but filelist line 0 is run 14121). We therefore match on the robust
identifier (run, subrun, event) using the official files' `larlite_id_tree`.

KEY FACTS this relies on (see docs/Gen2_Flat_Ntuple_Spec.md):
  - The on-disk path encodes the run: run R -> zero-pad to 8 digits -> split into
    four 2-digit dirs, e.g. run 16934 -> 00016934 -> <data_root>/00/01/69/34/ .
    All files for a run live in that one directory.
  - A merged_dlreco file can hold MULTIPLE (run, subrun) pairs (it is NOT one
    subrun per file), so we index each file under every (run,subrun) in its id-tree.
  - `larlite_id_tree` is a plain TTree with POD branches _run_id/_subrun_id/_event_id
    (readable with bare ROOT, no larlite libraries).

So to find a signal event's file we scan the run's directory, index every file by
the (run,subrun) pairs it contains, then confirm the event id is present.

Capped subsample: by default we map only the first --nfiles unique (run,subrun)
pairs found in the signal CSV, to keep the first cascade pass small.

Outputs (into --outdir):
  signal_files_subsample.txt   one official merged_dlreco path per line (cascade INPUTLIST)
  signal_file_map.csv          run,subrun,filepath,n_signal_events,n_events_confirmed

Run inside the pointcept container (after sourcing setenv_pointcept_container.sh):
  python3 map_signal_to_files.py --nfiles 20 --outdir workdir
"""
import argparse
import csv
import glob
import os
import ROOT as rt

DATA_ROOT = ("/cluster/tufts/wongjiradlab/larbys/data/mcc9/"
             "mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/data")


def run_to_dir(data_root, run):
    p = "%08d" % run
    return os.path.join(data_root, p[0:2], p[2:4], p[4:6], p[6:8])


def read_idtree(path):
    """return (set of (run,subrun,event), set of (run,subrun)) present in the file."""
    f = rt.TFile(path)
    t = f.Get("larlite_id_tree")
    if not t or t.GetEntries() == 0:
        f.Close()
        return set(), set()
    rse = set()
    rs = set()
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        r, s, e = int(t._run_id), int(t._subrun_id), int(t._event_id)
        rse.add((r, s, e))
        rs.add((r, s))
    f.Close()
    return rse, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-csv", default="workdir/signal_events.csv")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--nfiles", type=int, default=20,
                    help="cap: map only N unique (run,subrun) files (-1 = all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth unique (run,subrun) (default 1 = consecutive "
                         "head). Use to spread the subsample across the whole sample "
                         "instead of clustering in the first ntuple region.")
    ap.add_argument("--spread", action="store_true",
                    help="auto-pick --stride so the --nfiles selection is spread evenly "
                         "across all unique files (overrides --stride)")
    ap.add_argument("--outdir", default="workdir")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- group signal events by (run, subrun) ----
    by_rs = {}            # (run,subrun) -> list of (run,subrun,event)
    order = []            # preserve first-seen order of (run,subrun)
    with open(args.signal_csv) as fh:
        for row in csv.DictReader(fh):
            r, s, e = int(row["run"]), int(row["subrun"]), int(row["event"])
            key = (r, s)
            if key not in by_rs:
                by_rs[key] = []
                order.append(key)
            by_rs[key].append((r, s, e))

    n_unique = len(order)
    stride = args.stride
    if args.spread and args.nfiles > 0:
        stride = max(1, n_unique // args.nfiles)
    if stride < 1:
        stride = 1
    order = order[::stride]
    if args.nfiles >= 0:
        order = order[:args.nfiles]
    print("signal events span %d unique (run,subrun) files; "
          "selecting %d (stride=%d%s)"
          % (n_unique, len(order), stride, ", spread" if args.spread else ""))

    # ---- resolve each (run,subrun) to a file via its run directory ----
    run_dir_cache = {}    # run -> { (run,subrun): (path, rse_set) }

    def scan_run_dir(run):
        if run in run_dir_cache:
            return run_dir_cache[run]
        d = run_to_dir(args.data_root, run)
        mapping = {}    # (run,subrun) -> (path, rse_set)
        for path in sorted(glob.glob(os.path.join(d, "merged_dlreco_*.root"))):
            rse, rs_set = read_idtree(path)
            for rs in rs_set:
                mapping[rs] = (path, rse)   # index file under every (run,subrun) it holds
        run_dir_cache[run] = mapping
        return mapping

    map_rows = []
    file_list = []
    n_ok = n_missing = 0
    for (r, s) in order:
        mapping = scan_run_dir(r)
        hit = mapping.get((r, s))
        if hit is None:
            print("  MISSING file for run=%d subrun=%d (dir=%s)"
                  % (r, s, run_to_dir(args.data_root, r)))
            n_missing += 1
            map_rows.append((r, s, "NOT_FOUND", len(by_rs[(r, s)]), 0))
            continue
        path, rse = hit
        wanted = by_rs[(r, s)]
        confirmed = sum(1 for ev in wanted if ev in rse)
        map_rows.append((r, s, path, len(wanted), confirmed))
        file_list.append(path)
        n_ok += 1
        if confirmed != len(wanted):
            print("  WARNING run=%d subrun=%d: %d/%d signal events confirmed in id-tree"
                  % (r, s, confirmed, len(wanted)))

    # ---- write outputs ----
    list_path = os.path.join(args.outdir, "signal_files_subsample.txt")
    seen = set()
    uniq_files = []
    for p in file_list:
        if p not in seen:
            seen.add(p)
            uniq_files.append(p)
    with open(list_path, "w") as fh:
        for p in uniq_files:
            fh.write(p + "\n")
    file_list = uniq_files
    map_path = os.path.join(args.outdir, "signal_file_map.csv")
    with open(map_path, "w") as fh:
        fh.write("run,subrun,filepath,n_signal_events,n_events_confirmed\n")
        for row in map_rows:
            fh.write("%d,%d,%s,%d,%d\n" % row)

    print("-" * 60)
    print("files resolved : %d" % n_ok)
    print("files missing  : %d" % n_missing)
    print("wrote: %s  (%d files for cascade INPUTLIST)" % (list_path, len(file_list)))
    print("wrote: %s" % map_path)


if __name__ == "__main__":
    main()
