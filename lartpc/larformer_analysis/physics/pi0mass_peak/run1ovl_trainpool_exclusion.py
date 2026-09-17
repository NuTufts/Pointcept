"""Build the row-exclusion list for the run-1 BNB-overlay HALF ntuple.

The half production converted the mcc9_v28_run1_bnboverlay TRAINPOOL list
(filenos 1-4740). Its first 350 filenos were ALSO converted earlier as the
"generic-mu tranche 1" pilot (overlay_train/mcc9_v28_run1_bnboverlay, same
fileno numbering, byte-identical files) and 7,930 of those events (the
nu-deposit-filtered subset) sit in the stage-3 segmenter training list
(training_data_ledger/cachelist_rebalanced_train_v2.txt). Analysis MC must
not contain events a chain model trained on, so we drop those FILES whole
(a whole-file drop keeps the POT bookkeeping exact and avoids the
composition bias an event-level drop of a nu-deposit-filtered subset would
introduce) and re-sum the POT over the surviving files from the truth
sidecars (one potTree row per fileno, run/subrun/event per entry).

Ntuple rows are mapped to filenos through the sidecars' per-entry
(run, subrun, event) attrs -- no reliance on hadd ordering.

    python3 run1ovl_trainpool_exclusion.py \
        --ntuple .../dlgen2_larformer_ntuple_bnbovl_run1_half.root \
        --sidecar-dir .../run1_bnboverlay_half/truth_sidecar \
        --out run1ovl_trainpool_exclusion.npz
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import uproot

K = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
LEDGER = os.path.join(K, "lartpc/data_prep/uboone_official/training_data_ledger")
SAMPLE = "mcc9_v28_run1_bnboverlay"


def training_filenos(lists):
    """(filenos, (fileno, entry) pairs) of SAMPLE events in the given lists."""
    fns, pairs = set(), set()
    rx = re.compile(SAMPLE + r"_fileno(\d+)_entry(\d+)")
    for path in lists:
        with open(path) as f:
            for line in f:
                m = rx.search(line)
                if m:
                    fns.add(int(m.group(1)))
                    pairs.add((int(m.group(1)), int(m.group(2))))
    return fns, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--train-lists", nargs="+", default=[
        os.path.join(LEDGER, "cachelist_rebalanced_train_v2.txt"),
        os.path.join(LEDGER, "h5list_generic_mu_tranche1_all.txt")],
        help="lists whose SAMPLE entries define the trained/pilot files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_fn, train_pairs = training_filenos(args.train_lists)
    print(f">>> training/pilot lists: {len(train_pairs)} events over "
          f"{len(train_fn)} filenos (range {min(train_fn)}-{max(train_fn)})")

    # sidecars: fileno -> POT, (run,subrun,event) -> (fileno, entry)
    rse2fe, pot = {}, {}
    files = sorted(glob.glob(os.path.join(args.sidecar_dir, "truth_fileno*.h5")))
    for path in files:
        fn = int(re.search(r"truth_fileno(\d+)\.h5", path).group(1))
        with h5py.File(path, "r") as f:
            pot[fn] = (float(f.attrs["totPOT"]), float(f.attrs["totGoodPOT"]))
            for k in f.keys():
                if not k.startswith("entry_"):
                    continue
                g = f[k].attrs
                rse2fe[(int(g["run"]), int(g["subrun"]), int(g["event"]))] = (
                    fn, int(k.split("_")[1]))
    print(f">>> {len(files)} sidecars, {len(rse2fe)} entries, "
          f"total POT {sum(p[0] for p in pot.values()):.4e}")

    t = uproot.open(args.ntuple)
    a = t["EventTree"].arrays(["run", "subrun", "event"], library="np")
    n = len(a["run"])
    fe = np.full((n, 2), -1, np.int64)
    for i in range(n):
        fe[i] = rse2fe.get((int(a["run"][i]), int(a["subrun"][i]),
                            int(a["event"][i])), (-1, -1))
    unmatched = int((fe[:, 0] < 0).sum())
    print(f">>> ntuple {n} rows; unmatched to sidecars: {unmatched}")
    if unmatched:
        raise SystemExit("row->fileno mapping incomplete; refusing to write")
    pt = t["potTree"].arrays(library="np")
    pot_tree = float(np.sum(pt["totGoodPOT"]) or np.sum(pt["totPOT"]))
    present = sorted(set(fe[:, 0].tolist()))
    pot_present = sum(pot[fn][1] or pot[fn][0] for fn in present)
    print(f">>> filenos present in ntuple {len(present)} | potTree rows "
          f"{len(pt['totPOT'])} POT {pot_tree:.4e} | sidecar POT of present "
          f"filenos {pot_present:.4e}")

    drop_fn = sorted(train_fn & set(present))
    drop = np.isin(fe[:, 0], drop_fn)
    pairs_in = sum(1 for (f_, e_) in zip(fe[:, 0], fe[:, 1])
                   if (int(f_), int(e_)) in train_pairs)
    kept_pot = sum(pot[fn][1] or pot[fn][0] for fn in present
                   if fn not in train_fn)
    print(f">>> dropping {len(drop_fn)} filenos = {int(drop.sum())} rows "
          f"({drop.mean()*100:.2f}%), of which {pairs_in} rows are listed "
          f"training/pilot events | kept POT {kept_pot:.4e} "
          f"({kept_pot/pot_present*100:.2f}% of present)")
    np.savez(args.out, rows=np.nonzero(drop)[0].astype(np.int64),
             fileno=fe[:, 0], entry=fe[:, 1],
             dropped_filenos=np.asarray(drop_fn, np.int64),
             kept_pot=kept_pot, total_pot=pot_present, pot_tree=pot_tree)
    print(f">>> wrote {args.out}")


if __name__ == "__main__":
    main()
