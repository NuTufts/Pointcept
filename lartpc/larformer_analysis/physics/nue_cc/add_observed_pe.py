"""Augment a nue_cc table (.npz) with the in-time beam-flash total observed PE.

The observed PE is NOT in the gen2ntuple -- it lives in the cascade
keypoint2_streams nu-slice files as `flash/observed_pe` (32 PMTs). This scans a
cascade dir once, builds {(run,subrun,event): sum(observed_pe)} (cached), maps it
onto the table's events by (run,subrun,event), and re-saves the table with an
`observed_pe` column (NaN where no cascade match).

Motivation (light-yield / run-period test): EXT + MC are Run 3, bnb5e19 data is
Run 1. Run 1 has higher PMT light yield, so in-time cosmics produce more PE and
more pass the light trigger -> expect the Run-1 data observed-PE distribution
shifted to higher PE vs the Run-3-based prediction, with the flash-chi2 excess
piling up at lower observed PE.

    PYTHONPATH=. python3 add_observed_pe.py --table tables/data.npz \
        --cascade-dir <keypoint2_streams> --cache caches/pe_data.npz
"""
import argparse
import os

import numpy as np
import h5py


def scan_pe(cascade_dir, cache_npz=None):
    """{(run,subrun,event): sum(flash/observed_pe)} over nu-slice files; cached."""
    if cache_npz and os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        return {tuple(int(x) for x in k): float(v)
                for k, v in zip(z["keys"], z["pe"])}
    m = {}
    nfile = 0
    for root, _dirs, files in os.walk(cascade_dir):
        for n in files:
            if (not n.startswith("keypoint2_event") or not n.endswith("_0.h5")
                    or n.endswith("_fm_0.h5")):
                continue
            try:
                with h5py.File(os.path.join(root, n), "r") as f:
                    key = (int(f.attrs["run"]), int(f.attrs["subrun"]),
                           int(f.attrs["event"]))
                    if "flash" in f and "observed_pe" in f["flash"]:
                        m[key] = float(np.sum(f["flash/observed_pe"][()]))
            except Exception:                                  # noqa: BLE001
                continue
            nfile += 1
            if nfile % 20000 == 0:
                print(f"    scanned {nfile} files, {len(m)} RSE ...", flush=True)
    if cache_npz and m:
        os.makedirs(os.path.dirname(os.path.abspath(cache_npz)), exist_ok=True)
        np.savez(cache_npz, keys=np.array(list(m.keys()), np.int64),
                 pe=np.array(list(m.values()), np.float64))
        print(f">>> cached {len(m)} RSE-PE to {cache_npz}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    pe_map = scan_pe(args.cascade_dir, args.cache)
    print(f">>> {len(pe_map)} RSE->PE from {args.cascade_dir}")

    tab = dict(np.load(args.table))
    run = tab["run"]; sub = tab["subrun"]; evt = tab["event"]
    n = len(run)
    obs_pe = np.full(n, np.nan)
    hit = 0
    for i in range(n):
        v = pe_map.get((int(run[i]), int(sub[i]), int(evt[i])))
        if v is not None:
            obs_pe[i] = v; hit += 1
    tab["observed_pe"] = obs_pe
    np.savez(args.table, **tab)
    print(f">>> matched {hit}/{n} events; wrote observed_pe into {args.table}")


if __name__ == "__main__":
    main()
