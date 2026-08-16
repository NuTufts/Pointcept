"""Attach REAL LArMatch scores to overlay merged_sp H5s (replacing the
stepA dummy lm_score == 1.0), joining the stepA0 larmatchme larlite hits to
the H5 triplets by the unique (tick, uwire, vwire, ywire) key — the same
identity the training pipeline's reco/truth merge relies on (both pipelines
build triplets with the same proposal machinery).

Triplets with no larmatch hit above the deploy floor get score 0.0 (the
"would have been cut at production" population). Output = copies of the
input H5s with only triplet_data/lm_score replaced (+ provenance attrs);
originals untouched. Apply the training-parity cut downstream via the
dataset's lm-score threshold (0.15).

    python3 attach_larmatch_scores.py \
        --h5 merged_..._fileno00001_entry000000.h5 [more.h5 ...] \
        --larmatch larmatchme_fileno00001_larlite.root \
        --out-dir <dir> [--producer larmatch]

Requires the ubdl environment (larlite) — run inside the pointcept container
after `source setenv_pointcept_container.sh` in UBDL_DIR.
"""
import argparse
import os
import shutil

import numpy as np
import h5py


def load_hit_scores(larlite_path, producer):
    """(tick,u,v,y) -> score map per (run,subrun,event) from a larmatchme
    larlite file."""
    from larlite import larlite as ll
    io = ll.storage_manager(ll.storage_manager.kREAD)
    io.add_in_filename(larlite_path)
    io.open()
    out = {}
    ientry = 0
    while io.go_to(ientry):
        hits = io.get_data(ll.data.kLArFlow3DHit, producer)
        rse = (int(io.run_id()), int(io.subrun_id()), int(io.event_id()))
        m = {}
        for i in range(hits.size()):
            h = hits.at(i)
            key = (int(h.tick), int(h.targetwire[0]), int(h.targetwire[1]),
                   int(h.targetwire[2]))
            s = float(h[9])
            if key not in m or s > m[key]:
                m[key] = s
        out[rse] = m
        ientry += 1
    io.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--larmatch", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--producer", default="larmatch")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_rse = load_hit_scores(args.larmatch, args.producer)
    print(f">>> larmatch file: {len(by_rse)} events, "
          f"{sum(len(m) for m in by_rse.values())} scored triplets")

    for h5path in args.h5:
        outp = os.path.join(args.out_dir, os.path.basename(h5path))
        with h5py.File(h5path, "r") as f:
            a = f["entry_0"].attrs if f["entry_0"].attrs else f.attrs
            rse = (int(a["run"]), int(a["subrun"]), int(a["event"]))
        m = by_rse.get(rse)
        if m is None:
            print(f"  [MISS] {os.path.basename(h5path)}: rse {rse} not in "
                  f"larmatch file — skipped")
            continue
        shutil.copyfile(h5path, outp)
        with h5py.File(outp, "r+") as f:
            td = f["entry_0/triplet_data"]
            tick = td["tick"][()].astype(np.int64)
            uw = td["uwire"][()].astype(np.int64)
            vw = td["vwire"][()].astype(np.int64)
            yw = td["ywire"][()].astype(np.int64)
            n = len(tick)
            scores = np.zeros(n, np.float32)
            nm = 0
            for i in range(n):
                s = m.get((int(tick[i]), int(uw[i]), int(vw[i]), int(yw[i])))
                if s is not None:
                    scores[i] = s
                    nm += 1
            del td["lm_score"]
            td.create_dataset("lm_score", data=scores)
            f["entry_0"].attrs["lm_score_source"] = "larmatchme_attached"
            f["entry_0"].attrs["lm_score_match_frac"] = nm / max(n, 1)
        print(f"  {os.path.basename(h5path)}: {nm}/{n} triplets matched "
              f"({nm / max(n, 1):.3f}); frac>=0.15: "
              f"{(scores >= 0.15).mean():.3f}")


if __name__ == "__main__":
    main()
