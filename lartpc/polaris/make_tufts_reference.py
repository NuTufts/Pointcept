#!/usr/bin/env python3
"""(Tufts side) Record a per-event reference of the bnb5e19 production for
spot-checking the Polaris re-run (needs h5py; run in the container).

  python3 lartpc/polaris/make_tufts_reference.py --tree <keypoint2_streams> --max-index 499 \
      --out lartpc/polaris/tufts_ref_bnb5e19_idx0-499.json

Per event index: src_file, run/subrun/event, n_particles, the sorted per-particle
point counts (identical list = identical partition), nu_vertex_cm, flash_chi2,
gamma_eff and whether an fm-stream file exists. Events without a nu file are
recorded as absent. The JSON is committed and consumed by compare_to_tufts.py.
"""
import argparse
import json
import os

import h5py


def event_dir(tree, i):
    return os.path.join(tree, "%03d" % (i // 10000), "%03d" % ((i % 10000) // 250))


def summarize(path):
    with h5py.File(path, "r") as h:
        a = h.attrs
        src = a.get("src_file", "")
        src = src.decode() if isinstance(src, bytes) else str(src)
        npts = sorted(int(h["particle"][k]["point_idx"].shape[0]) for k in h["particle"].keys()) \
            if "particle" in h else []
        rec = {
            "src_file": src,
            "run": int(a.get("run", -1)), "subrun": int(a.get("subrun", -1)), "event": int(a.get("event", -1)),
            "n_particles": int(a.get("n_particles", len(npts))),
            "particle_npts": npts,
            "n_slice_points": int(h["slice"]["coord_cm"].shape[0]) if "slice" in h else -1,
            "nu_vertex_cm": [float(x) for x in h["nu_vertex_cm"][()]] if "nu_vertex_cm" in h else None,
            "flash_chi2": float(a.get("flash_chi2", float("nan"))),
            "gamma_eff": float(h["flash"].attrs.get("gamma_eff", float("nan"))) if "flash" in h else None,
        }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--max-index", type=int, default=499)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    events = {}
    n_present = 0
    for i in range(args.max_index + 1):
        d = event_dir(args.tree, i)
        nu = os.path.join(d, f"keypoint2_event{i:05d}_0.h5")
        fm = os.path.join(d, f"keypoint2_event{i:05d}_fm_0.h5")
        if os.path.exists(nu):
            rec = summarize(nu); rec["has_fm"] = os.path.exists(fm); n_present += 1
        else:
            rec = {"absent": True, "has_fm": os.path.exists(fm)}
        events[str(i)] = rec
    out = {"source": args.tree, "max_index": args.max_index, "n_present": n_present, "events": events}
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=0, separators=(",", ":"))
    print(f">>> {n_present}/{args.max_index + 1} events with a nu file -> {args.out}")


if __name__ == "__main__":
    main()
