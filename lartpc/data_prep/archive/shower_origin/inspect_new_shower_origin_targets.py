"""
Sanity-check the redefined shower-origin training targets.

For each fragment in each input HDF5 event file, prints the four target-routing
inputs (type, trackid, istrunk, nu_vertex_is_visible, num visible nu showers)
and the resulting score-peak and regression targets emitted by
ShowerOriginDataset.get_all_fragments() — so you can verify the routing
matrix matches expectations:

    type=0, vertex observable     -> origin_coord = stored originpt
    type=0, vertex unobservable   -> origin_coord = trunk_start_coord
    type=1                        -> origin_coord = trunk_start_coord
    type=2                        -> origin_coord = stored originpt

Usage:
    python lartpc/data_prep/archive/shower_origin/inspect_new_shower_origin_targets.py \
        --data-list example_showerorigin_trainingdata.txt --num-events 3

    python lartpc/data_prep/archive/shower_origin/inspect_new_shower_origin_targets.py \
        --input /path/to/single.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np

# Allow running from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pointcept.datasets.shower_origin import ShowerOriginDataset  # noqa: E402


TYPE_NAMES = {0: "nu-inside", 1: "outside", 2: "cosmic-inside"}


def routing_label(frag_type, nu_vertex_observable):
    if frag_type == 0:
        return "origin=originpt" if nu_vertex_observable else "origin=trunk_start (unobservable)"
    if frag_type == 1:
        return "origin=trunk_start (outside-TPC)"
    if frag_type == 2:
        return "origin=originpt (cosmic)"
    return f"origin=? (unknown type {frag_type})"


def peek_event(filepath):
    """Read raw HDF5 to compute the same routing decision the dataset makes."""
    with h5py.File(filepath, "r") as f:
        sf = f["entry_0/shower_fragments"]
        num = int(sf.attrs.get("num_fragments", 0))
        nu_vis = int(sf["nu_vertex_is_visible"][()]) if "nu_vertex_is_visible" in sf else 1
        types = sf["type"][:]
        counts = sf["pointindices_counts"][:]
        # Visible nu-showers (matches min_fragment_points=20 default below)
        visible_nu = set()
        for i in range(num):
            if int(counts[i]) >= 20 and int(types[i]) == 0:
                visible_nu.add(int(sf["trackid"][i]))
        observable = (nu_vis == 1) or (len(visible_nu) >= 2)
    return num, nu_vis, len(visible_nu), observable


def fmt_pt(p):
    return f"({p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f})"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Single HDF5 file")
    g.add_argument("--data-list", help="Text file listing HDF5 files")
    ap.add_argument("--num-events", type=int, default=5)
    ap.add_argument("--min-fragment-points", type=int, default=20)
    args = ap.parse_args()

    if args.input:
        files = [args.input]
    else:
        with open(args.data_list) as f:
            files = [line.strip() for line in f
                     if line.strip() and not line.startswith("#")]
        files = files[: args.num_events]

    # Build a temp data-list file the dataset can consume (it expects a path)
    tmp_list = "/tmp/_inspect_sht_list.txt"
    with open(tmp_list, "w") as fh:
        for p in files:
            fh.write(p + "\n")

    ds = ShowerOriginDataset(
        split="train",
        data_root="/",
        data_list_file=tmp_list,
        coord_scale=1.0,
        include_ghosts=False,
        include_cosmic_showers=True,
        min_fragment_points=args.min_fragment_points,
        transform=None,
    )

    print(f"Built dataset with {len(ds.event_index)} events")
    for ev_idx in range(len(ds.event_index)):
        file_idx = ds.event_index[ev_idx]
        path = ds.data_list[file_idx]
        num_raw, nu_vis, n_vis_nu, observable = peek_event(path)
        print()
        print("=" * 100)
        print(f"Event {ev_idx}: {os.path.basename(path)}")
        print(f"  raw num_fragments={num_raw}  nu_vertex_is_visible={nu_vis}  "
              f"visible_nu_showers={n_vis_nu}  -> nu_vertex_observable={observable}")

        try:
            frags = ds.get_all_fragments(ev_idx)
        except Exception as e:
            print(f"  [error] get_all_fragments failed: {e}")
            continue

        print(f"  kept fragments: {len(frags)}")
        print(f"  {'tid':>5} {'pid':>4} {'type':<14} "
              f"{'origin_coord':>26} {'trunk_start':>26} {'start':>26}  routing")
        for fr in frags:
            t = int(fr["origin_type"])
            tname = TYPE_NAMES.get(t, f"t{t}")
            o = fr["origin_coord"]
            ts = fr["trunk_start_coord"]
            st = fr["start_coord"]
            print(f"  {fr['trackid']:>5} {fr['pid']:>4} {tname:<14} "
                  f"{fmt_pt(o)} {fmt_pt(ts)} {fmt_pt(st)}  "
                  f"{routing_label(t, observable)}")
            # Cross-check: when origin should equal trunk_start, it does.
            should_eq_trunk = (
                t == 1 or (t == 0 and not observable)
            )
            if should_eq_trunk:
                if not np.allclose(o, ts):
                    print(f"     [MISMATCH] origin_coord != trunk_start_coord "
                          f"(diff={np.linalg.norm(o-ts):.3f})")
            else:
                # type=0 observable, type=2: origin should NOT necessarily equal trunk_start
                pass


if __name__ == "__main__":
    main()
