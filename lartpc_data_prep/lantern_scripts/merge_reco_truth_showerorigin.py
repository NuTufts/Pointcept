#!/usr/bin/env python3
"""
Merge reco shower fragments with truth labels.

Takes a reco HDF5 (from convert_larlite_to_showerorigin_h5.py) and a truth HDF5
(from process_dlmerged_to_hdf5_event_files.py) for the same event, matches reco
points to truth points via (tick, uwire, vwire, ywire) lookup, labels each reco
DBSCAN fragment, and writes a new merged HDF5 with truth fields filled in.

Usage:
    python lartpc_data_prep/merge_reco_truth_showerorigin.py \
        --reco-h5 showerorigin_event_000000.h5 \
        --truth-h5 pointceptdata_event_000000.h5 \
        --output merged_event_000000.h5

See docs/uboone_recopipeline_for_shower_origin.md Step 3 for details.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

SHOWER_PIDS = {11, -11, 22}

TRACK_PIDS = {13, -13, 211, -211, 321, -321, 2212}

# Fragment type codes
TYPE_INSIDE = 0
TYPE_OUTSIDE = 1
TYPE_ON_TRACK = 2
TYPE_GHOST = 3
TYPE_TRUE_TRACK = 4

TYPE_NAMES = {
    0: "inside", 1: "outside", 2: "on_track",
    3: "ghost", 4: "true_track",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge reco shower fragments with truth labels"
    )
    parser.add_argument(
        "--reco-h5", required=True, type=str,
        help="Reco HDF5 file (from convert_larlite_to_showerorigin_h5.py)"
    )
    parser.add_argument(
        "--truth-h5", required=True, type=str,
        help="Truth HDF5 file (from process_dlmerged_to_hdf5_event_files.py)"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output merged HDF5 file"
    )
    parser.add_argument(
        "--stats-json", type=str, default=None,
        help="Optional path for JSON stats sidecar (default: <output>.stats.json)"
    )
    return parser.parse_args()


# ── Point Matching ────────────────────────────────────────────────────────────

def build_truth_lookup(truth_f):
    """
    Build a dict from (tick, uwire, vwire, ywire) -> truth point index.

    Both reco and truth points are produced by the same triplet-making
    algorithm, so (tick, u, v, y) uniquely identifies a spacepoint.
    """
    triplet = truth_f["entry_0"]["triplet_data"]
    tick = triplet["tick"][:].astype(np.int32)
    uwire = triplet["uwire"][:].astype(np.int32)
    vwire = triplet["vwire"][:].astype(np.int32)
    ywire = triplet["ywire"][:].astype(np.int32)

    lookup = {}
    for i in range(len(tick)):
        key = (int(tick[i]), int(uwire[i]), int(vwire[i]), int(ywire[i]))
        lookup[key] = i

    return lookup


def match_reco_to_truth(reco_f, truth_lookup):
    """
    Match each reco point to its truth index via wire coordinate lookup.

    Returns:
        match_idx: (N_reco,) int64 array, truth index per reco point (-1 if unmatched)
    """
    triplet = reco_f["entry_0"]["triplet_data"]
    tick = triplet["tick"][:].astype(np.int32)
    uwire = triplet["uwire"][:].astype(np.int32)
    vwire = triplet["vwire"][:].astype(np.int32)
    ywire = triplet["ywire"][:].astype(np.int32)

    n_reco = len(tick)
    match_idx = np.full(n_reco, -1, dtype=np.int64)

    for i in range(n_reco):
        key = (int(tick[i]), int(uwire[i]), int(vwire[i]), int(ywire[i]))
        tidx = truth_lookup.get(key, -1)
        match_idx[i] = tidx

    return match_idx


def get_per_point_truth(truth_f, match_idx):
    """
    Map truth per-point arrays to reco points using match_idx.

    Returns dict with trackid, pid, origin arrays (N_reco,).
    """
    triplet = truth_f["entry_0"]["triplet_data"]
    truth_trackid = triplet["trackid"][:].astype(np.int64)
    truth_pid = triplet["pid"][:].astype(np.int64)
    truth_origin = triplet["origin"][:].astype(np.int64)

    n_reco = len(match_idx)
    matched = match_idx >= 0

    reco_trackid = np.full(n_reco, -1, dtype=np.int64)
    reco_pid = np.full(n_reco, -1, dtype=np.int64)
    reco_origin = np.full(n_reco, -1, dtype=np.int64)

    valid = match_idx[matched]
    reco_trackid[matched] = truth_trackid[valid]
    reco_pid[matched] = truth_pid[valid]
    reco_origin[matched] = truth_origin[valid]

    return {
        "trackid": reco_trackid,
        "pid": reco_pid,
        "origin": reco_origin,
    }


# ── Truth Fragment Origin Lookup ──────────────────────────────────────────────

def load_truth_shower_fragments(truth_f):
    """
    Load truth shower_fragments metadata for origin lookup.

    Returns dict with arrays or None if no shower_fragments in truth file.
    """
    entry = truth_f["entry_0"]
    if "shower_fragments" not in entry:
        return None

    sf = entry["shower_fragments"]
    num_frags = int(sf.attrs.get("num_fragments", 0))
    if num_frags == 0:
        return None

    result = {
        "trackid": sf["trackid"][:].astype(np.int64),
        "type": sf["type"][:].astype(np.int64),
        "originpt": sf["originpt"][:].astype(np.float32),
        "istrunk": sf["istrunk"][:].astype(np.int64),
        "startpt": sf["startpt"][:].astype(np.float32),
    }

    if "pret0shiftedoriginpt" in sf:
        result["pret0shiftedoriginpt"] = sf["pret0shiftedoriginpt"][:].astype(
            np.float32
        )

    if "nu_vertex_is_visible" in sf:
        result["nu_vertex_is_visible"] = int(sf["nu_vertex_is_visible"][()])
    else:
        result["nu_vertex_is_visible"] = 1

    return result


def lookup_origin(tid, truth_sf):
    """
    Look up origin info for a trackid from truth shower_fragments.

    Returns:
        (type, originpt, pret0shiftedoriginpt)
        Falls back to (TYPE_GHOST, zeros, zeros) if trackid not found.
    """
    if truth_sf is None:
        return TYPE_GHOST, np.zeros(3, np.float32), np.zeros(4, np.float32)

    matches = np.where(truth_sf["trackid"] == tid)[0]
    if len(matches) == 0:
        return TYPE_GHOST, np.zeros(3, np.float32), np.zeros(4, np.float32)

    # Prefer trunk fragment (istrunk==1)
    trunk = [m for m in matches if truth_sf["istrunk"][m] == 1]
    idx = trunk[0] if trunk else matches[0]

    origin_type = int(truth_sf["type"][idx])
    originpt = truth_sf["originpt"][idx].copy()
    if "pret0shiftedoriginpt" in truth_sf:
        pret0 = truth_sf["pret0shiftedoriginpt"][idx].copy()
    else:
        pret0 = np.zeros(4, np.float32)

    return origin_type, originpt, pret0


# ── Fragment Labeling ─────────────────────────────────────────────────────────

def label_fragments(reco_f, match_idx, per_point_truth, truth_sf):
    """
    Label each reco DBSCAN fragment using truth information.

    Returns list of dicts, one per fragment, with:
        trackid, pid, type, originpt, pret0shiftedoriginpt, startpt, purity, stats
    """
    entry = reco_f["entry_0"]
    reco_pos = entry["triplet_data"]["pos"][:].astype(np.float32)
    sf = entry["shower_fragments"]
    num_frags = int(sf.attrs.get("num_fragments", 0))
    if num_frags == 0:
        return []

    flat_indices = sf["pointindices_flat"][:].astype(np.int64)
    index_counts = sf["pointindices_counts"][:].astype(np.int64)

    reco_trackid = per_point_truth["trackid"]
    reco_pid = per_point_truth["pid"]

    results = []
    offset = 0

    for fi in range(num_frags):
        count = int(index_counts[fi])
        frag_idx = flat_indices[offset:offset + count]
        offset += count

        frag_tids = reco_trackid[frag_idx]
        frag_pids = reco_pid[frag_idx]
        frag_matched = match_idx[frag_idx] >= 0

        n_total = len(frag_idx)

        # Count points by category:
        # - "shower": matched, trackid > 0, pid in SHOWER_PIDS
        # - "non-shower": matched, trackid > 0, pid NOT in SHOWER_PIDS
        # - "ambiguous": unmatched or trackid <= 0 (truth ghost)
        #
        # Truth-ghost points (trackid=0/pid=0) passed reco ghost removal
        # but weren't matched to an MC particle. They are ambiguous and
        # should NOT count against a fragment being a shower. Only points
        # positively identified as non-shower particles count as non-shower.
        n_non_shower = 0
        n_track = 0
        n_shower = 0
        n_ambiguous = 0
        for j in range(n_total):
            if not frag_matched[j] or frag_tids[j] <= 0:
                n_ambiguous += 1  # unmatched or truth ghost — neutral
            elif int(frag_pids[j]) in SHOWER_PIDS:
                n_shower += 1
            else:
                n_non_shower += 1  # positively non-shower particle
                if int(frag_pids[j]) in TRACK_PIDS:
                    n_track += 1

        frag_info = {
            "n_total": n_total,
            "n_matched": int(frag_matched.sum()),
            "n_shower": n_shower,
            "n_non_shower": n_non_shower,
            "n_ambiguous": n_ambiguous,
            "n_track": n_track,
        }

        # Decision: compare shower vs non-shower among identified points.
        # If no identified points, or non-shower outnumbers shower, label
        # as ghost/true_track. Ambiguous points don't influence the decision.
        n_identified = n_shower + n_non_shower
        if n_identified == 0 or n_non_shower > n_shower:
            # Majority non-shower
            if n_identified > 0 and n_track > n_identified / 2:
                frag_type = TYPE_TRUE_TRACK
            else:
                frag_type = TYPE_GHOST
            frag_info.update({
                "trackid": -1,
                "pid": -1,
                "type": frag_type,
                "originpt": np.zeros(3, np.float32),
                "pret0shiftedoriginpt": np.zeros(4, np.float32),
                "purity": 0.0,
            })
        else:
            # Majority shower: plurality vote among shower trackids
            shower_mask = np.array([
                frag_matched[j] and int(frag_pids[j]) in SHOWER_PIDS
                for j in range(n_total)
            ])
            shower_tids = frag_tids[shower_mask]

            if len(shower_tids) == 0:
                # Edge case: majority-shower by count but no actual shower tids
                frag_info.update({
                    "trackid": -1,
                    "pid": -1,
                    "type": TYPE_GHOST,
                    "originpt": np.zeros(3, np.float32),
                    "pret0shiftedoriginpt": np.zeros(4, np.float32),
                    "purity": 0.0,
                })
            else:
                unique_tids, tid_counts = np.unique(
                    shower_tids, return_counts=True
                )
                best_idx = np.argmax(tid_counts)
                best_tid = int(unique_tids[best_idx])
                best_count = int(tid_counts[best_idx])

                # Get pid of the winning trackid
                winning_mask = (frag_tids == best_tid)
                best_pid = int(frag_pids[winning_mask][0])

                # Purity: fraction of fragment points matching winning trackid
                purity = best_count / n_total

                # Look up origin from truth shower_fragments
                origin_type, originpt, pret0 = lookup_origin(
                    best_tid, truth_sf
                )

                # Compute start point: fragment point closest to truth origin
                frag_pos = reco_pos[frag_idx]
                if np.any(originpt != 0):
                    dists = np.linalg.norm(
                        frag_pos - originpt.reshape(1, 3), axis=1
                    )
                    startpt = frag_pos[np.argmin(dists)].copy()
                else:
                    startpt = None  # keep reco PCA start

                frag_info.update({
                    "trackid": best_tid,
                    "pid": best_pid,
                    "type": origin_type,
                    "originpt": originpt,
                    "pret0shiftedoriginpt": pret0,
                    "startpt": startpt,
                    "purity": purity,
                })

        results.append(frag_info)

    # Apply nu_vertex_is_visible override logic
    if truth_sf is not None:
        _apply_nu_vertex_override(results, truth_sf)

    return results


def _apply_nu_vertex_override(frag_results, truth_sf):
    """
    Apply the nu_vertex_is_visible origin override.

    When the neutrino vertex is NOT visible and there is only one visible
    nu-origin (type=0) shower trackid, replace its origin with the trunk
    start point from the truth fragments. This matches the logic in
    shower_origin.py get_data().
    """
    nu_vtx_visible = truth_sf.get("nu_vertex_is_visible", 1)
    if nu_vtx_visible != 0:
        return

    # Count unique visible nu-origin trackids among shower-matched fragments
    nu_trackids = set()
    for r in frag_results:
        if r["type"] == TYPE_INSIDE:
            nu_trackids.add(r["trackid"])

    if len(nu_trackids) > 1:
        return  # multiple nu-origin showers, no override

    # Find trunk start point for the single nu-origin trackid
    for tid in nu_trackids:
        matches = np.where(truth_sf["trackid"] == tid)[0]
        trunk = [m for m in matches if truth_sf["istrunk"][m] == 1]
        if trunk:
            trunk_startpt = truth_sf["startpt"][trunk[0]].copy()
            # Override originpt for all fragments with this trackid
            for r in frag_results:
                if r["trackid"] == tid and r["type"] == TYPE_INSIDE:
                    r["originpt"] = trunk_startpt


# ── HDF5 Output ──────────────────────────────────────────────────────────────

def write_merged_h5(reco_f, truth_f, output_path, match_idx, per_point_truth,
                    frag_results, truth_sf):
    """
    Write the merged HDF5 file with same schema as reco but truth fields filled.
    """
    num_frags = len(frag_results)

    def _cd(group, name, data):
        # gzip+shuffle requires a chunked (non-scalar, non-empty) dataset
        arr = np.asarray(data)
        if arr.ndim == 0 or arr.size == 0:
            return group.create_dataset(name, data=arr)
        return group.create_dataset(
            name, data=arr,
            compression="gzip", compression_opts=4,
            shuffle=True, chunks=True,
        )

    def _copy_truth_group(src_group, dst_entry, name):
        dst = dst_entry.create_group(name)
        for ak, av in src_group.attrs.items():
            dst.attrs[ak] = av
        for key in src_group.keys():
            item = src_group[key]
            if isinstance(item, h5py.Dataset):
                _cd(dst, key, item[:])

    with h5py.File(output_path, "w") as out:
        entry = out.create_group("entry_0")

        # Propagate run/subrun/event attributes from reco H5 if present
        reco_entry = reco_f["entry_0"]
        for attr_name in ("run", "subrun", "event"):
            if attr_name in reco_entry.attrs:
                entry.attrs[attr_name] = int(reco_entry.attrs[attr_name])

        td = entry.create_group("triplet_data")

        # Copy all existing reco triplet_data datasets
        reco_td = reco_f["entry_0"]["triplet_data"]
        for key in reco_td.keys():
            _cd(td, key, reco_td[key][:])

        # Add truth per-point arrays
        _cd(td, "trackid", per_point_truth["trackid"])
        _cd(td, "pid", per_point_truth["pid"])
        _cd(td, "origin", per_point_truth["origin"])
        _cd(td, "truth_match", match_idx)

        # Remap per-point truth ssnet_label onto reco points (if present).
        truth_entry = truth_f["entry_0"]
        truth_td = truth_entry["triplet_data"]
        if "ssnet_label" in truth_td:
            truth_ssnet = truth_td["ssnet_label"][:]
            matched = match_idx >= 0
            reco_ssnet = np.full(len(match_idx), -1, dtype=truth_ssnet.dtype)
            reco_ssnet[matched] = truth_ssnet[match_idx[matched]]
            _cd(td, "ssnet_label", reco_ssnet)

        # Copy truth-only groups that are not per-spacepoint (mckeypoints,
        # mc_particle_tree). These are event-level so no remap is needed.
        for grp_name in ("mckeypoints", "mc_particle_tree"):
            if grp_name in truth_entry:
                _copy_truth_group(truth_entry[grp_name], entry, grp_name)

        # Write shower_fragments group
        sf = entry.create_group("shower_fragments")
        sf.attrs["num_fragments"] = num_frags

        if num_frags > 0:
            reco_sf = reco_f["entry_0"]["shower_fragments"]

            # Copy unchanged fields
            _cd(sf, "pointindices_flat", reco_sf["pointindices_flat"][:])
            _cd(sf, "pointindices_counts", reco_sf["pointindices_counts"][:])
            # Start points: use recomputed (closest to origin) when available,
            # otherwise keep the reco PCA start point.
            reco_startpts = reco_sf["startpt"][:].astype(np.float32)
            for fi, r in enumerate(frag_results):
                if r.get("startpt") is not None:
                    reco_startpts[fi] = r["startpt"]
            _cd(sf, "startpt", reco_startpts)
            _cd(sf, "istrunk", np.ones(num_frags, dtype=np.int64))

            # Write labeled fields from frag_results
            _cd(sf, "trackid",
                np.array([r["trackid"] for r in frag_results], dtype=np.int64))
            _cd(sf, "pid",
                np.array([r["pid"] for r in frag_results], dtype=np.int64))
            _cd(sf, "type",
                np.array([r["type"] for r in frag_results], dtype=np.int64))
            _cd(sf, "originpt",
                np.array([r["originpt"] for r in frag_results], dtype=np.float32))
            _cd(sf, "pret0shiftedoriginpt",
                np.array([r["pret0shiftedoriginpt"] for r in frag_results],
                         dtype=np.float32))

            # nu_vertex_is_visible from truth (scalar)
            nu_vis = 0
            if truth_sf is not None:
                nu_vis = truth_sf.get("nu_vertex_is_visible", 0)
            _cd(sf, "nu_vertex_is_visible", np.int64(nu_vis))


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(match_idx, frag_results, reco_path, output_path):
    """Compute and return summary statistics dict."""
    n_reco = len(match_idx)
    n_matched = int((match_idx >= 0).sum())
    n_unmatched = n_reco - n_matched

    type_counts = {name: 0 for name in TYPE_NAMES.values()}
    purities = []

    for r in frag_results:
        tname = TYPE_NAMES.get(r["type"], "unknown")
        type_counts[tname] = type_counts.get(tname, 0) + 1
        if r["type"] < TYPE_GHOST and r.get("purity", 0) > 0:
            purities.append(r["purity"])

    avg_purity = float(np.mean(purities)) if purities else 0.0

    stats = {
        "reco_file": os.path.basename(reco_path),
        "output_file": os.path.basename(output_path),
        "n_reco_points": n_reco,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "match_rate": n_matched / n_reco if n_reco > 0 else 0.0,
        "n_fragments": len(frag_results),
        "type_counts": type_counts,
        "avg_shower_purity": avg_purity,
        "per_fragment": [
            {
                "idx": i,
                "type": TYPE_NAMES.get(r["type"], "unknown"),
                "trackid": r["trackid"],
                "pid": r["pid"],
                "n_total": r["n_total"],
                "n_matched": r["n_matched"],
                "n_shower": r.get("n_shower", 0),
                "n_non_shower": r["n_non_shower"],
                "n_ambiguous": r.get("n_ambiguous", 0),
                "purity": r.get("purity", 0.0),
            }
            for i, r in enumerate(frag_results)
        ],
    }

    return stats


def print_stats(stats):
    """Print summary to stdout."""
    print(f"\nEvent: {stats['output_file']}")
    print(f"  Reco points: {stats['n_reco_points']}, "
          f"matched: {stats['n_matched']} "
          f"({stats['match_rate']:.2%}), "
          f"unmatched: {stats['n_unmatched']}")
    print(f"  Fragments: {stats['n_fragments']} total")
    for tname, count in stats["type_counts"].items():
        if count > 0:
            pct = count / stats["n_fragments"] * 100 if stats["n_fragments"] > 0 else 0
            print(f"    {tname:12s}: {count:4d}  ({pct:5.1f}%)")
    if stats["avg_shower_purity"] > 0:
        print(f"  Shower fragment purity (avg): {stats['avg_shower_purity']:.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.reco_h5):
        print(f"ERROR: reco file not found: {args.reco_h5}")
        sys.exit(1)
    if not os.path.exists(args.truth_h5):
        print(f"ERROR: truth file not found: {args.truth_h5}")
        sys.exit(1)

    print(f"Reco:   {args.reco_h5}")
    print(f"Truth:  {args.truth_h5}")
    print(f"Output: {args.output}")

    # Open both files
    with h5py.File(args.truth_h5, "r") as truth_f:
        # Build lookup table
        print("Building truth point lookup...")
        truth_lookup = build_truth_lookup(truth_f)
        print(f"  Truth points: {len(truth_lookup)}")

        # Load truth shower fragments for origin lookup
        truth_sf = load_truth_shower_fragments(truth_f)
        if truth_sf is not None:
            print(f"  Truth shower fragments: {len(truth_sf['trackid'])}")
        else:
            print("  No truth shower_fragments found (will label all as ghost)")

        with h5py.File(args.reco_h5, "r") as reco_f:
            # Match reco points to truth
            print("Matching reco points to truth...")
            match_idx = match_reco_to_truth(reco_f, truth_lookup)
            n_matched = int((match_idx >= 0).sum())
            n_reco = len(match_idx)
            print(f"  Matched: {n_matched}/{n_reco} "
                  f"({n_matched / n_reco:.2%})")

            if n_matched == 0:
                print("WARNING: No points matched! Check that reco and truth "
                      "files are from the same event.")

            # Get per-point truth labels
            per_point_truth = get_per_point_truth(truth_f, match_idx)

            # Label fragments
            print("Labeling fragments...")
            frag_results = label_fragments(
                reco_f, match_idx, per_point_truth, truth_sf
            )

            # Write output
            print(f"Writing merged file: {args.output}")
            write_merged_h5(
                reco_f, truth_f, args.output, match_idx, per_point_truth,
                frag_results, truth_sf,
            )

    # Stats
    stats = compute_stats(match_idx, frag_results, args.reco_h5, args.output)
    print_stats(stats)

    # Write stats JSON
    stats_path = args.stats_json
    if stats_path is None:
        stats_path = args.output + ".stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to: {stats_path}")


if __name__ == "__main__":
    main()
