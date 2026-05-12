"""
Convert larlite ROOT files (output of deploy_larmatchme.py / Step 1)
into per-event HDF5 files compatible with ShowerOriginDataset.

The larlite file contains larflow3dhit objects with:
  - [0:3]  : x, y, z coordinates (cm)
  - [9]    : larmatch triplet score
  - [10:17]: SSNet scores (7 values)
  - [17:23]: keypoint scores (6 values)
  - [23:26]: plane charge (3 values)
  - [26:29]: 3D flow direction
  - [29:77]: larmatch features (48 values)
  - hit.targetwire[0,1,2]: U, V, Y wire coordinates
  - hit.renormed_shower_score: combined shower SSNet score

This script:
  1. Reads the larlite file using larlite.storage_manager
  2. Optionally filters hits by larmatch score
  3. Identifies shower-like points using renormed_shower_score
  4. Runs DBSCAN clustering on shower points to form fragments
  5. Computes a start point for each fragment (most upstream point)
  6. Writes per-event HDF5 files in the ShowerOriginDataset schema

Usage:
    python convert_larlite_to_showerorigin_h5.py \
        --input-larlite output_larlite.root \
        --output-dir ./showerorigin_h5/ \
        [--input-larcv output_larcv.root] \
        [--min-score 0.5] \
        [--shower-threshold 0.5] \
        [--dbscan-eps 3.0] \
        [--dbscan-min-samples 4] \
        [--min-fragment-points 20]
"""

import os
import sys
import argparse
import numpy as np
import h5py

from sklearn.cluster import DBSCAN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert larlite larflow3dhit ROOT file to "
                    "ShowerOriginDataset-compatible HDF5 files."
    )
    parser.add_argument(
        "-i", "--input-larlite", required=True, type=str,
        help="Input larlite ROOT file (from deploy_larmatchme.py)."
    )
    parser.add_argument(
        "--input-larcv", type=str, default=None,
        help="Input larcv ROOT file (for wire-plane pixel values). "
             "If not provided, pixval will be filled with ones."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=str,
        help="Output directory for per-event HDF5 files."
    )
    parser.add_argument(
        "-n", "--nentries", type=int, default=-1,
        help="Max entries to process (-1 = all)."
    )
    parser.add_argument(
        "--start-entry", type=int, default=0,
        help="First entry to process."
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="If set, apply additional larmatch score filter "
             "(hits below this threshold are removed). "
             "Ghosts are already removed by deploy_larmatchme.py, "
             "but a stricter threshold can reduce noise."
    )
    parser.add_argument(
        "--shower-threshold", type=float, default=0.5,
        help="Threshold on renormed_shower_score to classify a hit "
             "as shower-like for DBSCAN clustering (default: 0.5)."
    )
    parser.add_argument(
        "--dbscan-eps", type=float, default=3.0,
        help="DBSCAN eps (neighborhood radius in cm). "
             "Default 3.0 matches the C++ ShowerFragmentOriginMaker."
    )
    parser.add_argument(
        "--dbscan-min-samples", type=int, default=4,
        help="DBSCAN min_samples. Default 4 matches C++ code."
    )
    parser.add_argument(
        "--min-fragment-points", type=int, default=20,
        help="Minimum points in a DBSCAN cluster to keep as a fragment "
             "(default: 20, matching training data loader)."
    )
    parser.add_argument(
        "--hit-producer", type=str, default="larmatch",
        help="Name of the larflow3dhit producer (default: 'larmatch')."
    )
    parser.add_argument(
        "--adc", type=str, default="wire",
        help="Name of the ADC wire image producer (default: 'wire')."
    )
    parser.add_argument(
        "-tb","--tick-backward", default=False, action='store_true',
        help="If flag provided, input file is expected to be in reverse-tick format and will be reversed upon loading."
    )
    parser.add_argument(
        "--fileno-tag", type=str, default="",
        help="Optional tag (e.g. 'fileno00001') inserted into output filenames "
             "between the 'showerorigin_' prefix and the input basename, to "
             "disambiguate intermediates produced from different input files."
    )
    parser.add_argument(
        "--max-hits", type=int, default=1_000_000,
        help="Skip processing of events with more than this many larmatch "
             "hits (default: 1_000_000). Oversized events are written as "
             "empty placeholder H5 files with entry_0.attrs.oversized=1 so "
             "the resume logic still sees a complete entry but the "
             "validation step can drop them. Set <= 0 to disable."
    )
    return parser.parse_args()


def extract_hits(io, ientry, producer, min_score=None):
    """
    Extract hit data from a larlite event.

    Returns dict with:
        pos:       (N, 3) float32 — x, y, z in cm
        tick:      (N,)   int32   — image row (tick) per hit
        uwire:     (N,)   float32 — U wire index
        vwire:     (N,)   float32 — V wire index
        ywire:     (N,)   float32 — Y wire index
        shower_score: (N,) float32 — renormed shower score
        lm_score:  (N,)   float32 — larmatch score (index [9])
        pixval:    (N, 3) float32 — placeholder (ones), filled later
        larmatch_feats: (N, 48) float32 — larmatch features (indices [39:77])
    """
    from larlite import larlite as ll

    io.go_to(ientry)
    event_hits = io.get_data(ll.data.kLArFlow3DHit, producer)
    nhits = event_hits.size()

    if nhits == 0:
        return None

    pos = np.zeros((nhits, 3), dtype=np.float32)
    tick = np.zeros(nhits, dtype=np.int32)
    uwire = np.zeros(nhits, dtype=np.float32)
    vwire = np.zeros(nhits, dtype=np.float32)
    ywire = np.zeros(nhits, dtype=np.float32)
    shower_score = np.zeros(nhits, dtype=np.float32)
    lm_score = np.zeros(nhits, dtype=np.float32)
    larmatch_feats = np.zeros((nhits, 48), dtype=np.float32)
    has_larmatch_feats = False

    for ihit in range(nhits):
        hit = event_hits.at(ihit)
        pos[ihit, 0] = hit[0]
        pos[ihit, 1] = hit[1]
        pos[ihit, 2] = hit[2]
        tick[ihit] = hit.tick
        lm_score[ihit] = hit[9]
        shower_score[ihit] = hit.renormed_shower_score
        uwire[ihit] = hit.targetwire[0]
        vwire[ihit] = hit.targetwire[1]
        ywire[ihit] = hit.targetwire[2]
        if hit.size() > 29:
            nmax = min(hit.size(), 78)
            for ii in range(29, nmax):
                larmatch_feats[ihit, ii-29] = hit[ii]
            has_larmatch_feats = True

    # if has_larmatch_feats:
    #     #For debug
    #     print("hit has larmatch feat data!")
    #     print(larmatch_feats[:3,:])
    #     print(" num nonzero: ", np.sum(larmatch_feats!=0.0))

    # Optional stricter score filtering
    if min_score is not None:
        mask = lm_score >= min_score
        pos = pos[mask]
        tick = tick[mask]
        uwire = uwire[mask]
        vwire = vwire[mask]
        ywire = ywire[mask]
        shower_score = shower_score[mask]
        lm_score = lm_score[mask]

    # Placeholder pixval — replaced by load_pixval_from_larcv if larcv provided
    pixval = np.ones((pos.shape[0], 3), dtype=np.float32)

    return {
        "pos": pos,
        "tick": tick,
        "uwire": uwire,
        "vwire": vwire,
        "ywire": ywire,
        "shower_score": shower_score,
        "lm_score": lm_score,
        "pixval": pixval,
        "larmatch_feats": larmatch_feats,
        "has_larmatch_feats": has_larmatch_feats,
    }


class LArCVPixelReader:
    """
    Manages a larcv IOManager for reading wire-plane pixel values.

    Opened once and reused across entries to avoid repeated open/close.
    Call close() when done processing all entries.
    """

    def __init__(self, larcv_file, wire_producer="wire", is_tick_backward=False):
        from larcv import larcv
        self.larcv = larcv
        self.wire_producer = wire_producer
        self.tick_direction=larcv.IOManager.kTickForward
        if is_tick_backward:
            self.tick_direction=larcv.IOManager.kTickBackward

        self.iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv_pixval", self.tick_direction)
        self.iolcv.add_in_file(larcv_file)
        # Only read what we need
        self.iolcv.specify_data_read("image2d", wire_producer)
        if is_tick_backward:
            self.iolcv.reverse_all_products()

        self.iolcv.set_verbosity(2)
        self.iolcv.initialize()

        self._current_entry = -1
        self._img_v = None

    def _load_entry(self, ientry):
        """Load images for a specific entry (cached)."""
        if ientry == self._current_entry:
            return
        self.iolcv.read_entry(ientry)
        ev_img = self.iolcv.get_data("image2d", self.wire_producer)
        self._img_v = ev_img.as_vector()
        self._current_entry = ientry

    def get_pixval(self, ientry, tick, uwire, vwire, ywire):
        """
        Sample pixel values from wire-plane images for each hit.

        Each hit has a tick (image row) and wire index per plane
        (image column). We sample the ADC value at (tick, wire)
        from each plane's Image2D.

        Args:
            ientry: event index
            tick:  (N,) int — image row (tick) per hit
            uwire: (N,) int/float — U plane wire index per hit
            vwire: (N,) int/float — V plane wire index per hit
            ywire: (N,) int/float — Y plane wire index per hit

        Returns:
            pixval: (N, 3) float32 — ADC values [U, V, Y] per hit
        """
        self._load_entry(ientry)

        n_hits = len(tick)
        pixval = np.zeros((n_hits, 3), dtype=np.float32)

        if self._img_v is None or self._img_v.size() < 3:
            print(f"  Warning: no wire images for entry {ientry}, "
                  f"returning zeros.")
            return pixval

        # Wire arrays per plane: [U, V, Y] = planes [0, 1, 2]
        wire_arrays = [
            np.asarray(uwire, dtype=np.int32),
            np.asarray(vwire, dtype=np.int32),
            np.asarray(ywire, dtype=np.int32),
        ]
        tick_arr = np.asarray(tick, dtype=np.int32)

        for plane_idx in range(3):
            img = self._img_v.at(plane_idx)
            meta = img.meta()

            # Image bounds
            min_tick = int(meta.min_y())
            max_tick = int(meta.max_y())
            min_wire = int(meta.min_x())
            max_wire = int(meta.max_x())
            n_rows = int(meta.rows())
            n_cols = int(meta.cols())

            wire_col = wire_arrays[plane_idx]

            for ihit in range(n_hits):
                t = tick_arr[ihit]
                w = wire_col[ihit]

                # Bounds check
                if t < min_tick or t >= max_tick:
                    continue
                if w < min_wire or w >= max_wire:
                    continue

                # Convert physical coordinates to pixel indices
                row = meta.row(t)
                col = meta.col(w)

                if row < n_rows and col < n_cols:
                    pixval[ihit, plane_idx] = img.pixel(row, col)

        return pixval

    def close(self):
        """Finalize the larcv IOManager."""
        self.iolcv.finalize()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def cluster_shower_fragments(pos, shower_score, shower_threshold,
                             dbscan_eps, dbscan_min_samples,
                             min_fragment_points):
    """
    Identify shower-like hits and cluster them with DBSCAN.

    Args:
        pos: (N, 3) all hit positions
        shower_score: (N,) renormed_shower_score per hit
        shower_threshold: threshold on shower_score
        dbscan_eps: DBSCAN eps parameter (cm)
        dbscan_min_samples: DBSCAN min_samples
        min_fragment_points: minimum cluster size to keep

    Returns:
        fragments: list of dicts, each with:
            - 'point_indices': array of indices into the full pos array
            - 'startpt': (3,) most upstream point (simple: PCA-based)
    """
    # Select shower-like hits
    shower_mask = shower_score >= shower_threshold
    shower_indices = np.where(shower_mask)[0]

    if len(shower_indices) < dbscan_min_samples:
        return []

    shower_pos = pos[shower_indices]

    # Run DBSCAN
    clustering = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        metric='euclidean',
        n_jobs=-1,
    ).fit(shower_pos)

    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    fragments = []
    for label in sorted(unique_labels):
        cluster_mask = labels == label
        cluster_indices_in_shower = np.where(cluster_mask)[0]

        if len(cluster_indices_in_shower) < min_fragment_points:
            continue

        # Map back to full-event indices
        full_indices = shower_indices[cluster_indices_in_shower]
        cluster_pos = pos[full_indices]

        # Compute start point using PCA direction
        startpt = compute_start_point(cluster_pos)

        fragments.append({
            "point_indices": full_indices,
            "startpt": startpt,
        })

    return fragments


def compute_start_point(cluster_pos):
    """
    Compute the most upstream point of a cluster using PCA.

    The start point is the point with the smallest projection
    along the principal axis direction (most upstream).

    Args:
        cluster_pos: (M, 3) positions of points in the cluster

    Returns:
        startpt: (3,) the most upstream point
    """
    centroid = cluster_pos.mean(axis=0)
    centered = cluster_pos - centroid

    # PCA via SVD
    if len(cluster_pos) < 3:
        return cluster_pos[0]

    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = Vt[0]  # first principal component
    except np.linalg.LinAlgError:
        return cluster_pos[0]

    # Project onto principal axis
    projections = centered @ principal_axis

    # Most upstream = smallest projection
    start_idx = np.argmin(projections)
    return cluster_pos[start_idx].copy()


def write_event_h5(output_path, hit_data, fragments, run=-1, subrun=-1, event=-1):
    """
    Write a single event to an HDF5 file in ShowerOriginDataset format.

    Args:
        output_path: path to the output .h5 file
        hit_data: dict from extract_hits()
        fragments: list of fragment dicts from cluster_shower_fragments()
    """
    num_frags = len(fragments)
    n_points = hit_data["pos"].shape[0]

    with h5py.File(output_path, 'w') as f:
        entry = f.create_group("entry_0")

        # Store run/subrun/event for downstream matching
        entry.attrs["run"] = int(run)
        entry.attrs["subrun"] = int(subrun)
        entry.attrs["event"] = int(event)

        # --- triplet_data group ---
        triplet = entry.create_group("triplet_data")
        triplet.create_dataset("pos", data=hit_data["pos"])
        triplet.create_dataset("pixval", data=hit_data["pixval"])
        triplet.create_dataset("uwire", data=hit_data["uwire"])
        triplet.create_dataset("vwire", data=hit_data["vwire"])
        triplet.create_dataset("ywire", data=hit_data["ywire"])
        triplet.create_dataset("tick", data=hit_data["tick"])
        # All hits are kept (no ghost filtering needed at load time)
        triplet.create_dataset(
            "hasmatch",
            data=np.ones(n_points, dtype=np.int64),
        )

        # Store the shower score for downstream analysis
        triplet.create_dataset(
            "shower_score",
            data=hit_data["shower_score"],
        )
        # Store the larmatch score
        triplet.create_dataset(
            "lm_score",
            data=hit_data["lm_score"],
        )
        # Store the larmatch features
        if hit_data["has_larmatch_feats"]:
            # hits have larmatch feats
            triplet.create_dataset(
                "larmatch_feats",
                data=hit_data["larmatch_feats"],
            )
        else:
            # hits do not have larmatch feats
            triplet.create_dataset(
                "larmatch_feats",
                data=np.array([], dtype=np.float32),
            )
        
        # --- shower_fragments group ---
        sf = entry.create_group("shower_fragments")
        sf.attrs["num_fragments"] = num_frags

        if num_frags > 0:
            # Build flat index arrays
            all_indices = []
            index_counts = np.zeros(num_frags, dtype=np.int64)
            startpts = np.zeros((num_frags, 3), dtype=np.float32)

            for i, frag in enumerate(fragments):
                indices = frag["point_indices"]
                all_indices.append(indices)
                index_counts[i] = len(indices)
                startpts[i] = frag["startpt"]

            flat_indices = np.concatenate(all_indices).astype(np.int64)

            sf.create_dataset("pointindices_flat", data=flat_indices)
            sf.create_dataset("pointindices_counts", data=index_counts)
            sf.create_dataset("startpt", data=startpts)

            # Dummy fields for reco data (no MC truth available)
            # trackid: sequential IDs
            sf.create_dataset(
                "trackid",
                data=np.arange(num_frags, dtype=np.int64),
            )
            # pid: mark all as photon (22) — generic shower
            sf.create_dataset(
                "pid",
                data=np.full(num_frags, 22, dtype=np.int64),
            )
            # istrunk: all fragments are independent (1=trunk)
            sf.create_dataset(
                "istrunk",
                data=np.ones(num_frags, dtype=np.int64),
            )
            # type: -1 = unknown (model will predict this)
            sf.create_dataset(
                "type",
                data=np.full(num_frags, -1, dtype=np.int64),
            )
            # originpt: placeholder zeros (model will predict this)
            sf.create_dataset(
                "originpt",
                data=np.zeros((num_frags, 3), dtype=np.float32),
            )
            # pret0shiftedoriginpt: placeholder (no MC truth)
            sf.create_dataset(
                "pret0shiftedoriginpt",
                data=np.zeros((num_frags, 4), dtype=np.float32),
            )
            # nu_vertex_is_visible: unknown for reco data
            sf.create_dataset(
                "nu_vertex_is_visible",
                data=np.int64(0),
            )
        else:
            # Empty fragment arrays
            sf.create_dataset(
                "pointindices_flat", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "pointindices_counts", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "startpt", data=np.zeros((0, 3), dtype=np.float32)
            )
            sf.create_dataset(
                "trackid", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "pid", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "istrunk", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "type", data=np.array([], dtype=np.int64)
            )
            sf.create_dataset(
                "originpt", data=np.zeros((0, 3), dtype=np.float32)
            )
            sf.create_dataset(
                "pret0shiftedoriginpt",
                data=np.zeros((0, 4), dtype=np.float32),
            )
            sf.create_dataset(
                "nu_vertex_is_visible", data=np.int64(0),
            )


def write_oversized_placeholder(output_path, n_hits, max_hits,
                                run=-1, subrun=-1, event=-1):
    """
    Write a same-schema placeholder H5 for an event whose larmatch-hit count
    exceeds the `--max-hits` cap.

    Downstream consumers that just look at array shapes see an empty event
    (zero spacepoints, zero fragments). The `entry_0.attrs.oversized=1`
    marker (plus n_hits_seen / max_hits_cap) tells the merger and the
    validation script that this is an intentional skip, not a corruption.
    """
    with h5py.File(output_path, "w") as f:
        entry = f.create_group("entry_0")
        entry.attrs["run"] = int(run)
        entry.attrs["subrun"] = int(subrun)
        entry.attrs["event"] = int(event)
        entry.attrs["oversized"] = 1
        entry.attrs["n_hits_seen"] = int(n_hits)
        entry.attrs["max_hits_cap"] = int(max_hits)

        td = entry.create_group("triplet_data")
        td.create_dataset("pos", data=np.zeros((0, 3), dtype=np.float32))
        td.create_dataset("pixval", data=np.zeros((0, 3), dtype=np.float32))
        td.create_dataset("uwire", data=np.zeros(0, dtype=np.float32))
        td.create_dataset("vwire", data=np.zeros(0, dtype=np.float32))
        td.create_dataset("ywire", data=np.zeros(0, dtype=np.float32))
        td.create_dataset("tick", data=np.zeros(0, dtype=np.int32))
        td.create_dataset("hasmatch", data=np.zeros(0, dtype=np.int64))
        td.create_dataset("shower_score", data=np.zeros(0, dtype=np.float32))
        td.create_dataset("lm_score", data=np.zeros(0, dtype=np.float32))
        td.create_dataset(
            "larmatch_feats", data=np.zeros((0, 48), dtype=np.float32)
        )

        sf = entry.create_group("shower_fragments")
        sf.attrs["num_fragments"] = 0
        sf.create_dataset(
            "pointindices_flat", data=np.array([], dtype=np.int64)
        )
        sf.create_dataset(
            "pointindices_counts", data=np.array([], dtype=np.int64)
        )
        sf.create_dataset("startpt", data=np.zeros((0, 3), dtype=np.float32))
        sf.create_dataset("trackid", data=np.array([], dtype=np.int64))
        sf.create_dataset("pid", data=np.array([], dtype=np.int64))
        sf.create_dataset("istrunk", data=np.array([], dtype=np.int64))
        sf.create_dataset("type", data=np.array([], dtype=np.int64))
        sf.create_dataset(
            "originpt", data=np.zeros((0, 3), dtype=np.float32)
        )
        sf.create_dataset(
            "pret0shiftedoriginpt", data=np.zeros((0, 4), dtype=np.float32)
        )
        sf.create_dataset("nu_vertex_is_visible", data=np.int64(0))


def main():
    args = parse_args()

    # Import ROOT/larlite here so argparse --help works without ROOT
    import ROOT as rt
    from larlite import larlite

    os.makedirs(args.output_dir, exist_ok=True)

    # Open larcv reader if provided
    larcv_reader = None
    if args.input_larcv is not None:
        larcv_reader = LArCVPixelReader(args.input_larcv, wire_producer=args.adc, is_tick_backward=args.tick_backward)

    # Open larlite file
    io = larlite.storage_manager(larlite.storage_manager.kREAD)
    io.add_in_filename(args.input_larlite)
    io.open()

    nentries = io.get_entries()
    start = args.start_entry
    end = nentries if args.nentries < 0 else min(start + args.nentries, nentries)

    # Derive base filename from input
    base = os.path.basename(args.input_larlite)
    if base.endswith(".root"):
        base = base[:-5]
    # Remove _larlite suffix if present
    if base.endswith("_larlite"):
        base = base[:-8]

    print(f"Processing {args.input_larlite}")
    print(f"  Entries: {start} to {end - 1} ({end - start} total)")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Shower threshold: {args.shower_threshold}")
    print(f"  DBSCAN eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}")
    print(f"  Min fragment points: {args.min_fragment_points}")
    if args.min_score is not None:
        print(f"  Additional larmatch score filter: >= {args.min_score}")

    total_fragments = 0
    events_with_fragments = 0
    events_oversized = 0
    max_hits_cap = args.max_hits if args.max_hits and args.max_hits > 0 else None

    for ientry in range(start, end):
        # Peek at the hit count before allocating any per-hit arrays. The
        # giant events (>~1M hits) blow out memory inside extract_hits if
        # we let them allocate the parallel float arrays (especially the
        # 48-dim larmatch_feats) — but the larflow3dhit vector itself is
        # cheap to size-check.
        if max_hits_cap is not None:
            io.go_to(ientry)
            event_hits = io.get_data(larlite.data.kLArFlow3DHit,
                                     args.hit_producer)
            nhits_pre = event_hits.size()
            if nhits_pre > max_hits_cap:
                rse_run = io.run_id()
                rse_subrun = io.subrun_id()
                rse_event = io.event_id()
                tag_part = f"{args.fileno_tag}_" if args.fileno_tag else ""
                outname = (
                    f"showerorigin_{tag_part}{base}_entry{ientry:06d}.h5"
                )
                outpath = os.path.join(args.output_dir, outname)
                write_oversized_placeholder(
                    outpath, nhits_pre, max_hits_cap,
                    run=rse_run, subrun=rse_subrun, event=rse_event,
                )
                events_oversized += 1
                print(
                    f"  [{ientry}] OVERSIZED: {nhits_pre} hits > "
                    f"{max_hits_cap} cap, wrote empty placeholder "
                    f"-> {outname}"
                )
                continue

        hit_data = extract_hits(
            io, ientry, args.hit_producer, min_score=args.min_score
        )

        if hit_data is None or hit_data["pos"].shape[0] == 0:
            print(f"  [{ientry}] No hits, skipping.")
            continue

        # Extract run/subrun/event from larlite storage_manager
        rse_run = io.run_id()
        rse_subrun = io.subrun_id()
        rse_event = io.event_id()

        nhits = hit_data["pos"].shape[0]

        # Load pixval from larcv if provided
        if larcv_reader is not None:
            hit_data["pixval"] = larcv_reader.get_pixval(
                ientry,
                hit_data["tick"],
                hit_data["uwire"],
                hit_data["vwire"],
                hit_data["ywire"],
            )

        # Cluster shower fragments
        fragments = cluster_shower_fragments(
            hit_data["pos"],
            hit_data["shower_score"],
            args.shower_threshold,
            args.dbscan_eps,
            args.dbscan_min_samples,
            args.min_fragment_points,
        )

        nfrags = len(fragments)
        total_fragments += nfrags
        if nfrags > 0:
            events_with_fragments += 1

        # Write HDF5
        tag_part = f"{args.fileno_tag}_" if args.fileno_tag else ""
        outname = f"showerorigin_{tag_part}{base}_entry{ientry:06d}.h5"
        outpath = os.path.join(args.output_dir, outname)
        write_event_h5(outpath, hit_data, fragments,
                       run=rse_run, subrun=rse_subrun, event=rse_event)

        frag_sizes = [len(f["point_indices"]) for f in fragments]
        n_shower = (hit_data["shower_score"] >= args.shower_threshold).sum()
        print(
            f"  [{ientry}] {nhits} hits, {n_shower} shower hits, "
            f"{nfrags} fragments {frag_sizes} -> {outname}"
        )

    io.close()
    if larcv_reader is not None:
        larcv_reader.close()

    print(f"\nDone. {end - start} events processed.")
    print(f"  {events_with_fragments} events with fragments")
    print(f"  {total_fragments} total fragments")
    if max_hits_cap is not None:
        print(f"  {events_oversized} oversized events (>{max_hits_cap} "
              f"hits, written as placeholders)")


if __name__ == "__main__":
    main()
