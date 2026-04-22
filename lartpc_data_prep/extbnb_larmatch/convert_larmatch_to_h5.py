"""
Convert larmatch larlite ROOT output + dlmerged ROOT file to HDF5 event files
with per-point larmatch ghost scores.

Produces HDF5 files compatible with Pointcept's LArTPCDataset, with the addition
of a `larmatch_score` field (float32, [0,1]) for each spacepoint.

This script must run inside the pointcept container (needs larlite, larcv, h5py).

Usage:
    python3 convert_larmatch_to_h5.py \
        -i larmatchme_larlite.root \
        --input-larcv merged_dlreco_with_ssnet.root \
        -o /output/dir/ \
        --adc wire -tb
"""

import os
import sys
import argparse
import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert larmatch output to HDF5 with ghost scores."
    )
    parser.add_argument(
        "-i", "--input-larlite", required=True, type=str,
        help="Input larlite ROOT file from deploy_larmatchme.py."
    )
    parser.add_argument(
        "--input-larcv", required=True, type=str,
        help="Input merged_dlreco ROOT file (for ADC pixel values)."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=str,
        help="Output directory for HDF5 files."
    )
    parser.add_argument(
        "--adc", type=str, default="wire",
        help="Name of the ADC wire image producer (default: 'wire')."
    )
    parser.add_argument(
        "-tb", "--tick-backward", default=False, action='store_true',
        help="Reverse tick order when loading larcv data."
    )
    parser.add_argument(
        "--hit-producer", type=str, default="larmatch",
        help="Name of the larflow3dhit producer (default: 'larmatch')."
    )
    parser.add_argument(
        "-n", "--nentries", type=int, default=-1,
        help="Number of entries to process (-1 = all)."
    )
    parser.add_argument(
        "--file-id", type=str, default=None,
        help="Unique identifier for this input file (e.g. line number in input list). "
             "Embedded in output filenames to prevent collisions across files."
    )
    parser.add_argument(
        "--original-filename", type=str, default=None,
        help="Original ROOT filename (before renaming in Step 1). "
             "Used to extract UUID for output naming."
    )
    return parser.parse_args()


def extract_hits(io, ientry, producer):
    """
    Extract hit data from a larlite event.

    Returns dict with pos, tick, uwire, vwire, ywire, lm_score, pixval
    or None if no hits.
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
    lm_score = np.zeros(nhits, dtype=np.float32)

    for ihit in range(nhits):
        hit = event_hits.at(ihit)
        pos[ihit, 0] = hit[0]
        pos[ihit, 1] = hit[1]
        pos[ihit, 2] = hit[2]
        tick[ihit] = hit.tick
        lm_score[ihit] = hit[9]
        uwire[ihit] = hit.targetwire[0]
        vwire[ihit] = hit.targetwire[1]
        ywire[ihit] = hit.targetwire[2]

    # Placeholder pixval -- filled by LArCVPixelReader
    pixval = np.ones((nhits, 3), dtype=np.float32)

    return {
        "pos": pos,
        "tick": tick,
        "uwire": uwire,
        "vwire": vwire,
        "ywire": ywire,
        "lm_score": lm_score,
        "pixval": pixval,
    }


class LArCVPixelReader:
    """
    Reads wire-plane ADC pixel values from a larcv file.
    Opened once and reused across entries.
    """

    def __init__(self, larcv_file, wire_producer="wire", is_tick_backward=False):
        from larcv import larcv
        self.larcv = larcv
        self.wire_producer = wire_producer
        self.tick_direction = larcv.IOManager.kTickForward
        if is_tick_backward:
            self.tick_direction = larcv.IOManager.kTickBackward

        self.iolcv = larcv.IOManager(
            larcv.IOManager.kREAD, "larcv_pixval", self.tick_direction
        )
        self.iolcv.add_in_file(larcv_file)
        self.iolcv.specify_data_read("image2d", wire_producer)
        if is_tick_backward:
            self.iolcv.reverse_all_products()
        self.iolcv.set_verbosity(2)
        self.iolcv.initialize()

        self._current_entry = -1
        self._img_v = None

    def _load_entry(self, ientry):
        if ientry == self._current_entry:
            return
        self.iolcv.read_entry(ientry)
        ev_img = self.iolcv.get_data("image2d", self.wire_producer)
        self._img_v = ev_img.as_vector()
        self._current_entry = ientry

    def get_pixval(self, ientry, tick, uwire, vwire, ywire):
        """Sample ADC values at each hit's (tick, wire) from each plane."""
        self._load_entry(ientry)

        n_hits = len(tick)
        pixval = np.zeros((n_hits, 3), dtype=np.float32)

        if self._img_v is None or self._img_v.size() < 3:
            print(f"  Warning: no wire images for entry {ientry}, returning zeros.")
            return pixval

        wire_arrays = [
            np.asarray(uwire, dtype=np.int32),
            np.asarray(vwire, dtype=np.int32),
            np.asarray(ywire, dtype=np.int32),
        ]
        tick_arr = np.asarray(tick, dtype=np.int32)

        for plane_idx in range(3):
            img = self._img_v.at(plane_idx)
            meta = img.meta()

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

                if t < min_tick or t >= max_tick:
                    continue
                if w < min_wire or w >= max_wire:
                    continue

                row = meta.row(t)
                col = meta.col(w)

                if row < n_rows and col < n_cols:
                    pixval[ihit, plane_idx] = img.pixel(row, col)

        return pixval

    def close(self):
        self.iolcv.finalize()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def write_event_h5(output_path, pos, pixval, tick, uwire, vwire, ywire, lm_score):
    """Write a single event to HDF5 in Pointcept-compatible format."""
    # Use gzip compression + chunking to match original extbnb file format (~3-4x smaller)
    chunk_1d = (min(10000, len(pos)),)
    chunk_2d = (min(10000, len(pos)), 3)
    compress = dict(compression="gzip", shuffle=True)

    with h5py.File(output_path, "w") as f:
        grp = f.create_group("entry_0/triplet_data")
        grp.create_dataset("pos", data=pos, dtype="float32", chunks=chunk_2d, **compress)
        grp.create_dataset("pixval", data=pixval, dtype="float32", chunks=chunk_2d, **compress)
        grp.create_dataset("edep", data=pixval, dtype="float32", chunks=chunk_2d, **compress)
        grp.create_dataset("tick", data=tick.astype("int32"), dtype="int32", chunks=chunk_1d, **compress)
        grp.create_dataset("uwire", data=uwire.astype("int32"), dtype="int32", chunks=chunk_1d, **compress)
        grp.create_dataset("vwire", data=vwire.astype("int32"), dtype="int32", chunks=chunk_1d, **compress)
        grp.create_dataset("ywire", data=ywire.astype("int32"), dtype="int32", chunks=chunk_1d, **compress)
        grp.create_dataset("larmatch_score", data=lm_score, dtype="float32", chunks=chunk_1d, **compress)


def main():
    args = parse_args()

    # Import ROOT-based libraries (only available inside container)
    from larlite import larlite as ll

    os.makedirs(args.output_dir, exist_ok=True)

    # Build output filename stem with unique identifier
    # Priority: use original filename (has UUID), fall back to file-id
    if args.original_filename:
        orig = os.path.basename(args.original_filename)
        if orig.endswith(".root"):
            orig = orig[:-5]
        stem = orig  # e.g. "merged_dlreco_00deb37b-c348-4dd4-b886-2b2aa166b24f"
    elif args.file_id:
        stem = f"merged_dlreco_fileid{args.file_id}"
    else:
        # Last resort: use larcv filename (may collide if renamed)
        input_basename = os.path.basename(args.input_larcv)
        if input_basename.endswith(".root"):
            input_basename = input_basename[:-5]
        stem = input_basename.replace("merged_dlreco_with_ssnet", "merged_dlreco")

    # Open larlite input
    ioll = ll.storage_manager(ll.storage_manager.kREAD)
    ioll.add_in_filename(args.input_larlite)
    ioll.set_verbosity(2)
    ioll.open()

    n_entries = ioll.get_entries()
    if args.nentries > 0:
        n_entries = min(n_entries, args.nentries)

    print(f"Processing {n_entries} entries from {args.input_larlite}")
    print(f"LArCV file: {args.input_larcv}")
    print(f"Output dir: {args.output_dir}")

    # Open larcv reader for pixel values
    pix_reader = LArCVPixelReader(
        args.input_larcv,
        wire_producer=args.adc,
        is_tick_backward=args.tick_backward,
    )

    n_written = 0
    n_skipped = 0

    for ientry in range(n_entries):
        hit_data = extract_hits(ioll, ientry, args.hit_producer)

        if hit_data is None or len(hit_data["pos"]) == 0:
            print(f"  Entry {ientry}: no hits, skipping")
            n_skipped += 1
            continue

        # Fill in real pixel values from wire images
        pixval = pix_reader.get_pixval(
            ientry,
            hit_data["tick"],
            hit_data["uwire"],
            hit_data["vwire"],
            hit_data["ywire"],
        )

        # Output filename
        out_name = f"pointceptdata_{stem}_entry{ientry:06d}.h5"
        out_path = os.path.join(args.output_dir, out_name)

        write_event_h5(
            out_path,
            pos=hit_data["pos"],
            pixval=pixval,
            tick=hit_data["tick"],
            uwire=hit_data["uwire"],
            vwire=hit_data["vwire"],
            ywire=hit_data["ywire"],
            lm_score=hit_data["lm_score"],
        )

        n_hits = len(hit_data["pos"])
        mean_score = hit_data["lm_score"].mean()
        print(f"  Entry {ientry}: {n_hits} hits, mean_lm_score={mean_score:.3f} -> {out_name}")
        n_written += 1

    pix_reader.close()
    ioll.close()

    print(f"\nDone: {n_written} events written, {n_skipped} skipped")


if __name__ == "__main__":
    main()
