"""
convert_dlmerged_to_larformer_h5.py

Single-stage conversion for the LArFormer cascade data path.

merged_dlreco.root  ->  per-event  merged_<TAG>_<fileno-tag>_entry<N>.h5

The LArFormer cascade does its own deghosting (the LoRA-finetuned Sonata
Stage-1 deghoster), so LArMatch is NOT run. The 3D spacepoints are the full
`SimChTripletLabelMaker` triplet proposals (ghost-included; `hasmatch` 0/1) —
the exact point set the deghoster/slicer/particle stages were trained on.

For each entry this script:

  1. Runs `larflow.prep.SimChTripletLabelMaker` (same configuration as the
     training Step-3 `process_dlmerged_to_hdf5_event_files.py`) and saves the
     C++ HDF5 record (`entry_0/triplet_data` + `mc_particle_tree` + ...).
  2. Re-opens the file with h5py and folds in everything the C++ record lacks
     but the inference path needs:
       - `entry_0/triplet_data/lm_score`  (dummy 1.0 — LArFormerDataset reads
         it unconditionally; the cascade config disables the lm_score
         pre-filter, so the value is behaviourally irrelevant)
       - `entry_0` attrs `run / subrun / event`
       - `entry_0/flashes/` (simpleFlashBeam + simpleFlashCosmic) and
         `entry_0/pmt_positions`  (detector-level flash info, no separate file)
       - any per-SP field LArFormerDataset reads unconditionally
         (`trackid`, `pid`) that a `--is-data` run might omit -> filled with -1

Modes (match the training conventions):
  - newer sim (bnb_nu_pi0filter_corsika):  --adc wiremc
  - older sim (mcc9_v29e nue overlay):     --adc wire -tb --mcc9
  - real data (bnb5e19):                   --adc wire -tb --is-data

Usage:
    python convert_dlmerged_to_larformer_h5.py \
        -i merged_dlreco.root -o ./out_h5/ \
        --tag bnb_nu_pi0filter_corsika --fileno-tag fileno00001 \
        --adc wiremc -n 2
"""

import os
import argparse
import numpy as np
import h5py


# ----------------------------------------------------------------------------
# Flash / PMT constants (mirror prepare_flashinfo_h5.py).
# ----------------------------------------------------------------------------
USEC_PER_TICK = 0.5
TRIGGER_TICK = 3200
N_PMTS = 32
PMT_CHANNEL_OFFSET_PER_PRODUCER = (0, 200)  # beam=[0,31], cosmic=[200,231]


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert merged_dlreco -> LArFormer per-event HDF5 "
                    "(SimChTripletLabelMaker triplets + folded-in flash).")
    p.add_argument("-i", "--input-dlmerged", required=True,
                   help="Input merged_dlreco/dlmerged ROOT file.")
    p.add_argument("-o", "--output-dir", required=True,
                   help="Output directory for per-event HDF5 files.")
    p.add_argument("--tag", default="larformer",
                   help="Dataset TAG used in the output filename.")
    p.add_argument("--fileno-tag", default="",
                   help="e.g. 'fileno00001' inserted into the output filename.")
    p.add_argument("-n", "--nentries", type=int, default=-1,
                   help="Max entries to process (-1 = all).")
    p.add_argument("-e", "--start-entry", type=int, default=0,
                   help="First entry index to process.")
    p.add_argument("--adc", default="wiremc",
                   help="ADC image producer name (wiremc for sim, wire for data).")
    p.add_argument("-tb", "--tick-backward", default=False, action="store_true",
                   help="larcv data is tick-backward; reverse on load.")
    p.add_argument("--mcc9", default=False, action="store_true",
                   help="Use MCC9 truth processing (older official sim).")
    p.add_argument("-d", "--is-data", default=False, action="store_true",
                   help="Real detector data: no MC truth processed.")
    p.add_argument("--no-flash", default=False, action="store_true",
                   help="Do not fold in flash info.")
    p.add_argument("--out-prefix", default="merged",
                   help="Output filename prefix (default: 'merged').")
    p.add_argument("-v", "--verbosity", type=int, default=2,
                   help="SimChTripletLabelMaker verbosity (0=debug..2=normal).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Flash extraction (detector-level; lifted from prepare_flashinfo_h5.py).
# ----------------------------------------------------------------------------
def build_channel_to_opdet_map():
    from larlite import larutil
    geom = larutil.Geometry.GetME()
    positions = np.zeros((N_PMTS, 3), dtype=np.float32)
    for od in range(N_PMTS):
        xyz = np.zeros(3, dtype=np.float64)
        geom.GetOpDetPosition(od, xyz)
        positions[od] = xyz.astype(np.float32)
    ch_to_od = {}
    for prod_id, offset in enumerate(PMT_CHANNEL_OFFSET_PER_PRODUCER):
        for k in range(N_PMTS):
            ch = offset + k
            ch_to_od[ch] = int(geom.OpDetFromOpChannel(ch))
    return ch_to_od, positions


def extract_flashes(ioll, channel_to_opdet):
    """simpleFlashBeam (producer_id 0) + simpleFlashCosmic (1). PE indexed by
    OpDet so pe[i] and pmt_positions[i] are the same physical PMT."""
    from larlite import larlite as ll
    flashes = []
    for prod_id, producer in enumerate(("simpleFlashBeam", "simpleFlashCosmic")):
        ev = ioll.get_data(ll.data.kOpFlash, producer)
        if ev is None:
            continue
        channel_offset = PMT_CHANNEL_OFFSET_PER_PRODUCER[prod_id]
        for i in range(ev.size()):
            f = ev[i]
            nch = int(f.nOpDets())
            pe = np.zeros(N_PMTS, dtype=np.float32)
            for k in range(N_PMTS):
                src_ch = channel_offset + k
                if src_ch >= nch:
                    continue
                od = channel_to_opdet.get(src_ch)
                if od is None or not (0 <= od < N_PMTS):
                    continue
                pe[od] = f.PE(src_ch)
            time_us = float(f.Time())
            flashes.append({
                "pe": pe, "total_pe": float(pe.sum()), "time_us": time_us,
                "tpc_tick": float(time_us / USEC_PER_TICK + TRIGGER_TICK),
                "producer_id": np.int32(prod_id), "flash_index": np.int32(i),
                "y_center": float(f.YCenter()) if hasattr(f, "YCenter") else 0.0,
                "z_center": float(f.ZCenter()) if hasattr(f, "ZCenter") else 0.0,
            })
    return flashes


def fold_into_h5(path, run, subrun, event, flashes, pmt_positions):
    """Re-open the C++-written per-event H5 and add the inference-path extras."""
    with h5py.File(path, "a") as f:
        entry = f["entry_0"]
        entry.attrs["run"] = int(run)
        entry.attrs["subrun"] = int(subrun)
        entry.attrs["event"] = int(event)

        td = entry["triplet_data"]
        n_sp = td["pos"].shape[0]
        # LArFormerDataset reads these unconditionally — guarantee presence.
        if "lm_score" not in td:
            td.create_dataset("lm_score", data=np.ones(n_sp, dtype=np.float32))
        if "trackid" not in td:
            td.create_dataset("trackid", data=np.full(n_sp, -1, dtype=np.int64))
        if "pid" not in td:
            td.create_dataset("pid", data=np.full(n_sp, -1, dtype=np.int64))

        if flashes is None:
            return
        entry.attrs["usec_per_tick"] = USEC_PER_TICK
        entry.attrs["trigger_tick"] = int(TRIGGER_TICK)
        entry.attrs["n_pmts"] = N_PMTS
        F = len(flashes)
        fl = entry.create_group("flashes")
        fl.attrs["num_flashes"] = F
        pe = (np.stack([x["pe"] for x in flashes]).astype(np.float32)
              if F else np.zeros((0, N_PMTS), np.float32))

        def col(key, dtype):
            return (np.array([x[key] for x in flashes], dtype=dtype)
                    if F else np.zeros(0, dtype))

        fl.create_dataset("pe", data=pe, compression="gzip", compression_opts=6)
        fl.create_dataset("total_pe", data=col("total_pe", np.float32))
        fl.create_dataset("time_us", data=col("time_us", np.float32))
        fl.create_dataset("tpc_tick", data=col("tpc_tick", np.float32))
        fl.create_dataset("producer_id", data=col("producer_id", np.int32))
        fl.create_dataset("flash_index", data=col("flash_index", np.int32))
        fl.create_dataset("y_center", data=col("y_center", np.float32))
        fl.create_dataset("z_center", data=col("z_center", np.float32))
        if pmt_positions is not None:
            entry.create_dataset("pmt_positions",
                                 data=pmt_positions.astype(np.float32))


def main():
    args = parse_args()

    import ROOT  # noqa: F401
    from larlite import larlite
    from larcv import larcv
    from larflow import larflow

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- SimChTripletLabelMaker (matches training Step-3 setup) ----
    simchmaker = larflow.prep.SimChTripletLabelMaker()
    simchmaker.set_verbosity(args.verbosity)
    simchmaker.save_truth_tripletinfo(True)
    simchmaker.set_adc_treename(args.adc)
    if args.is_data:
        print("Running in DATA mode (no MC truth).")
        simchmaker.set_is_data()
    simchmaker._mcpixelmaker.set_verbosity(args.verbosity)
    simchmaker._mcpixelmaker.set_driftwc_source()
    simchmaker._mcpixelmaker.set_dwire(1)
    simchmaker._mcpixelmaker.set_drow(0)
    if args.mcc9:
        print("RUNNING IN MCC9 MODE")
        simchmaker.process_mcc9_sim()
    simchmaker._shower_fragment_maker.set_verbosity(args.verbosity)

    # ---- IO ----
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(args.input_dlmerged)
    ioll.set_verbosity(2)
    ioll.open()

    tick_dir = (larcv.IOManager.kTickBackward if args.tick_backward
                else larcv.IOManager.kTickForward)
    iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv", tick_dir)
    iolcv.add_in_file(args.input_dlmerged)
    for prod in ("wire", "wiremc", "thrumu", "ancestor", "segment",
                 "instance", "larflow"):
        iolcv.specify_data_read("image2d", prod)
    iolcv.specify_data_read("chstatus", "wire")
    iolcv.specify_data_read("chstatus", "wiremc")
    if args.tick_backward:
        print("REVERSE TICK-ORDER OF LARCV DATA")
        iolcv.reverse_all_products()
    iolcv.set_verbosity(0)
    iolcv.initialize()

    # ---- Flash setup ----
    channel_to_opdet = None
    pmt_positions = None
    if not args.no_flash:
        channel_to_opdet, pmt_positions = build_channel_to_opdet_map()

    nentries = ioll.get_entries()
    start = max(0, args.start_entry)
    end = nentries if args.nentries < 0 else min(start + args.nentries, nentries)

    base = os.path.basename(args.input_dlmerged)
    if base.endswith(".root"):
        base = base[:-5]
    tag_part = f"{args.fileno_tag}_" if args.fileno_tag else ""

    print(f"Processing {args.input_dlmerged}")
    print(f"  Entries : {start}..{end - 1} ({end - start})")
    print(f"  Output  : {args.output_dir}")
    print(f"  adc={args.adc} tb={args.tick_backward} mcc9={args.mcc9} "
          f"is_data={args.is_data}  flash={'off' if args.no_flash else 'on'}")

    for ientry in range(start, end):
        ioll.go_to(ientry)
        iolcv.read_entry(ientry)

        outname = (f"{args.out_prefix}_{args.tag}_{tag_part}"
                   f"entry{ientry:06d}.h5")
        outpath = os.path.join(args.output_dir, outname)

        simchmaker.process(ioll, iolcv)
        simchmaker.open_hdf_file(outpath)
        simchmaker.save_entry("/entry_0")
        simchmaker.close_hdf_file()

        flashes = None
        if not args.no_flash:
            flashes = extract_flashes(ioll, channel_to_opdet)
        fold_into_h5(outpath, ioll.run_id(), ioll.subrun_id(), ioll.event_id(),
                     flashes, pmt_positions)

        nfl = 0 if flashes is None else len(flashes)
        print(f"  [{ientry}] -> {outname}  ({nfl} flashes)")

    simchmaker.close_hdf_file()
    ioll.close()
    print(f"\nDone. {end - start} entries processed.")


if __name__ == "__main__":
    main()
