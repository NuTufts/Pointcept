"""prepare_flashinfo_h5.py — build the auxiliary flash-info H5 paired with a
merged training H5.

For one dlmerged ROOT file and one corresponding merged H5 (for a single
entry index), produce ``flashinfo_<basename>_entry<NNNN>.h5`` containing:

    entry_0/
        flashes/                          per-flash arrays from simpleFlashBeam
                                          + simpleFlashCosmic
        pmt_positions                     (32, 3) MicroBooNE PMT positions
        mc_particle_start_times/          parallel to mc_particle_tree/trackid
        slice_flash_matches/              one row per slice, greedy nearest-flash
                                          match within dtick_threshold (3 ticks
                                          default)

This is the **single-entry interactive script** intended for one ROOT entry +
one merged H5 file. A batch driver that iterates over a dlmerged file's
entries is a follow-up.

Usage:
    python prepare_flashinfo_h5.py \\
        --input-dlmerged   /path/to/dlmerged_X.root \\
        --entry            N \\
        --merged-h5        /path/to/merged_<basename>_entry<NNNN>.h5 \\
        --output-h5        /path/to/flashinfo_<basename>_entry<NNNN>.h5 \\
        [--dtick-threshold 3.0]
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_labels import compute_slice_labels  # noqa: E402

from larlite import larlite  # noqa: E402  (only available inside the container)


# ----------------------------------------------------------------------------
# Conversion constants. These mirror CrossingPointsAnaMethods::getTrueTick(...)
# in the form used by FlashMatcherV2.cxx (trig_time = 4050.0, the default).
# The 4050 ns is a calibration knob — passing trig_time=4050.0 zeros out the
# subtractive offset, so the effective formula is simply:
#     true_tick = TRIGGER_TICK + t_ns / NS_PER_TICK
# i.e., t_ns=0 in MCTrack/MCShower already corresponds to the trigger tick.
# (See ublarcvapp/MCTools/crossingPointsAnaMethods.cxx l.82 for the C++ source.)
# Stored as H5 attrs so a downstream consumer never has to guess.
# ----------------------------------------------------------------------------
TRIGGER_OFFSET_NS = 4050.0          # calibration tag; not a subtractive offset
USEC_PER_TICK = 0.5                 # TPC tick width
NS_PER_TICK = USEC_PER_TICK * 1000.0
TRIGGER_TICK = 3200                 # tick where the optical trigger sits
DRIFT_VELOCITY_CM_PER_US = 0.1098   # MicroBooNE nominal
CM_PER_TICK = DRIFT_VELOCITY_CM_PER_US * USEC_PER_TICK

# MicroBooNE 2D image window in tick space (matches FlashMatcherV2.cxx l.501).
IMAGE_TICK_MIN = 2400.0
IMAGE_TICK_MAX = 8448.0

N_PMTS = 32

# MicroBooNE uses two readout electronics chains on the same 32 physical PMTs:
#   - beam readout puts the 32 PMTs on OpChannels [0, 31]
#   - cosmic readout puts them on OpChannels [200, 231]
# OpChannel != OpDet — the channel->opdet mapping is a non-trivial permutation
# (e.g., OpChannel 0 -> OpDet 3, ch 4 -> OpDet 0, ...). Positions are indexed
# by OpDet. We store flash PE indexed by OpDet so that pe[i] always refers to
# the same physical PMT as pmt_positions[i]. The channel->opdet map is built
# at the top of process_entry() from larutil::Geometry::OpDetFromOpChannel.
PMT_CHANNEL_OFFSET_PER_PRODUCER = (0, 200)  # index = producer_id (0=beam, 1=cosmic)


def true_tick_from_ns(t_ns):
    """MCTrack/MCShower time (ns) -> true TPC tick (no drift).

    See module docstring constants block — trigger is at t_ns = 0 and tick =
    TRIGGER_TICK; no additional subtraction is applied here.
    """
    return TRIGGER_TICK + t_ns / NS_PER_TICK


def reco_tick_from_true(true_tick, x_cm):
    """Apparent (drift-shifted) tick at position x_cm."""
    return true_tick + x_cm / CM_PER_TICK


# ----------------------------------------------------------------------------
# ROOT readers
# ----------------------------------------------------------------------------

def build_trackid_start_map(ioll):
    """Return {trackid: (t_ns, x, y, z, source)} for every mctrack/mcshower
    in the currently-loaded entry. source: 0=mctrack, 1=mcshower."""
    out = {}
    ev_trk = ioll.get_data(larlite.data.kMCTrack, "mcreco")
    for i in range(ev_trk.size()):
        t = ev_trk[i]
        s = t.Start()
        out[int(t.TrackID())] = (
            float(s.T()), float(s.X()), float(s.Y()), float(s.Z()), 0,
        )
    ev_sh = ioll.get_data(larlite.data.kMCShower, "mcreco")
    for i in range(ev_sh.size()):
        sh = ev_sh[i]
        s = sh.Start()
        out[int(sh.TrackID())] = (
            float(s.T()), float(s.X()), float(s.Y()), float(s.Z()), 1,
        )
    return out


def build_trackid_traj_reco_tick_bounds(ioll):
    """For each mctrack, compute (min, max) reco tick over its trajectory.
    Used to flag image-boundary-crossing primaries. mcshowers are treated
    as point-like (start only) for this check.
    """
    out = {}
    ev_trk = ioll.get_data(larlite.data.kMCTrack, "mcreco")
    for i in range(ev_trk.size()):
        t = ev_trk[i]
        n = t.size()
        if n == 0:
            continue
        tmin = float("inf")
        tmax = float("-inf")
        for j in range(n):
            step = t[j]
            tt = true_tick_from_ns(step.T())
            rt = reco_tick_from_true(tt, step.X())
            if rt < tmin:
                tmin = rt
            if rt > tmax:
                tmax = rt
        out[int(t.TrackID())] = (tmin, tmax)

    ev_sh = ioll.get_data(larlite.data.kMCShower, "mcreco")
    for i in range(ev_sh.size()):
        sh = ev_sh[i]
        s = sh.Start()
        tt = true_tick_from_ns(s.T())
        rt = reco_tick_from_true(tt, s.X())
        out[int(sh.TrackID())] = (rt, rt)
    return out


def build_channel_to_opdet_map():
    """Return (channel_to_opdet, opdet_positions) where:
        channel_to_opdet : dict {OpChannel -> OpDet} for the 64 channels we read
                           (beam stream [0,31] + cosmic stream [200,231])
        opdet_positions  : (32, 3) float32 positions indexed by OpDet
    """
    from larlite import larutil
    geom = larutil.Geometry.GetME()
    n_opdets = int(geom.NOpDets())
    assert n_opdets >= N_PMTS, f"unexpected NOpDets={n_opdets}"

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


def extract_flashes(ioll, channel_to_opdet, verbose=False):
    """Pull flashes from simpleFlashBeam + simpleFlashCosmic into a list of
    dicts. producer_id: 0=beam, 1=cosmic.

    PE is stored indexed by OpDet so pe[i] and pmt_positions[i] always refer
    to the same physical PMT. We read larlite::opflash::PE(channel) for the
    channels of each stream's PMT range, then assign into pe[channel_to_opdet[ch]].
    """
    flashes = []
    producers = ("simpleFlashBeam", "simpleFlashCosmic")
    for prod_id, producer in enumerate(producers):
        ev = ioll.get_data(larlite.data.kOpFlash, producer)
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
            tpc_tick = time_us / USEC_PER_TICK + TRIGGER_TICK
            flashes.append({
                "pe": pe,
                "total_pe": float(pe.sum()),
                "time_us": time_us,
                "tpc_tick": float(tpc_tick),
                "producer_id": np.int32(prod_id),
                "flash_index": np.int32(i),
                "y_center": float(f.YCenter()) if hasattr(f, "YCenter") else 0.0,
                "z_center": float(f.ZCenter()) if hasattr(f, "ZCenter") else 0.0,
            })
        if verbose:
            print(f"  producer[{producer}] n_flashes={ev.size()} "
                  f"channel_offset={channel_offset}")
    return flashes


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------

def match_slices_to_flashes(slice_keys, slice_ticks, flashes, dtick_threshold):
    """Greedy nearest-flash match per slice.

    Returns:
        matched_flash_idx (S,) int32   -1 if no flash within threshold
        match_dtick       (S,) float32 |Δtick| for that match
        flash_match_slice (F,) int64   for each flash, the slice key of the
                                       closest matched slice, -1 otherwise
        flash_match_dtick (F,) float32
    """
    S = len(slice_keys)
    F = len(flashes)
    flash_ticks = np.array([f["tpc_tick"] for f in flashes], dtype=np.float32)

    matched_flash_idx = np.full(S, -1, dtype=np.int32)
    match_dtick = np.full(S, np.nan, dtype=np.float32)
    flash_match_slice = np.full(F, -1, dtype=np.int64)
    flash_match_dtick = np.full(F, np.nan, dtype=np.float32)

    for si in range(S):
        if F == 0:
            continue
        dt = np.abs(flash_ticks - float(slice_ticks[si]))
        best = int(np.argmin(dt))
        if dt[best] <= dtick_threshold:
            matched_flash_idx[si] = best
            match_dtick[si] = float(dt[best])
            if (flash_match_slice[best] == -1
                    or dt[best] < flash_match_dtick[best]):
                flash_match_slice[best] = int(slice_keys[si])
                flash_match_dtick[best] = float(dt[best])

    return matched_flash_idx, match_dtick, flash_match_slice, flash_match_dtick


def check_image_boundary_crossing(slice_info, traj_bounds):
    """Flag slices whose any member trackid trajectory leaves
    [IMAGE_TICK_MIN, IMAGE_TICK_MAX]. Coarse — uses only trackids that are
    present in larlite mctrack/mcshower."""
    n = len(slice_info["primary_trackid"])
    crosses = np.zeros(n, dtype=np.int8)
    for k in range(n):
        for tid in slice_info["slice_member_trackids"][k]:
            b = traj_bounds.get(int(tid))
            if b is None:
                continue
            tmin, tmax = b
            if tmin < IMAGE_TICK_MIN or tmax > IMAGE_TICK_MAX:
                crosses[k] = 1
                break
    return crosses


# ----------------------------------------------------------------------------
# PMT positions
# ----------------------------------------------------------------------------

def get_pmt_positions():
    """Return (N_PMTS, 3) MicroBooNE PMT positions in cm indexed by OpDet."""
    try:
        _, positions = build_channel_to_opdet_map()
        return positions
    except Exception as exc:
        print(f"[warn] could not read PMT positions from larutil.Geometry: {exc}")
        return None


# ----------------------------------------------------------------------------
# H5 writer
# ----------------------------------------------------------------------------

def write_flashinfo_h5(
    out_path, run, subrun, event,
    flashes, pmt_positions,
    mc_starts,
    slice_info, slice_tick,
    slice_matched_flash_idx, slice_match_dtick,
    slice_crosses_boundary,
    flash_matched_slice, flash_match_dtick,
    dtick_threshold,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        e = f.create_group("entry_0")
        e.attrs["run"] = int(run)
        e.attrs["subrun"] = int(subrun)
        e.attrs["event"] = int(event)
        e.attrs["dtick_threshold"] = float(dtick_threshold)
        e.attrs["trigger_offset_ns"] = TRIGGER_OFFSET_NS
        e.attrs["usec_per_tick"] = USEC_PER_TICK
        e.attrs["trigger_tick"] = int(TRIGGER_TICK)
        e.attrs["drift_velocity_cm_per_us"] = DRIFT_VELOCITY_CM_PER_US
        e.attrs["image_tick_min"] = IMAGE_TICK_MIN
        e.attrs["image_tick_max"] = IMAGE_TICK_MAX
        e.attrs["n_pmts"] = N_PMTS

        # ---- flashes -------------------------------------------------------
        fl = e.create_group("flashes")
        F = len(flashes)
        if F > 0:
            pe = np.stack([fl_["pe"] for fl_ in flashes], axis=0).astype(np.float32)
        else:
            pe = np.zeros((0, N_PMTS), dtype=np.float32)

        def _f32(key):
            return np.array([fl_[key] for fl_ in flashes], dtype=np.float32)
        def _i32(key):
            return np.array([fl_[key] for fl_ in flashes], dtype=np.int32)

        fl.create_dataset("pe", data=pe, compression="gzip", compression_opts=6)
        fl.create_dataset("total_pe", data=_f32("total_pe") if F else np.zeros(0, np.float32))
        fl.create_dataset("time_us",  data=_f32("time_us")  if F else np.zeros(0, np.float32))
        fl.create_dataset("tpc_tick", data=_f32("tpc_tick") if F else np.zeros(0, np.float32))
        fl.create_dataset("producer_id", data=_i32("producer_id") if F else np.zeros(0, np.int32))
        fl.create_dataset("flash_index", data=_i32("flash_index") if F else np.zeros(0, np.int32))
        fl.create_dataset("y_center", data=_f32("y_center") if F else np.zeros(0, np.float32))
        fl.create_dataset("z_center", data=_f32("z_center") if F else np.zeros(0, np.float32))
        fl.create_dataset("matched_slice_id", data=flash_matched_slice.astype(np.int64))
        fl.create_dataset("match_dtick", data=flash_match_dtick.astype(np.float32))

        # ---- pmt positions -------------------------------------------------
        if pmt_positions is not None:
            e.create_dataset("pmt_positions", data=pmt_positions.astype(np.float32))

        # ---- mc_particle_start_times ---------------------------------------
        m = e.create_group("mc_particle_start_times")
        m.create_dataset("trackid", data=mc_starts["trackid"].astype(np.int32))
        m.create_dataset("start_t_ns", data=mc_starts["start_t_ns"].astype(np.float64),
                         compression="gzip", compression_opts=6)
        m.create_dataset("start_tpc_tick_nodrift",
                         data=mc_starts["start_tpc_tick_nodrift"].astype(np.float32))
        m.create_dataset("source", data=mc_starts["source"].astype(np.int8))

        # ---- slice_flash_matches -------------------------------------------
        S = len(slice_info["primary_trackid"])
        sl = e.create_group("slice_flash_matches")
        sl.create_dataset("slice_id",
                          data=slice_info["primary_trackid"].astype(np.int64))
        sl.create_dataset("primary_origin",
                          data=slice_info["primary_origin"].astype(np.int32))
        sl.create_dataset("primary_tpc_tick", data=slice_tick.astype(np.float32))
        sl.create_dataset("matched_flash_idx", data=slice_matched_flash_idx.astype(np.int32))
        sl.create_dataset("match_dtick", data=slice_match_dtick.astype(np.float32))
        sl.create_dataset("is_null_flash",
                          data=(slice_matched_flash_idx == -1).astype(np.int8))
        sl.create_dataset("crosses_image_boundary",
                          data=slice_crosses_boundary.astype(np.int8))

        total_pe = np.zeros(S, dtype=np.float32)
        for si in range(S):
            idx = int(slice_matched_flash_idx[si])
            if 0 <= idx < F:
                total_pe[si] = flashes[idx]["total_pe"]
        sl.create_dataset("total_pe_matched", data=total_pe)


# ----------------------------------------------------------------------------
# Per-entry driver
# ----------------------------------------------------------------------------

def process_entry(input_dlmerged, ientry, merged_h5, output_h5,
                  dtick_threshold, verbose=True):
    if not os.path.exists(input_dlmerged):
        raise FileNotFoundError(f"dlmerged ROOT not found: {input_dlmerged}")
    if not os.path.exists(merged_h5):
        raise FileNotFoundError(f"merged H5 not found: {merged_h5}")

    if verbose:
        print(f"[entry {ientry}] reading {input_dlmerged}")
        print(f"             paired merged H5 {merged_h5}")
        print(f"             output           {output_h5}")
        print(f"             dtick_threshold  {dtick_threshold} ticks "
              f"({dtick_threshold * USEC_PER_TICK * 1000.0} ns)")

    # 1. open larlite and seek
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(input_dlmerged)
    ioll.set_verbosity(2)
    ioll.open()

    nentries = ioll.get_entries()
    if ientry < 0 or ientry >= nentries:
        ioll.close()
        raise IndexError(f"entry {ientry} out of range (file has {nentries} entries)")

    ioll.go_to(ientry)
    run = int(ioll.run_id())
    subrun = int(ioll.subrun_id())
    event = int(ioll.event_id())
    if verbose:
        print(f"  larlite (run, subrun, event) = ({run}, {subrun}, {event})")

    # 2. read mc_particle_tree + triplet_data from merged H5; compute slices
    with h5py.File(merged_h5, "r") as mh:
        e0 = mh["entry_0"]
        mh_run = int(e0.attrs.get("run", -1))
        mh_sr  = int(e0.attrs.get("subrun", -1))
        mh_ev  = int(e0.attrs.get("event", -1))
        if (mh_run, mh_sr, mh_ev) != (run, subrun, event):
            print(f"  [warn] (run,subrun,event) mismatch: ROOT=({run},{subrun},{event}) "
                  f"merged_h5=({mh_run},{mh_sr},{mh_ev})")
        mpt = e0["mc_particle_tree"]
        td = e0["triplet_data"]
        mc_tids = np.asarray(mpt["trackid"][:]).astype(np.int64)
        slice_info = compute_slice_labels(
            mpt, td["trackid"][:], td["hasmatch"][:],
        )

    # 3. trackid -> start time, and per-track trajectory tick bounds
    trackid_starts = build_trackid_start_map(ioll)
    traj_bounds = build_trackid_traj_reco_tick_bounds(ioll)

    # 4. mc_particle_start_times parallel to mc_particle_tree/trackid
    n_mc = len(mc_tids)
    mc_starts = {
        "trackid": np.asarray(mc_tids, dtype=np.int32),
        "start_t_ns": np.zeros(n_mc, dtype=np.float64),
        "start_tpc_tick_nodrift": np.zeros(n_mc, dtype=np.float32),
        "source": np.full(n_mc, -1, dtype=np.int8),
    }
    for i, tid in enumerate(mc_tids):
        info = trackid_starts.get(int(tid))
        if info is None:
            # Geant4 secondary — only known through the graph, not mctrack/mcshower
            continue
        t_ns, _, _, _, src = info
        mc_starts["start_t_ns"][i] = t_ns
        mc_starts["start_tpc_tick_nodrift"][i] = true_tick_from_ns(t_ns)
        mc_starts["source"][i] = src

    # 5. lead-primary tick per slice
    S = len(slice_info["primary_trackid"])
    slice_tick = np.zeros(S, dtype=np.float32)
    for k in range(S):
        members = slice_info["slice_member_trackids"][k]
        slice_tick[k] = np.nan
        # Try the slice_id key first, then any member
        candidates = [int(slice_info["primary_trackid"][k])] + [int(m) for m in members]
        for cand in candidates:
            info = trackid_starts.get(cand)
            if info is not None:
                slice_tick[k] = true_tick_from_ns(info[0])
                break

    # 6. extract flashes (build channel->opdet map first so PE is OpDet-indexed)
    channel_to_opdet, pmt_positions = build_channel_to_opdet_map()
    flashes = extract_flashes(ioll, channel_to_opdet, verbose=verbose)

    # 7. match
    slice_matched_flash_idx, slice_match_dtick, flash_match_slice, flash_match_dtick = (
        match_slices_to_flashes(
            slice_info["primary_trackid"], slice_tick, flashes, dtick_threshold,
        )
    )

    # 8. image-boundary crossing
    slice_crosses = check_image_boundary_crossing(slice_info, traj_bounds)

    # 9. write (pmt_positions already built alongside channel->opdet map above)
    write_flashinfo_h5(
        output_h5, run, subrun, event,
        flashes, pmt_positions,
        mc_starts,
        slice_info, slice_tick,
        slice_matched_flash_idx, slice_match_dtick,
        slice_crosses,
        flash_match_slice, flash_match_dtick,
        dtick_threshold,
    )

    if verbose:
        n_matched = int((slice_matched_flash_idx >= 0).sum())
        n_cross = int(slice_crosses.sum())
        n_nu = int((slice_info["primary_origin"] == 1).sum())
        n_cos = int((slice_info["primary_origin"] == 2).sum())
        print(f"  flashes={len(flashes)}  slices={S} (nu={n_nu}, cos={n_cos})  "
              f"matched={n_matched}/{S}  boundary_crossing={n_cross}")
        print(f"  wrote {output_h5}")

    ioll.close()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dlmerged", required=True)
    p.add_argument("--entry", type=int, required=True)
    p.add_argument("--merged-h5", required=True)
    p.add_argument("--output-h5", required=True)
    p.add_argument("--dtick-threshold", type=float, default=3.0,
                   help="Max |Δtick| between slice primary and matched flash. "
                        "Default 3 ticks = 1500 ns.")
    args = p.parse_args()

    process_entry(
        args.input_dlmerged, args.entry,
        args.merged_h5, args.output_h5,
        args.dtick_threshold,
    )


if __name__ == "__main__":
    main()
