"""Build / load the per-sample calibration record file.

One pass over the sample's kp2 nu-stream list (gidx order, the same index the
nu_reco shards and the ntuple exporter use). Per event we store the flash
choice, the observed PE, the live mask and the production's provenance attrs;
for every loose calibration-muon candidate we store its kinematics, isolation
variables, and its OWN-points prediction at GAMMA_BEAM_REF. Every protocol cut
is applied later by fit_gamma.py, so cuts can be scanned without rebuilding.

Layout of the npz: `ev_<key>` arrays (one row per event), `mu_<key>` arrays
(one row per candidate; `mu_ev` indexes into the ev arrays), and `meta_*`.
"""
import os
import sys
import time

import numpy as np
import h5py

from lartpc.larformer_reco.utils import read_list
from lartpc.larformer_reco.export.export_gen2ntuple import (
    build_reco_map, _attr_str)

from . import GAMMA_BEAM_REF, VERSION
from .flash import read_flashes, choose_flash, MAX_FLASHES
from .masks import live_mask
from .points import MspCharge, slice_charge
from .predict import predict_pe_ref, cos_sim, centroid_yz
from .muon_select import event_particles, candidate_muons

_STR_KEYS = {"kp2_path", "msp_path", "dead_file", "chi2_masked_file", "sat_file"}


def _truth_ctx(msp_path):
    """MC only: nu-origin truth context from flashmatch_quality (None if the
    merged_sp has no mc_particle_tree)."""
    try:
        from lartpc.larformer_reco.eval.flashmatch_quality import _nu_truth_ctx
        with h5py.File(msp_path, "r") as f:
            if "mc_particle_tree" not in f["entry_0"]:
                return None
        return _nu_truth_ctx(msp_path)
    except Exception:
        return None


def _nu_qfrac(kp, ctx):
    try:
        from lartpc.larformer_reco.eval.flashmatch_quality import _slice_qfrac
        return float(_slice_qfrac(kp, ctx))
    except Exception:
        return np.nan


def _gt_lookup(msp_path):
    """MC only: trackid -> (pdg, origin) from mc_particle_tree."""
    try:
        with h5py.File(msp_path, "r") as f:
            mt = f["entry_0/mc_particle_tree"]
            tid = np.asarray(mt["trackid"][()], np.int64)
            origin = np.asarray(mt["origin"][()], np.int64)
            pdg = (np.asarray(mt["pdg"][()], np.int64) if "pdg" in mt
                   else np.zeros(len(tid), np.int64))
        return {int(t): (int(p), int(o)) for t, p, o in zip(tid, pdg, origin)}
    except Exception:
        return {}


def build_records(sample, start, n, out_path, min_len=30.0, device="cpu",
                  broad=(2.0, 7.0), union_every=False, verbose=True):
    t_start = time.time()
    kp_list = read_list(sample["kp2_nu_list"])
    msp_list = read_list(sample["merged_sp_list"])
    msp_by_base = {os.path.basename(p): p for p in msp_list}
    stop = len(kp_list) if n is None or n < 0 else min(start + n, len(kp_list))
    if verbose:
        print(f">>> {sample['tag']}: events [{start}:{stop}) of {len(kp_list)} "
              f"| kind={sample['kind']} period={sample['period']} "
              f"| union_every={union_every}", flush=True)
    reco_map = build_reco_map(sample["nu_reco_dir"])
    shard_handles = {}

    def reco_group(gidx):
        hit = reco_map.get(gidx)
        if hit is None:
            return None
        shard, ev = hit
        if shard not in shard_handles:
            shard_handles[shard] = h5py.File(shard, "r")
        f = shard_handles[shard]
        return f[ev] if ev in f else None

    ev_rows, mu_rows = [], []
    counters = dict(events=0, no_kp2=0, no_flash=0, no_reco=0, no_cand=0,
                    no_msp=0, with_cand=0, muons=0, union_pred=0)
    for gidx in range(start, stop):
        counters["events"] += 1
        kp_path = kp_list[gidx]
        try:
            kp = h5py.File(kp_path, "r")
        except Exception:
            counters["no_kp2"] += 1
            continue
        with kp:
            a = kp.attrs
            fa = kp["flash"].attrs if "flash" in kp else {}
            run = int(a.get("run", 0))
            ev = dict(
                gidx=gidx, run=run, subrun=int(a.get("subrun", 0)),
                event=int(a.get("event", 0)), kp2_path=kp_path,
                msp_path="", has_gt=int(bool(a.get("has_gt", False))),
                gamma_beam_file=float(fa.get("gamma_beam", np.nan)),
                gamma_scale_file=float(fa.get("gamma_scale", np.nan)),
                gamma_eff_file=float(fa.get("gamma_eff", np.nan)),
                dead_file=_attr_str(fa, "dead_opdets"),
                chi2_masked_file=_attr_str(fa, "chi2_masked_opdets"),
                cascade_flash_time_us=float(fa.get("time_us", np.nan)),
                nu_chi2_file=np.nan, pred_nu_file=np.full(32, np.nan, np.float32),
                n_union_pts=0, q_union=np.nan, n_unmatched=0,
                pred_union_ref=np.full(32, np.nan, np.float32),
                nu_qfrac=np.nan, has_flash=0, flash_time_us=np.nan,
                flash_total_pe=np.nan, flash_times=np.full(MAX_FLASHES, np.nan, np.float32),
                flash_pes=np.full(MAX_FLASHES, np.nan, np.float32),
                obs_pe=np.full(32, np.nan, np.float32), live=np.zeros(32, bool),
                sat_file="", n_cand=0)
            if "slices" in kp:
                labs = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in kp["slices/label"][()]]
                if "nu" in labs:
                    j = labs.index("nu")
                    ev["pred_nu_file"] = np.asarray(kp["slices/pred_pe"][()][j], np.float32)
                    ev["nu_chi2_file"] = float(kp["slices/chi2"][()][j])
            if "slice" in kp and "coord_cm" in kp["slice"]:
                ev["n_union_pts"] = int(kp["slice/coord_cm"].shape[0])
            fl = read_flashes(kp)
            best, times, pes = choose_flash(fl, broad=broad)
            ev["flash_times"], ev["flash_pes"] = times, pes
            if best < 0:
                counters["no_flash"] += 1
                ev_rows.append(ev)
                continue
            obs = np.clip(fl["pe"][best], 0, None).astype(np.float32)
            t0 = float(fl["time_us"][best])
            live, dead, sat = live_mask(run, obs)
            ev.update(has_flash=1, flash_time_us=t0,
                      flash_total_pe=float(fl["total_pe"][best]), obs_pe=obs,
                      live=live, sat_file=",".join(str(s) for s in sat))
            gr = reco_group(gidx)
            if gr is None:
                counters["no_reco"] += 1
            parts = event_particles(kp, gr)
            cands = candidate_muons(parts, min_len=min_len)
            ev["n_cand"] = len(cands)
            need_msp = bool(cands) or union_every
            if not cands:
                counters["no_cand"] += 1
            if not need_msp or ev["n_union_pts"] == 0:
                ev_rows.append(ev)
                continue
            src = os.path.basename(_attr_str(a, "src_file"))
            msp_path = msp_by_base.get(src)
            if msp_path is None or not os.path.exists(msp_path):
                counters["no_msp"] += 1
                ev_rows.append(ev)
                continue
            ev["msp_path"] = msp_path
            msp = MspCharge(msp_path)
            coords = kp["slice/coord_cm"][()].astype(np.float32)
            q_pts, n_un = slice_charge(msp, coords)
            ev["q_union"] = float(q_pts.sum()); ev["n_unmatched"] = n_un
            ev["pred_union_ref"] = predict_pe_ref(coords, q_pts, t0, device=device)
            counters["union_pred"] += 1
            tctx = _truth_ctx(msp_path) if sample["kind"] == "mc" else None
            if tctx is not None:
                ev["nu_qfrac"] = _nu_qfrac(kp, tctx)
            gt = _gt_lookup(msp_path) if sample["kind"] == "mc" else {}
            if cands:
                counters["with_cand"] += 1
            ev_index = len(ev_rows)
            for c in cands:
                pidx = kp[f"particle/{c['inst']}/point_idx"][()].astype(np.int64)
                pts = coords[pidx]; q = q_pts[pidx]
                pred = predict_pe_ref(pts, q, t0, device=device)
                pcy, pcz = centroid_yz(pred, live); ocy, ocz = centroid_yz(obs, live)
                gpdg, gorig = gt.get(int(c["gt_trackid"]), (0, -1))
                mu_rows.append(dict(
                    ev=ev_index, inst=int(c["inst"]), orphan=int(c["orphan"]),
                    vertexed=int(c["vertexed"]), is_secondary=int(c["is_secondary"]),
                    cls=int(c["cls"]), pdg=int(c["pdg"]),
                    start=np.asarray(c["start"], np.float32),
                    end=np.asarray(c["end"], np.float32),
                    length=float(c["length"]), n_boundary=int(c["n_boundary"]),
                    boundary_end_x=float(c["boundary_end_x"]), ke=float(c["ke"]),
                    charge_reco=float(c["charge"]), q_mu=float(q.sum()),
                    npts=int(len(pidx)),
                    iso_track_ke_max=float(c["iso_track_ke_max"]),
                    n_other_tracks=int(c["n_other_tracks"]),
                    n_other_tracks_orphan=int(c["n_other_tracks_orphan"]),
                    proton_ke_max=float(c["proton_ke_max"]),
                    iso_shower_e_max=float(c["iso_shower_e_max"]),
                    n_showers=int(c["n_showers"]),
                    larpid_pid=int(c["larpid_pid"]), larpid_mu=float(c["larpid_mu"]),
                    gt_trackid=int(c["gt_trackid"]), gt_pdg=int(gpdg),
                    gt_origin=int(gorig), true_ke=float(c["true_ke"]),
                    pred_mu_ref=np.asarray(pred, np.float32),
                    cos_sim=cos_sim(obs[live], pred[live]),
                    dcy=abs(pcy - ocy), dcz=abs(pcz - ocz)))
                counters["muons"] += 1
            ev_rows.append(ev)
        if verbose and (gidx - start + 1) % 500 == 0:
            print(f"    {gidx - start + 1} events, {counters['muons']} muon rows, "
                  f"{time.time() - t_start:.0f}s", flush=True)
    for f in shard_handles.values():
        f.close()

    out = {}
    for pref, rows in (("ev_", ev_rows), ("mu_", mu_rows)):
        if not rows:
            continue
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if k in _STR_KEYS:
                out[pref + k] = np.asarray(vals, dtype=str)
            else:
                out[pref + k] = np.asarray(vals)
    out["meta_sample"] = np.asarray(sample["tag"])
    out["meta_kind"] = np.asarray(sample["kind"])
    out["meta_period"] = np.asarray(int(sample["period"]))
    out["meta_chain"] = np.asarray(sample["chain"])
    out["meta_window_us"] = np.asarray(sample["flash_window_us"], np.float64)
    out["meta_start"] = np.asarray(start); out["meta_stop"] = np.asarray(stop)
    out["meta_gamma_ref"] = np.asarray(GAMMA_BEAM_REF)
    out["meta_version"] = np.asarray(VERSION)
    out["meta_min_len"] = np.asarray(min_len)
    for k, v in counters.items():
        out["meta_count_" + k] = np.asarray(v)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **out)
    if verbose:
        print(f">>> {counters} | {time.time() - t_start:.0f}s -> {out_path}",
              flush=True)
    return out_path, counters


def load_records(paths):
    """Concatenate shard npz files into one dict of arrays (mu_ev re-offset)."""
    if isinstance(paths, str):
        paths = [paths]
    ev, mu, meta = {}, {}, {}
    n_ev = 0
    for p in sorted(paths):
        z = np.load(p, allow_pickle=False)
        keys = list(z.keys())
        ne = len(z["ev_gidx"]) if "ev_gidx" in keys else 0
        for k in keys:
            if k.startswith("ev_"):
                ev.setdefault(k[3:], []).append(z[k])
            elif k.startswith("mu_"):
                v = z[k]
                if k == "mu_ev":
                    v = v + n_ev
                mu.setdefault(k[3:], []).append(v)
            elif k.startswith("meta_") and k not in meta:
                meta[k[5:]] = z[k]
        n_ev += ne
    ev = {k: np.concatenate(v) for k, v in ev.items()}
    mu = {k: np.concatenate(v) for k, v in mu.items()}
    return ev, mu, meta
