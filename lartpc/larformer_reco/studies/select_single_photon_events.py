"""Isolate SINGLE-VISIBLE-PHOTON nu interactions (truth only).

Signal topology: an interaction whose ONLY visible particle is a single photon
above a visible-energy threshold -- no visible electron / muon / pion / proton and
no second visible photon. Physically this is almost always a pi0 whose OTHER
photon is sub-threshold or escapes the detector (a rare true single-gamma final
state aside), so it mimics a nu_e CC electron shower. This is an important
background/signal class the experiment wants to study.

"Visible energy" is charge-based, matching the reco: E_vis = a_gamma * Q, where Q
is the de-double-counted pixel sum (calo.dedup_charge `comb`, Y-else-mean(U,V)) of
the particle's true triplet_data spacepoints (trackid == t). Truth only -- reads
merged_sp/entry_0/{mc_particle_tree,triplet_data}; no keypoint2 / nu_reco.

Selection (per event): among nu-origin (origin==1) reconstructable particles
(e/gamma/mu/pi/p), count those with E_vis > --thresh-mev.  Qualify iff exactly one
is visible AND it is a photon (and no e/mu/pi/p exceeds --veto-thresh-mev, default
= thresh).  Writes the qualifying merged_sp paths to --out (feed to the visualizer
via --browse-list).  Shardable with --start/--n.

    python select_single_photon_events.py --merged-sp-list inputlists/..._all.txt \
        --out inputlists/merged_sp_valdata_singlephoton.txt [--thresh-mev 50]
"""
import os
import sys
import argparse

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from lartpc.larformer_reco.utils import read_list  # noqa: E402
from lartpc.larformer_reco.trajfit.calo import dedup_charge            # noqa: E402
from lartpc.larformer_reco.trajfit.particle_momentum import load_shower_calib  # noqa: E402

PID2SP = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
          211: "pi", -211: "pi", 2212: "p"}
VETO_SP = {"e", "mu", "pi", "p"}


def visible_particles(msp_path, a_vis):
    """[(species, trackid, true_ke, E_vis, parent_pid), ...] for nu-origin
    reconstructable particles (E_vis charge-based). Also returns has_pi0."""
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        mt = e["mc_particle_tree"]
        tid = np.asarray(mt["trackid"][()], np.int64)
        pid = np.asarray(mt["pid"][()], np.int64)
        org = np.asarray(mt["origin"][()], np.int64)
        ke = np.asarray(mt["energy_mev"][()], np.float64)
        par = np.asarray(mt["parent_trackid"][()], np.int64)
        pid_by = {int(tid[i]): int(pid[i]) for i in range(len(tid))}
        has_pi0 = bool(np.any((org == 1) & (pid == 111)))
        # nu-origin reconstructable trackids
        keep = {int(tid[i]): (PID2SP[int(pid[i])], float(ke[i]),
                              pid_by.get(int(par[i])))
                for i in range(len(tid))
                if org[i] == 1 and int(pid[i]) in PID2SP}
        if not keep:
            return [], has_pi0
        td = e["triplet_data"]
        ttid = np.asarray(td["trackid"][()], np.int64)
        sel = np.isin(ttid, np.asarray(list(keep), np.int64))
        out = []
        if sel.any():
            tsel = ttid[sel]
            _, qc = dedup_charge(np.asarray(td["pixval"][()])[sel],
                                 np.asarray(td["tick"][()])[sel],
                                 np.asarray(td["uwire"][()])[sel],
                                 np.asarray(td["vwire"][()])[sel],
                                 np.asarray(td["ywire"][()])[sel])
            q_by = {}
            for t in np.unique(tsel):
                q_by[int(t)] = float(qc[tsel == t].sum())
        else:
            q_by = {}
    for t, (sp, tke, ppid) in keep.items():
        out.append((sp, t, tke, a_vis * q_by.get(t, 0.0), ppid))
    return out, has_pi0


def classify(vis, thresh, veto_thresh):
    """Return (qualifies, info). Single visible photon > thresh, nothing else."""
    visible = [v for v in vis if v[3] > thresh]
    veto = [v for v in vis if v[0] in VETO_SP and v[3] > veto_thresh]
    photons = [v for v in visible if v[0] == "gamma"]
    ok = (len(visible) == 1 and len(photons) == 1 and not veto)
    # the OTHER nu-origin photons (below thresh: sub-threshold deposit OR escaped
    # with E_vis==0) -- the lost-photon signature of a pi0 with one photon gone
    other_g = [v for v in vis if v[0] == "gamma" and v[3] <= thresh]
    return ok, dict(photon=photons[0] if photons else None,
                    n_visible=len(visible), n_other_gamma=len(other_g))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--out", required=True, help="qualifying merged_sp list")
    ap.add_argument("--thresh-mev", type=float, default=50.0,
                    help="single photon must exceed this visible energy")
    ap.add_argument("--veto-thresh-mev", type=float, default=None,
                    help="e/mu/pi/p veto threshold (default = --thresh-mev)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1, help="-1 = to end")
    ap.add_argument("--max-hits", type=int, default=-1,
                    help="stop after this many qualifiers (quick sampling)")
    ap.add_argument("--diagnose", action="store_true",
                    help="print the visible composition of each qualifying event")
    args = ap.parse_args()
    veto_thresh = args.thresh_mev if args.veto_thresh_mev is None \
        else args.veto_thresh_mev

    calib = load_shower_calib()
    a_vis = float(calib.get("gamma", 0.0253)) or 0.0253
    msps = read_list(args.merged_sp_list)
    lo = args.start
    hi = len(msps) if args.n < 0 else min(len(msps), args.start + args.n)
    print(f">>> {len(msps)} merged_sp; scanning [{lo}:{hi}]  a_vis(gamma)="
          f"{a_vis:.4f} MeV/ADC  thresh={args.thresh_mev} veto={veto_thresh}",
          flush=True)

    hits, n_gamma, n_scan, sub_ct, pi0_ct = [], 0, 0, 0, 0
    for i in range(lo, hi):
        n_scan += 1
        try:
            vis, has_pi0 = visible_particles(msps[i], a_vis)
        except Exception:
            continue
        if any(v[0] == "gamma" for v in vis):
            n_gamma += 1
        ok, info = classify(vis, args.thresh_mev, veto_thresh)
        if not ok:
            continue
        hits.append(msps[i])
        if info["n_other_gamma"] > 0:
            sub_ct += 1
        if has_pi0:
            pi0_ct += 1
        if args.diagnose:
            g = info["photon"]
            print(f"  [{i}] {os.path.basename(msps[i])}  photon E_vis="
                  f"{g[3]:.0f} MeV (trueKE={g[2]:.0f})  other_gamma="
                  f"{info['n_other_gamma']}  has_pi0={has_pi0}", flush=True)
        if 0 < args.max_hits <= len(hits):
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(hits) + ("\n" if hits else ""))
    print(f"\n>>> scanned {n_scan} ({n_gamma} with a nu-gamma) -> "
          f"{len(hits)} single-visible-photon events -> {args.out}")
    if hits:
        print(f"    of those: {pi0_ct}/{len(hits)} have a true pi0, "
              f"{sub_ct}/{len(hits)} have >=1 other (lost/sub-threshold) nu photon "
              f"(pi0-with-lost-photon signature)")


if __name__ == "__main__":
    main()
