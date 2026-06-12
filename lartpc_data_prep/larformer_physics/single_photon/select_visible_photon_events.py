"""
Phase-2 truth selection for the scaled single-photon study.

Scans Stage-A merged_sp H5 and, per event, computes from truth:
  - visible photons: nu-origin photons (pid==22) whose GT shower-fragment trunk has
    nSP_trunk >= --nsp-threshold (proxy for >=20 MeV in a single cluster; the old
    mcc9 sim has no per-point edep). Each photon's energy and distance to the nu
    vertex are recorded.
  - 1g+0X classification: the event is 1-gamma-0X if it has exactly 1 visible
    photon AND no other visible nu-induced particle, where "visible X" is a
    nu-origin particle whose true KINETIC energy exceeds a per-type threshold:
        proton>50, pi+-/K+- >30, muon>30, electron>20 MeV   (neutrons ignored).
    (A 2nd visible photon makes n_vis_gamma>1, so it is not 1-gamma by counting.)

Outputs (into --outdir):
  visible_photon_events.csv  one row per event with >=1 visible photon
  photon_truth.csv           one row per visible photon (energy, vertex distance)
  cascade_inputs.txt         capped, spread list of visible-photon event H5 for Stage B

Run inside the pointcept container (needs h5py):
  python3 select_visible_photon_events.py --merged-dir <merged_sp> --outdir <out> \
      --ncap 3000 --nsp-threshold 80
"""
import argparse
import glob
import os
import h5py
import numpy as np

PHOTON_PDG = 22
NU_ORIGIN = 1

# true KINETIC-energy thresholds (MeV) for a particle to count as a visible "X"
X_KE_THRESH = {2212: 50.0, 211: 30.0, 321: 30.0, 13: 30.0, 11: 20.0}
# particle masses (MeV) to convert total energy -> KE
MASS_MEV = {2212: 938.272, 2112: 939.565, 211: 139.570, 321: 493.677,
            13: 105.658, 11: 0.511, 22: 0.0}


def photon_fragments(e, nsp_thr):
    """per nu-origin photon trackid: (nSP_trunk, origin_pt_cm). visible if nSP_trunk>=thr."""
    out = {}
    if "shower_fragments" not in e or int(e["shower_fragments"].attrs.get("num_fragments", 0)) == 0:
        return out
    sf = e["shower_fragments"]
    pid = sf["pid"][:]; tid = sf["trackid"][:]; istrunk = sf["istrunk"][:]
    cnt = sf["pointindices_counts"][:]; originpt = sf["originpt"][:]
    for t in np.unique(tid[pid == PHOTON_PDG]):
        fr = np.where((tid == t) & (pid == PHOTON_PDG))[0]
        trunk = fr[istrunk[fr] == 1]
        nsp_trunk = int(cnt[trunk].sum()) if len(trunk) else 0
        op = originpt[trunk[0]] if len(trunk) else originpt[fr[0]]
        out[int(t)] = (nsp_trunk, np.asarray(op, dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", required=True,
                    help="dir holding Stage-A merged_*_entry*.h5 (searched recursively)")
    ap.add_argument("--outdir", default="workdir_scale")
    ap.add_argument("--tag", default="sp_bnb_nu_overlay_scale")
    ap.add_argument("--nsp-threshold", type=int, default=80)
    ap.add_argument("--ncap", type=int, default=3000,
                    help="cap on visible-photon events written to cascade_inputs (-1=all)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.merged_dir, "**",
                                          "merged_%s_fileno*_entry*.h5" % args.tag),
                             recursive=True))
    print("merged_sp H5 found:", len(files))

    ev_rows = []      # per visible-photon event
    ph_rows = []      # per visible photon
    n_scanned = 0
    for path in files:
        try:
            f = h5py.File(path, "r")
        except Exception as ex:
            print("  WARN open failed:", path, ex); continue
        with f:
            e = f["entry_0"]
            n_scanned += 1
            run = int(e.attrs.get("run", -1)); sub = int(e.attrs.get("subrun", -1))
            evt = int(e.attrs.get("event", -1))
            mt = e["mc_particle_tree"]
            pid = mt["pid"][:]; tid = mt["trackid"][:]
            E = mt["energy_mev"][:]; org = mt["origin"][:]
            nuvtx = mt["nu_vertices"][:][0] if mt["nu_vertices"].shape[0] > 0 else np.array([np.nan]*3)

            # --- visible photons (nSP proxy) ---
            frag = photon_fragments(e, args.nsp_threshold)
            tid2E = {int(t): float(en) for t, en in zip(tid, E)}
            tid2org = {int(t): int(o) for t, o in zip(tid, org)}
            vis_photons = []   # (trackid, nSP, trueE, dist_to_vtx)
            for t, (nsp, op) in frag.items():
                if tid2org.get(t, 1) != NU_ORIGIN:
                    continue
                if nsp < args.nsp_threshold:
                    continue
                d = float(np.linalg.norm(op - nuvtx)) if np.isfinite(nuvtx).all() else -1.0
                vis_photons.append((t, nsp, tid2E.get(t, -1.0), d))
            n_vis_gamma = len(vis_photons)
            if n_vis_gamma == 0:
                continue

            # --- visible X (per-type true KE thresholds) ---
            n_vis_X = 0; xbreak = {}
            for i in range(len(pid)):
                if org[i] != NU_ORIGIN:
                    continue
                p = abs(int(pid[i]))
                if p not in X_KE_THRESH:
                    continue
                ke = float(E[i]) - MASS_MEV.get(p, 0.0)
                if ke > X_KE_THRESH[p]:
                    n_vis_X += 1
                    xbreak[p] = xbreak.get(p, 0) + 1
            is_1g0X = int(n_vis_gamma == 1 and n_vis_X == 0)

            # leading photon (most spacepoints)
            vis_photons.sort(key=lambda x: -x[1])
            lead = vis_photons[0]
            ev_rows.append((run, sub, evt, path, n_vis_gamma, n_vis_X, is_1g0X,
                            lead[2], lead[3],
                            xbreak.get(2212, 0), xbreak.get(211, 0) + xbreak.get(321, 0),
                            xbreak.get(13, 0), xbreak.get(11, 0)))
            for (t, nsp, te, d) in vis_photons:
                ph_rows.append((run, sub, evt, t, nsp, te, d, is_1g0X))

    # ---- write per-event + per-photon CSVs ----
    ev_path = os.path.join(args.outdir, "visible_photon_events.csv")
    with open(ev_path, "w") as fh:
        fh.write("run,subrun,event,filepath,n_vis_gamma,n_vis_X,is_1g0X,"
                 "lead_photon_E,lead_photon_dist,nX_p,nX_pi,nX_mu,nX_e\n")
        for r in ev_rows:
            fh.write("%d,%d,%d,%s,%d,%d,%d,%.3f,%.3f,%d,%d,%d,%d\n" % r)
    ph_path = os.path.join(args.outdir, "photon_truth.csv")
    with open(ph_path, "w") as fh:
        fh.write("run,subrun,event,trackid,nSP_trunk,trueE_mev,dist_to_vtx_cm,is_1g0X\n")
        for r in ph_rows:
            fh.write("%d,%d,%d,%d,%d,%.3f,%.3f,%d\n" % r)

    # ---- capped, spread cascade input list ----
    paths = [r[3] for r in ev_rows]
    if args.ncap >= 0 and len(paths) > args.ncap:
        stride = len(paths) // args.ncap
        paths = paths[::stride][:args.ncap]
    list_path = os.path.join(args.outdir, "cascade_inputs.txt")
    with open(list_path, "w") as fh:
        for p in paths:
            fh.write(os.path.abspath(p) + "\n")

    n_1g0X = sum(r[6] for r in ev_rows)
    print("-" * 60)
    print("events scanned                : %d" % n_scanned)
    print("events with >=1 visible photon: %d" % len(ev_rows))
    print("  of which 1gamma+0X          : %d (%.1f%%)"
          % (n_1g0X, 100.0 * n_1g0X / max(len(ev_rows), 1)))
    print("visible photons total         : %d" % len(ph_rows))
    print("cascade events written        : %d (cap=%d)" % (len(paths), args.ncap))
    print("wrote: %s" % ev_path)
    print("wrote: %s" % ph_path)
    print("wrote: %s" % list_path)


if __name__ == "__main__":
    main()
