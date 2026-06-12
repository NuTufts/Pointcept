"""
Pass-2 truth detectability: which true neutrino-origin photons are "visible".

The single-photon signal needs a photon that creates >=20 MeV of ionization in a
single cluster. The older mcc9 sim files have NO true per-spacepoint energy (the
H5 `edep` field just mirrors the wire-plane ADC `pixval`), so we use proxies built
from the GT shower fragments that Stage A already writes:

  entry_0/shower_fragments : per-fragment pid, trackid, istrunk, pointindices_*
  entry_0/triplet_data     : pixval (N,3) wire-plane ADC per spacepoint
  entry_0/mc_particle_tree : energy_mev per particle (matched by trackid)

For each neutrino-origin photon (pid==22) we group its GT fragments by trackid and
report:
  nSP_total   total spacepoints across the photon's fragments  (count proxy)
  nSP_trunk   spacepoints in the trunk fragment(s)             (count proxy)
  pixval_sum  sum of wire-plane ADC over the photon's unique spacepoints (charge proxy)
  trueE_mev   true initial photon energy from mc_particle_tree (for calibrating a
              threshold against ~20 MeV)

Output photon_detectability.csv (one row per true nu-origin photon). Use it to pick
the nSP (or pixval) threshold that defines "detectable", then count detectable
photons per event for the true-positive denominator.

Run inside the pointcept container (needs h5py):
  python3 compute_photon_detectability.py --inputs workdir/cascade_inputs.txt \
      --out workdir/photon_detectability.csv [--nsp-threshold 0]
"""
import argparse
import glob
import os
import h5py
import numpy as np

PHOTON_PDG = 22
NU_ORIGIN = 1


def list_h5(inputs):
    if os.path.isfile(inputs) and inputs.endswith(".txt"):
        with open(inputs) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    if os.path.isdir(inputs):
        return sorted(glob.glob(os.path.join(inputs, "**", "merged_*_entry*.h5"),
                                recursive=True))
    return [inputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="workdir/cascade_inputs.txt",
                    help="a .txt list of merged H5, a dir, or a single H5")
    ap.add_argument("--out", default="workdir/photon_detectability.csv")
    ap.add_argument("--nsp-threshold", type=int, default=0,
                    help="report a detectable flag using nSP_total >= this (0 = no cut, just dump)")
    args = ap.parse_args()

    files = list_h5(args.inputs)
    print("merged H5 to scan:", len(files))

    rows = []
    n_events = 0
    for path in files:
        with h5py.File(path, "r") as f:
            e = f["entry_0"]
            run = int(e.attrs.get("run", -1))
            sub = int(e.attrs.get("subrun", -1))
            evt = int(e.attrs.get("event", -1))
            n_events += 1

            if "shower_fragments" not in e:
                continue
            sf = e["shower_fragments"]
            if int(sf.attrs.get("num_fragments", 0)) == 0:
                continue

            pid = sf["pid"][:]
            tid = sf["trackid"][:]
            istrunk = sf["istrunk"][:]
            cnt = sf["pointindices_counts"][:]
            flat = sf["pointindices_flat"][:]
            offs = np.concatenate([[0], np.cumsum(cnt)])

            pixval = e["triplet_data"]["pixval"][:]
            origin_sp = e["triplet_data"]["origin"][:]

            # mc_particle_tree: trackid -> (energy_mev, origin)
            mt = e["mc_particle_tree"]
            mt_tid = mt["trackid"][:]
            mt_e = mt["energy_mev"][:]
            mt_org = mt["origin"][:]
            tid2e = {int(t): float(en) for t, en in zip(mt_tid, mt_e)}
            tid2org = {int(t): int(o) for t, o in zip(mt_tid, mt_org)}

            # photon fragments grouped by trackid
            is_phot = pid == PHOTON_PDG
            for t in np.unique(tid[is_phot]):
                fr = np.where((tid == t) & is_phot)[0]
                idx_all = np.concatenate([flat[offs[i]:offs[i + 1]] for i in fr])
                idx_uniq = np.unique(idx_all)
                trunk_fr = fr[istrunk[fr] == 1]
                nsp_trunk = int(cnt[trunk_fr].sum()) if len(trunk_fr) else 0
                nsp_total = int(cnt[fr].sum())
                pix_sum = float(pixval[idx_uniq].sum())
                trueE = tid2e.get(int(t), -1.0)
                # photon origin: prefer mc_particle_tree, fall back to spacepoint origin
                org = tid2org.get(int(t), int(np.bincount(origin_sp[idx_uniq]).argmax())
                                  if len(idx_uniq) else -1)
                rows.append((run, sub, evt, int(t), len(fr), nsp_total, nsp_trunk,
                             pix_sum, trueE, org))

    # write csv
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("run,subrun,event,trackid,n_frag,nSP_total,nSP_trunk,"
                 "pixval_sum,trueE_mev,origin\n")
        for r in rows:
            fh.write("%d,%d,%d,%d,%d,%d,%d,%.6g,%.4f,%d\n" % r)

    # summary
    nu_rows = [r for r in rows if r[9] == NU_ORIGIN]
    nsp = np.array([r[5] for r in nu_rows]) if nu_rows else np.array([])
    print("-" * 60)
    print("events scanned          : %d" % n_events)
    print("true photons (all)      : %d" % len(rows))
    print("  nu-origin photons     : %d" % len(nu_rows))
    if len(nsp):
        for q in (10, 25, 50, 75, 90):
            print("  nSP_total %2dth pctile : %.0f" % (q, np.percentile(nsp, q)))
        print("  nSP_total min/max     : %d / %d" % (nsp.min(), nsp.max()))
    if args.nsp_threshold > 0 and nu_rows:
        det = sum(1 for r in nu_rows if r[5] >= args.nsp_threshold)
        ev_det = len(set((r[0], r[1], r[2]) for r in nu_rows
                         if r[5] >= args.nsp_threshold))
        print("  detectable (nSP>=%d)  : %d photons in %d events"
              % (args.nsp_threshold, det, ev_det))
    print("wrote: %s" % args.out)


if __name__ == "__main__":
    main()
