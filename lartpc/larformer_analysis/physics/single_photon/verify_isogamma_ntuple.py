"""Verification of the isotropic single-photon gun ntuple against its truth
sidecars (Sample_Processing_Procedure.md STEP 5, "reco must follow truth").

    python3 verify_isogamma_ntuple.py <DATADIR> [--tag isogamma_run45]

Prints: entry/potTree counts, flashGammaEff + xsecWeight uniques, truth
coverage (every converted event has a row), vertex distance to the true
conversion point, and reco efficiency split by whether/where the photon
converted and by photon energy. The tick-order bug of 2026-10-07 showed up here
as foundVertex ~20% flat in all of these splits and vtxDistToTrue ~4 m.
"""
import argparse
import glob

import h5py
import numpy as np
import uproot

TPC = ((0, 256), (-116, 116), (0, 1036))
FID = ((10, 246), (-106, 106), (10, 1026))


def inside(p, box):
    return all(lo < v < hi for v, (lo, hi) in zip(p, box))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    ap.add_argument("--tag", default="isogamma_run45")
    args = ap.parse_args()
    D = args.datadir

    f = uproot.open(f"{D}/dlgen2_larformer_ntuple_{args.tag}.root")
    ev = f["EventTree"]
    a = ev.arrays(["run", "subrun", "event", "foundVertex", "vtxDistToTrue",
                   "flashGammaEff", "trueNuPDG", "trueNuE", "xsecWeight",
                   "nShowers", "nTracks"], library="np")
    n = ev.num_entries
    print(f"EventTree {n} entries, {len(ev.keys())} branches; potTree {f['potTree'].num_entries} rows")
    for k in ("flashGammaEff", "trueNuPDG", "xsecWeight"):
        u, c = np.unique(a[k], return_counts=True)
        print(f"  {k}: {dict(zip(u.tolist(), c.tolist()))}")
    idx = {k: i for i, k in enumerate(zip(a["run"], a["subrun"], a["event"]))}

    # truth sidecars -> per-row conversion point classification
    conv = np.zeros(n, bool)
    fid = np.zeros(n, bool)
    ntruth = nrow = 0
    for fn in glob.glob(f"{D}/truth_sidecar/truth_fileno*.h5"):
        with h5py.File(fn, "r") as h:
            for k in h:
                g = h[k]
                ntruth += 1
                i = idx.get((g.attrs["run"], g.attrs["subrun"], g.attrs["event"]))
                if i is None:
                    continue
                nrow += 1
                sp = g["trueSimPart"]
                ph = np.where(sp["PDG"][:] == 22)[0]
                if len(ph) == 0:
                    continue
                p = (sp["EDepX"][ph[0]], sp["EDepY"][ph[0]], sp["EDepZ"][ph[0]])
                if np.all(np.isfinite(p)) and inside(p, TPC):
                    conv[i] = True
                    fid[i] = inside(p, FID)
    print(f"truth events {ntruth}, with an ntuple row {nrow} "
          f"(sidecars include files dropped from merged_sp; rows must == EventTree)")

    fv = a["foundVertex"] == 1
    d = a["vtxDistToTrue"][fv & conv]
    print(f"vtxDistToTrue (found vertex, photon converts in TPC): "
          f"median {np.median(d):.1f} cm, <5 cm {np.mean(d < 5):.3f}, <10 cm {np.mean(d < 10):.3f}")
    E = a["trueNuE"]
    slc = a["flashGammaEff"] > 0
    sh = a["nShowers"] >= 1

    def row(lbl, m):
        print(f"  {lbl:30s} N={m.sum():6d}  nu-slice {slc[m].mean():.3f}  "
              f"foundVertex {fv[m].mean():.3f}  >=1 shower {sh[m].mean():.3f}  "
              f"vtx<5cm {np.mean(a['vtxDistToTrue'][m & fv] < 5) if (m & fv).any() else float('nan'):.3f}")

    print("efficiency vs truth:")
    row("all", np.ones(n, bool))
    row("no conversion in TPC", ~conv)
    row("converts in TPC", conv)
    row("converts >10 cm from walls", fid)
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .8), (.8, 2), (2, 10)]:
        row(f"  fid, E {lo}-{hi} GeV", fid & (E >= lo) & (E < hi))


if __name__ == "__main__":
    main()
