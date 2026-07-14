"""Build a visualizer browse-list of merged_sp files for the reco-NC pi0
selection (nu-stream FV vertex + >=2 LArFormer photon showers >20 MeV, no
primary muon>100 MeV). The gen2ntuple is exported in merged_sp-list order
(entry i <-> line i), so passing entries map straight to merged_sp paths; the
linkage is verified against each cascade file's run/subrun/event.

    python3 make_ncpi0_browse_list.py --ntuple <ntuple.root> \
        --merged-sp-list <merged_sp_TAG.txt> --cascade-dir <keypoint2_streams> \
        --out <browse_ncpi0.txt> [--eq2]
"""
import argparse
import os

import numpy as np
import uproot
import awkward as ak
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eq2", action="store_true",
                    help="exactly 2 photons (default: >=2)")
    ap.add_argument("--chi2-min", type=float, default=None,
                    help="keep only events with primary-vtx flash chi2 >= this")
    ap.add_argument("--chi2-max", type=float, default=None,
                    help="keep only events with primary-vtx flash chi2 <= this")
    ap.add_argument("--max-events", type=int, default=None,
                    help="cap the output (evenly sampled across the survivors)")
    args = ap.parse_args()

    msp = [l.strip() for l in open(args.merged_sp_list) if l.strip()]
    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "vtxX", "vtxY", "vtxZ",
                  "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
                  "recoVtxFlashChi2", "showerLArFormerPID", "showerRecoE",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    n = len(a["run"])
    if n != len(msp):
        print(f"[warn] ntuple entries {n} != merged_sp lines {len(msp)}; "
              "order mapping may be unsafe")

    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > 20)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > 100))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))
    sel = vok & (n_g >= 2) & ~reco_cc
    if args.eq2:
        sel &= (n_g == 2)

    # primary (nu-stream) vertex flash chi2, same coord-match as
    # flashchi2_ncpi0.py, for the optional chi2-range filter
    if args.chi2_min is not None or args.chi2_max is not None:
        chi2 = np.full(n, np.nan)
        for i in np.nonzero(sel)[0]:
            rx = ak.to_numpy(a["recoVtxX"][i]); ry = ak.to_numpy(a["recoVtxY"][i])
            rz = ak.to_numpy(a["recoVtxZ"][i]); st = ak.to_numpy(a["recoVtxStream"][i])
            c2 = ak.to_numpy(a["recoVtxFlashChi2"][i])
            if not len(rx):
                continue
            d = np.sqrt((rx - a["vtxX"][i])**2 + (ry - a["vtxY"][i])**2
                        + (rz - a["vtxZ"][i])**2)
            d = np.where(st == a["primaryVtxStream"][i], d, 1e9)
            j = int(np.argmin(d))
            if d[j] < 1.0 and c2[j] >= 0:
                chi2[i] = c2[j]
        keep = np.isfinite(chi2)
        if args.chi2_min is not None:
            keep &= chi2 >= args.chi2_min
        if args.chi2_max is not None:
            keep &= chi2 <= args.chi2_max
        sel &= keep

    idx = np.nonzero(sel)[0]
    if args.max_events and len(idx) > args.max_events:
        idx = idx[np.linspace(0, len(idx) - 1, args.max_events).astype(int)]
    cr = ("" if args.chi2_min is None and args.chi2_max is None
          else f" | flash chi2 in [{args.chi2_min}, {args.chi2_max}]")
    print(f">>> {len(idx)} reco-NC {'eq2' if args.eq2 else 'ge2'} events "
          f"of {n}{cr}")

    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])
    lines, nver, nbad = [], 0, 0
    for i in idx:
        path = msp[i]
        # verify entry<->line via the cascade file's run/subrun/event
        casc = os.path.join(args.cascade_dir, f"keypoint2_event{i:05d}_0.h5")
        if os.path.exists(casc):
            try:
                with h5py.File(casc, "r") as f:
                    ok = (int(f.attrs["run"]) == run[i]
                          and int(f.attrs["subrun"]) == sub[i]
                          and int(f.attrs["event"]) == evt[i])
                nver += 1
                if not ok:
                    nbad += 1
                    print(f"  [mismatch] entry {i}: ntuple "
                          f"({run[i]},{sub[i]},{evt[i]}) vs cascade "
                          f"({int(f.attrs['run'])},{int(f.attrs['subrun'])},"
                          f"{int(f.attrs['event'])})")
            except Exception as e:
                print(f"  [warn] cascade {i}: {e}")
        lines.append(path)
    print(f">>> linkage verified on {nver} events, {nbad} mismatches")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f">>> wrote {len(lines)} merged_sp paths -> {args.out}")


if __name__ == "__main__":
    main()
