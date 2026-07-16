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
    ap.add_argument("--reco-cc", action="store_true",
                    help="select the reco-CC stream (default reco-NC)")
    ap.add_argument("--corrected-chi2", action="store_true",
                    help="use the dead-PMT-masked nu-slice chi2 recomputed from "
                         "the cascade (matches plots_flashfix) instead of the "
                         "ntuple's uncorrected recoVtxFlashChi2. Needs "
                         "--cascade-dir.")
    ap.add_argument("--dead-channels", default="15")
    args = ap.parse_args()

    msp = [l.strip() for l in open(args.merged_sp_list) if l.strip()]
    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "vtxX", "vtxY", "vtxZ",
                  "recoVtxX", "recoVtxY", "recoVtxZ", "recoVtxStream",
                  "recoVtxFlashChi2", "showerLArFormerPID", "showerRecoE",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    n = len(a["run"])
    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])
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
    sel = vok & (n_g >= 2) & (reco_cc == bool(args.reco_cc))
    if args.eq2:
        sel &= (n_g == 2)

    # ---- flash chi2 for the optional range filter --------------------------
    if (args.chi2_min is not None or args.chi2_max is not None) \
            and args.corrected_chi2:
        # dead-PMT-masked nu-slice chi2 from the cascade (matches plots_flashfix)
        import sys as _sys
        _sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "larformer_analysis", "physics", "pi0mass_peak"))
        from flash_correction import corrected_chi2_by_rse
        dead = tuple(int(x) for x in args.dead_channels.split(",") if x != "")
        cache = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "larformer_analysis", "physics", "pi0mass_peak",
            "rse_" + os.path.basename(os.path.dirname(
                args.cascade_dir.rstrip("/"))) + "_"
            + os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")
        cc_map = corrected_chi2_by_rse(args.cascade_dir, run, sub, evt,
                                       np.nonzero(sel)[0], dead, cache)
        chi2 = np.full(n, np.nan)
        for i, v in cc_map.items():
            chi2[i] = v
        keep = np.isfinite(chi2)
        if args.chi2_min is not None:
            keep &= chi2 >= args.chi2_min
        if args.chi2_max is not None:
            keep &= chi2 <= args.chi2_max
        sel &= keep
    elif args.chi2_min is not None or args.chi2_max is not None:
        # primary (nu-stream) vertex flash chi2 straight from the ntuple
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
    print(f">>> {len(idx)} reco-{'CC' if args.reco_cc else 'NC'} "
          f"{'eq2' if args.eq2 else 'ge2'} events of {n}{cr}"
          + ("  [corrected/dead-masked chi2]" if args.corrected_chi2 else ""))

    # Verify entry i <-> merged_sp line i using the MERGED_SP file's own
    # run/subrun/event (entry_0 attrs). This is the mapping the browse list
    # actually depends on (the exporter walks the merged_sp list in order).
    # Do NOT verify via the cascade index: for the veto-surgered bnb5e19 the
    # cascade index != ntuple entry index (that is why flash_correction matches
    # by RSE), so a cascade-index check reports spurious mismatches.
    lines, nver, nbad = [], 0, 0
    for i in idx:
        path = msp[i]
        try:
            with h5py.File(path, "r") as f:
                at = f["entry_0"].attrs
                ok = (int(at["run"]) == run[i] and int(at["subrun"]) == sub[i]
                      and int(at["event"]) == evt[i])
            nver += 1
            if not ok:
                nbad += 1
                if nbad <= 3:
                    print(f"  [mismatch] entry {i}: ntuple "
                          f"({run[i]},{sub[i]},{evt[i]}) vs merged_sp "
                          f"{os.path.basename(path)}")
        except Exception as e:
            print(f"  [warn] merged_sp {i}: {e}")
        lines.append(path)
    print(f">>> merged_sp linkage verified on {nver} events, {nbad} mismatches")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f">>> wrote {len(lines)} merged_sp paths -> {args.out}")


if __name__ == "__main__":
    main()
