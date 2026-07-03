"""
Phase-A calibration for the expanded segmenter flow (flash-matched slice recovery).

The expanded cascade would segment, in addition to nu-class slices, the top-K slices
by flash Neyman-chi2 (restricted to OOB <= oob_max and chi2 <= chi2_max). Here we use
the existing 1g+0X events' slice data (no cascade re-run) to pick K / chi2_max / oob_max:

per event we know each slice's flash chi2 (gamma=5.25), OOB fraction, and whether it is
a "photon slice" (>=50% of its points true-photon) and whether it's nu-class.

For each (K, oob_max, chi2_max) we report, over events where the photon is NOT already
in a nu-class slice (i.e. the ones the flash path must rescue):
  capture   = frac of those events where >=1 photon slice is in the flash top-K  (signal)
  contam    = mean # of NON-photon slices added by the flash top-K               (background)
This is the efficiency/purity trade for the added segmentation path.

Run in the pointcept container (GPU for the flash predictions):
  python3 calibrate_flash_path.py --event-list workdir_scale/inspect_1g0X.csv
"""
import argparse
import csv
import glob
import math
import os

import view_1g0X_flash as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-list", default="workdir_scale/inspect_1g0X.csv")
    ap.add_argument("--merged-dir", default=None)
    ap.add_argument("--gamma", type=float, default=5.25)
    args = ap.parse_args()

    events = list(csv.DictReader(open(args.event_list)))
    datadir = os.path.dirname(os.path.dirname(events[0]["stage3pred_path"]))
    merged_dir = args.merged_dir or os.path.join(datadir, "merged_sp")
    merged = {os.path.basename(p): p
              for p in glob.glob(os.path.join(merged_dir, "**", "merged_*entry*.h5"),
                                 recursive=True)}

    # per-event slice list: (chi2, oob, is_photon), + already-nu-photon flag
    per_event = []
    for r in events:
        base = os.path.basename(r["stage3pred_path"])[len("stage3pred_"):]
        mp = merged.get(base)
        if not mp:
            continue
        try:
            d = V.load_event(r["stage3pred_path"], mp)
        except Exception:
            continue
        g = args.gamma
        nu_photon = any(s["cls"] == V.NU_CLASS and s["is_photon"] for s in d["slices"])
        sl = [(V.chi2(d["pe_obs"], g * s["pe_pred"]), s["oob"], s["is_photon"])
              for s in d["slices"]]
        per_event.append((nu_photon, sl))

    n = len(per_event)
    n_need = sum(1 for nuph, _ in per_event if not nuph)   # photon NOT in a nu slice
    print("events: %d   (already in nu-slice: %d ; flash must rescue: %d)"
          % (n, n - n_need, n_need))

    def eval_cut(K, oob_max, chi2_max):
        cap = 0; contam_sum = 0
        for nuph, sl in per_event:
            if nuph:
                continue
            cand = sorted([(c, ip) for (c, oob, ip) in sl
                           if oob <= oob_max and c <= chi2_max])[:K]
            if any(ip for _, ip in cand):
                cap += 1
            contam_sum += sum(1 for _, ip in cand if not ip)
        return cap / max(n_need, 1), contam_sum / max(n_need, 1)

    print("\n%-3s %-7s %-9s %8s %9s" % ("K", "oob_max", "chi2_max", "capture", "contam/ev"))
    for oob_max in [0.05, 0.10]:
        for chi2_max in [50, 100, 200, 500, math.inf]:
            for K in [1, 2, 3]:
                cap, con = eval_cut(K, oob_max, chi2_max)
                print("%-3d %-7.2f %-9s %8.3f %9.2f"
                      % (K, oob_max, ("inf" if math.isinf(chi2_max) else "%d" % chi2_max),
                         cap, con))
        print()


if __name__ == "__main__":
    main()
