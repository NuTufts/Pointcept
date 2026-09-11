"""Pipeline checks on a record file.

  1. The full nu-union prediction rebuilt at GAMMA_BEAM_REF, scaled by the
     file's gamma_eff, must reproduce the stored slices/pred_pe (validates the
     charge convention, the drift correction and the PhotonLib path).
  2. The muon's slice-wide-dedup charge should track nu_reco part_charge (which
     dedups over the particle union): report the ratio distribution.
  3. Provenance: one distinct production gamma_eff per sample.

    PYTHONPATH=$K python3 scripts/validate_records.py --sample <tag> [--records glob]
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))

from lartpc.larformer_analysis.flashmodel_calib.gammacal import CALIB_DIR, GAMMA_BEAM_REF, samples  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import load_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", required=True, choices=sorted(samples.SAMPLES))
    ap.add_argument("--records", default=None)
    args = ap.parse_args()
    s = samples.get(args.sample)
    paths = sorted(glob.glob(args.records or os.path.join(
        CALIB_DIR, "results", s["chain"], "records", f"{s['tag']}*.npz")))
    ev, mu, meta = load_records(paths)
    n = len(ev["gidx"])
    print(f">>> {s['tag']}: {len(paths)} shards, {n} events, {len(mu.get('ev', []))} muon candidates")
    ge = ev["gamma_eff_file"]
    print(f"   production gamma_eff values: {sorted(set(np.round(ge[np.isfinite(ge)], 4)))} "
          f"| gamma_scale: {sorted(set(np.round(ev['gamma_scale_file'][np.isfinite(ev['gamma_scale_file'])], 4)))}")
    ok = (np.isfinite(ev["pred_union_ref"]).all(1) & np.isfinite(ev["pred_nu_file"]).all(1)
          & (np.abs(ev["flash_time_us"] - ev["cascade_flash_time_us"]) < 1e-3))
    if ok.any():
        a = ev["pred_union_ref"][ok] * (ge[ok] / GAMMA_BEAM_REF)[:, None]
        b = ev["pred_nu_file"][ok]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1.0)
        print(f"   [1] union pred vs stored (same t0): N={ok.sum()} | max rel diff {rel.max():.2e} "
              f"| events with max rel > 1e-3: {(rel.max(1) > 1e-3).sum()}")
    dif = ok.sum() < np.isfinite(ev["pred_union_ref"]).all(1).sum()
    if dif:
        print(f"   [1] note: {np.isfinite(ev['pred_union_ref']).all(1).sum() - ok.sum()} events skipped "
              f"where the chosen in-window flash differs from the cascade's max-PE flash")
    if mu:
        m = np.isfinite(mu["charge_reco"]) & (mu["charge_reco"] > 0)
        if m.any():
            rq = mu["q_mu"][m] / mu["charge_reco"][m]
            print(f"   [2] q_mu / part_charge: N={m.sum()} median {np.median(rq):.3f} p16-84 "
                  f"{np.percentile(rq, 16):.3f}-{np.percentile(rq, 84):.3f} (slice-wide vs union dedup)")
        else:
            print("   [2] no nu_reco charge to compare (calib stream: kp2 instances only)")
        print(f"   [2] orphan (vertex-less) candidates: {int(mu['orphan'].sum())}")
    print(f"   [3] events with in-window flash: {int(ev['has_flash'].sum())} | with candidates: "
          f"{int((ev['n_cand'] > 0).sum())} | unmatched slice coords total: {int(ev['n_unmatched'].sum())}")


if __name__ == "__main__":
    main()
