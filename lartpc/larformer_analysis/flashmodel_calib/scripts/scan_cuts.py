"""Muon-arm cut variants side by side for one sample (robustness table).

    PYTHONPATH=$K python3 scripts/scan_cuts.py --sample <tag> [--records glob]

Prints N, s, bootstrap error, p16-84, core median/fraction and gate for the
protocol defaults and a fixed set of variants. Nothing is written.
"""
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
sys.path.insert(0, _HERE)

import fit_gamma as FG  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal import CALIB_DIR, samples  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import load_records  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.estimators import ratio_stats  # noqa: E402

VARIANTS = [
    ("protocol defaults", []),
    ("cos >= 0.95", ["--cos-min", "0.95"]),
    ("cos >= 0.98", ["--cos-min", "0.98"]),
    ("cos >= 0.8", ["--cos-min", "0.8"]),
    ("no track/shower isolation", ["--iso-track-ke", "1e9", "--iso-shower-e", "1e9"]),
    ("no isolation, cos >= 0.95", ["--iso-track-ke", "1e9", "--iso-shower-e", "1e9", "--cos-min", "0.95"]),
    ("no isolation, no q_frac, cos >= 0.95", ["--iso-track-ke", "1e9", "--iso-shower-e", "1e9",
                                             "--q-frac-min", "0", "--cos-min", "0.95"]),
    ("q_frac >= 0.9", ["--q-frac-min", "0.9"]),
    ("boundary x > 0 (any side)", ["--x-boundary-min", "0"]),
    ("boundary x > 200", ["--x-boundary-min", "200"]),
    ("two-boundary muons", ["--n-boundary", "2", "--x-boundary-min", "0"]),
    ("contained (0 boundary)", ["--n-boundary", "0"]),
    ("length > 100", ["--min-len", "100"]),
    ("min pred 200", ["--min-pred", "200"]),
    ("window margin 0.5", ["--window-margin", "0.5"]),
    ("allow secondary", ["--allow-secondary"]),
]


def main():
    ap = FG.build_parser()
    args0 = ap.parse_args()
    s = samples.get(args0.sample)
    paths = sorted(glob.glob(args0.records or os.path.join(
        CALIB_DIR, "results", s["chain"], "records", f"{s['tag']}*.npz")))
    ev, mu, meta = load_records(paths)
    window = tuple(s["flash_window_us"])
    print(f">>> {s['tag']} ({s['kind']}, period {s['period']}) muon-arm cut variants | "
          f"{len(mu['ev'])} candidates")
    print(f"{'variant':38s} {'N':>6s} {'s':>7s} {'err':>7s} {'p16-84':>13s} {'coreMed':>8s} {'core':>5s} gate")
    for name, extra in VARIANTS:
        a = ap.parse_args(["--sample", args0.sample] + extra)
        m, drops, aux = FG.select_muons(ev, mu, a, window)
        r = aux["obs_l"][m] / aux["pred_l"][m]
        st = ratio_stats(r, nboot=300)
        print(f"{name:38s} {st['N']:6d} {st['median']:7.4f} {st['err_boot']:7.4f} "
              f"{st['p16']:6.3f}-{st['p84']:6.3f} {st.get('core_median', np.nan):8.4f} "
              f"{st['core_frac']:5.2f} {'OK' if st['gate_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
