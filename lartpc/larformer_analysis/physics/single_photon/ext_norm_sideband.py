"""EXT (beam-off) normalization from the pure-cosmic flash-chi2 sideband.

The run-1 EXT spill count is believed ~25-30% low (versions doc section 3c), so
the spill-ratio scale over-predicts cosmics. This measures the scale directly:
in the high-chi2 sideband of the pre-chi2 selection the sample is cosmic-
dominated, so  scale = (data - MC_nu) / EXT_raw  per sideband window.

Run it on the tables written by single_photon_selection.py --out-dir.

    python3 ext_norm_sideband.py --workdir workdir_run1_novtx [--step 3]
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--step", type=int, default=3,
                    help="cut-step index the sideband is measured in (default 3 "
                         "= the step before the flash-chi2 cut)")
    ap.add_argument("--applied-ext-scale", type=float, default=None,
                    help="the --ext-scale the tables were made with; default = "
                         "read back from the EXT table's nonzero weights")
    args = ap.parse_args()
    M = np.load(f"{args.workdir}/mc_table.npz")
    X = np.load(f"{args.workdir}/ext_table.npz")
    D = np.load(f"{args.workdir}/data_table.npz")
    applied = args.applied_ext_scale
    if applied is None:
        nz = X["w"][X["w"] > 0]
        applied = float(np.median(nz)) if len(nz) else 1.0
    w, wx = M["w"], X["w"] / applied
    mm, xx, dd = M["steps"][args.step], X["steps"][args.step], D["steps"][args.step]
    lg = lambda c: np.log10(np.clip(c, 1e-9, None))
    print(f">>> {args.workdir}: step {args.step}, tables built with ext-scale {applied:g}")
    print(f"{'sideband':>16}{'data':>8}{'MC nu':>9}{'EXT raw':>10}{'-> scale':>10}")
    for lo in (3.0, 3.5, 4.0, 4.5):
        b = lambda c, m: m & (c > 0) & (lg(c) > lo)
        da = int(b(D["chi2"], dd).sum())
        mc = w[b(M["chi2"], mm)].sum()
        ex = wx[b(X["chi2"], xx)].sum()
        print(f"{'log10chi2 > ' + str(lo):>16}{da:8d}{mc:9.1f}{ex:10.1f}"
              f"{(da - mc) / max(ex, 1e-9):10.4f}")


if __name__ == "__main__":
    main()
