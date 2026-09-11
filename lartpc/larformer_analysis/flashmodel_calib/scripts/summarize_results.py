"""Markdown table of every result JSON in results/<chain>/ (for the log).

    PYTHONPATH=$K python3 scripts/summarize_results.py [--chain s1ep2p8cew6] [--arm muon]
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
from lartpc.larformer_analysis.flashmodel_calib.gammacal import CALIB_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chain", default="s1ep2p8cew6")
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(CALIB_DIR, "results", args.chain, "*.json"))):
        with open(p) as f:
            j = json.load(f)
        if args.arm and j.get("arm") != args.arm:
            continue
        st = j.get("stats", {})
        rows.append((j["sample"], j["kind"], j["period"], j["arm"], j["N"],
                     j.get("scale_abs"), j.get("scale_err"), st.get("p16"), st.get("p84"),
                     st.get("core_median"), st.get("core_frac"), st.get("gate_ok"),
                     j.get("g_neyman_median"), (j.get("scans") or {}).get("mip_q_per_cm_median"),
                     j.get("production", {}).get("gamma_scale"), os.path.basename(p)))
    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"
    print("| sample | kind | period | arm | N | s = median(obs/pred_ref) | p16-84 | core median | core frac | gates | g_neyman | MIP q/cm | produced at scale | result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {f(r[5], 4)} +- {f(r[6], 4)} | "
              f"{f(r[7])}-{f(r[8])} | {f(r[9])} | {f(r[10], 2)} | {'OK' if r[11] else 'FAIL'} | "
              f"{f(r[12], 4)} | {f(r[13], 1)} | {r[14]} | `{r[15]}` |")


if __name__ == "__main__":
    main()
