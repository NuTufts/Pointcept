"""Build the calibration record file for one sample (one shard).

    PYTHONPATH=$K python3 scripts/build_records.py --sample extbnb200k_cew6 \
        [--start 0 --n 25000] [--out results/<chain>/records/<tag>.shard0000000.npz]
        [--union-every]   # also predict the full nu-union for every event
                          # (needed for the MC truth-nu arm; ~1 merged_sp read/event)

Runs on CPU (PhotonLib lookup on torch CPU). Shard it with slurm/run_build_records.sh.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))

from lartpc.larformer_analysis.flashmodel_calib.gammacal import CALIB_DIR  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal import samples     # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import build_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", required=True, choices=sorted(samples.SAMPLES))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1, help="-1 = to the end")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-len", type=float, default=30.0,
                    help="loose candidate length floor [cm]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--union-every", action="store_true")
    args = ap.parse_args()
    s = samples.get(args.sample)
    out = args.out or os.path.join(CALIB_DIR, "results", s["chain"], "records",
                                   f"{s['tag']}.shard{args.start:07d}.npz")
    build_records(s, args.start, args.n, out, min_len=args.min_len,
                  device=args.device, union_every=args.union_every)


if __name__ == "__main__":
    main()
