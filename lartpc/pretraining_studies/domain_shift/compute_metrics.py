#!/usr/bin/env python3
"""
Run the full metric battery on a pair of extracted-feature files.

  python3 compute_metrics.py \
      --a features/P5B.1_img6144000_mc-diag1k_cosmic.npz \
      --b features/P5B.1_img6144000_extbnb-diag1k_cosmic.npz \
      --label "P5B.1@6.1M tier=cosmic" \
      --out results/p5b1_cosmic.json

Computes on the pooled event embeddings: PAD (linear+kNN, with bootstrap CI
and same-domain null bands) and MMD^2 (permutation p-value + bootstrap CI),
plus the prototype-occupancy JSD (bootstrap over events). Writes one JSON
with everything needed for the F2/F3 figures; safe to rerun (overwrites).

CPU-only; a 1k-vs-1k pair with default settings takes minutes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import bootstrap_ci, null_split  # noqa: E402
from domain_metrics import mmd2, pad, proto_jsd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help=".npz for side A (e.g. MC)")
    ap.add_argument("--b", required=True, help=".npz for side B (e.g. data)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True, help="output .json")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-pca-mmd", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    fa, fb = np.load(args.a), np.load(args.b)
    Xa, Xb = fa["pooled"], fb["pooled"]
    ha, hb = fa["proto_hist"], fb["proto_hist"]
    meta = {
        "label": args.label,
        "a": {"file": args.a, "n": len(Xa),
              **json.loads(str(fa["meta"]))},
        "b": {"file": args.b, "n": len(Xb),
              **json.loads(str(fb["meta"]))},
        "n_boot": args.n_boot, "n_null": args.n_null,
        "n_perm": args.n_perm, "seed": args.seed,
    }
    print(f"[battery] {args.label}: n_a={len(Xa)} n_b={len(Xb)}")

    res = {"meta": meta}

    print("  PAD + bootstrap ...")
    res["pad"] = bootstrap_ci(pad, Xa, Xb, n_boot=args.n_boot,
                              seed=args.seed)
    print("  PAD null splits (a-vs-a, b-vs-b) ...")
    res["pad_null_a"] = null_split(pad, Xa, n_splits=args.n_null,
                                   seed=args.seed)
    res["pad_null_b"] = null_split(pad, Xb, n_splits=args.n_null,
                                   seed=args.seed)

    print("  MMD + permutation + bootstrap ...")
    # no 'seed' here -- it would collide with bootstrap_ci/null_split's own
    # seed kwarg; mmd2's internal permutation seed stays at its default.
    mmd_kw = dict(n_pca=args.n_pca_mmd, n_perm=args.n_perm)
    res["mmd"] = bootstrap_ci(mmd2, Xa, Xb, n_boot=max(50, args.n_boot // 4),
                              seed=args.seed, **mmd_kw)
    res["mmd_null_a"] = null_split(mmd2, Xa, n_splits=args.n_null,
                                   seed=args.seed, **mmd_kw)

    print("  prototype JSD + bootstrap ...")
    res["proto"] = bootstrap_ci(proto_jsd, ha, hb, n_boot=args.n_boot,
                                seed=args.seed)
    res["proto_null_a"] = null_split(proto_jsd, ha, n_splits=args.n_null,
                                     seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    p = res["pad"]["auc_linear"]
    n = res["pad_null_a"]["auc_linear"]
    print(f"[result] {args.label}")
    print(f"  AUC(linear) = {p['value']:.4f} "
          f"[{p['lo']:.4f}, {p['hi']:.4f}]  "
          f"(null a-vs-a: {n['mean']:.4f} +- {n['std']:.4f})")
    print(f"  PAD(linear) = {res['pad']['pad_linear']['value']:.4f}")
    m = res["mmd"]["mmd2"]
    print(f"  MMD^2       = {m['value']:.5g} [{m['lo']:.5g}, {m['hi']:.5g}] "
          f" p={res['mmd']['mmd2_p']['value']:.4f}")
    j = res["proto"]["proto_jsd"]
    print(f"  proto JSD   = {j['value']:.4f} [{j['lo']:.4f}, {j['hi']:.4f}]"
          f"  excl_a={res['proto']['proto_excl_a']['value']}"
          f"  excl_b={res['proto']['proto_excl_b']['value']}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
