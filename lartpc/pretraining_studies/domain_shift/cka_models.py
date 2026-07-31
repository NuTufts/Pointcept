#!/usr/bin/env python3
"""
Cross-MODEL representation comparison (plan figure F6-lite).

Given feature files for the same (diag-set, tier) extracted through
different model snapshots, compute pairwise linear CKA on the pooled
event embeddings, aligned by event name. Answers: does the mixture model
(P5B.1) represent events like the MC-only model (P1A.2), the data-only
model (P1A.3), neither, or between?

Also computes the normalized cross-model gap table from the battery
JSONs: MMD^2 divided by its own permutation-null 95th percentile -- the
scale-free effect size that IS comparable across different embedding
spaces (raw MMD^2 is not).

CPU, seconds. Example:
  python3 cka_models.py \
    --models P5B.1=features/P5B.1_img6144000 P1A.2=features/P1A.2_img6144000 \
             P1A.3=features/P1A.3_img6144000 \
    --tiers mc_cosmic data_raw \
    --results-dir results --out results/cka_models_img6144000.json
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_metrics import cka  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="NAME=feature-prefix entries")
    ap.add_argument("--tiers", nargs="+",
                    default=["mc_cosmic", "data_raw"])
    ap.add_argument("--results-dir", default="results",
                    help="battery JSONs for the normalized-gap table")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def aligned(fa, fb):
    na = {str(n): i for i, n in enumerate(fa["names"])}
    nb = {str(n): i for i, n in enumerate(fb["names"])}
    common = sorted(set(na) & set(nb))
    ia = np.array([na[c] for c in common])
    ib = np.array([nb[c] for c in common])
    return fa["pooled"][ia], fb["pooled"][ib], len(common)


def main():
    args = parse_args()
    models = dict(m.split("=", 1) for m in args.models)
    res = {"meta": {"models": models, "tiers": args.tiers}, "cka": {},
           "normalized_gap": {}}

    for tier in args.tiers:
        feats = {}
        for name, prefix in models.items():
            path = f"{prefix}_{tier}.npz"
            if os.path.exists(path):
                feats[name] = np.load(path)
            else:
                print(f"[warn] missing {path}; {name} skipped for {tier}")
        rows = {}
        for a, b in itertools.combinations(sorted(feats), 2):
            Fa, Fb, n = aligned(feats[a], feats[b])
            v = cka(Fa, Fb)["cka_linear"]
            rows[f"{a}|{b}"] = {"cka_linear": v, "n_events": n}
            print(f"  [{tier}] CKA({a}, {b}) = {v:.4f}  (n={n})")
        res["cka"][tier] = rows

    # normalized gap table from battery JSONs: mmd2 / its perm-null 95th
    for path in sorted(glob.glob(os.path.join(args.results_dir,
                                              "*_tier*.json"))):
        try:
            r = json.load(open(path))
            m = r["mmd"]["mmd2"]["value"]
            n95 = r["mmd"]["mmd2_null_95"]["value"]
            res["normalized_gap"][os.path.basename(path)] = {
                "mmd2": m, "null95": n95,
                "mmd2_over_null95": m / n95 if n95 > 0 else float("nan"),
                "auc_knn": r["pad"]["auc_knn"]["value"],
            }
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {path}: {e}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] -> {args.out}")
    print("\nnormalized gap (MMD^2 / perm-null-95, cross-model comparable):")
    for k, v in sorted(res["normalized_gap"].items()):
        print(f"  {k:45s} {v['mmd2_over_null95']:8.1f}x   "
              f"auc_knn={v['auc_knn']:.4f}")


if __name__ == "__main__":
    main()
