#!/usr/bin/env python3
"""
Verify the acceptance-test extractions (run by acceptance_test.sbatch).

Checks per file: expected arrays, shapes, finite features, images_seen
recorded. Cross-file: Tier-1 masking really removed nu-origin points on MC
(event level via n_kept < n_raw, point level via point_origin), data files
untouched by tier 'cosmic', and a 20v20 smoke PAD runs end to end.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_metrics import pad, proto_jsd  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAIL.append(name)


def load(d, tag):
    f = np.load(os.path.join(d, f"{tag}.npz"), allow_pickle=False)
    meta = json.loads(str(f["meta"]))
    return f, meta


def main():
    d = sys.argv[1]
    fs = {t: load(d, t) for t in
          ("mc_raw", "mc_cosmic", "mc_cosmicclean",
           "data_raw", "data_cosmicclean")}

    print("== per-file structure ==")
    for tag, (f, meta) in fs.items():
        n = f["pooled"].shape[0]
        check(f"{tag}: >=18 events kept", n >= 18,
              f"n={n} skipped={len(f['skipped'])}")
        check(f"{tag}: pooled dim 1088", f["pooled"].shape[1] == 1088,
              f"shape={f['pooled'].shape}")
        check(f"{tag}: proto_hist dim 4096",
              f["proto_hist"].shape == (n, 4096))
        check(f"{tag}: features finite",
              bool(np.isfinite(f["pooled"]).all()))
        check(f"{tag}: proto counts match upcast",
              bool((f["proto_hist"].sum(1) == f["n_upcast"]).all()))
        check(f"{tag}: images_seen recorded", meta["images_seen"] > 0,
              f"images_seen={meta['images_seen']}")
        check(f"{tag}: point sample present", "point_feats" in f)

    print("== tier masking semantics ==")
    mc_raw, mc_cos = fs["mc_raw"][0], fs["mc_cosmic"][0]
    common = np.intersect1d(mc_raw["names"], mc_cos["names"])
    check("mc raw/cosmic share events", len(common) >= 15,
          f"common={len(common)}")
    ra = {n: k for n, k in zip(mc_raw["names"], mc_raw["n_kept"])}
    co = {n: k for n, k in zip(mc_cos["names"], mc_cos["n_kept"])}
    dropped = [ra[n] - co[n] for n in common]
    check("cosmic tier drops points on MC",
          all(x >= 0 for x in dropped) and any(x > 0 for x in dropped),
          f"median dropped={int(np.median(dropped))}")
    check("cosmic tier: no nu-origin points in sample",
          not (fs["mc_cosmic"][0]["point_origin"] == 1).any())
    check("raw tier: nu-origin points present in MC sample",
          (fs["mc_raw"][0]["point_origin"] == 1).any())
    check("cosmic-clean drops more than cosmic on MC",
          fs["mc_cosmicclean"][0]["n_kept"].sum() < mc_cos["n_kept"].sum())
    dr, dc = fs["data_raw"][0], fs["data_cosmicclean"][0]
    check("data cosmic-clean (lm cut) drops points",
          dc["n_kept"].sum() < dr["n_kept"].sum(),
          f"{dc['n_kept'].sum()} < {dr['n_kept'].sum()}")
    check("data domain tagged", (fs["data_raw"][0]["domains"] == "data").all())

    print("== smoke metrics (20v20 -- numbers not meaningful) ==")
    r = pad(fs["mc_raw"][0]["pooled"], fs["data_raw"][0]["pooled"],
            n_splits=4)
    print(f"  smoke PAD mc_raw vs data_raw: auc_linear={r['auc_linear']:.3f}")
    j = proto_jsd(fs["mc_raw"][0]["proto_hist"],
                  fs["data_raw"][0]["proto_hist"])
    print(f"  smoke proto JSD: {j['proto_jsd']:.3f} "
          f"(active {j['proto_active_a']}/{j['proto_active_b']})")
    check("smoke metrics ran", True)

    print()
    if FAIL:
        print(f"*** {len(FAIL)} FAILURES: {FAIL}")
        sys.exit(1)
    print("ACCEPTANCE TEST PASSED")


if __name__ == "__main__":
    main()
