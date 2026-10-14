#!/usr/bin/env python3
"""
Python semi-analytic model vs LArSoft (larsim SemiAnalyticalModel compiled from
source at extraction, /validation_optical in the detinfo file).

    python lartpc/flashmatch/model/tests/test_semianalytic.py [--detector sbnd] [--bench]
    pytest lartpc/flashmatch/model/tests/test_semianalytic.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from lartpc.detinfo import load  # noqa: E402
from lartpc.detinfo.io import read_table  # noqa: E402
from lartpc.flashmatch.model.visibility import SemiAnalyticVisibility  # noqa: E402

RTOL, ATOL = 1e-6, 1e-9


def reference(di):
    import h5py
    with h5py.File(di.path, "r") as f:
        return read_table(f["validation_optical"])


def compare(name, got, ref, verbose=True):
    got = np.asarray(got, dtype=np.float64)
    nan_ok = np.array_equal(np.isnan(got), np.isnan(ref))
    m = ~np.isnan(ref)
    err = np.abs(got[m] - ref[m])
    ok = nan_ok and bool(np.all(err <= ATOL + RTOL * np.abs(ref[m])))
    worst = np.argmax(err / (ATOL + RTOL * np.abs(ref[m]))) if err.size else 0
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max |diff| {err.max():.2e}, "
              f"max rel {np.max(err / np.maximum(np.abs(ref[m]), 1e-300)):.2e} "
              f"(ref max {np.abs(ref[m]).max():.3g}); NaN pattern {'equal' if nan_ok else 'DIFFERENT'}"
              f"; nonzero ref {np.count_nonzero(ref[m])}/{m.sum()}")
        if not ok:
            idx = np.argwhere(m)[worst]
            print(f"      worst at (point, opdet) {tuple(idx)}: got {got[tuple(idx)]:.8g} ref {ref[tuple(idx)]:.8g}")
    return ok


def run(di, verbose=True):
    R = reference(di)
    sam = SemiAnalyticVisibility.from_detinfo(di)
    d, r = sam.visibility(R["point"])
    ok = compare("direct (all opdets)", d.numpy(), R["direct"], verbose)
    ok &= compare("reflected (all opdets)", r.numpy(), R["reflected"], verbose)
    # subset (PMTs only) must equal the corresponding columns
    types = di.conventions.opdet_types()
    pmts = np.nonzero(np.array(["pmt" in t for t in types]))[0]
    sub = SemiAnalyticVisibility.from_detinfo(di, opdets=pmts)
    ds, rs = sub.visibility(R["point"])
    ok &= compare(f"direct (PMT subset, {len(pmts)})", ds.numpy(), R["direct"][:, pmts], verbose)
    ok &= compare(f"reflected (PMT subset)", rs.numpy(), R["reflected"][:, pmts], verbose)
    # float32: report only
    s32 = SemiAnalyticVisibility.from_detinfo(di, dtype=torch.float32)
    d32, r32 = s32.visibility(R["point"])
    m = ~np.isnan(R["reflected"])
    if verbose:
        print(f"  [info] float32: direct max rel {np.max(np.abs(d32.numpy() - R['direct']) / np.maximum(R['direct'], 1e-12)):.2e}"
              f" (where ref > 1e-12), reflected max |diff| {np.max(np.abs(r32.numpy()[m] - R['reflected'][m])):.2e}")
    return ok


def bench(di, n=100_000, device="cpu"):
    types = di.conventions.opdet_types()
    pmts = np.nonzero(np.array(["pmt" in t for t in types]))[0]
    a = di.geometry.GetTPC(0).active_bbox
    rng = np.random.default_rng(1)
    pts = rng.uniform(a.min, a.max, size=(n, 3))
    for dtype in (torch.float64, torch.float32):
        for ids, lab in ((pmts, "PMTs"), (None, "all")):
            s = SemiAnalyticVisibility.from_detinfo(di, opdets=ids, dtype=dtype, device=device)
            s.visibility(pts[:1000])                       # warm-up
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.time(); s.visibility(pts)
            if device != "cpu":
                torch.cuda.synchronize()
            dt = time.time() - t0
            D = len(ids) if ids is not None else di.geometry.NOpDets()
            print(f"  [bench] {n} points x {D} opdets ({lab}), {str(dtype)[6:]}, {torch.get_num_threads()} threads: "
                  f"{dt:.2f} s ({dt / n * 1e6:.1f} us/point, direct+reflected) on {device}")


def test_semianalytic_matches_larsoft():
    assert run(load(os.environ.get("DETINFO_TEST_DETECTOR", "sbnd")), verbose=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="sbnd")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-n", type=int, default=100_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    di = load(a.detector)
    print(f"semi-analytic vs LArSoft reference: {di.path}")
    ok = run(di)
    if a.bench:
        bench(di, n=a.bench_n, device=a.device)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
