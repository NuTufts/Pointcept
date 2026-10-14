#!/usr/bin/env python3
"""
UBooNEFlashModel must reproduce the legacy MicroBooNE flash code bit-for-bit.

Compared with np.array_equal (no tolerance):
  predict_legacy / predict_slice   vs flash_predict.predict_many_slices_pe / predict_slice_pe
  drift_correct / oob_mask         vs flash_chi2.drift_correct / oob_mask
  chi2 (neyman)                    vs flash_chi2.neyman_chi2
  channel_mask                     vs dead_channels.resolve_dead_opdets ∪ saturation.find_saturated
on synthetic slices and on real MicroBooNE events (merged_sp files: true-charge
points and the observed in-time flash).

    python lartpc/flashmatch/model/tests/test_uboone_regression.py [--files N]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from lartpc.flashmatch import dead_channels, flash_chi2, flash_predict, saturation  # noqa: E402
from lartpc.flashmatch.model import make_flash_model  # noqa: E402

# run-1 MC overlay (truth points) and run-3 EXT data (no truth; dead PMT 15, saturation)
MERGED_SP_GLOBS = (
    "/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_bnboverlay_half/merged_sp/000/*.h5",
    "/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9_v29e_dl_run3_G1_extbnb/merged_sp/000/00/*.h5",
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def synthetic(fm, rng):
    lo = np.array([flash_chi2.TPC_X_MIN, flash_chi2.TPC_Y_MIN, flash_chi2.TPC_Z_MIN])
    hi = np.array([flash_chi2.TPC_X_MAX, flash_chi2.TPC_Y_MAX, flash_chi2.TPC_Z_MAX])
    n = 20000
    pos = rng.uniform(lo - 20, hi + 20, size=(n, 3)).astype(np.float32)   # some outside the TPC
    q = rng.lognormal(3, 1, size=n).astype(np.float32)
    S = 7
    cid = rng.integers(0, S, size=n)
    for t0 in (0.0, 4.21875, -812.5, 1530.0):
        for pid in (0, 1):
            g = (5.25 * 0.8576, 5.25 * 0.5449)
            ref = flash_predict.predict_many_slices_pe(pos, q, cid, S, t0, producer_id=pid,
                                                       gamma_by_producer=g)
            got = fm.predict_legacy(pos, q, cid, S, t0, producer_id=pid, gamma_by_producer=g)
            check(f"predict_many_slices_pe t0={t0} pid={pid}", np.array_equal(ref, got),
                  f"max |diff| {np.max(np.abs(ref - got)):.3g}, sum PE {ref.sum():.4g}")
        ref1 = flash_predict.predict_slice_pe(pos[:500], q[:500], t0, 0, gamma_by_producer=(5.25, 5.25))
        got1 = fm.predict_slice(pos[:500], q[:500], t0, light_yield=fm.light_yield((5.25, 5.25), 0))
        check(f"predict_slice_pe t0={t0}", np.array_equal(ref1, got1))
        check(f"drift_correct t0={t0}", np.array_equal(flash_chi2.drift_correct(pos, t0),
                                                       fm.drift_correct(pos, t0)))
        pc = flash_chi2.drift_correct(pos, t0)
        check(f"oob_mask t0={t0}", np.array_equal(flash_chi2.oob_mask(pc), fm.oob_mask(pc)),
              f"oob frac {fm.oob_fraction(pos, t0):.3f}")
    check("empty slice", np.array_equal(fm.predict_legacy(pos[:0], q[:0], cid[:0], 3, 0.0),
                                        flash_predict.predict_many_slices_pe(pos[:0], q[:0], cid[:0], 3, 0.0, 0)))


def real_events(fm, files):
    import h5py
    n_ev = 0
    for path in files:
        with h5py.File(path, "r") as f:
            e = f["entry_0"]
            if "flashes" not in e:
                continue
            run = int(e.attrs.get("run", 0))
            td = e["triplet_data"]
            hm = (td["hasmatch"][()] == 1) if "hasmatch" in td else np.zeros(td["pos"].shape[0], bool)
            if hm.sum() >= 50:                       # MC: true points, clusters = trackid
                sel = np.nonzero(hm)[0]
                _, cid = np.unique(td["trackid"][()][sel], return_inverse=True)
            else:                                    # data: charged triplets, clusters = z bins
                pv = td["pixval"][()]
                sel = np.nonzero(pv[:, 2] > 10)[0][:20000]
                if sel.size < 50:
                    continue
                cid = np.clip((td["pos"][()][sel, 2] // 100).astype(np.int64), 0, 10)
                _, cid = np.unique(cid, return_inverse=True)
            pos = td["pos"][()][sel].astype(np.float32)
            q = flash_predict.select_charge_y_with_uv_fallback_np(td["pixval"][()][sel])
            S = int(cid.max()) + 1
            fl = e["flashes"]
            beam = np.nonzero(fl["producer_id"][()] == 0)[0]
            if beam.size == 0:
                continue
            bi = beam[np.argmax(fl["total_pe"][()][beam])]
            obs = fl["pe"][()][bi]
            t0 = float(fl["time_us"][()][bi])
        g = 5.25 * fm.gamma_scale("auto", run)
        ref = flash_predict.predict_many_slices_pe(pos, q, cid, S, t0, 0, gamma_by_producer=(g, g))
        got = fm.predict_legacy(pos, q, cid, S, t0, 0, gamma_by_producer=(g, g))
        dead = dead_channels.resolve_dead_opdets("auto", run)
        mask_ref = tuple(sorted(set(dead) | set(saturation.find_saturated(obs, dead=dead, max_masked=4))))
        mask = fm.channel_mask(obs, run)
        chi_ref = [flash_chi2.neyman_chi2(ref[i], obs, dead_opdets=mask_ref) for i in range(S)]
        chi = [fm.chi2(got[i], obs, mask=mask) for i in range(S)]
        ok = np.array_equal(ref, got) and mask == mask_ref and chi == chi_ref
        check(f"real event {os.path.basename(path)} (run {run}, {len(pos)} pts, {S} clusters, t0 {t0:.2f})",
              ok, f"mask {mask}, best chi2 {min(chi):.4g}")
        n_ev += 1
    return n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--glob", action="append", default=None, help="merged_sp file glob(s)")
    a = ap.parse_args()
    fm = make_flash_model("uboone")
    print(f"UBooNEFlashModel vs legacy flash code (photon library on {fm.device})")
    synthetic(fm, np.random.default_rng(7))
    files = [f for g in (a.glob or MERGED_SP_GLOBS) for f in sorted(glob.glob(g))[: a.files]]
    n = real_events(fm, files)
    print(f"real events checked: {n}")
    nfail = sum(not ok for _, ok in RESULTS)
    print(f"{len(RESULTS) - nfail}/{len(RESULTS)} checks passed")
    sys.exit(1 if nfail else 0)


def test_uboone_regression():
    fm = make_flash_model("uboone")
    synthetic(fm, np.random.default_rng(7))
    real_events(fm, [f for g in MERGED_SP_GLOBS for f in sorted(glob.glob(g))[:4]])
    assert all(ok for _, ok in RESULTS), [r for r in RESULTS if not r[1]]


if __name__ == "__main__":
    main()
