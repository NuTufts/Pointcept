#!/usr/bin/env python3
"""
Unit checks of SBNDFlashModel (consistency of the pieces; physics closure is separate).

    python lartpc/flashmatch/model/tests/test_sbnd_model.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from lartpc.flashmatch.model import ScoreConfig, make_flash_model, normalized_chi2  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def main():
    fm_all = make_flash_model("sbnd", detectors="all", device="cpu")
    fm = make_flash_model("sbnd", device="cpu")                         # PMTs
    di = fm.di
    print(f"SBND flash model: {len(fm.selected)} selected channels of {fm.n_channels} "
          f"(types {fm.detector_types}); v_drift {fm.v_drift_cm_per_us:.5f} cm/us")

    # 1. one point: model == hand calculation
    p = np.array([[-80.0, 30.0, 220.0]])
    q = np.array([1000.0])                                                # ADC area on plane 2
    pe = fm_all.predict(p, q, flash_t0_us=0.0, light_yield=2.0)[0]
    d, r = fm_all.sam.visibility(p)
    e = 1000.0 / fm_all.cal_area[2] * np.exp((fm_all.tpc_width[0] - 80.0) / fm_all.v_drift_cm_per_us
                                             / fm_all.lifetime_us) * fm_all.photons_per_electron
    e = np.float32(e)
    hand = 2.0 * e * (d.numpy()[0] * fm_all.eff_direct.numpy() + r.numpy()[0] * fm_all.eff_reflected.numpy())
    check("single point == SAM x efficiencies x charge model", np.allclose(pe, hand, rtol=1e-6, atol=1e-9),
          f"total PE {pe.sum():.4g}, photons {e:.4g}")
    check("no direct light on the far side of the cathode",
          np.all(d.numpy()[0][fm_all.di.geometry.OpDetPositions()[:, 0] > 0] == 0))

    # 2. PMT-only == PMT columns of the all-sensors model; others zero
    pts = np.random.default_rng(3).uniform([-200, -200, 5], [200, 200, 495], size=(500, 3))
    qq = np.random.default_rng(4).lognormal(5, 1, 500)
    a = fm_all.predict(pts, qq)[0]
    b = fm.predict(pts, qq)[0]
    check("PMT-only == PMT columns of all-sensors", np.allclose(a[fm.selected], b[fm.selected], rtol=1e-6)
          and np.all(b[np.setdiff1d(np.arange(312), fm.selected)] == 0))

    # 3. drift correction sign per TPC
    t = 10.0
    x = fm.drift_correct(np.array([[-100.0, 0, 250], [100.0, 0, 250], [-210.0, 0, 250], [215.0, 0, 250]]), t)[:, 0]
    v = fm.v_drift_cm_per_us
    check("drift: TPC0 x - v t, TPC1 x + v t (outside x -> nearest TPC)",
          np.allclose(x, [-100 - v * t, 100 + v * t, -210 - v * t, 215 + v * t], atol=1e-4), str(np.round(x, 3)))

    # 4. a point outside the AV uses the OUT library
    po = np.array([[-205.0, 20.0, 250.0]])
    pe_o = fm.predict(po, np.array([1000.0]))[0]
    lib = fm.photonlib
    n = fm.photons(np.array([1000.0]), po)[0]
    hand_o = n * (lib.GetVisibility(po)[0] * fm.eff_direct.numpy()
                  + lib.GetVisibility(po, reflected=True)[0] * fm.eff_reflected.numpy())
    hand_o[np.setdiff1d(np.arange(312), fm.selected)] = 0
    check("outside-AV point == library x efficiencies", np.allclose(pe_o, hand_o, rtol=1e-6, atol=1e-9),
          f"total PE {pe_o.sum():.4g}")

    # 5. masks and scores
    m = fm.channel_mask(np.ones(312))
    check("mask = skip list ∪ unselected", set(fm.skip_channels) <= set(m)
          and set(np.setdiff1d(np.arange(312), fm.selected)) <= set(m), f"{len(m)} masked")
    obs = b * 1.1
    c_n = fm.chi2(b, obs, mask=m)
    fmn = make_flash_model("sbnd", device="cpu", score=ScoreConfig(kind="normalized"))
    c_q = fmn.chi2(b, obs, mask=m, tpc=0)
    check("normalized chi2 of a scaled copy ~ 0 (shape-only)", abs(c_q) < 1e-9, f"neyman {c_n:.3g}, normalized {c_q:.3g}")
    nc = normalized_chi2(np.array([1., 2, 3, 0]), np.array([2., 2, 2, 5]), mask=[3])
    hand_q = np.sum((np.array([1/3, 1/3, 1/3]) - np.array([1/6, 2/6, 3/6])) ** 2 /
                    (np.array([1/6, 2/6, 3/6]) + (0.1 * np.array([1/6, 2/6, 3/6])) ** 2)) / 3
    check("normalized chi2 hand example", abs(nc - hand_q) < 1e-12, f"{nc:.6f} vs {hand_q:.6f}")

    nf = sum(not ok for _, ok in RESULTS)
    print(f"{len(RESULTS) - nf}/{len(RESULTS)} checks passed")
    return nf == 0


def test_sbnd_model():
    assert main()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
