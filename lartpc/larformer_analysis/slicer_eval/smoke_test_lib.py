"""Smoke test for the lib/* modules — runs end-to-end on ONE real event triple.

Picks up the (merged_h5, flashinfo_h5, inference_h5) triple for
fileno00013_entry000004 (the one event present in all three local data
roots) and exercises:

  - lib.categorize.categorize_event           — gracefully returns OTHER
                                                 on legacy flashinfo (no
                                                 event_truth group)
  - lib.flash_chi2.neyman_chi2                — finite scalar
  - lib.flash_chi2.chi2_with_oob              — runs the OOB-threshold
                                                 sweep on a synthetic
                                                 slice
  - lib.flash_predict.predict_slice_pe        — produces (32,) PE vector
                                                 for the in-time-matched
                                                 GT-nu slice via PhotonLib
  - end-to-end: predict PE → compute chi-2 vs observed in-time flash

The test is meant to be run inside the pointcept_cuml container (needs
torch + numpy + h5py + photonlib cache present).

Usage:
  ./run_in_container.sh python lartpc/larformer_analysis/slicer_eval/smoke_test_lib.py
"""

import os
import sys

import numpy as np
import h5py


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from lartpc.larformer_analysis.slicer_eval.lib import (  # noqa: E402
    categorize, flash_chi2, flash_predict,
)


# Single test event triple — confirmed present locally.
MERGED_H5 = (
    "/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/"
    "merged_h5/000/000/"
    "merged_bnb_nu_pi0filter_corsika_fileno00013_entry000004.h5"
)
FLASHINFO_H5 = (
    "/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/"
    "flashinfo_h5/000/000/"
    "flashinfo_bnb_nu_pi0filter_corsika_fileno00013_entry000004.h5"
)
INFERENCE_H5 = (
    f"{REPO_ROOT}/exp/larformer_slicer_v1_cascaded_crosslevelrefiner_"
    f"mixedq_maskdn/inference_iter_18619/"
    f"slicerpred_merged_bnb_nu_pi0filter_corsika_fileno00013_entry000004.h5"
)


def main():
    for label, p in (("merged_h5", MERGED_H5),
                     ("flashinfo_h5", FLASHINFO_H5),
                     ("inference_h5", INFERENCE_H5)):
        if not os.path.exists(p):
            sys.exit(f"missing {label}: {p}")
        print(f"  ok  {label}  {os.path.basename(p)}")

    # ------------------------------------------------------------------
    # 1. categorize — should return OTHER on legacy flashinfo (no
    #    event_truth group). After running the extended prepare_flashinfo
    #    on this file, should change to a real category.
    # ------------------------------------------------------------------
    print("\n=== Test 1: categorize ===")
    with h5py.File(FLASHINFO_H5, "r") as f:
        e0 = f["entry_0"]
        has_truth = "event_truth" in e0
        has_nu_showers = "nu_showers" in e0
        mask = categorize.categorize_event(e0)
        n_vis_g = categorize.count_visible_nu_gammas(e0)
        n_pi0 = categorize.count_primary_pi0(e0)
    print(f"  flashinfo has event_truth: {has_truth}")
    print(f"  flashinfo has nu_showers:  {has_nu_showers}")
    print(f"  n_visible_nu_gammas: {n_vis_g}, n_primary_pi0: {n_pi0}")
    print(f"  category mask = 0b{mask:05b} = {categorize.category_str(mask)}")
    if not has_truth:
        assert mask == categorize.OTHER, \
            f"legacy flashinfo without event_truth should be OTHER; got {mask}"
        print("  legacy-flashinfo falls through to OTHER  CHECK PASSED")
    else:
        print("  flashinfo has truth fields  CHECK PASSED")

    # ------------------------------------------------------------------
    # 2. flash_chi2 — Neyman chi-2 + OOB rejection
    # ------------------------------------------------------------------
    print("\n=== Test 2: flash_chi2 ===")
    pe_pred = np.array([10.0] * 32, dtype=np.float32)
    pe_obs = np.array([12.0] * 32, dtype=np.float32)
    chi2 = flash_chi2.neyman_chi2(pe_pred, pe_obs, f_sys=0.10, eps=1.0)
    # Hand-check: per-PMT term = (12-10)^2 / (12 + 1.44 + 1) = 4/14.44 ≈ 0.277
    # times 32 PMTs ≈ 8.87
    print(f"  uniform offset chi-2 = {chi2:.4f}  (expected ~8.87)")
    assert 8.0 < chi2 < 10.0
    # Synthetic OOB test: put 3 of 10 SPs outside TPC bounds; sweep thresholds.
    pos = np.array([
        [128.0,    0.0,  500.0],   # in-FV
        [130.0,   50.0,  600.0],   # in-FV
        [200.0,  -50.0,  700.0],   # in-FV
        [128.0,    0.0,  800.0],   # in-FV
        [128.0,    0.0,  900.0],   # in-FV
        [128.0,    0.0, 1000.0],   # in-FV
        [128.0,    0.0,  100.0],   # in-FV
        [-50.0,    0.0,  500.0],   # OOB (x<0)
        [300.0,    0.0,  500.0],   # OOB (x>256)
        [128.0,  200.0,  500.0],   # OOB (y>117)
    ], dtype=np.float32)
    pe_pred_slice = np.full(32, 5.0, dtype=np.float32)
    pe_obs_slice  = np.full(32, 6.0, dtype=np.float32)
    out = flash_chi2.chi2_with_oob(
        pos_cm=pos, pe_pred=pe_pred_slice, pe_obs=pe_obs_slice,
        flash_t0_us=0.0,
        oob_thresholds=np.array([0.0, 0.05, 0.20, 0.50], dtype=np.float32),
    )
    print(f"  oob_frac = {out['oob_frac']:.2f} ({out['n_oob']}/{out['n_sp']})")
    print(f"  chi2 sweep = {out['chi2']}  "
          f"(expected: NaN where oob_frac > threshold)")
    assert out["oob_frac"] == 0.3
    assert np.isnan(out["chi2"][0])   # threshold 0.0 → reject
    assert np.isnan(out["chi2"][1])   # threshold 0.05 → reject
    assert np.isnan(out["chi2"][2])   # threshold 0.20 → reject
    assert not np.isnan(out["chi2"][3])  # threshold 0.50 → keep
    print("  OOB-sweep behavior  CHECK PASSED")

    # ------------------------------------------------------------------
    # 3. flash_predict — load photonlib, compute PE for the in-time
    #    GT-nu slice. Needs torch + the photonlib cache.
    # ------------------------------------------------------------------
    print("\n=== Test 3: flash_predict (PhotonLib) ===")
    try:
        import torch                                       # noqa: F401
    except Exception as exc:
        print(f"  SKIP (torch unavailable): {exc}")
        return
    if not os.path.exists(flash_predict.PHOTONLIB_DEFAULT_CACHE):
        print(f"  SKIP (no cache): {flash_predict.PHOTONLIB_DEFAULT_CACHE}")
        return

    # Identify the in-time GT nu slice from flashinfo:
    #   - filter slice_flash_matches to primary_origin==1 (nu)
    #   - pick the one with matched_flash_idx >= 0 closest to t=0
    with h5py.File(FLASHINFO_H5, "r") as f:
        e0 = f["entry_0"]
        sl = e0["slice_flash_matches"]
        slice_id          = sl["slice_id"][:]
        primary_origin    = sl["primary_origin"][:]
        matched_flash_idx = sl["matched_flash_idx"][:]
        fl = e0["flashes"]
        pe_obs_all  = fl["pe"][:]
        time_us_all = fl["time_us"][:]
        producer_id_all = fl["producer_id"][:]
        print(f"  n_slices={len(slice_id)}, n_flashes={len(time_us_all)}")
        nu_mask = (primary_origin == 1) & (matched_flash_idx >= 0)
        if not nu_mask.any():
            print("  SKIP: no GT-nu slice matched to a flash in this event")
            return
        nu_slice_idx = int(np.flatnonzero(nu_mask)[0])
        in_time_flash_idx = int(matched_flash_idx[nu_slice_idx])
        nu_primary_trackid = int(slice_id[nu_slice_idx])
        flash_t0_us = float(time_us_all[in_time_flash_idx])
        pe_obs = pe_obs_all[in_time_flash_idx].astype(np.float32)
        producer_id = int(producer_id_all[in_time_flash_idx])
        print(f"  in-time flash: idx={in_time_flash_idx}, "
              f"t0={flash_t0_us:.3f} us, producer={producer_id}, "
              f"total_PE={float(pe_obs.sum()):.1f}")

    # Pull the GT nu slice's SPs from the merged H5. The merged H5's
    # triplet_data has per-SP (x, y, z) AND pixval, and shower_fragments
    # (or compute_slice_labels) tells us which SPs belong to the GT
    # nu primary's slice. For the smoke test we'll use a simpler proxy:
    # the inference H5's `post/slice_id_gt` field directly gives the
    # GT slice id per surviving SP.
    with h5py.File(INFERENCE_H5, "r") as f:
        post_coord = f["post/coord"][:]                # (N, 3) cm
        post_slice_id_gt = f["post/slice_id_gt"][:]    # GT slice ids
        # pre/pixval has 3 channels (U, V, Y); pre/keep maps to post.
        pre_pixval = f["pre/pixval"][:]                # (N_pre, 3)
        pre_keep = f["pre/keep"][:]                    # bool (N_pre,)
        post_pixval = pre_pixval[pre_keep]
        print(f"  n_post={len(post_coord)}, n_pre={len(pre_pixval)}, "
              f"n_keep={int(pre_keep.sum())}")
    sp_mask = (post_slice_id_gt == nu_primary_trackid)
    n_nu_sp = int(sp_mask.sum())
    print(f"  GT-nu slice: trackid={nu_primary_trackid}, n_post_SPs={n_nu_sp}")
    if n_nu_sp == 0:
        print("  SKIP: GT-nu slice has 0 SPs after deghosting")
        return

    pos_nu  = post_coord[sp_mask].astype(np.float32)
    px_nu   = post_pixval[sp_mask].astype(np.float32)
    charge  = flash_predict.select_charge_y_with_uv_fallback_np(px_nu)
    pe_pred = flash_predict.predict_slice_pe(
        pos_cm=pos_nu, charge=charge,
        flash_t0_us=flash_t0_us, producer_id=producer_id,
        gamma_by_producer=(1.0, 1.0),
    )
    print(f"  PhotonLib PE (sum) = {float(pe_pred.sum()):.1f}, "
          f"max-PMT = {float(pe_pred.max()):.1f}")
    assert pe_pred.shape == (32,)
    assert np.isfinite(pe_pred).all()

    # End-to-end chi-2 (with OOB sweep).
    out = flash_chi2.chi2_with_oob(
        pos_cm=pos_nu, pe_pred=pe_pred, pe_obs=pe_obs,
        flash_t0_us=flash_t0_us,
        oob_thresholds=np.array([0.0, 0.05, 0.10, 0.20, 0.50], dtype=np.float32),
    )
    print(f"  OOB-frac = {out['oob_frac']:.3f}  "
          f"(n_oob={out['n_oob']}/{out['n_sp']})")
    print(f"  chi-2 (Neyman) sweep = {out['chi2']}")
    if out["oob_frac"] < 0.5:
        assert not np.isnan(out["chi2"][-1])
    print("  predict_slice_pe + chi2_with_oob  CHECK PASSED")

    print("\nPass-1 lib smoke test PASSED.")


if __name__ == "__main__":
    main()
