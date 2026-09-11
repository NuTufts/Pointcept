"""Calibration sample registry.

One entry per (production, chain). `kind` is what produced the IN-TIME light:
  data : beam-on or beam-off real data (EXT) -- real scintillation light
  mc   : overlay (simulated nu on unbiased-trigger beam-off data) -- the in-time
         flash is the SIMULATED nu light; a coincident data cosmic is rare.
`period` is the MicroBooNE run period (dead_channels.run_period).

`flash_window_us` is the software-trigger / beam-gate window in which the
in-time flash sits for this sample (from the flash-time histogram; see
CALIBRATION_LOG.md). `window_source` says whether it was measured on this
sample or assumed from a sibling.

`calib_only` marks TRAINPOOL samples that must never be used for physics.
"""
import os

from . import REPO_ROOT, CALIB_DIR

_D = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts"
_L = "/cluster/tufts/wongjiradlab/larbys/data/larformer"
_IN = os.path.join(REPO_ROOT, "lartpc", "larformer_reco", "inputlists")
_OUT = os.path.join(REPO_ROOT, "lartpc", "larformer_reco", "outputlists")
_TRUTH_RUN3 = os.path.join(REPO_ROOT, "lartpc", "larformer_reco", "output",
                           "mcc9_bnbnu_overlay_1500_full_satfix", "truth_sidecar")

SAMPLES = {
    "mcoverlay67k_cew6": dict(
        kind="mc", period=3, chain="s1ep2p8cew6",
        description="run-3b BNB nu overlay, 67,211 events (mcc9_v29e)",
        kp2_nu_list=os.path.join(_OUT, "keypoint2_out_mc_overlay_s1ep2p8cew6_run3_nu.txt"),
        nu_reco_dir=f"{_D}/larformer_mcoverlay67k_s1ep2p8cew6/nu_reco_larpid_nu",
        merged_sp_list=os.path.join(_IN, "merged_sp_mc_overlay_s1ep2p8cew6_run3.txt"),
        ntuple=f"{_D}/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root",
        truth_sidecar=_TRUTH_RUN3,
        flash_window_us=(3.6, 5.2), window_source="measured 2026-09-11 (600 ev)",
        calib_only=False),
    "extbnb200k_cew6": dict(
        kind="data", period=3, chain="s1ep2p8cew6",
        description="run-3 EXT-BNB (beam-off) 200k subset (mcc9_v29e)",
        kp2_nu_list=os.path.join(_OUT, "keypoint2_out_extbnb200k_s1ep2p8cew6_nu.txt"),
        nu_reco_dir=f"{_D}/larformer_extbnb200k_s1ep2p8cew6/nu_reco_larpid_nu",
        merged_sp_list=os.path.join(_IN, "merged_sp_extbnb200k_s1ep2p8cew6.txt"),
        ntuple=f"{_D}/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root",
        truth_sidecar=None,
        flash_window_us=(3.2, 5.4), window_source="measured 2026-09-11 (600 ev)",
        calib_only=False),
    "bnb5e19_cew6": dict(
        kind="data", period=1, chain="s1ep2p8cew6",
        description="run-1 BNB beam-on 5e19 (mcc9_v28), 176,302 events",
        kp2_nu_list=os.path.join(_OUT, "keypoint2_out_bnb5e19_s1ep2p8cew6_nu.txt"),
        nu_reco_dir=f"{_D}/larformer_bnb5e19_s1ep2p8cew6/nu_reco_larpid_nu",
        merged_sp_list=os.path.join(_IN, "merged_sp_bnb5e19_s1ep2p8cew6.txt"),
        ntuple=f"{_D}/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root",
        truth_sidecar=None,
        flash_window_us=(2.8, 5.0), window_source="measured 2026-09-11 (600 ev)",
        calib_only=False),
    "nueoverlay79k_cew6": dict(
        kind="mc", period=3, chain="s1ep2p8cew6",
        description="run-3b intrinsic-nue overlay (shower-dominated cross-check)",
        kp2_nu_list=os.path.join(_OUT, "keypoint2_out_nue_overlay_s1ep2p8cew6_run3_nu.txt"),
        nu_reco_dir=f"{_D}/larformer_nueoverlay79k_s1ep2p8cew6/nu_reco_larpid_nu",
        merged_sp_list=os.path.join(_IN, "merged_sp_nue_overlay_s1ep2p8cew6_run3.txt"),
        ntuple=f"{_D}/larformer_nueoverlay79k_s1ep2p8cew6/dlgen2_larformer_ntuple_nue_overlay_s1ep2p8cew6_run3.root",
        truth_sidecar=None,
        flash_window_us=(3.6, 5.2), window_source="assumed = run-3 MC overlay",
        calib_only=False),
    "gammapilot_run1ovl_g100": dict(
        kind="mc", period=1, chain="s1ep2p8cew6",
        description="run-1 BNB nu overlay pilot (mcc9_v28), 12,812 TRAINPOOL events; "
                    "produced at --gamma-run-scale 1.0; CALIBRATION USE ONLY",
        kp2_nu_list=os.path.join(_OUT, "keypoint2_out_gammapilot_run1ovl_g100_nu.txt"),
        nu_reco_dir=f"{_L}/gammapilot_run1ovl_g100/nu_reco_larpid_nu",
        merged_sp_list=os.path.join(_IN, "merged_sp_gammapilot_run1ovl_g100.txt"),
        ntuple=f"{_L}/gammapilot_run1ovl_g100/dlgen2_larformer_ntuple_gammapilot_run1ovl_g100.root",
        truth_sidecar=None,
        flash_window_us=(3.6, 5.2), window_source="measured 2026-09-12 (1500 ev)",
        calib_only=True),
    # ---- flash-calibration stream outputs (cascade --flash-calib-mode) ----
    # kp2 list = every keypoint2_event*_calib_0.h5 under the output dir (build
    # it with scripts/make_calib_list.py); no nu_reco stage.
    "extbnb200k_calib_test300": dict(
        kind="data", period=3, chain="s1ep2p8cew6",
        description="run-3 EXT, 300-event flash-calib-mode test (gidx 30000-30299)",
        kp2_nu_list=os.path.join(CALIB_DIR, "results", "s1ep2p8cew6", "records", "calib_test",
                                 "kp2_ext300_calib.txt"),
        nu_reco_dir=None,
        merged_sp_list=os.path.join(_IN, "merged_sp_extbnb200k_s1ep2p8cew6.txt"),
        ntuple=None, truth_sidecar=None,
        flash_window_us=(3.2, 5.4), window_source="= extbnb200k_cew6",
        calib_only=False),
    "extbnb200k_calib3k": dict(
        kind="data", period=3, chain="s1ep2p8cew6",
        description="run-3 EXT, flash-calib-mode, first 3000 events of the 200k list",
        kp2_nu_list=os.path.join(CALIB_DIR, "results", "s1ep2p8cew6", "records", "calib_kp2",
                                 "kp2_extbnb200k_calib.txt"),
        nu_reco_dir=None,
        merged_sp_list=os.path.join(_IN, "merged_sp_extbnb200k_s1ep2p8cew6.txt"),
        ntuple=None, truth_sidecar=None,
        flash_window_us=(3.2, 5.4), window_source="= extbnb200k_cew6", calib_only=False),
    "bnb5e19_calib3k": dict(
        kind="data", period=1, chain="s1ep2p8cew6",
        description="run-1 bnb5e19, flash-calib-mode, first 3000 events",
        kp2_nu_list=os.path.join(CALIB_DIR, "results", "s1ep2p8cew6", "records", "calib_kp2",
                                 "kp2_bnb5e19_calib.txt"),
        nu_reco_dir=None,
        merged_sp_list=os.path.join(_IN, "merged_sp_bnb5e19_s1ep2p8cew6.txt"),
        ntuple=None, truth_sidecar=None,
        flash_window_us=(2.8, 5.0), window_source="= bnb5e19_cew6", calib_only=False),
    "mcoverlay67k_calib3k": dict(
        kind="mc", period=3, chain="s1ep2p8cew6",
        description="run-3 numu overlay, flash-calib-mode, first 3000 events",
        kp2_nu_list=os.path.join(CALIB_DIR, "results", "s1ep2p8cew6", "records", "calib_kp2",
                                 "kp2_mcoverlay67k_calib.txt"),
        nu_reco_dir=None,
        merged_sp_list=os.path.join(_IN, "merged_sp_mc_overlay_s1ep2p8cew6_run3.txt"),
        ntuple=None, truth_sidecar=_TRUTH_RUN3,
        flash_window_us=(3.6, 5.2), window_source="= mcoverlay67k_cew6", calib_only=False),
    "run1ovl_calib6k": dict(
        kind="mc", period=1, chain="s1ep2p8cew6",
        description="run-1 BNB nu overlay pilot (mcc9_v28, TRAINPOOL), flash-calib-mode, "
                    "first 6000 events; CALIBRATION USE ONLY",
        kp2_nu_list=os.path.join(CALIB_DIR, "results", "s1ep2p8cew6", "records", "calib_kp2",
                                 "kp2_run1ovl_calib.txt"),
        nu_reco_dir=None,
        merged_sp_list=os.path.join(_IN, "merged_sp_gammapilot_run1ovl_g100.txt"),
        ntuple=None, truth_sidecar=None,
        flash_window_us=(3.6, 5.2), window_source="measured 2026-09-12 (1500 ev of the pilot production)",
        calib_only=True),
    # placeholders: fill in when the productions exist
    # "extbnb_run1_cew6":   dict(kind="data", period=1, ...),
    # "bnboverlay_run1_cew6": dict(kind="mc", period=1, ...),
}


def get(tag):
    if tag not in SAMPLES:
        raise KeyError(f"unknown sample {tag!r}; known: {sorted(SAMPLES)}")
    s = dict(SAMPLES[tag])
    s["tag"] = tag
    return s
