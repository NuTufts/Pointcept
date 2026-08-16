"""CONTROL (2026-08-16): the OLD production chain (LoRA deghoster + OLD
slicer + OLD stage-3) with the deghost threshold loosened to tau=0.20 —
matched-footing baseline for the hybrid (new-slicer) overlay tests.
Only the thresholds change vs the production kp2 config."""

_base_ = ["./larformer-keypoint2-fullcascade.py"]

model = dict(
    cascade=dict(
        cascaded_slicer=dict(
            deghost_threshold_min=0.2,
            deghost_threshold_max=0.2,
            deghost_threshold_val=0.2,
        ),
    ),
)

save_path = "exp/larformer_keypoint2_fullcascade_oldslicer_tau020_infer"
