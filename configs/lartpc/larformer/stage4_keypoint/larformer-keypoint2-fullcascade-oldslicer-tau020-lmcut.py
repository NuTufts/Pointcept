"""CONTROL + lm>=0.15 cut (2026-08-16): old chain at tau=0.2 on
LArMatch-score-attached inputs with the training-parity cut."""

_base_ = ["./larformer-keypoint2-fullcascade-oldslicer-tau020.py"]

data = dict(test=dict(lm_score_val_threshold=0.15))

save_path = "exp/larformer_keypoint2_fullcascade_oldslicer_tau020_lmcut_infer"
