"""LArFormer Keypoint v2 — full-cascade INFERENCE on the V2 PRODUCTION CHAIN.

Same single-config wiring as `larformer-keypoint2-fullcascade.py`, but the
cascade is imported from the v2 production overlay
    configs/lartpc/larformer/stage3_particle/larformer-fullcascade-production-v2-tau020.py
which (after `_base_` merge) resolves to:

    Stage 1  PTv3-decoder deghoster (ft full-event, xformers, tau=0.20)
    Stage 2  m2frecipe-v2 slicer epoch_4 (48 queries, refiner 16392)
    Stage 3  particle segmenter — SAME architecture as v1; weights swapped
             below to the m2frecipe retrain's model_best (epoch_8)

The keypoint model itself is UNCHANGED (attempt-2 ckpt trained against the
OLD-slicer predicted-mask cache) — it is expected to degrade on new-chain
slices, which is exactly what the {pred,true}-vertex comparison quantifies
(run_nu_reco.py --true-vertex decouples downstream reco from it). Retraining
stage 4 against the v2 cache is the follow-up that closes this gap.

Deghoster/slicer weights load inside the cascade config (explicit
deghoster_weight / cascaded_slicer_weight paths, strict=False);
`particle_weights` / `keypoint_weights` are read from here by the inference
tool AFTER build (env vars / CLI flags still override).
"""

import os

import pointcept
from pointcept.utils.config import Config

# NOTE: __file__ is unreliable here — Pointcept's Config loader copies this file
# to a temp dir before exec, so __file__ resolves to /tmp/... Locate the sibling
# configs via the installed pointcept package instead (PYTHONPATH=<repo>).
_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(pointcept.__file__))),
    "configs", "lartpc")

# --- pull the two parent configs in as plain dicts ---------------------------
_casc = Config.fromfile(
    os.path.join(_CFG_DIR, "larformer", "stage3_particle",
                 "larformer-fullcascade-production-v2-tau020.py"))
_kp = Config.fromfile(
    os.path.join(_CFG_DIR, "larformer", "stage4_keypoint",
                 "larformer-keypoint2-particle-predmask-cached-v1.py"))

# Fixed normalization the dataset uses for the GT mckeypoints (denormalized with
# THESE, not the recentered per-slice affine). Carried through from the cascade.
coord_center = tuple(_casc.coord_center)
coord_scale = float(_casc.coord_scale)

# =============================================================================
# Weights loaded by the inference tool after build (CLI flags override).
# =============================================================================
_REPO = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

# Stage-3 particle segmenter — the m2frecipe retrain on the v2 tau=0.20 cache
# (model_best = epoch_8, 2026-08-11).
particle_weights = os.environ.get("LARFORMER_KP_PARTICLE_CKPT", "").strip() or (
    f"{_REPO}/exp/larformer_particle_v2_cached_ptv3crosslevel_m2frecipe/"
    "model/model_best.pth")

# Trained attempt-2 keypoint model (OLD-chain ckpt — see docstring).
keypoint_weights = os.environ.get("LARFORMER_KP_KEYPOINT_CKPT", "").strip() or (
    f"{_REPO}/exp/larformer_keypoint2_particle_cachedpredmask_v1/"
    "model/epoch_30.pth")

# =============================================================================
# Model — the full CascadedKeypoint (frozen cascade + keypoint model).
# =============================================================================
particle_source = "predicted"     # production path (live Stage-3 predicted masks)
model = dict(
    type="CascadedKeypoint",
    cascade=dict(_casc.model),
    keypoint_model=dict(_kp.model),
    particle_source=particle_source,
    no_object_class_id=int(_kp.model.get("num_classes", 8)) - 1,
)

# =============================================================================
# Inference dataset — raw per-event merged_h5 (the cascade's test split).
# `data_list_file` is set on the command line by the inference tool.
# =============================================================================
# The v2 production cap (cap-study value: bites ~1.6% of events, guards
# worst-case OOM) — matches the stage-1+2 cache build the v2 stage-3 model
# trained against.
max_spacepoints = 300_000

_test = dict(_casc.data.test)
_test["max_spacepoints"] = max_spacepoints
# Surface MC keypoints so outputs carry GT (matched particle + GT start/end/
# nu-vertex) for the side-by-side visualizer. Sim/overlay only; the tool's
# --no-gt flips these off for real data.
_test["emit_keypoints"] = True
_test["gt_source"] = "particle"

data = dict(
    num_classes=_casc.data.num_classes,
    ignore_index=_casc.data.ignore_index,
    names=list(_casc.data.names),
    test=_test,
)

# =============================================================================
# Misc inference knobs read by the tool.
# =============================================================================
nu_thresh = 0.3                  # dense nu-vertex head decode threshold
save_path = "exp/larformer_keypoint2_fullcascade_v2_infer"

# Pointcept's Config deep-copies every top-level name; drop modules/Config
# handles + scratch parent-config objects that can't be pickled/deepcopied
# (mirrors the `del os, _env` at the bottom of the cascade config).
del os, pointcept, Config, _casc, _kp, _CFG_DIR, _REPO, _test
