"""LArFormer Keypoint v2 — SINGLE-CONFIG full-cascade INFERENCE.

Wires the whole attempt-2 keypoint inference pipeline from ONE config so that
`tools/run_larformer_keypoint2_cascade_inference.py` needs only `--config`
(replacing the old --cascade-config + --keypoint-config + --particle-weights +
--keypoint-weights quartet).

Composition (`CascadedKeypoint`):

    Stage 1  LoRA deghoster          ─┐
    Stage 2  ptv3crosslevel slicer    ├─ CascadedParticleSegmenter   (`cascade`)
    Stage 3  particle segmenter      ─┘
       │  predicted nu-slice + per-particle predicted masks
       ▼
    attempt-2 keypoint model on the nu slice                   (`keypoint_model`)

The cascade model + the raw-merged_h5 test dataset are imported VERBATIM from
    configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py
The keypoint model is imported from the config that produced the trained ckpt
    configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-particle-predmask-cached-v1.py

Note on the keypoint model's `main_source='predicted_cached'`: that knob only
matters during TRAINING (read precomputed predicted masks from the stage-1+2+3
cache). At inference the cascade runs LIVE and feeds the keypoint model the
predicted masks via `particle_instances_per_event`, which
`LArFormer._particle_main_instances` consumes FIRST — so the cached-mask path is
inert here and no extra segmenter is built inside the keypoint model.

Weights:
  * deghoster / slicer / sonata-pretrain — loaded INSIDE the cascade config from
    its own env-overridable defaults (LARFORMER_DEGHOSTER_CKPT, etc.).
  * `particle_weights` (Stage-3 segmenter) and `keypoint_weights` (the keypoint
    model) — loaded by the inference tool AFTER build (the tool reads them from
    this config; CLI flags still override). `particle_weights` points at the
    EXACT Stage-3 checkpoint whose predicted masks built the training cache, for
    train/inference fidelity.
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
                 "larformer-particle-fullcascade-ptv3crosslevel.py"))
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
_REPO = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept"

# Stage-3 particle segmenter — the SAME weights that generated the predicted
# masks cached for keypoint training (resume3_cosinedecay / epoch_12).
particle_weights = os.environ.get("LARFORMER_KP_PARTICLE_CKPT", "").strip() or (
    f"{_REPO}/exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_"
    "resume3_cosinedecay/model/epoch_12.pth")

# Trained attempt-2 keypoint model.
keypoint_weights = os.environ.get("LARFORMER_KP_KEYPOINT_CKPT", "").strip() or (
    "exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth")

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
# The keypoint model trained on a cache rebuilt with a HIGH spacepoint cap; only
# very noisy events hit it. Match that here so the live nu-slice density tracks
# training (see lartpc_data_prep/larformer_keypoint_v2/README.md).
max_spacepoints = 500_000

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
save_path = "exp/larformer_keypoint2_fullcascade_infer"

# Pointcept's Config deep-copies every top-level name; drop modules/Config
# handles + scratch parent-config objects that can't be pickled/deepcopied
# (mirrors the `del os, _env` at the bottom of the cascade config).
del os, pointcept, Config, _casc, _kp, _CFG_DIR, _REPO, _test
