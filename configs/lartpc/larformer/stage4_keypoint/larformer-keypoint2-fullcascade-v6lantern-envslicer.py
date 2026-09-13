"""Keypoint2 full-cascade INFERENCE on the HYBRID interim chain:
OLD LoRA deghoster @ tau=0.2 + NEW m2frecipe slicer + NEW stage-3 segmenter
+ OLD attempt-2 keypoint model. See
larformer-fullcascade-v6lantern-envslicer-tau020.py for the rationale.

Identical to larformer-keypoint2-fullcascade-v2.py except the cascade import.
"""

import os

import pointcept
from pointcept.utils.config import Config

_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(pointcept.__file__))),
    "configs", "lartpc")

_casc = Config.fromfile(
    os.path.join(_CFG_DIR, "larformer", "stage3_particle",
                 "larformer-fullcascade-v6lantern-envslicer-tau020.py"))
_kp = Config.fromfile(
    os.path.join(_CFG_DIR, "larformer", "stage4_keypoint",
                 "larformer-keypoint2-particle-predmask-cached-v1.py"))

coord_center = tuple(_casc.coord_center)
coord_scale = float(_casc.coord_scale)

# repo root: env LARFORMER_KPV2_ROOT overrides (portability to other clusters)
_REPO = os.environ.get("LARFORMER_KPV2_ROOT", "").strip() or \
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

particle_weights = os.environ.get("LARFORMER_KP_PARTICLE_CKPT", "").strip() or (
    f"{_REPO}/exp/larformer_particle_v2_cached_ptv3crosslevel_m2frecipe/"
    "model/model_best.pth")

keypoint_weights = os.environ.get("LARFORMER_KP_KEYPOINT_CKPT", "").strip() or (
    f"{_REPO}/exp/larformer_keypoint2_particle_cachedpredmask_v1/"
    "model/epoch_30.pth")

# The keypoint model inherits its Sonata pretrain path from
# larformer-keypoint2-slice-v1.py (hard-coded Tufts old-repo path; the weights
# are overwritten by keypoint_weights at load time but the file must exist).
# Off-site (Polaris) the assets env.sh exports LARFORMER_SONATA_PRETRAIN; honour
# it here like the cascade configs do (found by the Tufts smoke of the Polaris
# scripts, 2026-09-12).
_kp_model = dict(_kp.model)
_sonata = os.environ.get("LARFORMER_SONATA_PRETRAIN", "").strip()
if _sonata:
    _kp_model["backbone_weight"] = _sonata

particle_source = "predicted"
model = dict(
    type="CascadedKeypoint",
    cascade=dict(_casc.model),
    keypoint_model=_kp_model,
    particle_source=particle_source,
    no_object_class_id=int(_kp.model.get("num_classes", 8)) - 1,
)

max_spacepoints = 500_000   # inference: use the old roomy cap (the 300k
                            # training-cache guard cost 5% of photon charge
                            # on overlay events)

_test = dict(_casc.data.test)
_test["max_spacepoints"] = max_spacepoints
_test["emit_keypoints"] = True
_test["gt_source"] = "particle"

data = dict(
    num_classes=_casc.data.num_classes,
    ignore_index=_casc.data.ignore_index,
    names=list(_casc.data.names),
    test=_test,
)

nu_thresh = 0.3
save_path = "exp/larformer_keypoint2_fullcascade_hybrid_infer"

del os, pointcept, Config, _casc, _kp, _CFG_DIR, _REPO, _test, _kp_model, _sonata
