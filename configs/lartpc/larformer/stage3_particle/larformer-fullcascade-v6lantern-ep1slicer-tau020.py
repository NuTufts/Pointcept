"""Early-checkpoint (UNDERFIT) slicer probe: v6-lantern deghoster + the
m2frecipe-v2 slicer at EPOCH 1 (of 4; OneCycleLR still ramping).
Mechanism test (fit-sharpness/underfit-robustness): if the ep1 slicer's
overlay in-slice photon charge beats ep4's while val is lower, the v2
recipe's better optimization is trading transfer for in-domain fit.
"""

_base_ = ["./larformer-fullcascade-v6lantern-tau020.py"]

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

model = dict(
    cascaded_slicer_weight=(
        f"{_KPV2}/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_"
        "cap300k_m2frecipe_v2/model/epoch_1.pth"
    ),
)
