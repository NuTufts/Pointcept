"""Hybrid cascade with the v6-lantern LoRA deghoster (data-isolation cell,
ep25 — overlay keep == deployed LoRA at matched ga, better in-domain;
DOMAIN_STUDY_RESULTS.md section 24) + the deployed m2frecipe-v2 slicer
(ep4) + new stage-3. tau stays 0.2. Same LoRA architecture as the
deployed deghoster — only the weight path differs from
larformer-fullcascade-hybrid-loradeghost-tau020.py.
"""

_base_ = ["./larformer-fullcascade-hybrid-loradeghost-tau020.py"]

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

model = dict(
    cascaded_slicer=dict(
        deghoster_weight=(
            f"{_KPV2}/sonata/lora_deghost_v6noghosts_lantern/model/"
            "epoch_25.pth"
        ),
    ),
)
