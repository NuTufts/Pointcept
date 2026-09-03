"""PINNED cascade for the S1 stage-3 cache build (SLICER_RETRAIN_PLAN,
2026-08-21): v6-lantern deghoster (ep25) + S1 mix-enriched slicer at
EPOCH 2 (pilot checkpoint; C1-selected final checkpoint gets its own
cache after ep5). tau inherited (0.2-era cascade settings).

Pinned file (no env vars) so the cache provenance is unambiguous."""

_base_ = ["./larformer-fullcascade-v6lantern-tau020.py"]

_KPV2 = "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept"

model = dict(
    cascaded_slicer_weight=(
        f"{_KPV2}/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth"
    ),
)
