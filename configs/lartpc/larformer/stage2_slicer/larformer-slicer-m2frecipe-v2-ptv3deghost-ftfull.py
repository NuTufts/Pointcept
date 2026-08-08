"""Cascade eval config: ep4 slicer + FULL-EVENT-FINE-TUNED deghoster (v2).

Same as larformer-slicer-m2frecipe-v2-ptv3deghost.py but with the deghoster
weights from the full-event fine-tune (exp/deghost_ptv3decoder_v2_fullevent_ft,
job 2227461). Default tau=0.5 for the pred-mode columns; the saved pre/p_real
is threshold-independent, so the offline tau-sweep runs from the same output.
"""

_base_ = ["./larformer-slicer-m2frecipe-v2-ptv3deghost.py"]

model = dict(
    deghoster_weight=(
        "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
        "exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth"
    ),
)
save_path = "exp/larformer_slicer_m2frecipe_v2_ptv3deghost_ftfull_eval"
