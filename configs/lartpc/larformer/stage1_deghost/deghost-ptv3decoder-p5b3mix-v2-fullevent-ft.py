"""Full-event fine-tune of the P5B.3-mix-encoder deghoster — stage 2 of the
encoder-swap A/B (see deghost-ptv3decoder-p5b3mix-v1.py). Identical recipe
to deghost-ptv3decoder-v2-fullevent-ft.py; only the warm-start checkpoint
(the p5b3mix crop stage's model_best) and save_path differ.

Do NOT launch until the crop stage has finished and its model_best exists.
"""

_base_ = ["./deghost-ptv3decoder-v2-fullevent-ft.py"]

weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
    "exp/deghost_ptv3decoder_p5b3mix_v1/model/model_best.pth"
)

save_path = "exp/deghost_ptv3decoder_p5b3mix_v2_fullevent_ft"
