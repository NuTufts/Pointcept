"""Rebalanced stage-3 retrain, WARM START from the production epoch_8
checkpoint (user request 2026-09-04): retain photon reco, fix the muon
class prior. Deltas vs larformer-particle-s1cache-m2frecipe-rebal-v2:
  - weight = s1cache_m2frecipe epoch_8 (CheckpointLoader: model weights
    only, fresh optimizer). The original run had weight=None (Sonata
    backbone init), so this is a true fine-tune.
  - epoch 8 -> 4 and OneCycleLR max_lr 1e-4 -> 2.5e-5: a short, cool
    schedule so the restart peak does not blow away the checkpoint's
    calibration. Everything else (losses, clipping, masked_no_object,
    valprobe) unchanged.
Acceptance unchanged: mu->e << 0.105 on the confusion matrix by epoch
1-2, e/gamma rows held; if mu->e plateaus high (the confident mode is
engraved), fall back to the from-scratch rebal-v2 config.
"""

_base_ = ["./larformer-particle-s1cache-m2frecipe-rebal-v2.py"]

weight = ("/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
          "exp/larformer_particle_s1cache_m2frecipe/model/epoch_8.pth")

epoch = 4
eval_epoch = 4
scheduler = dict(max_lr=2.5e-5)
optimizer = dict(lr=2.5e-5)

save_path = "exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm"
