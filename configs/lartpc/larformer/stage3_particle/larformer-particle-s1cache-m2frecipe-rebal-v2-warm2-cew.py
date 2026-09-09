"""Continuation of the rebalanced warm retrain (user decision 2026-09-05):
6 more OneCycle epochs at max_lr 2.5e-5 from rebal_v2_warm epoch_4, now
with per-species CE weights = inverse-sqrt-frequency from the FULL-corpus
census (jobs 3296385; instances e 214.6k / gamma 248.1k / mu 114.7k /
pi 153.6k / p 360.7k over the 281,452-event rebalanced list):

    w = [e 0.95, gamma 0.89, mu 1.30, pi 1.13, p 0.73]  (mean-normalized)

applied through the loss's ce_weights buffer (query CE incl. the
masked-no-object path; no_object stays 0.1). Rationale: warm epochs 1-4
moved mu->mu 0.723->0.815, mu->e 0.164->0.107 on the 2000-event test
subset with gamma held — continue in the same direction; the census says
the residual imbalance is modest, so gentle expectation-level weights +
more epochs. Acceptance unchanged (confusion table on
merged_sp_confusion_test2000.txt).
"""

_base_ = ["./larformer-particle-s1cache-m2frecipe-rebal-v2.py"]

weight = ("/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
          "exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm/model/epoch_4.pth")

epoch = 6
eval_epoch = 6
scheduler = dict(max_lr=2.5e-5)
optimizer = dict(lr=2.5e-5)

model = dict(
    loss_kwargs=dict(
        class_weights=[0.95, 0.89, 1.30, 1.13, 0.73],
    ),
)

save_path = "exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew"
