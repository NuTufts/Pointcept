"""STAGE-3 SEGMENTER RETRAIN, rebalanced cache list v2 (SLICER_RETRAIN_PLAN
2026-09-04): mu/e class-prior fix. Identical to
larformer-particle-s1cache-m2frecipe (the epoch_8 production recipe) except
the train split reads an explicit cache-file list instead of crawling:

  cachelist_rebalanced_train_v2.txt = 281,452 events
    nue-group DOWN-SAMPLED to 40% (deterministic md5 per-file): 97,110
    cpi+ sim 68,059 | pi0filter sim 54,708 | pi0 overlay 40,587
    + NEW generic numu overlay 20,988 (run3b 496 + run1 324 filenos,
      label-completed, nu-deposit>=20pts dirt filter — entering-particle
      events KEPT per user 2026-09-04)

Projected instance balance e:mu 1.96 (was 3.7; e 232k / gamma 269k /
mu 119k). Motivation: segmenter mu->e confusion 10.5%, KE-flat,
high-confidence, clean instances — class-prior/topology imbalance from
the nue-heavy mix (diagnosis chain, SLICER_RETRAIN_PLAN 2026-09-03).
ACCEPTANCE (confusion matrix, primaries KE>50, purity>0.5):
mu->e << 0.105 (target toward larpid's 0.011); e/gamma rows held;
mu->pi low-KE unchanged or better.
"""

_base_ = ["./larformer-particle-s1cache-m2frecipe.py"]

data = dict(
    train=dict(
        data_list_file=("/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/"
                        "kpv2_pointcept/lartpc/data_prep/uboone_official/"
                        "training_data_ledger/cachelist_rebalanced_train_v2.txt"),
    ),
)

save_path = "exp/larformer_particle_s1cache_m2frecipe_rebal_v2"
