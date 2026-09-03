"""STAGE-3 SEGMENTER RETRAIN on the S1 cache (SLICER_RETRAIN_PLAN,
2026-08-22): the m2frecipe recipe unchanged, pointed at the cache built
with v6-lantern deghoster + S1-ep2 slicer over the MIX v1 corpus
(186.5k overlay + 219.8k LANTERN enriched, label-completed;
verification PASSED 2826660).

Deltas vs larformer-particle-v2-cached-...-m2frecipe.py:
  - CACHE_ROOT -> s1ep2_v6lantern_tau020 cache (val split = the 1500
    completed-copy files: like-for-like labels for the valprobe).
  - masked_no_object=True: overlay events' cached slices contain real
    cosmic-contamination points with NO GT instance — exclude unmatched
    queries concentrated on unlabeled points from the no-object CE
    (enrichment-bar criterion; same machinery as the S1 slicer).
  - mask_denoising min_gt_instances=2 (partial-truth DN guard).
Success criterion (full-chain battery): overlay mu-ID 83% -> toward the
old stage-3's 99%, at preserved photon-lane performance.
"""

_base_ = ["./larformer-particle-v2-cached-ptv3crosslevel-m2frecipe.py"]

CACHE_ROOT = ("/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/"
              "larformer_cache_stage12__s1ep2_v6lantern_tau020/")
TRAIN_ROOT = f"{CACHE_ROOT}/train"
VAL_ROOT = f"{CACHE_ROOT}/val"

data = dict(
    train=dict(data_root=TRAIN_ROOT),
    val=dict(data_root=VAL_ROOT),
    test=dict(data_root=VAL_ROOT),
)

model = dict(
    loss_kwargs=dict(
        masked_no_object=True,
    ),
    mask_denoising=dict(
        min_gt_instances=2,
    ),
)

save_path = "exp/larformer_particle_s1cache_m2frecipe"
