"""LArFormer Keypoint v2 — Phase 2: slice model + per-particle query decoder.

Inherits the Phase-1 slice config (`larformer-keypoint2-slice-v1.py`: frozen
encoder + trainable PTv3 decoder + dec2 "object" / spacepoint "nu_vertex" dense
heads) and turns ON the per-particle keypoint query decoder
(`enable_particle_keypoint`). For each GT particle instance the decoder predicts
typed start/end keypoints, seeded from the dec2 object head, supervised by the
instance's origin/end (Hungarian + CE + smooth-L1). The evaluator additionally
logs per-particle start/end localization (val/pkp_*).

Phase 2b (noised-GT denoising + predicted-mask matching) is a follow-up.
"""

_base_ = ["./larformer-keypoint2-slice-v1.py"]

# Turn on the per-particle decoder (merges into the inherited model dict).
model = dict(
    enable_particle_keypoint=True,
    particle_keypoint_cfg=dict(
        num_queries=8,
        num_layers=3,
        num_heads=4,
        mlp_ratio=4.0,
        pos_emb_kind="sinusoidal",
        # seed queries from the slice-level dec2 object head (top-N tokens by
        # object score among the particle's points).
        seed_level="ptv3_dec2",
        seed_head="object",
        n_seed_queries=4,
        weight_class=1.0,
        weight_pos=5.0,
        coord_scale=179.55,
        # Phase 2b denoising path: DN queries seeded at jittered GT keypoints
        # (class-embedding content, KNOWN assignment) over a drop/add-noised
        # point set — robustness to imperfect masks + the bridge to the
        # predicted-mask path (Phase 3). Noise is resampled per step.
        denoise=dict(
            enable=True, n_groups=3, anchor_jitter_std=0.03,
            drop_frac=0.2, add_frac=0.2, weight=1.0),
    ),
)

save_path = "exp/larformer_keypoint2_particle_v1"
