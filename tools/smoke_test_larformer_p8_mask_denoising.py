"""Phase-B smoke test: Mask DINO-style mask denoising.

Verifies the four new pieces in isolation (no backbone, no real data):

  1. MaskDenoiser direct construction:
       - dn_groups × n_gt expansion with correct gt_target_idx / group_id
       - class-embedding content init
       - Gaussian anchor jitter around origin_coord_norm
       - max_dn_per_event cap
       - empty (n_gt == 0) returns empty DNInit cleanly
       - no_object-class GTs are filtered out

  2. build_self_attn_mask structure:
       - regular ↔ regular allowed
       - regular ↔ DN blocked (both directions)
       - DN cross-group blocked, within-group allowed

  3. Mask2FormerDecoder with concatenated [regular | DN] queries:
       - forward + backward run end-to-end, finite gradients
       - self_attn mask is actually used (compare output with vs without)

  4. LArFormerLoss.compute_dn_loss:
       - returns finite scalar total
       - backward through the DN loss reaches the decoder's heads

Run inside the pointcept container:
  ./run_in_container.sh python tools/smoke_test_larformer_p8_mask_denoising.py
"""

import os
import sys
from collections import OrderedDict

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    sys.path.insert(0, REPO_ROOT)

    from pointcept.models.LArFormer.builders.base import LevelOutput
    from pointcept.models.LArFormer.decoder import Mask2FormerDecoder
    from pointcept.models.LArFormer.losses import LArFormerLoss
    from pointcept.models.LArFormer.query_denoising import (
        DNInit, MaskDenoiser,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    DIM = 32
    NUM_CLASSES = 3
    NUM_REG = 8
    DN_GROUPS = 3
    SRC = "voxel_8cm"
    N_TOKENS = 60
    DEVICE = torch.device("cpu")  # CPU is fine for the smoke test

    # ------------------------------------------------------------------
    # 1. MaskDenoiser
    # ------------------------------------------------------------------
    print("=== Test 1: MaskDenoiser ===")

    denoiser = MaskDenoiser(
        token_dim=DIM, num_classes=NUM_CLASSES,
        dn_groups=DN_GROUPS, max_dn_per_event=100,
        anchor_jitter_std=0.05,
    )

    # 5 fake GT instances at distinct origins, mixed classes 0/1
    # (no_object class = NUM_CLASSES - 1 = 2 is never used as a GT label
    # in the normal path — but we also test below that any leaks are
    # filtered).
    gt_instances = [
        {"origin_type": 0, "origin_coord_norm": np.array([-0.7, -0.7, -0.7], np.float32)},
        {"origin_type": 1, "origin_coord_norm": np.array([+0.7, +0.7, +0.7], np.float32)},
        {"origin_type": 0, "origin_coord_norm": np.array([-0.5, +0.5, +0.0], np.float32)},
        {"origin_type": 1, "origin_coord_norm": np.array([+0.3, -0.3, +0.4], np.float32)},
        {"origin_type": 0, "origin_coord_norm": np.array([+0.0, +0.0, -0.8], np.float32)},
    ]
    K = len(gt_instances)
    dn_init = denoiser(gt_instances=gt_instances, device=DEVICE, dtype=torch.float32)
    assert isinstance(dn_init, DNInit)
    expected_Q = K * DN_GROUPS
    assert dn_init.n_queries == expected_Q
    assert dn_init.init_q.shape == (expected_Q, DIM)
    assert dn_init.init_anchor.shape == (expected_Q, 3)
    # gt_target_idx should be arange(K) repeated DN_GROUPS times.
    expected_target = torch.arange(K).repeat(DN_GROUPS)
    assert torch.equal(dn_init.gt_target_idx, expected_target)
    expected_group = torch.arange(DN_GROUPS).repeat_interleave(K)
    assert torch.equal(dn_init.group_id, expected_group)
    # Anchor should be close to (but distinct from) gt origin per query
    # — within ~4σ of the GT centroid.
    gt_origins = torch.tensor(
        np.stack([g["origin_coord_norm"] for g in gt_instances], axis=0),
        dtype=torch.float32,
    )
    anchor_err = (dn_init.init_anchor - gt_origins[dn_init.gt_target_idx]).abs()
    assert anchor_err.max() < 4 * 0.05, \
        f"anchor jitter > 4σ: max {anchor_err.max().item():.4f}"
    # ...but NOT identical (otherwise denoising target == input position).
    assert anchor_err.max() > 1e-6
    print(f"  full Q_dn={dn_init.n_queries} (K={K} × G={DN_GROUPS}), "
          f"anchor max-jitter {anchor_err.max().item():.4f}  CHECK PASSED")

    # --- 1a. Cap test
    denoiser_cap = MaskDenoiser(
        token_dim=DIM, num_classes=NUM_CLASSES,
        dn_groups=DN_GROUPS, max_dn_per_event=10,
    )
    dn_cap = denoiser_cap(gt_instances, DEVICE, torch.float32)
    assert dn_cap.n_queries == 10
    assert dn_cap.init_q.shape == (10, DIM)
    print(f"  cap test: max_dn_per_event=10 → Q_dn={dn_cap.n_queries}  CHECK PASSED")

    # --- 1b. Empty
    dn_empty = denoiser([], DEVICE, torch.float32)
    assert dn_empty.n_queries == 0
    assert dn_empty.init_q.shape == (0, DIM)
    print(f"  empty event: Q_dn=0  CHECK PASSED")

    # --- 1c. no_object filter
    bad = [
        {"origin_type": 0, "origin_coord_norm": np.zeros(3, np.float32)},
        {"origin_type": NUM_CLASSES - 1,  # no_object
         "origin_coord_norm": np.zeros(3, np.float32)},
    ]
    dn_bad = denoiser(bad, DEVICE, torch.float32)
    # Only the class-0 GT should survive → DN_GROUPS queries total.
    assert dn_bad.n_queries == DN_GROUPS, dn_bad.n_queries
    print(f"  no_object class filtered: Q_dn={dn_bad.n_queries}  CHECK PASSED")

    # ------------------------------------------------------------------
    # 2. build_self_attn_mask
    # ------------------------------------------------------------------
    print("\n=== Test 2: build_self_attn_mask ===")
    group_id = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    n_reg = 4
    mask = MaskDenoiser.build_self_attn_mask(n_regular=n_reg, group_id=group_id)
    assert mask.shape == (n_reg + 6, n_reg + 6)
    assert mask.dtype == torch.bool
    # Regular-regular: all False (allowed).
    assert not mask[:n_reg, :n_reg].any()
    # Regular-DN both directions: all True (blocked).
    assert mask[:n_reg, n_reg:].all()
    assert mask[n_reg:, :n_reg].all()
    # DN within group 0 (indices n_reg, n_reg+1): allowed.
    assert not mask[n_reg:n_reg + 2, n_reg:n_reg + 2].any()
    # DN cross-group (e.g. group 0 ↔ group 1).
    assert mask[n_reg:n_reg + 2, n_reg + 2:n_reg + 4].all()
    assert mask[n_reg + 2:n_reg + 4, n_reg:n_reg + 2].all()
    print(f"  ({n_reg + 6}, {n_reg + 6}) mask structure  CHECK PASSED")
    # Empty DN group_id should give an all-False (n_reg, n_reg) mask.
    empty_mask = MaskDenoiser.build_self_attn_mask(
        n_regular=n_reg, group_id=torch.zeros(0, dtype=torch.long),
    )
    assert empty_mask.shape == (n_reg, n_reg)
    assert not empty_mask.any()
    print(f"  empty DN → ({n_reg}, {n_reg}) all-False  CHECK PASSED")

    # ------------------------------------------------------------------
    # 3. Mask2FormerDecoder with combined queries
    # ------------------------------------------------------------------
    print("\n=== Test 3: Mask2FormerDecoder with [regular | DN] queries ===")
    decoder = Mask2FormerDecoder(
        dim=DIM, scale_pattern=[SRC, SRC],
        num_queries=NUM_REG, num_classes=NUM_CLASSES,
        num_heads=4, mlp_ratio=2.0, enable_origin_head=False,
    )
    # Mimic Phase A active: zero-init the learnable query embeddings.
    torch.nn.init.zeros_(decoder.query_content)
    torch.nn.init.zeros_(decoder.query_pos)
    decoder.train()

    # Synthetic source-level event.
    tokens = torch.randn(N_TOKENS, DIM) * 0.1
    coords = torch.randn(N_TOKENS, 3) * 0.5
    sp_to_level = torch.arange(N_TOKENS, dtype=torch.long)
    levels = OrderedDict([
        (SRC, LevelOutput(tokens=tokens, coords=coords,
                          sp_to_level_id=sp_to_level, name=SRC)),
    ])

    # DN queries: take dn_init from Test 1 (Q_dn=15), pad regular queries
    # with random anchors so the selector path is bypassed cleanly.
    reg_init_q = torch.randn(NUM_REG, DIM) * 0.1
    reg_init_anchor = torch.rand(NUM_REG, 3) * 2 - 1
    init_q = torch.cat([reg_init_q, dn_init.init_q], dim=0)
    init_anchor = torch.cat([reg_init_anchor, dn_init.init_anchor], dim=0)
    dn_mask = MaskDenoiser.build_self_attn_mask(
        n_regular=NUM_REG, group_id=dn_init.group_id,
    )
    Q_total = NUM_REG + dn_init.n_queries

    out = decoder(
        levels,
        init_query_content=init_q,
        init_anchor_coords=init_anchor,
        dn_self_attn_mask=dn_mask,
    )
    cl = out["final"]["class_logits"]
    ml = out["final"]["mask_logits"][SRC]
    assert cl.shape == (Q_total, NUM_CLASSES)
    assert ml.shape == (Q_total, N_TOKENS)
    assert torch.isfinite(cl).all() and torch.isfinite(ml).all()
    print(f"  decoder forward: class_logits {tuple(cl.shape)}, "
          f"mask_logits[{SRC}] {tuple(ml.shape)}  CHECK PASSED")

    loss_smoke = cl.pow(2).mean() + ml.pow(2).mean()
    loss_smoke.backward()
    n_grad = 0; n_nan = 0
    for n, p in decoder.named_parameters():
        if p.grad is not None:
            n_grad += 1
            if not torch.isfinite(p.grad).all():
                n_nan += 1
                print(f"    NaN/Inf grad on {n}")
    assert n_nan == 0
    print(f"  backward: {n_grad} params got grads, 0 NaN/Inf  CHECK PASSED")

    # Confirm the self-attn mask is actually applied: re-run with an
    # all-False mask (no blocking) on identical inputs and confirm at
    # least one output element changes. With the eval+zero-init heads
    # the FINAL outputs are mostly 0 (designed identity at init), so we
    # hook the layer-0 self-attn input AFTER the cross-attn residual
    # adds and compare against running with vs without the mask.
    decoder.eval()
    captured = {}
    def _hook(_mod, inp, out_t):
        captured.setdefault("self_attn_out", []).append(out_t[0].detach().clone())
    h = decoder.layers[0].self_attn.register_forward_hook(_hook)
    with torch.no_grad():
        decoder(levels, init_query_content=init_q,
                init_anchor_coords=init_anchor,
                dn_self_attn_mask=dn_mask)
        # Same inputs, no mask
        decoder(levels, init_query_content=init_q,
                init_anchor_coords=init_anchor,
                dn_self_attn_mask=None)
    h.remove()
    out_with, out_without = captured["self_attn_out"]
    # With zero-init self_attn.out_proj the OUTPUT is 0 either way. So
    # the test should hook the ATTENTION COMPUTATION itself, not the
    # projected output. Easiest workaround: undo the zero-init on
    # self_attn.out_proj before this re-run so the mask's effect leaks
    # through.
    if (out_with - out_without).abs().max().item() < 1e-6:
        # Undo zero-init of self_attn.out_proj on layer 0 and rerun.
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(
                decoder.layers[0].self_attn.out_proj.weight,
            )
            torch.nn.init.zeros_(decoder.layers[0].self_attn.out_proj.bias)
        captured.clear()
        h = decoder.layers[0].self_attn.register_forward_hook(_hook)
        with torch.no_grad():
            decoder(levels, init_query_content=init_q,
                    init_anchor_coords=init_anchor,
                    dn_self_attn_mask=dn_mask)
            decoder(levels, init_query_content=init_q,
                    init_anchor_coords=init_anchor,
                    dn_self_attn_mask=None)
        h.remove()
        out_with, out_without = captured["self_attn_out"]
    diff = (out_with - out_without).abs().mean().item()
    print(f"  self_attn output differs with vs without mask: "
          f"mean|Δ| = {diff:.6f}")
    assert diff > 1e-6, "dn_self_attn_mask had no effect on self_attn output"
    print("  dn_self_attn_mask CHECK PASSED")

    # ------------------------------------------------------------------
    # 4. LArFormerLoss.compute_dn_loss
    # ------------------------------------------------------------------
    print("\n=== Test 4: LArFormerLoss.compute_dn_loss ===")
    levels_cfg_dn = [
        dict(name=SRC, builder="VoxelBuilder",
             builder_cfg=dict(),
             supervision=dict(mask=dict(weight=5.0, mode="primary"))),
    ]
    loss_fn = LArFormerLoss(
        levels_cfg=levels_cfg_dn,
        num_classes=NUM_CLASSES,
        weight_class=2.0,
        weight_mask_primary=5.0,
        weight_dice_primary=5.0,
        weight_origin=0.0,
        num_sample_points=32,
        weight_dn_loss=1.0,
    )

    # Per-level GT mask: K=3 instances, each owns ~5 tokens.
    K_gt = 3
    per_level_gt_mask = OrderedDict([(SRC, torch.zeros(K_gt, N_TOKENS))])
    per_level_gt_mask[SRC][0, 0:5] = 1.0
    per_level_gt_mask[SRC][1, 5:10] = 1.0
    per_level_gt_mask[SRC][2, 10:15] = 1.0
    gt_classes = torch.tensor([0, 1, 0], dtype=torch.long)
    gt_target_idx = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)  # 2 DN per GT
    Q_dn_loss = gt_target_idx.shape[0]

    # Build a fake DN decoder output (init + 1 layer + final) with grad.
    def _fake_layer():
        return {
            "class_logits": torch.randn(Q_dn_loss, NUM_CLASSES, requires_grad=True),
            "origin":       torch.zeros(Q_dn_loss, 3),
            "mask_logits":  OrderedDict([
                (SRC, torch.randn(Q_dn_loss, N_TOKENS, requires_grad=True)),
            ]),
        }
    init_l = _fake_layer()
    layer1 = _fake_layer()
    fake_dec = {
        "init":   init_l,
        "layers": [layer1],
        "final":  layer1,
    }
    dn_loss = loss_fn.compute_dn_loss(
        decoder_output_dn=fake_dec,
        per_level_gt_mask=per_level_gt_mask,
        gt_classes=gt_classes,
        gt_origin=None,
        gt_target_idx=gt_target_idx,
    )
    assert "total" in dn_loss
    assert dn_loss["total"].dim() == 0
    assert torch.isfinite(dn_loss["total"])
    assert int(dn_loss["n_dn"]) == Q_dn_loss
    print(f"  compute_dn_loss: total={dn_loss['total'].item():.4f} "
          f"(cls={dn_loss['cls'].item():.4f}, "
          f"mask={dn_loss['mask_primary'].item():.4f}, "
          f"dice={dn_loss['dice_primary'].item():.4f})  CHECK PASSED")
    dn_loss["total"].backward()
    assert init_l["class_logits"].grad is not None
    assert layer1["class_logits"].grad is not None
    assert torch.isfinite(init_l["class_logits"].grad).all()
    assert torch.isfinite(layer1["class_logits"].grad).all()
    assert torch.isfinite(init_l["mask_logits"][SRC].grad).all()
    print(f"  backward through DN loss: finite grads on init + layer  CHECK PASSED")

    # Empty DN: returns finite zero total.
    dn_empty_loss = loss_fn.compute_dn_loss(
        decoder_output_dn={
            "init":   {"class_logits": torch.zeros(0, NUM_CLASSES),
                       "origin":       torch.zeros(0, 3),
                       "mask_logits":  OrderedDict([(SRC, torch.zeros(0, N_TOKENS))])},
            "layers": [],
            "final":  {"class_logits": torch.zeros(0, NUM_CLASSES),
                       "origin":       torch.zeros(0, 3),
                       "mask_logits":  OrderedDict([(SRC, torch.zeros(0, N_TOKENS))])},
        },
        per_level_gt_mask=per_level_gt_mask,
        gt_classes=gt_classes,
        gt_origin=None,
        gt_target_idx=torch.zeros(0, dtype=torch.long),
    )
    assert torch.isfinite(dn_empty_loss["total"])
    assert int(dn_empty_loss["total"]) == 0
    print(f"  empty DN → total=0  CHECK PASSED")

    print("\nPhase-B smoke test PASSED.")


if __name__ == "__main__":
    main()
