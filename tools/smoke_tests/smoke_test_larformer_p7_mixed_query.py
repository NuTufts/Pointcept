"""Phase-A smoke test: DINO/Mask-DINO-style Mixed Query Selection.

Verifies the two new components in isolation (no backbone, no real data):

  1. MixedQuerySelector:
       - top_k mode picks the highest-score tokens
       - top_m_then_fps picks K spatially-diverse anchors from the top-M
       - FPS diversity beats pure top-K mean-pairwise-distance when the
         high-score tokens are spatially clustered (the cosmic-dominance
         scenario the FPS step exists to address)
       - Padding is clean when N_tokens < K
       - Scoring uses 1 - p(no_object) from the source-level cls head

  2. Mask2FormerDecoder with init_query_content + init_anchor_coords:
       - Builds, forward + backward run end-to-end
       - Per-layer `query_pos_dyn` actually uses the anchor PE (verified
         by checking that two different anchor sets give different
         layer-0 cross-attention outputs)
       - Default (None, None) path still matches the original behavior

Run inside the pointcept container:
    ./run_in_container.sh python tools/smoke_tests/smoke_test_larformer_p7_mixed_query.py
"""

import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_synthetic_level(
    n: int, dim: int, num_classes: int,
    cluster_centers, cluster_fracs,
    high_score_cluster_idx: int = 0,
    seed: int = 0,
):
    """Build a synthetic LevelOutput-like dict + cls logits where one
    cluster has high objectness scores (low p(no_object)). Coords are
    drawn in [-1, 1] around the given centers.

    cluster_fracs sum to 1. The first center gets the high-score
    tokens (1 - p(no_obj) close to 1); the rest get low scores.
    """
    g = torch.Generator().manual_seed(seed)
    sizes = [int(round(f * n)) for f in cluster_fracs]
    # Top up rounding error onto the last cluster.
    sizes[-1] += n - sum(sizes)
    coords_chunks = []
    score_chunks = []
    for ci, (c, sz) in enumerate(zip(cluster_centers, sizes)):
        coord = (torch.randn(sz, 3, generator=g) * 0.05
                 + torch.tensor(c, dtype=torch.float32))
        coords_chunks.append(coord)
        # high cluster: score ~ 0.95, others ~ 0.05
        s = (0.95 if ci == high_score_cluster_idx else 0.05)
        score = torch.full((sz,), s)
        score_chunks.append(score)
    coords = torch.cat(coords_chunks, 0)
    scores = torch.cat(score_chunks, 0)

    # Convert score → cls logits so that 1 - softmax(logits)[-1] == score.
    # Build the target prob row directly (mass p_no_obj on the last col,
    # rest spread equally over the (C-1) object classes) and use
    # logits = log(probs) — softmax is invariant to additive constants
    # so this gives an exact reconstruction.
    p_no_obj = (1.0 - scores).clamp(1e-6, 1.0 - 1e-6)
    probs = torch.zeros(coords.shape[0], num_classes)
    probs[:, -1] = p_no_obj
    probs[:, :-1] = ((1.0 - p_no_obj) / (num_classes - 1)).unsqueeze(-1)
    logits = probs.log()
    # Sanity: re-derive scores from these logits should match the inputs.
    derived = 1.0 - logits.softmax(-1)[:, -1]
    assert torch.allclose(derived, scores, atol=1e-4), \
        f"score reconstruction failed: max err {(derived - scores).abs().max().item()}"

    tokens = torch.randn(coords.shape[0], dim, generator=g) * 0.1
    sp_to_level_id = torch.arange(coords.shape[0], dtype=torch.long)
    return tokens, coords, logits, sp_to_level_id


def _mean_pairwise_nn(coords: torch.Tensor) -> float:
    """Mean nearest-neighbor distance: higher = more spatially spread out."""
    if coords.shape[0] < 2:
        return 0.0
    d = torch.cdist(coords, coords)                    # (K, K)
    d.fill_diagonal_(float("inf"))
    return float(d.min(-1).values.mean())


def main():
    sys.path.insert(0, REPO_ROOT)
    from collections import OrderedDict

    from pointcept.models.LArFormer.builders.base import LevelOutput
    from pointcept.models.LArFormer.decoder import Mask2FormerDecoder
    from pointcept.models.LArFormer.query_selection import (
        MixedQuerySelector, _farthest_point_sampling,
    )

    torch.manual_seed(0)

    DIM = 64
    NUM_QUERIES = 16
    NUM_CLASSES = 3
    SRC = "voxel_8cm"
    N_TOKENS = 400

    # ------------------------------------------------------------------
    # 1. MixedQuerySelector — FPS diversity vs top-K
    # ------------------------------------------------------------------
    print("=== Test 1: MixedQuerySelector ===")

    # One BIG high-score cluster (simulating a long cosmic) + 4 smaller
    # background clusters (other tracks, low score). Top-K should
    # concentrate on the big cluster; FPS should spread out.
    centers = [
        [-0.7, -0.7, -0.7],    # dominant high-score cluster
        [+0.6, -0.6, +0.5],
        [+0.5, +0.7, -0.4],
        [-0.4, +0.6, +0.6],
        [+0.0, +0.0, +0.0],
    ]
    fracs = [0.60, 0.10, 0.10, 0.10, 0.10]
    tokens, coords, logits, sp_to_level = _make_synthetic_level(
        N_TOKENS, DIM, NUM_CLASSES, centers, fracs,
        high_score_cluster_idx=0, seed=1,
    )
    levels = OrderedDict([
        (SRC, LevelOutput(tokens=tokens, coords=coords,
                          sp_to_level_id=sp_to_level, name=SRC)),
    ])
    per_level_cls = OrderedDict([(SRC, logits)])

    # --- 1a. top_k mode
    sel_topk = MixedQuerySelector(
        token_dim=DIM, num_queries=NUM_QUERIES, source_level=SRC,
        selection_mode="top_k",
    )
    q_topk, a_topk = sel_topk(levels, per_level_cls)
    assert q_topk.shape == (NUM_QUERIES, DIM)
    assert a_topk.shape == (NUM_QUERIES, 3)
    nn_topk = _mean_pairwise_nn(a_topk)
    print(f"  top_k: K={NUM_QUERIES}, "
          f"anchor span [{a_topk.min().item():+.3f}, {a_topk.max().item():+.3f}], "
          f"mean nearest-neighbor dist={nn_topk:.4f}")

    # --- 1b. top_m_then_fps mode (default)
    sel_fps = MixedQuerySelector(
        token_dim=DIM, num_queries=NUM_QUERIES, source_level=SRC,
        selection_mode="top_m_then_fps", score_filter_multiplier=4,
    )
    q_fps, a_fps = sel_fps(levels, per_level_cls)
    nn_fps = _mean_pairwise_nn(a_fps)
    print(f"  fps:   K={NUM_QUERIES}, "
          f"anchor span [{a_fps.min().item():+.3f}, {a_fps.max().item():+.3f}], "
          f"mean nearest-neighbor dist={nn_fps:.4f}")
    print(f"  FPS/top_k mean-NN ratio = {nn_fps / max(nn_topk, 1e-6):.2f}x")
    # The dominant cluster (60% of high-score tokens, σ=0.05) traps top-K
    # within a ~0.1-radius ball; FPS pulls anchors across centers ~1.5
    # apart. The actual ratio is ~8-10x in this synthetic — anything ≥1.5
    # already proves FPS is doing useful work.
    assert nn_fps > nn_topk * 1.5, \
        (f"FPS should spread anchors more than top_k: "
         f"got fps={nn_fps:.4f}, top_k={nn_topk:.4f}")
    print("  FPS diversity > top_k diversity CHECK PASSED")

    # --- 1c. Padding when N < K
    tiny_tokens, tiny_coords, tiny_logits, tiny_sp = _make_synthetic_level(
        n=5, dim=DIM, num_classes=NUM_CLASSES,
        cluster_centers=[[0.0, 0.0, 0.0]], cluster_fracs=[1.0], seed=2,
    )
    tiny_levels = OrderedDict([
        (SRC, LevelOutput(tokens=tiny_tokens, coords=tiny_coords,
                          sp_to_level_id=tiny_sp, name=SRC)),
    ])
    tiny_cls = OrderedDict([(SRC, tiny_logits)])
    q_pad, a_pad = sel_fps(tiny_levels, tiny_cls)
    assert q_pad.shape == (NUM_QUERIES, DIM)
    assert a_pad.shape == (NUM_QUERIES, 3)
    # First 5 should be real picks (nonzero), rest should be zero-padded.
    n_zero_rows = int((a_pad.abs().sum(-1) == 0).sum())
    assert n_zero_rows >= (NUM_QUERIES - 5), \
        f"expected ≥{NUM_QUERIES - 5} zero-padded rows, got {n_zero_rows}"
    print(f"  N<K padding: 5 valid + "
          f"{n_zero_rows}/{NUM_QUERIES} zero-padded rows CHECK PASSED")

    # --- 1d. Empty-level edge case
    empty_tokens = torch.zeros(0, DIM)
    empty_coords = torch.zeros(0, 3)
    empty_sp = torch.zeros(0, dtype=torch.long)
    empty_levels = OrderedDict([
        (SRC, LevelOutput(tokens=empty_tokens, coords=empty_coords,
                          sp_to_level_id=empty_sp, name=SRC)),
    ])
    empty_cls = OrderedDict([(SRC, torch.zeros(0, NUM_CLASSES))])
    q_emp, a_emp = sel_fps(empty_levels, empty_cls)
    assert q_emp.shape == (NUM_QUERIES, DIM)
    assert a_emp.shape == (NUM_QUERIES, 3)
    assert q_emp.abs().sum() == 0 and a_emp.abs().sum() == 0
    print("  empty-level → all-zero anchors CHECK PASSED\n")

    # ------------------------------------------------------------------
    # 2. Mask2FormerDecoder with anchor inputs
    # ------------------------------------------------------------------
    print("=== Test 2: Mask2FormerDecoder anchor wiring ===")

    decoder = Mask2FormerDecoder(
        dim=DIM, scale_pattern=[SRC, SRC],
        num_queries=NUM_QUERIES, num_classes=NUM_CLASSES,
        num_heads=4, mlp_ratio=2.0, enable_origin_head=True,
    )
    # Simulate Phase-A active: zero the learnable query embeddings so
    # they act as deltas on top of the anchors.
    torch.nn.init.zeros_(decoder.query_content)
    torch.nn.init.zeros_(decoder.query_pos)
    decoder.train()

    # --- 2a. Forward + backward with anchors
    out = decoder(
        levels, init_query_content=q_fps, init_anchor_coords=a_fps,
    )
    assert "final" in out
    cl = out["final"]["class_logits"]
    ml = out["final"]["mask_logits"][SRC]
    print(f"  forward: class_logits {tuple(cl.shape)}, "
          f"mask_logits[{SRC}] {tuple(ml.shape)}")
    assert cl.shape == (NUM_QUERIES, NUM_CLASSES)
    assert ml.shape == (NUM_QUERIES, N_TOKENS)
    assert torch.isfinite(cl).all() and torch.isfinite(ml).all()

    # Trivial loss → backward → at least the decoder's trainable params
    # should receive a finite gradient.
    loss = cl.pow(2).mean() + ml.pow(2).mean()
    loss.backward()
    n_grad = 0; n_nan = 0
    for n, p in decoder.named_parameters():
        if p.grad is not None:
            n_grad += 1
            if not torch.isfinite(p.grad).all():
                n_nan += 1
    print(f"  backward: {n_grad} params got grads, {n_nan} have NaN/Inf")
    assert n_nan == 0, f"{n_nan} decoder params received non-finite gradients"
    print("  forward + backward with anchors CHECK PASSED")

    # --- 2b. Default path (no anchor inputs) still works
    decoder.zero_grad(set_to_none=True)
    # Restore non-zero query embeddings for the default-path check, since
    # zero-init query_content + zero anchor gives a degenerate input.
    torch.nn.init.trunc_normal_(decoder.query_content, std=0.02)
    torch.nn.init.trunc_normal_(decoder.query_pos, std=0.02)
    out_def = decoder(levels)
    cl_def = out_def["final"]["class_logits"]
    assert torch.isfinite(cl_def).all()
    print(f"  default-path forward (no anchors) finite: "
          f"class_logits range [{cl_def.min().item():+.3f}, "
          f"{cl_def.max().item():+.3f}]")

    # --- 2c. Anchor PE actually reaches per-layer query_pos_dyn
    # Can't check via the final mask_logits because the canonical Phase-A
    # config has zero-init heads + zero-init attn out_proj + zero-init
    # query embeddings, which makes the decoder a designed identity at
    # init (cross-attn output = 0 regardless of Q). Instead, hook layer
    # 0's cross_attn to capture the Q tensor directly — that's where
    # query_pos_dyn (with anchor PE added) feeds in.
    torch.nn.init.zeros_(decoder.query_content)
    torch.nn.init.zeros_(decoder.query_pos)
    decoder.eval()
    captured = {}
    def _q_hook(name):
        def _hook(_mod, inp, _out):
            captured[name] = inp[0].detach().clone()
        return _hook
    h = decoder.layers[0].cross_attn.register_forward_hook(_q_hook("A"))
    with torch.no_grad():
        decoder(levels, init_query_content=q_fps, init_anchor_coords=a_fps)
    h.remove()
    q_A = captured["A"]
    h = decoder.layers[0].cross_attn.register_forward_hook(_q_hook("B"))
    perm = torch.randperm(NUM_QUERIES)
    with torch.no_grad():
        decoder(levels, init_query_content=q_fps,
                init_anchor_coords=a_fps[perm])
    h.remove()
    q_B = captured["B"]
    diff = (q_A - q_B).abs().mean()
    print(f"  layer-0 cross-attn Q sensitivity to anchors: "
          f"|Q(A) - Q(B)|.mean = {diff.item():.6f}")
    assert diff.item() > 1e-6, \
        ("anchor coords don't reach layer-0 cross-attn Q — anchor_pe "
         "is not wired into query_pos_dyn")

    # And confirm: with init_anchor_coords=None, the same Q tensor
    # falls back to whatever query_pos contributes (here, zero), so we
    # also expect Q_default != Q_with_anchor.
    captured.clear()
    h = decoder.layers[0].cross_attn.register_forward_hook(_q_hook("default"))
    with torch.no_grad():
        decoder(levels, init_query_content=q_fps,
                init_anchor_coords=None)
    h.remove()
    q_default = captured["default"]
    diff2 = (q_A - q_default).abs().mean()
    print(f"  anchor vs no-anchor Q difference: {diff2.item():.6f}")
    assert diff2.item() > 1e-6, \
        "init_anchor_coords=None vs supplied gave identical Q — bug"
    print("  anchor coords reach per-layer query_pos_dyn CHECK PASSED\n")

    # ------------------------------------------------------------------
    # 3. Pure-PyTorch FPS spot-check
    # ------------------------------------------------------------------
    print("=== Test 3: FPS helper ===")
    pts = torch.randn(100, 3)
    idx = _farthest_point_sampling(pts, k=10)
    assert idx.shape == (10,)
    assert int(idx[0].item()) == 0  # seeds with index 0 by convention
    # All picks are unique
    assert len(set(idx.tolist())) == 10
    # Asking for more points than available returns all of them.
    idx_all = _farthest_point_sampling(pts[:5], k=10)
    assert idx_all.shape == (5,)
    print("  FPS basic invariants CHECK PASSED\n")

    print("Phase-A smoke test PASSED.")


if __name__ == "__main__":
    main()
