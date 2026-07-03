"""Phase 2 unit test: KeypointQueryHead + keypoint_query_loss mechanics.

Validates the query-based keypoint head and its uncertainty-aware loss in
ISOLATION — no Stage 3, no backbone. Synthetic per-query embeddings + GT
particles + a fixed query→GT assignment. Overfitting the head must drive the
start/end position errors to ~0 and (for betanll) shrink the predicted
variance on the well-fit points. This pins down the head/loss before they are
wired to the real LArFormer matcher.

    python lartpc/larformer_analysis/archive/keypoint_v1/test_phase2_keypoint_query.py
"""

import sys

import torch

from pointcept.models.LArFormer.keypoint_query import (
    KeypointQueryHead, keypoint_query_loss,
)

COORD_SCALE = 179.55


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    Q, K = 12, 4                      # 12 queries, 4 GT particles
    # Distinct, fixed per-query input embeddings (the "frozen Stage-3 output").
    query_embed = torch.randn(Q, dim, device=device)

    # GT particles: random normalized start/end; first 3 are tracks (has_end),
    # last is a shower (no end).
    gt_start = torch.randn(K, 3, device=device) * 0.5
    gt_end = torch.randn(K, 3, device=device) * 0.5
    gt_has_end = torch.tensor([1, 1, 1, 0], device=device, dtype=torch.float32)

    # Fixed assignment: queries 0..3 → GT 0..3 (the rest are "no_object").
    q_idx = torch.arange(K, device=device)
    k_idx = torch.arange(K, device=device)

    head = KeypointQueryHead(dim=dim, predict_end=True,
                             predict_uncertainty=True).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)

    first = None
    for step in range(600):
        opt.zero_grad()
        pred = head(query_embed)
        ld = keypoint_query_loss(
            pred, q_idx, k_idx, gt_start, gt_end, gt_has_end,
            reg_kind="betanll", beta=0.5, coord_scale=COORD_SCALE,
        )
        ld["total"].backward()
        opt.step()
        if first is None:
            first = float(ld["total"])
        if step % 60 == 0 or step == 599:
            print(f"  step {step:3d}  total={float(ld['total']):+.4f}  "
                  f"start_err={float(ld['kpq_start_err_cm']):.3f}cm  "
                  f"end_err={float(ld['kpq_end_err_cm']):.3f}cm  "
                  f"end_exist={float(ld['kpq_end_exist']):.4f}")

    s_err = float(ld["kpq_start_err_cm"])
    e_err = float(ld["kpq_end_err_cm"])
    n_match = int(float(ld["kpq_n_match"]))
    n_end = int(float(ld["kpq_n_end"]))
    print(f"\n[phase2] matched={n_match} (expect {K}), with-end={n_end} (expect 3)")
    print(f"[phase2] final start_err={s_err:.4f} cm  end_err={e_err:.4f} cm")

    # Empty-match path must be finite zeros.
    empty = keypoint_query_loss(
        pred, q_idx[:0], k_idx[:0], gt_start, gt_end, gt_has_end)
    assert torch.isfinite(empty["total"]) and float(empty["total"]) == 0.0

    ok = (n_match == K and n_end == 3 and s_err < 1.0 and e_err < 1.0
          and torch.isfinite(ld["total"]))
    print(f"\n[phase2] {'PASS' if ok else 'FAIL'}: head fits start/end to <1cm, "
          f"matching counts correct, empty-match finite")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
