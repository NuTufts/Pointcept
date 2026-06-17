"""Query-based keypoint head + loss for the LArFormer keypoint module (Phase 2).

Design 2A (see lartpc_data_prep/larformer_keypoint/README.md §4): a per-query
head that turns a Stage-3 particle query's embedding into a typed 3D keypoint:

    start (3)        — the particle's start point (= the existing origin head)
    end   (3)        — the particle's end point (track-like particles only)
    end_logit (1)    — predicted "this particle has a track-end" gate
    *_logvar (3)     — per-axis log-variance for uncertainty-aware regression

Supervision rides on the Stage-3 query→GT-particle Hungarian match (no new
matcher for start/end): each matched query is supervised against its GT
particle's `origin_coord_norm` (start) and `end_coord_norm` (end, masked by
`has_end`). The nu vertex is handled separately by dedicated vertex queries
(a later Phase-2 sub-step).

`keypoint_query_loss` is matcher-agnostic: it takes (q_idx, k_idx) plus the
per-GT targets, so it can be unit-tested with a fixed assignment and later
fed the LArFormer matcher's output unchanged.

Regression loss kinds:
    "l1"        — mean |mu - target|
    "smooth_l1" — Huber
    "betanll"   — Gaussian NLL with β-weighting (Seitzer et al. 2022):
                  decouples the gradient magnitude from the predicted
                  variance so rare high-error keypoints aren't explained
                  away as noise. Uses *_logvar; falls back to L1 if a head
                  has no uncertainty output.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reg_loss(mu, target, logvar=None, kind="betanll", beta=0.5):
    """Per-dim regression loss, averaged. mu/target/logvar: (M, D)."""
    if mu.shape[0] == 0:
        return mu.new_zeros(())
    if kind == "l1":
        return (mu - target).abs().mean()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(mu, target, reduction="mean")
    if kind == "betanll":
        if logvar is None:
            return (mu - target).abs().mean()
        logvar = logvar.clamp(-10.0, 10.0)
        var = logvar.exp()
        nll = 0.5 * ((mu - target) ** 2 / var + logvar)        # (M, D)
        if beta > 0:
            nll = nll * (var.detach() ** beta)
        return nll.mean()
    raise ValueError(f"unknown reg loss kind {kind!r}")


class KeypointQueryHead(nn.Module):
    """Per-query keypoint head (start / end / end-exist / uncertainty).

    Args:
        dim:                  query embedding dim (= decoder token_dim).
        hidden_dim:           MLP width (default = dim).
        predict_end:          add the end-point + end-exist heads.
        predict_uncertainty:  add per-axis log-variance heads (for betanll).

    forward(query_embed (Q, dim)) → dict of:
        start (Q,3), end (Q,3)|None, end_logit (Q,)|None,
        start_logvar (Q,3)|None, end_logvar (Q,3)|None
    """

    def __init__(self, dim, hidden_dim=None, predict_end=True,
                 predict_uncertainty=True,
                 pos_emb_kind: Optional[str] = None,
                 pos_emb_num_freq: Optional[int] = None,
                 pos_emb_max_freq: float = 256.0,
                 pos_emb_hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        hidden_dim = int(hidden_dim) if hidden_dim else int(dim)
        self.predict_end = bool(predict_end)
        self.predict_uncertainty = bool(predict_uncertainty)
        # Opt-in anchor pos-emb added to the query before the heads (parity
        # with the integrated _PerLayerHeads path). Off by default.
        from .decoder import build_coord_pos_emb
        self.pos_emb = build_coord_pos_emb(
            pos_emb_kind, self.dim, num_freq=pos_emb_num_freq,
            max_freq=pos_emb_max_freq, hidden_dim=pos_emb_hidden_dim,
            zero_init=True)  # ADDED to the query → identity at init

        def _mlp(out_dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.start_head = _mlp(3)
        self.end_head = _mlp(3) if predict_end else None
        self.end_exist_head = nn.Linear(dim, 1) if predict_end else None
        if predict_uncertainty:
            self.start_logvar_head = _mlp(3)
            self.end_logvar_head = _mlp(3) if predict_end else None
            # Start uncertainty at logvar≈0 (var≈1) — zero-init the output.
            nn.init.zeros_(self.start_logvar_head[-1].weight)
            nn.init.zeros_(self.start_logvar_head[-1].bias)
            if self.end_logvar_head is not None:
                nn.init.zeros_(self.end_logvar_head[-1].weight)
                nn.init.zeros_(self.end_logvar_head[-1].bias)
        else:
            self.start_logvar_head = None
            self.end_logvar_head = None

    def forward(self, query_embed: torch.Tensor,
                anchor_coords: Optional[torch.Tensor] = None) -> dict:
        q = query_embed
        if self.pos_emb is not None and anchor_coords is not None:
            q = query_embed + self.pos_emb(anchor_coords)
        out = {"start": self.start_head(q)}
        out["start_logvar"] = (self.start_logvar_head(q)
                               if self.start_logvar_head is not None else None)
        if self.predict_end:
            out["end"] = self.end_head(q)
            out["end_logit"] = self.end_exist_head(q).squeeze(-1)
            out["end_logvar"] = (self.end_logvar_head(q)
                                 if self.end_logvar_head is not None else None)
        else:
            out["end"] = out["end_logit"] = out["end_logvar"] = None
        return out


def keypoint_query_loss(
    pred: dict,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    gt_start: torch.Tensor,            # (K, 3)
    gt_end: Optional[torch.Tensor],    # (K, 3) or None
    gt_has_end: Optional[torch.Tensor],  # (K,) bool/float or None
    reg_kind: str = "betanll",
    beta: float = 0.5,
    weight_start: float = 1.0,
    weight_end: float = 1.0,
    weight_end_exist: float = 0.5,
    coord_scale: float = 179.55,
) -> dict:
    """Per-query keypoint loss over a given query→GT assignment.

    Returns a dict with 'total' + component scalars + diagnostics
    (start_err_cm / end_err_cm in detector cm). All tensors 0-d.
    Empty match (no pairs) → finite zeros.
    """
    device = pred["start"].device
    zero = pred["start"].new_zeros(())
    out = {
        "total": zero, "kpq_start": zero, "kpq_end": zero,
        "kpq_end_exist": zero,
        "kpq_start_err_cm": zero, "kpq_end_err_cm": zero,
        "kpq_n_match": torch.zeros((), device=device),
        "kpq_n_end": torch.zeros((), device=device),
    }
    q_idx = q_idx.to(device=device, dtype=torch.long)
    k_idx = k_idx.to(device=device, dtype=torch.long)
    if q_idx.numel() == 0:
        return out

    # ---- start (all matched pairs) ----
    mu_s = pred["start"][q_idx]
    tgt_s = gt_start.to(device)[k_idx]
    lv_s = pred["start_logvar"][q_idx] if pred.get("start_logvar") is not None else None
    start_loss = _reg_loss(mu_s, tgt_s, lv_s, kind=reg_kind, beta=beta)
    with torch.no_grad():
        start_err_cm = (mu_s - tgt_s).norm(dim=-1).mean() * coord_scale

    total = weight_start * start_loss
    out.update({
        "kpq_start": start_loss,
        "kpq_start_err_cm": start_err_cm,
        "kpq_n_match": torch.tensor(float(q_idx.numel()), device=device),
    })

    # ---- end (matched pairs whose GT has an end) ----
    if (pred.get("end") is not None and gt_end is not None
            and gt_has_end is not None):
        has_end = gt_has_end.to(device).bool()[k_idx]            # (P,)
        if bool(has_end.any()):
            qe = q_idx[has_end]
            ke = k_idx[has_end]
            mu_e = pred["end"][qe]
            tgt_e = gt_end.to(device)[ke]
            lv_e = (pred["end_logvar"][qe]
                    if pred.get("end_logvar") is not None else None)
            end_loss = _reg_loss(mu_e, tgt_e, lv_e, kind=reg_kind, beta=beta)
            with torch.no_grad():
                end_err_cm = (mu_e - tgt_e).norm(dim=-1).mean() * coord_scale
            total = total + weight_end * end_loss
            out["kpq_end"] = end_loss
            out["kpq_end_err_cm"] = end_err_cm
            out["kpq_n_end"] = torch.tensor(float(qe.numel()), device=device)

        # ---- end-exist gate (BCE over all matched pairs) ----
        if pred.get("end_logit") is not None:
            tgt_exist = gt_has_end.to(device).float()[k_idx]
            exist_loss = F.binary_cross_entropy_with_logits(
                pred["end_logit"][q_idx], tgt_exist, reduction="mean")
            total = total + weight_end_exist * exist_loss
            out["kpq_end_exist"] = exist_loss

    out["total"] = total
    return out
