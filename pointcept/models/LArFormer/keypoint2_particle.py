"""Per-particle keypoint query decoder + loss (attempt-2 Phase 2).

For ONE particle (a point subset of the nu slice): a small DETR-style decoder
whose queries cross-attend to the particle's spacepoint tokens (FULL attention —
all points belong to the particle, no mask gating), self-attend, and each
predict a typed keypoint:

    class  ∈ {start, end, no_object}        (num_classes = 3)
    pos    ∈ R^3                             (keypoint position, coord_norm frame)

Queries are SEEDED (optionally) from the slice-level ptv3_dec2 "object" head:
the particle's highest-object-score dec2 tokens give per-query anchor coords +
content, so queries start at likely keypoint locations. The position head is
zero-init → at init `pos = anchor` (the seed), then it learns an offset.

Supervision (see `particle_keypoint_loss`): each particle's GT keypoints are
its instance's `origin_coord_norm` (start, always) and `end_coord_norm` (end,
if `has_end`). Queries are Hungarian-matched to those targets (cost = class CE
+ L2); matched queries get CE(class) + MSE(pos), unmatched → no_object CE. NO
mask supervision (the whole point: localization, not masking).

Reuses `_MaskedDecoderLayer` + `SinusoidalPosEmb3D` from the main decoder.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import _MaskedDecoderLayer, build_coord_pos_emb

# Class ids for the per-particle keypoint queries.
KP_CLS_START = 0
KP_CLS_END = 1
KP_CLS_NO_OBJECT = 2
KP_NUM_CLASSES = 3


class ParticleKeypointDecoder(nn.Module):
    """DETR-style per-particle keypoint decoder (full cross-attention).

    Args:
        dim:          token dim (= tokenizer token_dim).
        num_queries:  Q learnable queries per particle.
        num_classes:  start/end/no_object = 3.
        num_layers:   decoder layers.
        num_heads:    MHA heads.
        mlp_ratio:    FFN expansion.
        pos_emb_kind: "sinusoidal" (default) | "mlp" — for query_pos + key_pos.
    """

    def __init__(self, dim: int, num_queries: int = 8,
                 num_classes: int = KP_NUM_CLASSES, num_layers: int = 3,
                 num_heads: int = 4, mlp_ratio: float = 4.0,
                 pos_emb_kind: str = "sinusoidal",
                 pos_emb_num_freq: Optional[int] = None,
                 pos_emb_max_freq: float = 256.0):
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.query_content = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_content, std=0.02)
        self.query_pos = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.query_pos, std=0.02)
        # Per-class content embedding for denoising (DN) queries (seeded at a
        # jittered GT keypoint with its class's embedding — DN-DETR style).
        self.class_embedding = nn.Embedding(num_classes, dim)
        nn.init.trunc_normal_(self.class_embedding.weight, std=0.02)
        self.pos_emb = build_coord_pos_emb(
            pos_emb_kind, dim, num_freq=pos_emb_num_freq,
            max_freq=pos_emb_max_freq)
        self.layers = nn.ModuleList([
            _MaskedDecoderLayer(dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(int(num_layers))
        ])
        self.class_head = nn.Linear(dim, num_classes)
        self.pos_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        # zero-init pos head → pos == anchor (the seed) at init.
        nn.init.zeros_(self.pos_head[-1].weight)
        nn.init.zeros_(self.pos_head[-1].bias)

    def forward(self, sp_tokens: torch.Tensor, sp_coords: torch.Tensor,
                init_content: Optional[torch.Tensor] = None,
                init_anchor: Optional[torch.Tensor] = None,
                add_learnable_base: bool = True) -> List[dict]:
        """sp_tokens (M,D), sp_coords (M,3); optional seed (Q,D)/(Q,3).

        add_learnable_base=True (regular path): Q = num_queries, queries =
        learnable query_content (+ init_content), query_pos from the learnable
        self.query_pos. add_learnable_base=False (DN path): Q = init_content's
        count, queries = init_content alone (no learnable base / query_pos) —
        used for denoising queries seeded purely at jittered GT keypoints.
        Returns one dict per layer: {class_logits (Q,C), pos (Q,3)}.
        """
        if add_learnable_base:
            Q = self.num_queries
            queries0 = self.query_content.clone()
            if init_content is not None:
                queries0 = queries0 + init_content
            qpos_base = self.query_pos
        else:
            assert init_content is not None, "DN path needs init_content"
            Q = init_content.shape[0]
            queries0 = init_content
            qpos_base = init_content.new_zeros(Q, self.dim)
        anchor = (init_anchor if init_anchor is not None
                  else queries0.new_zeros(Q, 3))
        anchor_pe = self.pos_emb(anchor)                        # (Q,D)
        queries = queries0.unsqueeze(0)                        # (1,Q,D)

        if sp_tokens.shape[0] == 0:
            cls0 = self.class_head(queries.squeeze(0))
            return [{"class_logits": cls0, "pos": anchor}]

        keys = sp_tokens.unsqueeze(0)                           # (1,M,D)
        key_pos = self.pos_emb(sp_coords)                      # (M,D)
        out_layers: List[dict] = []
        cur_pos = anchor
        for li in range(len(self.layers)):
            query_pos = qpos_base + anchor_pe + self.pos_emb(cur_pos)
            queries = self.layers[li](
                queries, query_pos, keys, key_pos, attn_mask=None)
            h = queries.squeeze(0)                              # (Q,D)
            cls = self.class_head(h)
            cur_pos = anchor + self.pos_head(h)                 # offset from seed
            out_layers.append({"class_logits": cls, "pos": cur_pos})
        return out_layers


def noise_particle_indices(idx: torch.Tensor, n_event_sp: int,
                           drop_frac: float = 0.2, add_frac: float = 0.2):
    """Mask-domain noise: drop a fraction of a particle's points and add a few
    random non-particle event points (simulates an imperfect predicted mask).
    `idx` (M,) event-local indices → noised (M',) event-local indices."""
    dev = idx.device
    M = int(idx.numel())
    if M == 0:
        return idx
    keep_n = max(1, int(round(M * (1.0 - drop_frac))))
    kept = idx[torch.randperm(M, device=dev)[:keep_n]]
    add_n = int(round(M * add_frac))
    if add_n > 0 and n_event_sp > M:
        mask = torch.ones(n_event_sp, dtype=torch.bool, device=dev)
        mask[idx] = False
        pool = mask.nonzero(as_tuple=False).reshape(-1)
        if pool.numel() > 0:
            sel = pool[torch.randperm(pool.numel(), device=dev)[:add_n]]
            kept = torch.cat([kept, sel])
    return kept


def particle_keypoint_dn_loss(layer_preds: List[dict], dn_target_pos,
                              dn_target_cls, coord_scale: float = 179.55,
                              weight_class: float = 1.0, weight_pos: float = 5.0):
    """Denoising loss with KNOWN assignment (query i ↔ target i; no Hungarian).
    All DN queries are real keypoints (start/end). Deep-supervised."""
    device = layer_preds[-1]["class_logits"].device
    zero = layer_preds[-1]["pos"].new_zeros(())
    out = {"total": zero, "cls": zero, "pos": zero, "err_cm": zero.clone(),
           "n": torch.tensor(float(dn_target_pos.shape[0]), device=device)}
    if dn_target_pos.shape[0] == 0:
        return out
    total = zero
    n_layers = len(layer_preds)
    for li, lp in enumerate(layer_preds):
        cl, pos = lp["class_logits"], lp["pos"]
        cls_loss = F.cross_entropy(cl, dn_target_cls)
        pos_loss = F.smooth_l1_loss(pos, dn_target_pos)
        total = total + weight_class * cls_loss + weight_pos * pos_loss
        if li == n_layers - 1:
            out["cls"] = cls_loss.detach()
            out["pos"] = pos_loss.detach()
            with torch.no_grad():
                out["err_cm"] = ((pos - dn_target_pos).norm(dim=-1).mean()
                                 * coord_scale)
    out["total"] = total / max(n_layers, 1)
    return out


def _hungarian_match(class_logits, pos, gt_pos, gt_cls,
                     cost_class=1.0, cost_pos=1.0):
    """Match Q queries to T(<=Q) targets. Returns (q_idx, t_idx) numpy arrays.
    class_logits (Q,C), pos (Q,3); gt_pos (T,3), gt_cls (T,)."""
    from scipy.optimize import linear_sum_assignment
    if gt_pos.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    prob = class_logits.softmax(-1)                            # (Q,C)
    cls_cost = -prob[:, gt_cls]                                # (Q,T)
    l2 = torch.cdist(pos, gt_pos, p=2)                         # (Q,T)
    cost = (cost_class * cls_cost + cost_pos * l2).detach().cpu().numpy()
    qi, ti = linear_sum_assignment(cost)
    return qi.astype(np.int64), ti.astype(np.int64)


def particle_keypoint_loss(layer_preds: List[dict], gt_pos: torch.Tensor,
                           gt_cls: torch.Tensor, coord_scale: float = 179.55,
                           weight_class: float = 1.0, weight_pos: float = 5.0,
                           no_object_weight: float = 0.1,
                           cost_class: float = 1.0, cost_pos: float = 1.0):
    """Set-prediction loss for ONE particle, deep-supervised over layers.

    gt_pos (T,3) in coord_norm; gt_cls (T,) in {START, END}. Returns dict with
    'total' + diagnostics (start/end L2 in cm on the final layer).
    """
    device = layer_preds[-1]["class_logits"].device
    C = layer_preds[-1]["class_logits"].shape[-1]
    ce_w = torch.ones(C, device=device)
    ce_w[KP_CLS_NO_OBJECT] = no_object_weight
    zero = layer_preds[-1]["pos"].new_zeros(())
    out = {"total": zero, "kp_cls": zero, "kp_pos": zero,
           "kp_start_err_cm": zero.clone(), "kp_end_err_cm": zero.clone(),
           "n_start": torch.zeros((), device=device),
           "n_end": torch.zeros((), device=device)}
    total = zero
    n_layers = len(layer_preds)
    for li, lp in enumerate(layer_preds):
        cl, pos = lp["class_logits"], lp["pos"]
        Q = cl.shape[0]
        qi, ti = _hungarian_match(cl, pos, gt_pos, gt_cls,
                                  cost_class=cost_class, cost_pos=cost_pos)
        tgt_cls = torch.full((Q,), KP_CLS_NO_OBJECT, dtype=torch.long,
                             device=device)
        if qi.size > 0:
            q_t = torch.as_tensor(qi, device=device)
            t_t = torch.as_tensor(ti, device=device)
            tgt_cls[q_t] = gt_cls[t_t]
            cls_loss = F.cross_entropy(cl, tgt_cls, weight=ce_w)
            pos_loss = F.smooth_l1_loss(pos[q_t], gt_pos[t_t])
        else:
            cls_loss = F.cross_entropy(cl, tgt_cls, weight=ce_w)
            pos_loss = zero
        total = total + weight_class * cls_loss + weight_pos * pos_loss
        if li == n_layers - 1:
            out["kp_cls"] = cls_loss.detach()
            out["kp_pos"] = pos_loss.detach() if torch.is_tensor(pos_loss) else zero
            with torch.no_grad():
                for c, key, nkey in ((KP_CLS_START, "kp_start_err_cm", "n_start"),
                                     (KP_CLS_END, "kp_end_err_cm", "n_end")):
                    m = (gt_cls[t_t] == c) if qi.size > 0 else None
                    if m is not None and bool(m.any()):
                        err = (pos[q_t][m] - gt_pos[t_t][m]).norm(dim=-1).mean()
                        out[key] = err * coord_scale
                        out[nkey] = m.sum().float()
    out["total"] = total / max(n_layers, 1)
    return out
