"""Mask denoising for LArFormer (Mask DINO Phase B.1 adaptation).

Builds a set of "denoising queries" each anchored at a GT instance's
position-with-jitter and content-initialized to a learnable per-class
embedding. The decoder runs them in parallel with the regular queries —
attention is masked so DN ↔ regular doesn't cross-talk — and the loss
supervises each DN query DIRECTLY to its corresponding GT (no Hungarian).

The point: provide a strong auxiliary signal that bootstraps mask
specialization. The regular queries have to find their GT slice through
Hungarian matching, which is unstable in early training; the DN queries
already know which GT they're for, so their mask losses kick in
immediately. The decoder weights they update are shared with the regular
path → regular queries indirectly inherit the specialization.

What "noise" means in Phase B.1:
  - Anchor position: GT origin_coord_norm + N(0, σ²I) in coord_norm space.
  - Content: pure class-embedding lookup (independent of which SPs are
     in the slice — keeps init from leaking GT-mask structure into
     content vectors).

Optional later (Phase B.2+):
  - drop+add SP noise (mask-domain perturbation)
  - contrastive denoising (CDN positive/negative pairs)
  - per-group noise scheduling

Per-event budget:
  - dn_groups (default 3) DN queries per GT instance.
  - Total capped at max_dn_per_event (default 96). On events that
     would exceed the cap, a random subset of (group, gt) pairs is
     kept — logged so under-cap rate can be monitored.

See `docs/LArFormer.md` §18 for design context.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DNInit:
    """Per-event denoising-query bundle returned by `MaskDenoiser.forward`.

    All tensors live on the event's device and are on the same dtype.
    Empty (0-length) when the event has no GT instances or dn_groups==0.
    """
    init_q: torch.Tensor          # (Q_dn, D)   class-embedding lookup
    init_anchor: torch.Tensor     # (Q_dn, 3)   jittered GT origin_coord_norm
    gt_target_idx: torch.Tensor   # (Q_dn,)     long, ∈ [0, K) into gt_instances
    group_id: torch.Tensor        # (Q_dn,)     long, ∈ [0, dn_groups)

    @property
    def n_queries(self) -> int:
        return int(self.init_q.shape[0])


class MaskDenoiser(nn.Module):
    """Generates DN queries from per-event gt_instances.

    Args:
        token_dim:                decoder token dim D
        num_classes:              total class slots (incl. no_object, but
                                   no_object never gets a DN query — it's
                                   a "no-instance" marker)
        dn_groups:                how many DN copies per GT (default 3)
        max_dn_per_event:         cap on total DN queries per event
                                   (default 96). Random subsampling per
                                   event when exceeded.
        anchor_jitter_std:        σ of Gaussian noise added to
                                   origin_coord_norm anchors. Default 0.05
                                   ≈ 9 cm in detector space (one
                                   voxel_8cm cell).
        no_object_class_id:       which class id is no_object. Default
                                   None → num_classes-1 (matches LArFormer
                                   convention).
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        dn_groups: int = 3,
        max_dn_per_event: int = 96,
        anchor_jitter_std: float = 0.05,
        no_object_class_id: Optional[int] = None,
        min_gt_instances: int = 1,
    ):
        super().__init__()
        if dn_groups < 1:
            raise ValueError(
                f"dn_groups must be >=1; got {dn_groups}. "
                f"Set mask_denoising=None on the model to disable instead."
            )
        if max_dn_per_event < 1:
            raise ValueError(
                f"max_dn_per_event must be >=1; got {max_dn_per_event}"
            )
        self.token_dim = int(token_dim)
        self.num_classes = int(num_classes)
        self.dn_groups = int(dn_groups)
        self.max_dn_per_event = int(max_dn_per_event)
        self.anchor_jitter_std = float(anchor_jitter_std)
        self.no_object_class_id = (no_object_class_id
                                   if no_object_class_id is not None
                                   else num_classes - 1)
        # Partial-truth guard (SLICER_RETRAIN_PLAN A1-loss): DN groups
        # degenerate on events with very few GT instances (e.g. overlay
        # events with a single nu instance) — skip DN entirely there.
        self.min_gt_instances = int(min_gt_instances)

        # Per-class learnable content vector. no_object row is allocated
        # but never indexed (DN queries only exist for object-class GT
        # instances) — wastes one row but keeps indices identity-mapped
        # to the model's class id space, which is easier to reason about.
        self.class_embedding = nn.Embedding(num_classes, token_dim)
        nn.init.trunc_normal_(self.class_embedding.weight, std=0.02)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_per_event_dn(
        self,
        gt_instances: List[dict],
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> DNInit:
        """Build (gt_target_idx, group_id) then expand to (init_q, init_anchor)."""
        K = len(gt_instances)
        if K < max(1, self.min_gt_instances):
            return DNInit(
                init_q=torch.zeros(0, self.token_dim, device=device, dtype=dtype),
                init_anchor=torch.zeros(0, 3, device=device, dtype=dtype),
                gt_target_idx=torch.zeros(0, dtype=torch.long, device=device),
                group_id=torch.zeros(0, dtype=torch.long, device=device),
            )

        # Class id and anchor per GT instance.
        gt_classes = torch.tensor(
            [int(g["origin_type"]) for g in gt_instances],
            dtype=torch.long, device=device,
        )                                                            # (K,)
        # Drop any GTs whose class is no_object (shouldn't happen for
        # primary instances, but a safety net so we don't index the
        # no_object row of class_embedding).
        valid_gt = gt_classes != self.no_object_class_id
        if not valid_gt.all():
            valid_idx = valid_gt.nonzero(as_tuple=False).squeeze(-1)
            gt_instances = [gt_instances[int(i)] for i in valid_idx.tolist()]
            gt_classes = gt_classes[valid_gt]
            K = len(gt_instances)
            if K == 0:
                return DNInit(
                    init_q=torch.zeros(0, self.token_dim, device=device, dtype=dtype),
                    init_anchor=torch.zeros(0, 3, device=device, dtype=dtype),
                    gt_target_idx=torch.zeros(0, dtype=torch.long, device=device),
                    group_id=torch.zeros(0, dtype=torch.long, device=device),
                )

        # Anchors: prefer origin_coord_norm (always present in the
        # slicer-inferred schema, regardless of enable_origin_head). If
        # absent for any GT, use zero (the decoder's anchor PE will
        # still place this query at the origin — sub-optimal but
        # functional; covered by the per-event valid mask below).
        anchors = []
        any_missing = False
        for g in gt_instances:
            oc = g.get("origin_coord_norm")
            if oc is None:
                any_missing = True
                anchors.append(np.zeros(3, dtype=np.float32))
            elif isinstance(oc, torch.Tensor):
                anchors.append(oc.detach().cpu().numpy().astype(np.float32))
            else:
                anchors.append(np.asarray(oc, dtype=np.float32))
        gt_anchor = torch.tensor(
            np.stack(anchors, axis=0), dtype=dtype, device=device,
        )                                                            # (K, 3)
        # If everything is missing we still proceed — anchor PE just
        # collapses to pos_emb(0); not a great prior but loss can still
        # supervise. Warn-once via the structure being silently zero is
        # fine for now.
        del any_missing  # noqa: F841

        # Expand: dn_groups copies of each GT.
        gt_target_idx = torch.arange(K, device=device).repeat(self.dn_groups)   # (K*G,)
        group_id = (
            torch.arange(self.dn_groups, device=device)
            .repeat_interleave(K)
        )                                                                        # (K*G,)
        Q_dn = gt_target_idx.shape[0]

        # Cap per event: random subsample without replacement.
        if Q_dn > self.max_dn_per_event:
            if generator is not None:
                perm = torch.randperm(Q_dn, generator=generator, device=device)
            else:
                perm = torch.randperm(Q_dn, device=device)
            keep = perm[:self.max_dn_per_event]
            gt_target_idx = gt_target_idx[keep]
            group_id = group_id[keep]
            Q_dn = self.max_dn_per_event

        # Content: per-query class embedding (gradient flows into the
        # class_embedding table from the DN loss).
        init_q = self.class_embedding(gt_classes[gt_target_idx])     # (Q_dn, D)
        # Cast to the requested compute dtype.
        init_q = init_q.to(dtype=dtype)

        # Anchor: jittered GT origin. Jitter is sampled per-query (so
        # different DN groups for the SAME GT get different anchors —
        # otherwise the within-group queries would be redundant copies).
        anchor_base = gt_anchor[gt_target_idx]                       # (Q_dn, 3)
        if self.anchor_jitter_std > 0:
            if generator is not None:
                jitter = torch.randn(
                    Q_dn, 3, generator=generator, device=device, dtype=dtype,
                ) * self.anchor_jitter_std
            else:
                jitter = torch.randn(
                    Q_dn, 3, device=device, dtype=dtype,
                ) * self.anchor_jitter_std
            init_anchor = anchor_base + jitter
        else:
            init_anchor = anchor_base.clone()

        return DNInit(
            init_q=init_q,
            init_anchor=init_anchor,
            gt_target_idx=gt_target_idx,
            group_id=group_id,
        )

    def forward(
        self,
        gt_instances: List[dict],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> DNInit:
        """One event. Use the model's compute dtype for init_q/init_anchor."""
        return self._build_per_event_dn(
            gt_instances=gt_instances, device=device, dtype=dtype,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Self-attention mask builder (block-diagonal)
    # ------------------------------------------------------------------

    @staticmethod
    def build_self_attn_mask(
        n_regular: int,
        group_id: torch.Tensor,
    ) -> torch.Tensor:
        """Build the decoder self-attention mask covering [regular | DN] queries.

        Convention (matches `nn.MultiheadAttention.attn_mask` for bool):
        True = "block this attention edge".

        Layout:
          - regular ↔ regular: allowed
          - regular ↔ dn:       blocked (both directions)
          - dn(g)   ↔ dn(g):    allowed
          - dn(g)   ↔ dn(g'):   blocked

        Args:
          n_regular: K, number of regular queries
          group_id:  (Q_dn,) long, ∈ [0, dn_groups)

        Returns:
          (Q_total, Q_total) bool, with Q_total = n_regular + Q_dn.
        """
        device = group_id.device
        Q_dn = int(group_id.shape[0])
        Q_total = int(n_regular) + Q_dn
        mask = torch.zeros(Q_total, Q_total, dtype=torch.bool, device=device)
        if Q_dn == 0:
            return mask
        # Block cross-group edges between regular and DN.
        mask[:n_regular, n_regular:] = True
        mask[n_regular:, :n_regular] = True
        # Block DN cross-group.
        cross = (group_id.unsqueeze(0) != group_id.unsqueeze(1))     # (Q_dn, Q_dn)
        mask[n_regular:, n_regular:] = cross
        return mask
