"""
Cascade-filter helpers — pure functions for filtering an already-collated
LArFormer batch by a per-spacepoint keep mask, and for dropping batch
events whose surviving spacepoint count is zero.

Used by `CascadedSlicer.forward` (Stage-2 of the cascade) to apply the
frozen deghoster's per-SP `P(real) > τ` mask to the batched dict that
`larformer_collate` produced. The same helpers will also be useful for
Stage-3 (nu-slice carve), where the slicer's chosen nu-slice mask defines
the input set for the particle clusterer.

Why pure helpers (not methods on the model): the filter logic is
non-trivial (per-event remap of `truth_indices`, instance pruning, offset
recompute, optional fragment-index remap) and worth unit-testing in
isolation; the CascadedSlicer module just composes these.
"""

from typing import Iterable, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-SP filter
# ---------------------------------------------------------------------------

def _per_event_remap(keep_mask_flat: torch.Tensor,
                     offsets: List[int]) -> List[torch.Tensor]:
    """For each event, build a (n_sp_event,) LongTensor: old_local_idx → new_local_idx
    (−1 if dropped). Always returns CPU tensors so indexing into Python-side
    `gt_instances` dicts is cheap.
    """
    remaps: List[torch.Tensor] = []
    prev = 0
    keep_cpu = keep_mask_flat.detach().cpu()
    for end in offsets:
        ev_keep = keep_cpu[prev:end]
        n_keep = int(ev_keep.sum().item())
        remap = torch.full((end - prev,), -1, dtype=torch.long)
        survivors_local = torch.where(ev_keep)[0]
        remap[survivors_local] = torch.arange(n_keep, dtype=torch.long)
        remaps.append(remap)
        prev = end
    return remaps


def _remap_indices(idx, remap_cpu: torch.Tensor) -> torch.Tensor:
    """Remap a (M,) numpy array or LongTensor through a CPU remap table.
    Drops entries that mapped to −1. Returns a CPU LongTensor.
    """
    if isinstance(idx, np.ndarray):
        t = torch.as_tensor(idx, dtype=torch.long)
    elif isinstance(idx, torch.Tensor):
        t = idx.detach().cpu().to(torch.long)
    else:
        t = torch.as_tensor(idx, dtype=torch.long)
    if t.numel() == 0:
        return t
    new = remap_cpu[t]
    return new[new >= 0]


def filter_batch_by_keep_mask(
    data_dict: dict,
    keep_mask: torch.Tensor,
    drop_empty_instances: bool = True,
) -> dict:
    """Apply a per-spacepoint keep mask to an already-collated LArFormer batch.

    Args:
        data_dict:            output of `larformer_collate` (flat per-SP
                               tensors + `offset` + `gt_instances_per_event`).
        keep_mask:            (N_total,) bool tensor on the same device as
                               the per-SP fields. True = keep, False = drop.
        drop_empty_instances: if True, gt_instances whose remapped
                               `truth_indices` is empty are removed from
                               `gt_instances_per_event[ev]`.

    Returns:
        A new dict with the same keys as the input, where:
          - every per-SP flat tensor (shape[0] == N_total) is subset by
            `keep_mask`
          - `offset` and `n_spacepoints` are recomputed for survivors
          - `gt_instances_per_event[ev][k]["truth_indices"]` is remapped
            into the survivor index space; empty instances are pruned if
            `drop_empty_instances=True`
          - `n_gt_instances` is updated accordingly
          - `fragment_indices_per_event[ev][f]` (if present) is remapped;
            fragments with zero survivors are pruned, and the per-event
            `fragment_*` lists / `n_fragments` count are updated to match
          - per-event scalars (`names`, `runs`, `subruns`, `events`,
            `lm_score_threshold`) pass through unchanged (B is preserved;
            events with 0 survivors stay in the batch but their per-SP
            slice is empty — call `drop_empty_events` to prune them).

    Pure function: caller's `data_dict` is not mutated.
    """
    if "offset" not in data_dict:
        raise KeyError("data_dict has no 'offset' — is this a collated batch?")
    if keep_mask.dtype != torch.bool:
        raise ValueError(f"keep_mask must be bool, got {keep_mask.dtype}")
    n_total = int(data_dict["offset"][-1].item()) if int(data_dict["offset"].numel()) > 0 else 0
    if keep_mask.shape[0] != n_total:
        raise ValueError(
            f"keep_mask len ({keep_mask.shape[0]}) does not match "
            f"total spacepoint count ({n_total} from offset)"
        )

    offsets_cpu: List[int] = [int(x) for x in data_dict["offset"].detach().cpu().tolist()]
    device = keep_mask.device
    B = len(offsets_cpu)

    out: dict = {}

    # ---- Per-SP flat tensors: subset by keep_mask -----------------------
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor) and v.shape and v.shape[0] == n_total:
            out[k] = v[keep_mask]
    # 'offset' would have matched n_total only if N_total happened to equal B,
    # which is degenerate; we rewrite it explicitly below.
    if "offset" in out:
        del out["offset"]

    # ---- Per-event remaps + new offsets ---------------------------------
    remaps = _per_event_remap(keep_mask, offsets_cpu)
    new_n = torch.tensor([int((r >= 0).sum()) for r in remaps],
                         dtype=torch.long, device=device)
    out["n_spacepoints"] = new_n
    out["offset"] = torch.cumsum(new_n, dim=0)

    # ---- gt_instances_per_event: remap + optional prune -----------------
    if "gt_instances_per_event" in data_dict:
        new_gts: List[list] = []
        new_n_gts: List[int] = []
        for ev in range(B):
            remap = remaps[ev]
            new_ev_gts = []
            for g in data_dict["gt_instances_per_event"][ev]:
                old = g.get("truth_indices")
                if old is None:
                    new_ev_gts.append(dict(g))
                    continue
                new_idx = _remap_indices(old, remap)
                if drop_empty_instances and new_idx.numel() == 0:
                    continue
                new_g = dict(g)
                new_g["truth_indices"] = new_idx.numpy()
                new_g["n_truth_points"] = int(new_idx.numel())
                new_ev_gts.append(new_g)
            new_gts.append(new_ev_gts)
            new_n_gts.append(len(new_ev_gts))
        out["gt_instances_per_event"] = new_gts
        out["n_gt_instances"] = torch.tensor(
            new_n_gts, dtype=torch.long, device=device,
        )

    # ---- fragment_indices_per_event (if emit_fragments=True) ------------
    if "fragment_indices_per_event" in data_dict:
        new_frags: List[list] = []
        new_frag_trackid: List[torch.Tensor] = []
        new_frag_pid: List[torch.Tensor] = []
        new_frag_type: List[torch.Tensor] = []
        new_n_frags: List[int] = []
        for ev in range(B):
            remap = remaps[ev]
            ev_frags = data_dict["fragment_indices_per_event"][ev]
            ev_track = data_dict.get("fragment_trackid_per_event", [None] * B)[ev]
            ev_pid = data_dict.get("fragment_pid_per_event", [None] * B)[ev]
            ev_type = data_dict.get("fragment_type_per_event", [None] * B)[ev]
            keep_frag = []
            new_ev_frags = []
            for fi, idx_tensor in enumerate(ev_frags):
                new_idx = _remap_indices(idx_tensor, remap)
                if new_idx.numel() == 0:
                    continue
                new_ev_frags.append(new_idx)
                keep_frag.append(fi)
            new_frags.append(new_ev_frags)
            if ev_track is not None and len(keep_frag) > 0:
                new_frag_trackid.append(ev_track[keep_frag])
                new_frag_pid.append(ev_pid[keep_frag])
                new_frag_type.append(ev_type[keep_frag])
            else:
                new_frag_trackid.append(
                    ev_track.new_zeros(0) if ev_track is not None else None
                )
                new_frag_pid.append(
                    ev_pid.new_zeros(0) if ev_pid is not None else None
                )
                new_frag_type.append(
                    ev_type.new_zeros(0) if ev_type is not None else None
                )
            new_n_frags.append(len(new_ev_frags))
        out["fragment_indices_per_event"] = new_frags
        if "fragment_trackid_per_event" in data_dict:
            out["fragment_trackid_per_event"] = new_frag_trackid
        if "fragment_pid_per_event" in data_dict:
            out["fragment_pid_per_event"] = new_frag_pid
        if "fragment_type_per_event" in data_dict:
            out["fragment_type_per_event"] = new_frag_type
        out["n_fragments"] = torch.tensor(
            new_n_frags, dtype=torch.long, device=device,
        )

    # ---- Pass-through fields (not touched) ------------------------------
    for k, v in data_dict.items():
        if k not in out:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Per-event filter (drop events with zero survivors)
# ---------------------------------------------------------------------------

def drop_empty_events(data_dict: dict) -> dict:
    """Drop batch events that have 0 surviving spacepoints.

    Per-SP flat tensors are unchanged (dropped events already had 0 SPs);
    `offset`, `n_spacepoints`, per-event lists and per-event scalars are
    subset to surviving events.

    Returns the input dict if every event has ≥1 spacepoint (no copy).
    """
    n_per_event = data_dict["n_spacepoints"]
    keep_event = (n_per_event > 0)
    if bool(keep_event.all()):
        return data_dict

    keep_idx_cpu = torch.where(keep_event)[0].detach().cpu().tolist()
    device = n_per_event.device
    out = dict(data_dict)

    new_n = n_per_event[keep_event]
    out["n_spacepoints"] = new_n
    out["offset"] = torch.cumsum(new_n, dim=0)

    # Per-event Python-list fields
    for k in ("gt_instances_per_event", "names", "runs", "subruns", "events",
              "fragment_indices_per_event", "fragment_trackid_per_event",
              "fragment_pid_per_event", "fragment_type_per_event"):
        if k in data_dict:
            v = data_dict[k]
            out[k] = [v[i] for i in keep_idx_cpu]

    # Per-event scalar tensors
    for k in ("n_gt_instances", "n_fragments", "lm_score_threshold"):
        if k in data_dict and isinstance(data_dict[k], torch.Tensor):
            out[k] = data_dict[k][keep_event]

    return out
