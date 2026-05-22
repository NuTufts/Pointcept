"""
LArFormer ground-truth visualizer.

Reads a LArFormer training config, builds the same LArFormerDataset +
CompositeTokenizer the trainer would use, and renders per-level GT (instance
masks + per-token cls targets) in an interactive Dash app. The backbone is
NEVER instantiated — we run the level builders directly with zero-filled
per-spacepoint features. Only the geometry matters for GT visualization.

The viz queries the same code paths the loss uses:
    CompositeTokenizer   →  build_per_level_gt
so if the visualizer and trainer ever disagree, the bug is in shared code,
not in two parallel implementations. See `Pointcept/docs/LArFormer.md` §11.

Two filtering modes:

  - Default (no `--cascade-config`): uses only the dataset's lm_score
    augmentation filter (what LArFormerDataset does on every __getitem__).
    No backbone load; runs CPU-only.

  - With `--cascade-config <CascadedSlicer-cfg>`: ALSO builds the cascade's
    LoRA-tuned deghoster (loads its checkpoint) and applies the same
    `P(real) > τ` filter the cascade uses at training time. Adds a τ
    slider to the UI so you can sweep the threshold and see which
    spacepoints get dropped. Requires CUDA (Sonata backbone is heavy).

Usage:
    # Plain mode (no deghoster):
    ./run_in_container.sh python tools/visualize_larformer_gt.py \\
        --config configs/lartpc/larformer-slicer-v0.py

    # Cascade-deghoster mode (run the SAME trained deghoster as training):
    ./run_in_container.sh python tools/visualize_larformer_gt.py \\
        --config configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py \\
        --cascade-config configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py

(The two `--config` args can point at the same cascade config; `--config`
controls the dataset + levels for GT viz, `--cascade-config` controls the
deghoster model. Splitting them lets you visualize against a different
levels block than the deghoster's training config.)

Once running, open http://<host>:8050 in a browser.
"""

import argparse
import colorsys
import os
import sys

import numpy as np
import torch

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

# Repo root on sys.path so we can import pointcept + lartpc_data_prep + viz helpers.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectoroutline import DetectorOutline  # noqa: E402


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def instance_color(k: int, origin_type: int = -1) -> str:
    """Distinct rgb-ish color per instance, with origin-aware highlight.

    For slice GT, origin_type=0 is "nu" (always red so it's spotted instantly).
    Cosmic / shower instances are spread across HSV.
    """
    if origin_type == 0:
        return "rgba(255,60,60,1)"
    h = (k * 0.61803398875) % 1.0  # golden-ratio spread for visual contrast
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)"


def cls_color(c: int) -> str:
    """Stable per-class color. -1 (ignore) = light gray."""
    if c < 0:
        return "rgba(180,180,180,0.5)"
    palette = [
        "rgba(255,60,60,1)",     # 0
        "rgba(60,140,255,1)",    # 1
        "rgba(80,200,80,1)",     # 2
        "rgba(255,160,40,1)",    # 3
        "rgba(180,60,220,1)",    # 4
        "rgba(40,200,200,1)",    # 5
    ]
    return palette[c % len(palette)]


def _track_id_color(tid: int, origin_type: int = -1) -> str:
    """Color by trackid (consistent across GT and prediction panels).

    GT panel's `instance_color(k, ot)` previously hashed by instance index
    `k`, which means the same physical slice could get different colors in
    the GT vs prediction panel if their instance orderings differed.
    Hashing by trackid gives one color per physical track id, so a slice's
    color is identical wherever it shows up (top panel, bottom panel,
    cross-event browsing).

    origin_type=0 ("nu" in the slicer's class set) always renders red so
    the user can spot the nu slice instantly regardless of trackid hash.
    """
    if origin_type == 0:
        return "rgba(255,60,60,1)"
    # Golden-ratio hash on the absolute trackid; produces well-distributed
    # hues with no two consecutive trackids landing on the same color.
    h = (abs(int(tid)) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)"


# ---------------------------------------------------------------------------
# Prediction-panel hover-text helper
# ---------------------------------------------------------------------------

_PRED_HOVER_TEMPLATE = (
    "x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>"
    "mask_prob=%{customdata[0]:.3f}<br>"
    "pred_slice=%{customdata[1]}<br>"
    "gt_slice=%{customdata[2]}<extra></extra>"
)


def _pred_hover_kwargs(mask, customdata_full):
    """Return customdata + hovertemplate kwargs for a per-token Scatter3d.

    If `customdata_full` is None (older HDF5 written before per-token mask
    confidence was emitted), returns {} so the caller's Scatter3d falls
    back to the default plotly hover (no per-point mask prob shown).
    """
    if customdata_full is None:
        return {}
    return dict(
        customdata=customdata_full[mask],
        hovertemplate=_PRED_HOVER_TEMPLATE,
    )


def _build_pred_customdata(pred_mask_prob, pred_slice_id, slice_id_gt):
    """Stack the three per-token fields into a (N, 3) customdata array for
    plotly. Returns None if pred_mask_prob is missing (older HDF5)."""
    if pred_mask_prob is None:
        return None
    return np.stack([
        pred_mask_prob.astype(np.float32),
        pred_slice_id.astype(np.int64),
        slice_id_gt.astype(np.int64),
    ], axis=-1)


def _apply_pred_threshold(
    pred_mask_prob, pred_slice_id, pred_class, no_object_class_id,
    min_mask_prob,
):
    """Apply a per-token confidence threshold to a prediction triple.

    The cascade's inference rule is plain argmax over active queries — so
    a token with all queries weakly negative (e.g. max sigm ≈ 0.05) still
    gets bound to whichever query was the least bad. That's a no-threshold
    panoptic assignment, not a model commitment.

    With `min_mask_prob > 0`, tokens whose assigned-query sigm falls below
    the threshold are demoted to "unassigned" (pred_slice_id = -1, pred_class
    = no_object). The returned low_conf mask lets pred_correct coloring
    distinguish "model wrong" from "model unsure".

    Returns (pred_slice_id', pred_class', low_conf_mask). If
    pred_mask_prob is None (older HDF5) the arrays are returned unchanged
    and low_conf_mask is all-False.
    """
    n = pred_slice_id.shape[0]
    if pred_mask_prob is None or min_mask_prob <= 0.0 or n == 0:
        return pred_slice_id, pred_class, np.zeros(n, dtype=bool)
    low_conf = pred_mask_prob < float(min_mask_prob)
    out_slice = pred_slice_id.copy()
    out_class = pred_class.copy()
    out_slice[low_conf] = -1
    out_class[low_conf] = int(no_object_class_id)
    return out_slice, out_class, low_conf


# ---------------------------------------------------------------------------
# Slicer-prediction H5 reader
# ---------------------------------------------------------------------------

def _load_slicerpred(path: str) -> "dict | None":
    """Load one `slicerpred_<basename>.h5` file produced by
    `tools/run_slicer_inference.py` into a flat dict of numpy arrays +
    scalars. Returns None if the file doesn't exist (caller falls back to
    a "no prediction available" placeholder).

    The returned dict mirrors the H5's group/key structure as `group/key`
    strings (e.g. `post/coord`, `gt/primary_trackid`). Scalar attributes
    live under `meta/*` keys with the `meta_` prefix dropped.
    """
    import h5py
    if not os.path.exists(path):
        return None
    out: dict = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
        for k, v in f.attrs.items():
            # h5py stores scalars as numpy types; coerce to native for
            # ease of downstream use.
            if hasattr(v, "item"):
                out[k] = v.item()
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Cascade deghoster filter (the LoRA path)
# ---------------------------------------------------------------------------

class CascadeDeghostFilter:
    """Builds the cascade's deghoster + applies the same P(real) > τ filter.

    Holds:
      - `deghoster`: an nn.Module built from the cascade config's
        `deghoster` subconfig, with its trained weights loaded
      - `class_index_real`: which column of the deghoster's softmax is the
        real-class probability (0 for SonataLoRADeghostSegmentor, 1 for
        the LArFormer-flavored deghoster — same convention as the cascade
        config's `deghoster_class_index_real`)

    Used by the visualizer to mirror exactly what `CascadedSlicer.forward`
    does at training time: collate one event, run the deghoster, build
    the keep mask, filter via `filter_batch_by_keep_mask`. Then the rest
    of the GT-build pipeline sees only the deghosted spacepoints —
    matching what the slicer is trained on.
    """

    def __init__(
        self,
        deghoster_cfg: dict,
        deghoster_weight: "str | None",
        class_index_real: int,
        device: str = "cuda",
    ):
        from pointcept.models.builder import build_model
        from pointcept.models.LArFormer.cascaded import CascadedSlicer
        self.deghoster = build_model(deghoster_cfg).to(device).eval()
        if deghoster_weight is not None:
            CascadedSlicer._load_state_dict_into(
                self.deghoster, deghoster_weight, "deghoster",
            )
        self.class_index_real = int(class_index_real)
        self.device = device

    @torch.no_grad()
    def p_real(self, batched_dict: dict) -> torch.Tensor:
        """Run the deghoster + return flat (N_total,) P(real). Handles both
        SonataLoRA-style (`seg_logits`) and LArFormer-style (`predictions`)
        output conventions — same dispatch as
        CascadedSlicer._run_deghoster_p_real."""
        out = self.deghoster(batched_dict)
        if "seg_logits" in out:
            return out["seg_logits"].softmax(dim=-1)[:, self.class_index_real]
        if "predictions" in out:
            chunks = []
            for ev_pred in out["predictions"]:
                logits = ev_pred["per_level_cls"]["spacepoint"]
                chunks.append(
                    logits.softmax(dim=-1)[:, self.class_index_real]
                )
            return torch.cat(chunks, dim=0)
        raise RuntimeError(
            f"Unrecognized deghoster output keys: {list(out.keys())}"
        )


def _unpack_single_event_sample(batched: dict) -> dict:
    """Convert a 1-event (post-filter) batched dict back to a dataset-style
    per-event sample dict (numpy per-SP fields, scalar per-event fields).

    Mirrors what LArFormerDataset.__getitem__ would return for a single
    event, so downstream code (`build_event_gt`) doesn't need to branch.
    """
    assert int(batched["offset"].numel()) == 1, "expected 1-event batched dict"
    sample: dict = {}
    for k in ("coord", "coord_norm", "grid_coord", "feat", "lm_score", "wire",
              "trackid", "pid", "origin_label", "hasmatch", "ssnet_label",
              "slice_id"):
        if k in batched:
            v = batched[k]
            sample[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    sample["n_spacepoints"] = int(batched["n_spacepoints"][0].item())
    sample["gt_instances"] = batched["gt_instances_per_event"][0]
    sample["n_gt_instances"] = int(batched["n_gt_instances"][0].item()) \
        if "n_gt_instances" in batched else len(sample["gt_instances"])
    for plural, singular in (("names", "name"), ("runs", "run"),
                             ("subruns", "subrun"), ("events", "event")):
        if plural in batched and len(batched[plural]) > 0:
            sample[singular] = batched[plural][0]
    if "lm_score_threshold" in batched:
        sample["lm_score_threshold"] = float(batched["lm_score_threshold"][0])
    return sample


# ---------------------------------------------------------------------------
# GT extraction (the visualizer's only model-side call site)
# ---------------------------------------------------------------------------

def _replicate_ptv3_pool_chain(
    grid_coord: torch.Tensor,    # (N_sp, 3) integer grid coords
    coord_norm: torch.Tensor,    # (N_sp, 3) float, normalized
    stages_needed: list,          # e.g. ["dec1", "dec2", "dec3"]
    in_dims: dict,                # {stage_name: int (= dec_channels[stage])}
    encoder_strides: tuple = (2, 2, 2, 2),
) -> dict:
    """Reproduce PT-v3m2's GridPooling chain WITHOUT running the backbone.

    GridPooling at stage K is purely a function of `grid_coord` and `batch`:
    it computes `unique(batch + floor(grid_coord/stride))` and records the
    inverse map (sp_to_cluster). Since the visualizer only needs each
    decoder stage's `coords` and `sp_to_level_id` for GT plotting, we can
    replicate this deterministically without touching the model.

    Tokens are zero-filled (visualizer never renders them; only the
    geometry matters for GT viz).

    Returns a dict matching what `LArFormer._build_decoder_stages_per_event`
    produces for a single event:
        {stage_name: {"tokens": (M, in_dim) zeros,
                      "coords": (M, 3) cluster-mean coord_norm,
                      "sp_to_level_id": (N_sp,) cluster ID per SP}}
    """
    out = {}
    if grid_coord.dtype != torch.long:
        grid_coord = grid_coord.long()
    cum_stride = 1
    for stage_idx in range(1, len(encoder_strides) + 1):
        cum_stride *= int(encoder_strides[stage_idx - 1])
        stage_name = f"dec{stage_idx}"
        if stage_name not in stages_needed:
            continue
        stage_grid = torch.div(grid_coord, cum_stride, rounding_mode="trunc")
        # Single-event visualization: batch is identically 0, so the
        # batch-packing trick GridPooling uses isn't needed — plain
        # torch.unique over the (N_sp, 3) grid is equivalent and avoids
        # the int64-bit-shift gymnastics.
        _, inverse = torch.unique(
            stage_grid, dim=0, return_inverse=True, sorted=True,
        )
        M = int(inverse.max().item()) + 1
        # Cluster centroids in coord_norm space (matches GridPooling's
        # reduce="mean" on coord).
        coords_stage = torch.zeros(
            M, 3, dtype=coord_norm.dtype, device=coord_norm.device,
        )
        counts = torch.zeros(
            M, dtype=torch.long, device=coord_norm.device,
        )
        coords_stage.index_add_(0, inverse, coord_norm)
        counts.index_add_(0, inverse, torch.ones_like(inverse))
        coords_stage = coords_stage / counts.unsqueeze(-1).float().clamp(min=1)
        out[stage_name] = {
            "tokens": torch.zeros(
                M, int(in_dims[stage_name]),
                dtype=coord_norm.dtype, device=coord_norm.device,
            ),
            "coords": coords_stage,
            "sp_to_level_id": inverse.to(torch.long),
        }
    return out


def build_event_gt(model_levels_cfg, token_dim, dataset, idx,
                   deghoster_filter: "CascadeDeghostFilter | None" = None,
                   tau: float = 0.5):
    """Run the same level builders + GT lifters the loss uses, on one event.

    Returns:
        dict with keys:
            sample      — raw dataset[idx] output
            coord       — (N, 3) detector cm spacepoint coords
            coord_norm  — (N, 3) normalized
            levels      — OrderedDict[level_name → LevelOutput]
            per_level_gt — OrderedDict[level_name → {instance_mask, cls_target}]
            level_instance_per_token — OrderedDict[level_name → (M,) int64]
                          (which instance owns each token, or −1 if none)
    """
    from pointcept.models.LArFormer import (
        CompositeTokenizer, build_per_level_gt, filter_batch_by_keep_mask,
    )
    from pointcept.datasets import larformer_collate

    sample = dataset[idx]
    n_sp_pre = int(sample["n_spacepoints"])
    deghost_info: "dict | None" = None

    # Always-available "data composition" info from the dataset-side
    # hasmatch field (post-lm_score-filter, pre-deghoster). hasmatch=1 = real,
    # hasmatch=0 = ghost. Useful baseline even with the deghoster off.
    composition_info = None
    if "hasmatch" in sample:
        hm_pre = np.asarray(sample["hasmatch"])
        n_real_pre = int((hm_pre == 1).sum())
        n_ghost_pre = int((hm_pre == 0).sum())
        composition_info = {
            "n_real_pre": n_real_pre,
            "n_ghost_pre": n_ghost_pre,
            "real_frac": n_real_pre / max(n_real_pre + n_ghost_pre, 1),
        }

    # ---- Optional cascade-deghoster filter ---------------------------------
    if deghoster_filter is not None:
        # Mirror CascadedSlicer.forward: 1-event collate → deghoster →
        # P(real) > τ → filter_batch_by_keep_mask → unpack back to a
        # sample-like dict for the rest of the pipeline.
        batched = larformer_collate([sample])
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.to(deghoster_filter.device, non_blocking=True)
        p_real = deghoster_filter.p_real(batched)
        keep = p_real > float(tau)
        n_post = int(keep.sum().item())

        # Per-class confusion against the dataset's hasmatch GT, computed
        # from the BATCHED dict (still pre-filter at this point). Gives us
        # real-recall (fraction of true real points kept) and ghost-reject
        # (fraction of true ghosts dropped). These are the operational
        # quality numbers the user actually cares about.
        hm_t = batched.get("hasmatch")
        deghost_metrics = {}
        if hm_t is not None:
            real_mask = (hm_t == 1)
            ghost_mask = (hm_t == 0)
            n_real_pre_dg = int(real_mask.sum().item())
            n_ghost_pre_dg = int(ghost_mask.sum().item())
            n_real_kept = int((keep & real_mask).sum().item())
            n_ghost_kept = int((keep & ghost_mask).sum().item())
            deghost_metrics = {
                "n_real_pre": n_real_pre_dg,
                "n_ghost_pre": n_ghost_pre_dg,
                "n_real_kept": n_real_kept,
                "n_ghost_kept": n_ghost_kept,
                "real_recall": (n_real_kept / n_real_pre_dg
                                if n_real_pre_dg > 0 else float("nan")),
                "ghost_reject": (1.0 - n_ghost_kept / n_ghost_pre_dg
                                 if n_ghost_pre_dg > 0 else float("nan")),
            }

        if n_post == 0:
            # All SPs dropped at this τ — fall back to NO filter so the user
            # still gets something to look at, and surface the issue in the
            # metadata panel. Avoids a hard crash if τ is set unreasonably.
            deghost_info = {
                "tau": float(tau), "n_sp_pre": n_sp_pre,
                "n_sp_post": 0, "keep_frac": 0.0,
                **deghost_metrics,
                "warning": ("τ dropped every SP; showing pre-filter GT. "
                            "Lower τ to recover."),
            }
        else:
            filtered = filter_batch_by_keep_mask(batched, keep)
            sample = _unpack_single_event_sample(filtered)
            deghost_info = {
                "tau": float(tau), "n_sp_pre": n_sp_pre,
                "n_sp_post": int(sample["n_spacepoints"]),
                "keep_frac": int(sample["n_spacepoints"]) / max(n_sp_pre, 1),
                **deghost_metrics,
            }

    n_sp = int(sample["n_spacepoints"])
    # Build a minimal "event_dict" matching what LArFormer.forward() would
    # hand to a level builder.
    coord_norm = torch.from_numpy(sample["coord_norm"])
    feat = torch.from_numpy(sample["feat"])
    event_dict = {
        "coord_norm": coord_norm, "feat": feat, "n_sp": n_sp,
    }
    if "fragment_indices" in sample:
        event_dict["fragment_indices"] = [
            torch.from_numpy(idx) for idx in sample["fragment_indices"]
        ]

    # Detect PTv3DecoderStageLevel builders. If any are declared, we need
    # to populate event_dict["ptv3_dec_stages"] BEFORE the tokenizer runs
    # — the builder reads its coords + sp_to_level_id from there. We
    # replicate PT-v3m2's deterministic GridPooling chain directly off
    # the dataset's grid_coord (no backbone forward required for GT viz).
    ptv3_stages_in_dims = {}
    for lc in model_levels_cfg:
        if lc.get("builder") == "PTv3DecoderStageLevel":
            bcfg = lc.get("builder_cfg") or {}
            sk = bcfg.get("stage_key")
            id_ = bcfg.get("in_dim")
            if sk is not None and id_ is not None:
                ptv3_stages_in_dims[str(sk)] = int(id_)
    if ptv3_stages_in_dims:
        if "grid_coord" not in sample:
            raise KeyError(
                "PTv3DecoderStageLevel needs sample['grid_coord'] for "
                "pool-chain replication, but the dataset didn't emit it. "
                "Check that LArFormerDataset returns grid_coord."
            )
        gc = torch.from_numpy(sample["grid_coord"])
        event_dict["ptv3_dec_stages"] = _replicate_ptv3_pool_chain(
            grid_coord=gc,
            coord_norm=coord_norm,
            stages_needed=list(ptv3_stages_in_dims.keys()),
            in_dims=ptv3_stages_in_dims,
        )

    # We don't have a backbone — zero-fill per-SP features. The builders'
    # in_proj weights are random (the tokenizer is freshly constructed per
    # visualization session), but tokens are never rendered — only coords +
    # sp_to_level_id matter for GT viz. in_dim only fixes the linear
    # projection shape; 8 is arbitrary.
    in_dim = 8
    sp_feat = torch.zeros(n_sp, in_dim, dtype=torch.float32)

    tok = CompositeTokenizer(
        levels_cfg=model_levels_cfg, in_dim=in_dim, token_dim=token_dim,
    )
    levels = tok(sp_feat, coord_norm, event_dict)

    # Per-SP labels — pull every per-SP int field the dataset emits so any
    # level with supervision.cls can find its label_src.
    per_sp_labels = {}
    for k in ("hasmatch", "origin_label", "ssnet_label",
              "trackid", "pid", "slice_id"):
        if k in sample:
            per_sp_labels[k] = torch.from_numpy(sample[k]).to(torch.long)

    per_level_gt = build_per_level_gt(
        levels=levels,
        gt_instances=sample["gt_instances"],
        per_sp_labels=per_sp_labels,
        levels_cfg=model_levels_cfg,
    )

    # Reduce (K, M) instance mask → (M,) "which instance owns this token"
    # (argmax; −1 if no instance overlaps).
    level_inst = {}
    for name, lvl in levels.items():
        gt = per_level_gt[name]["instance_mask"]
        if gt is None or gt.shape[0] == 0 or gt.shape[1] == 0:
            level_inst[name] = -1 * np.ones(lvl.n_tokens, dtype=np.int64)
            continue
        has_any = (gt.sum(dim=0) > 0).cpu().numpy()
        argmax = gt.argmax(dim=0).cpu().numpy()
        out = np.where(has_any, argmax, -1).astype(np.int64)
        level_inst[name] = out

    return {
        "sample": sample,
        "coord": sample["coord"],
        "coord_norm": sample["coord_norm"],
        "levels": levels,
        "per_level_gt": per_level_gt,
        "level_instance_per_token": level_inst,
        "deghost_info": deghost_info,
        "composition_info": composition_info,
    }


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def make_detector_outline_trace():
    do = DetectorOutline()
    traces = []
    for line in (do.top_pts, do.bot_pts):
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        zs = [p[2] for p in line]
        traces.append(go.Scatter3d(
            x=zs, y=xs, z=ys, mode="lines",
            line=dict(color="rgba(120,120,120,0.4)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    # Verticals
    for i in range(4):
        a = do.top_pts[i]; b = do.bot_pts[i]
        traces.append(go.Scatter3d(
            x=[a[2], b[2]], y=[a[0], b[0]], z=[a[1], b[1]],
            mode="lines",
            line=dict(color="rgba(120,120,120,0.4)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def figure_for_event(event_data, level_name, color_by):
    """Build the 3D Plotly figure for one (event, level, color_by) triple."""
    sample = event_data["sample"]
    coord = event_data["coord"]
    levels = event_data["levels"]
    per_level_gt = event_data["per_level_gt"]
    level_inst = event_data["level_instance_per_token"]

    if level_name not in levels:
        return go.Figure(data=[],
                         layout=go.Layout(
                             title=f"level {level_name!r} not in this config"))

    lvl = levels[level_name]
    coords_norm = lvl.coords.cpu().numpy()
    # Denormalize back to detector cm to match the spacepoint context cloud
    coords_cm = coords_norm * sample.get("coord_scale_value", 179.55)  # we'll
    # not depend on that — pull from the dataset's own conversion: tokens
    # were built from coord_norm, and we have coord in cm. The two share an
    # affine relationship. Simpler: use coord_norm directly for the figure
    # axes; we plot SPACE in detector cm by reconstructing the inverse
    # transform from (coord, coord_norm) of the spacepoint level.
    sp_coord_norm = event_data["coord_norm"]
    sp_coord_cm = event_data["coord"]
    # Linear fit per axis (constant scale, single offset per axis — exact
    # since dataset uses (coord - center) / scale).
    scale = (sp_coord_cm.max(axis=0) - sp_coord_cm.min(axis=0)) / np.maximum(
        sp_coord_norm.max(axis=0) - sp_coord_norm.min(axis=0), 1e-9,
    )
    center = sp_coord_cm.mean(axis=0) - sp_coord_norm.mean(axis=0) * scale
    coords_plot = coords_norm * scale + center

    traces = make_detector_outline_trace()

    # Context: all spacepoints, gray, low-alpha
    sp_step = max(1, len(sp_coord_cm) // 50_000)  # subsample for browser perf
    sp_sub = sp_coord_cm[::sp_step]
    traces.append(go.Scatter3d(
        x=sp_sub[:, 2], y=sp_sub[:, 0], z=sp_sub[:, 1],
        mode="markers",
        marker=dict(size=1.0, color="rgba(170,170,170,0.25)"),
        name=f"all spacepoints ({len(sp_coord_cm)})",
        hoverinfo="skip",
    ))

    # Tokens at the chosen level
    if color_by == "instance_id":
        inst = level_inst[level_name]
        n_inst = max(int(inst.max()) + 1, 1) if inst.size else 0
        # Build one trace per instance for legend control
        gt_instances = sample["gt_instances"]
        used = sorted(set(int(x) for x in inst if x >= 0))
        if not used:
            traces.append(go.Scatter3d(
                x=coords_plot[:, 2], y=coords_plot[:, 0], z=coords_plot[:, 1],
                mode="markers",
                marker=dict(size=3, color="rgba(200,200,200,0.5)"),
                name=f"{level_name} (no GT)",
            ))
        else:
            # "no instance" tokens
            mask = (inst < 0)
            if mask.any():
                pts = coords_plot[mask]
                traces.append(go.Scatter3d(
                    x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                    mode="markers",
                    marker=dict(size=2.5, color="rgba(150,150,150,0.4)"),
                    name=f"{level_name}: unassigned ({int(mask.sum())})",
                ))
            for k in used:
                m = (inst == k)
                if not m.any():
                    continue
                gi = gt_instances[k] if k < len(gt_instances) else {}
                ot = int(gi.get("origin_type", -1))
                # Color by trackid (primary_trackid for slice mode,
                # trunk_trackid for shower mode, fall back to instance k
                # if neither is set). This way the same physical slice
                # gets the same color in the prediction panel below.
                tid = int(gi.get("primary_trackid",
                                 gi.get("trunk_trackid", k)))
                clr = _track_id_color(tid, origin_type=ot)
                pts = coords_plot[m]
                label_bits = [f"k={k}", f"tid={tid}"]
                if "origin_type" in gi:
                    label_bits.append(f"type={ot}")
                if "primary_origin" in gi:
                    po = int(gi['primary_origin'])
                    label_bits.append(
                        "nu" if po == 1 else ("cosmic" if po == 2 else f"po={po}")
                    )
                if "pid" in gi:
                    label_bits.append(f"pid={int(gi['pid'])}")
                if "n_truth_points" in gi:
                    label_bits.append(f"npt={int(gi['n_truth_points'])}")
                traces.append(go.Scatter3d(
                    x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                    mode="markers",
                    marker=dict(size=4, color=clr),
                    name=" | ".join(label_bits),
                ))
    elif color_by == "cls_target":
        cls = per_level_gt[level_name]["cls_target"]
        if cls is None:
            traces.append(go.Scatter3d(
                x=coords_plot[:, 2], y=coords_plot[:, 0], z=coords_plot[:, 1],
                mode="markers",
                marker=dict(size=3, color="rgba(200,200,200,0.5)"),
                name=f"{level_name}: no cls supervision",
            ))
        else:
            cls_np = cls.cpu().numpy()
            for c in sorted(set(int(x) for x in cls_np)):
                m = (cls_np == c)
                pts = coords_plot[m]
                clr = cls_color(c)
                traces.append(go.Scatter3d(
                    x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                    mode="markers",
                    marker=dict(size=4, color=clr),
                    name=f"cls={c} ({int(m.sum())})",
                ))
    elif color_by == "sp_to_level_id":
        # Recolor spacepoints by which level token they map to. Useful
        # sanity check for the voxel / fragment partition.
        sp_to_lvl = lvl.sp_to_level_id.cpu().numpy()
        unmapped = (sp_to_lvl < 0)
        if unmapped.any():
            pts = sp_coord_cm[unmapped]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.5, color="rgba(80,80,80,0.4)"),
                name=f"unmapped SPs ({int(unmapped.sum())})",
            ))
        mapped = (sp_to_lvl >= 0)
        if mapped.any():
            pts = sp_coord_cm[mapped]
            ids = sp_to_lvl[mapped]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=ids,
                    colorscale="Rainbow",
                    showscale=True,
                    colorbar=dict(title="level token id", thickness=12),
                ),
                name=f"mapped SPs ({int(mapped.sum())})",
            ))

    elif color_by == "sp_by_level_inst":
        # "Back-projected" rendering: plot ALL spacepoints at their original
        # cm coords, but color each SP by the GT instance the level's
        # *cluster* (containing that SP) was assigned. This shows, for
        # pyramid levels where cluster centroids coincide with member SPs
        # (e.g. PTv3 dec1 at 0.5 cm effective grid), exactly what GT
        # assignment the loss "sees" AT that level — including any two
        # physically distinct objects that happen to pool to the same
        # cluster and therefore inherit the same level-side GT id.
        inst = level_inst[level_name]                # (M,) int64
        sp_to_lvl = lvl.sp_to_level_id.cpu().numpy()  # (N_sp,)
        # Back-project: per-SP cluster GT id.
        sp_inst = np.full(sp_to_lvl.shape[0], -1, dtype=np.int64)
        mapped = (sp_to_lvl >= 0)
        sp_inst[mapped] = inst[sp_to_lvl[mapped]]

        gt_instances = sample["gt_instances"]
        # Unmapped / no-instance SPs go to a dim-gray bucket.
        unassigned = (sp_inst < 0)
        if unassigned.any():
            pts = sp_coord_cm[unassigned]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.2, color="rgba(150,150,150,0.4)"),
                name=f"unassigned SPs ({int(unassigned.sum())})",
            ))
        # Per-instance traces, color hashed by primary_trackid (matches
        # `instance_id` mode, the prediction panel, and cross-level
        # comparisons).
        used = sorted(set(int(x) for x in sp_inst if x >= 0))
        for k in used:
            m = (sp_inst == k)
            if not m.any():
                continue
            gi = gt_instances[k] if k < len(gt_instances) else {}
            ot = int(gi.get("origin_type", -1))
            tid = int(gi.get("primary_trackid",
                             gi.get("trunk_trackid", k)))
            clr = _track_id_color(tid, origin_type=ot)
            pts = sp_coord_cm[m]
            label = f"k={k} tid={tid} ({int(m.sum())} SPs)"
            if ot == 0:
                label += " [nu]"
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.5, color=clr),
                name=label,
            ))

    title = (f"event {sample.get('event', '?')} run={sample.get('run', '?')} "
             f"subrun={sample.get('subrun', '?')}  "
             f"level={level_name}  n_tokens={lvl.n_tokens}  "
             f"n_sp={sample['n_spacepoints']}  "
             f"n_gt={sample['n_gt_instances']}")
    fig = go.Figure(data=traces, layout=go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="z (cm)"),
            yaxis=dict(title="x (cm)"),
            zaxis=dict(title="y (cm)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    ))
    return fig


# ---------------------------------------------------------------------------
# Prediction figure
# ---------------------------------------------------------------------------

def _figure_for_prediction_at_level(pred: dict, color_by: str,
                                    level: str,
                                    min_mask_prob: float = 0.0) -> go.Figure:
    """Render the per-voxel prediction at a non-spacepoint `level`.

    Expects keys `levels/<level>/{coord, pred_query, pred_class,
    pred_slice_id, slice_id_gt, pred_correct}` written by
    `run_slicer_inference.py`. If the requested level isn't present,
    returns an explanatory placeholder figure.
    """
    coord_key = f"levels/{level}/coord"
    if coord_key not in pred:
        emitted = pred.get("voxel_levels", "")  # set in inference script meta
        return go.Figure(data=[], layout=go.Layout(
            title=f"Level {level!r} not in this prediction file. "
                  f"(emitted voxel levels: {emitted or '<none>'}). Re-run "
                  f"tools/run_slicer_inference.py with the current script "
                  f"version to regenerate.",
            margin=dict(l=0, r=0, b=0, t=40),
        ))

    traces = make_detector_outline_trace()
    coord = pred[coord_key]                                  # (M, 3) cm
    pred_slice_id = pred[f"levels/{level}/pred_slice_id"]    # (M,)
    pred_class = pred[f"levels/{level}/pred_class"]          # (M,)
    pred_correct = pred[f"levels/{level}/pred_correct"].astype(bool)
    slice_id_gt = pred[f"levels/{level}/slice_id_gt"]        # (M,)
    pred_mask_prob = pred.get(f"levels/{level}/pred_mask_prob")  # (M,) or None

    customdata_full = _build_pred_customdata(
        pred_mask_prob, pred_slice_id, slice_id_gt,
    )

    # Apply confidence threshold (no-op when min_mask_prob == 0 or when the
    # HDF5 doesn't carry per-token mask probs). Low-conf tokens become
    # "unassigned" for pred_slice_id / pred_class coloring; pred_correct
    # gets a separate low-conf bucket so it isn't conflated with "wrong".
    no_obj = int(pred.get("meta_no_object_class_id", 2))
    pred_slice_id, pred_class, low_conf = _apply_pred_threshold(
        pred_mask_prob, pred_slice_id, pred_class, no_obj, min_mask_prob,
    )
    pred_correct = pred_correct & ~low_conf

    # Per-voxel marker is bigger than the SP marker so a sparser level
    # stays visible. Tuned roughly by 1/cuberoot(M).
    M = int(coord.shape[0])
    #marker_size = max(2.5, 12.0 * (1000.0 / max(M, 100)) ** (1.0 / 3.0)) # too big, hard to see
    marker_size = 2.0

    primary_trackid_gt = pred.get("gt/primary_trackid", np.zeros(0, dtype=np.int64))
    origin_type_gt = pred.get("gt/origin_type", np.zeros(0, dtype=np.int64))
    tid_to_origin = {int(tid): int(ot) for tid, ot
                     in zip(primary_trackid_gt, origin_type_gt)}

    if color_by == "pred_correct":
        # 4 buckets when threshold > 0; the "low-conf" bucket only fires
        # when min_mask_prob > 0 AND pred_mask_prob is available, since
        # `low_conf` is all-False otherwise.
        correct_mask = pred_correct & (slice_id_gt >= 0) & ~low_conf
        wrong_mask = (~pred_correct) & (slice_id_gt >= 0) & ~low_conf
        low_conf_mask = low_conf & (slice_id_gt >= 0)
        unmatched_gt_mask = (slice_id_gt < 0)
        for mask, color, name in [
            (correct_mask, "rgba(60,200,80,0.9)",
             f"correct ({int(correct_mask.sum())})"),
            (wrong_mask, "rgba(255,60,60,0.9)",
             f"wrong ({int(wrong_mask.sum())})"),
            (low_conf_mask, "rgba(255,200,60,0.9)",
             f"low-conf (<{min_mask_prob:.2f}) ({int(low_conf_mask.sum())})"),
            (unmatched_gt_mask, "rgba(150,150,150,0.4)",
             f"no GT slice ({int(unmatched_gt_mask.sum())})"),
        ]:
            if not mask.any():
                continue
            pts = coord[mask]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size, color=color),
                name=name,
                **_pred_hover_kwargs(mask, customdata_full),
            ))

    elif color_by == "pred_slice_id":
        unmapped = (pred_slice_id < 0)
        if unmapped.any():
            pts = coord[unmapped]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size * 0.8,
                            color="rgba(150,150,150,0.4)"),
                name=f"unassigned ({int(unmapped.sum())})",
                **_pred_hover_kwargs(unmapped, customdata_full),
            ))
        for tid in sorted(set(int(t) for t in pred_slice_id if t >= 0)):
            m = (pred_slice_id == tid)
            ot = tid_to_origin.get(tid, -1)
            pts = coord[m]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size,
                            color=_track_id_color(tid, ot)),
                name=f"pred tid={tid} ({int(m.sum())})"
                     + (" [nu]" if ot == 0 else ""),
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "slice_id_gt":
        ghost = (slice_id_gt < 0)
        if ghost.any():
            pts = coord[ghost]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size * 0.8,
                            color="rgba(150,150,150,0.4)"),
                name=f"no GT slice ({int(ghost.sum())})",
                **_pred_hover_kwargs(ghost, customdata_full),
            ))
        for tid in sorted(set(int(t) for t in slice_id_gt if t >= 0)):
            m = (slice_id_gt == tid)
            ot = tid_to_origin.get(tid, -1)
            pts = coord[m]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size,
                            color=_track_id_color(tid, ot)),
                name=f"GT tid={tid} ({int(m.sum())})"
                     + (" [nu]" if ot == 0 else ""),
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "pred_class":
        no_obj = int(pred.get("meta_no_object_class_id", 2))
        for c in sorted(set(int(x) for x in pred_class)):
            m = (pred_class == c)
            pts = coord[m]
            label = "no_object" if c == no_obj else f"class={c}"
            color = "rgba(150,150,150,0.4)" if c == no_obj else cls_color(c)
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=marker_size, color=color),
                name=f"{label} ({int(m.sum())})",
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "p_real":
        # p_real is pre-deghost per-SP only; not meaningful at a voxel level.
        return go.Figure(data=[], layout=go.Layout(
            title=f"p_real coloring is spacepoint-only — not defined at "
                  f"level {level!r}. Switch level to 'spacepoint' or pick "
                  f"a different color mode.",
            margin=dict(l=0, r=0, b=0, t=40),
        ))

    n_pre = pred.get("meta_n_sp_pre", 0)
    n_post = pred.get("meta_n_sp_post", 0)
    tau = pred.get("meta_tau", float("nan"))
    title = (
        f"PREDICTION  level={level}  M={M}  "
        f"event={pred.get('meta_event', '?')} "
        f"run={pred.get('meta_run', '?')} "
        f"subrun={pred.get('meta_subrun', '?')}  |  "
        f"n_sp {n_pre} → {n_post}  τ={tau:.3f}  "
        f"matched={pred.get('meta_n_matched', 0)}/"
        f"{pred.get('meta_n_gt_instances', 0)}"
    )
    return go.Figure(data=traces, layout=go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="z (cm)"),
            yaxis=dict(title="x (cm)"),
            zaxis=dict(title="y (cm)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    ))


def figure_for_prediction(pred: "dict | None", color_by: str,
                          level: str = "spacepoint",
                          min_mask_prob: float = 0.0) -> go.Figure:
    """Build the 3D Plotly figure for the slicer's per-event prediction.

    Args:
        pred: dict loaded by `_load_slicerpred` (or None if no file).
        color_by: see color-mode list below.
        level: which level's mask predictions to render.
                - "spacepoint" (default) — uses the cascade's post-filter SP
                   predictions (post/pred_*). Most fine-grained view.
                - any other name (e.g. "voxel_10cm") — uses the per-voxel
                   per-level predictions emitted by `run_slicer_inference.py`
                   under `levels/<name>/*`. Useful for comparing merger
                   behavior across mask resolutions (the same mask_embed
                   produces masks at every level, so a merge present at the
                   spacepoint level should appear at the voxel levels too
                   — see docs/LArFormer.md §15).

    Color modes (spacepoint level supports all five; voxel levels skip
    `p_real` because it's pre-deghost-only):
      - "pred_correct"   green / red overlay on the post-filter / per-voxel
                          tokens (the fastest way to see where the model
                          goes wrong)
      - "pred_slice_id"  each token colored by its predicted slice id;
                          uses _track_id_color so the same trackid renders
                          the same color across panels
      - "slice_id_gt"    same tokens colored by their TRUE slice id (per-
                          voxel plurality vote at non-spacepoint levels)
      - "pred_class"     nu / cosmic / no_object class assignment
      - "p_real"         deghoster's P(real); continuous colorscale on
                          all pre-filter SPs (spacepoint level only)
    """
    if pred is None:
        return go.Figure(
            data=[],
            layout=go.Layout(
                title="No prediction file for this event "
                      "(set --slicerpred-dir on the CLI and run "
                      "tools/run_slicer_inference.py first).",
                margin=dict(l=0, r=0, b=0, t=40),
            ),
        )

    if level != "spacepoint":
        return _figure_for_prediction_at_level(
            pred, color_by, level, min_mask_prob=min_mask_prob,
        )

    traces = make_detector_outline_trace()

    pre_coord = pred["pre/coord"]                     # (n_pre, 3) cm
    keep_mask = pred["pre/keep"].astype(bool)          # (n_pre,) bool

    # The post-filter arrays in the H5 are aligned with `pre_coord[keep_mask]`
    # in original order. Recover the post-filter cm coords by indexing.
    post_coord_cm = pre_coord[keep_mask]
    post_slice_id_gt = pred["post/slice_id_gt"]
    pred_slice_id = pred["post/pred_slice_id"]
    pred_class = pred["post/pred_class"]
    pred_correct = pred["post/pred_correct"].astype(bool)
    pred_mask_prob = pred.get("post/pred_mask_prob")  # (n_post,) or None
    p_real_pre = pred["pre/p_real"]

    customdata_full = _build_pred_customdata(
        pred_mask_prob, pred_slice_id, post_slice_id_gt,
    )

    # Apply per-token confidence threshold (no-op when min_mask_prob == 0
    # or when older HDF5 lacks per-token mask probs). Low-conf tokens
    # become "unassigned" for the pred_slice_id / pred_class color modes;
    # pred_correct gets its own low-conf bucket (yellow) so it isn't
    # conflated with model-confidently-wrong (red).
    no_obj = int(pred.get("meta_no_object_class_id", 2))
    pred_slice_id, pred_class, low_conf = _apply_pred_threshold(
        pred_mask_prob, pred_slice_id, pred_class, no_obj, min_mask_prob,
    )
    pred_correct = pred_correct & ~low_conf

    # GT-instance metadata for nu/cosmic-aware coloring
    primary_trackid_gt = pred.get("gt/primary_trackid", np.zeros(0, dtype=np.int64))
    origin_type_gt = pred.get("gt/origin_type", np.zeros(0, dtype=np.int64))
    tid_to_origin = {int(tid): int(ot) for tid, ot
                     in zip(primary_trackid_gt, origin_type_gt)}

    if color_by == "pred_correct":
        # 4 buckets when threshold > 0; low_conf is all-False otherwise.
        correct_mask = pred_correct & (post_slice_id_gt >= 0) & ~low_conf
        wrong_mask = (~pred_correct) & (post_slice_id_gt >= 0) & ~low_conf
        low_conf_mask = low_conf & (post_slice_id_gt >= 0)
        unmatched_gt_mask = (post_slice_id_gt < 0)
        for mask, color, name in [
            (correct_mask, "rgba(60,200,80,0.9)",
             f"correct ({int(correct_mask.sum())})"),
            (wrong_mask, "rgba(255,60,60,0.9)",
             f"wrong ({int(wrong_mask.sum())})"),
            (low_conf_mask, "rgba(255,200,60,0.9)",
             f"low-conf (<{min_mask_prob:.2f}) ({int(low_conf_mask.sum())})"),
            (unmatched_gt_mask, "rgba(150,150,150,0.4)",
             f"no GT slice ({int(unmatched_gt_mask.sum())})"),
        ]:
            if not mask.any():
                continue
            pts = post_coord_cm[mask]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.6, color=color),
                name=name,
                **_pred_hover_kwargs(mask, customdata_full),
            ))

    elif color_by == "pred_slice_id":
        # Per-trackid coloring on post-filter SPs. Unassigned (pred_slice_id==-1)
        # is dim gray; everything else hashes by trackid.
        unmapped = (pred_slice_id < 0)
        if unmapped.any():
            pts = post_coord_cm[unmapped]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.5, color="rgba(150,150,150,0.4)"),
                name=f"unassigned ({int(unmapped.sum())})",
                **_pred_hover_kwargs(unmapped, customdata_full),
            ))
        used_tids = sorted(set(int(t) for t in pred_slice_id if t >= 0))
        for tid in used_tids:
            m = (pred_slice_id == tid)
            ot = tid_to_origin.get(tid, -1)
            pts = post_coord_cm[m]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=2.0, color=_track_id_color(tid, ot)),
                name=f"pred tid={tid} ({int(m.sum())})"
                     + (" [nu]" if ot == 0 else ""),
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "slice_id_gt":
        # GT-side coloring on the same post-filter SPs — direct visual
        # comparison anchor for `pred_slice_id`.
        ghost = (post_slice_id_gt < 0)
        if ghost.any():
            pts = post_coord_cm[ghost]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=1.5, color="rgba(150,150,150,0.4)"),
                name=f"ghost / no GT slice ({int(ghost.sum())})",
                **_pred_hover_kwargs(ghost, customdata_full),
            ))
        used_tids = sorted(set(int(t) for t in post_slice_id_gt if t >= 0))
        for tid in used_tids:
            m = (post_slice_id_gt == tid)
            ot = tid_to_origin.get(tid, -1)
            pts = post_coord_cm[m]
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=2.0, color=_track_id_color(tid, ot)),
                name=f"GT tid={tid} ({int(m.sum())})"
                     + (" [nu]" if ot == 0 else ""),
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "pred_class":
        no_obj = int(pred.get("meta_no_object_class_id", 2))
        for c in sorted(set(int(x) for x in pred_class)):
            m = (pred_class == c)
            pts = post_coord_cm[m]
            label = "no_object" if c == no_obj else f"class={c}"
            # Use cls_color so nu=red, cosmic=blue convention is reused.
            color = "rgba(150,150,150,0.4)" if c == no_obj else cls_color(c)
            traces.append(go.Scatter3d(
                x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                mode="markers",
                marker=dict(size=2.0, color=color),
                name=f"{label} ({int(m.sum())})",
                **_pred_hover_kwargs(m, customdata_full),
            ))

    elif color_by == "pred_mask_raw":
        # Show each matched query's FULL above-threshold mask (the set
        # pair_iou is computed against), regardless of panoptic argmax.
        # This is the "raw Q mask" view — SPs can appear in multiple
        # queries' overlap (over-claim). Distinguishes "Q's high-prob
        # claim" from "Q's argmax-won territory" (= the pred_slice_id
        # view). Big gap between the two = Q is over-claiming SPs that
        # other queries win.
        pred_mask_bool = pred.get("gt/pred_mask_bool")
        if pred_mask_bool is None:
            traces.append(go.Scatter3d(x=[], y=[], z=[], name=(
                "raw matched-Q mask requires gt/pred_mask_bool in the HDF5; "
                "re-run tools/run_slicer_inference.py with the latest script "
                "to regenerate.")))
        else:
            n_drawn = 0
            for k in range(min(len(primary_trackid_gt), pred_mask_bool.shape[0])):
                row = pred_mask_bool[k].astype(bool)
                if not row.any():
                    continue
                tid = int(primary_trackid_gt[k])
                ot = int(origin_type_gt[k])
                pts = post_coord_cm[row]
                traces.append(go.Scatter3d(
                    x=pts[:, 2], y=pts[:, 0], z=pts[:, 1],
                    mode="markers",
                    marker=dict(size=2.0, color=_track_id_color(tid, ot)),
                    name=f"raw matched-Q tid={tid} ({int(row.sum())})"
                         + (" [nu]" if ot == 0 else ""),
                    **_pred_hover_kwargs(row, customdata_full),
                ))
                n_drawn += 1
            if n_drawn == 0:
                traces.append(go.Scatter3d(x=[], y=[], z=[],
                                           name="(no matched queries had mask above threshold)"))

    elif color_by == "p_real":
        # Continuous colorscale on ALL pre-filter SPs. Shows the deghoster's
        # full output, not just the kept points.
        traces.append(go.Scatter3d(
            x=pre_coord[:, 2], y=pre_coord[:, 0], z=pre_coord[:, 1],
            mode="markers",
            marker=dict(
                size=1.5, color=p_real_pre,
                colorscale="RdYlBu",       # red = low P(real); blue = high
                cmin=0.0, cmax=1.0,
                showscale=True,
                colorbar=dict(title="P(real)", thickness=12),
            ),
            name=f"all SPs ({len(pre_coord)})",
            hovertemplate="P(real)=%{marker.color:.3f}<extra></extra>",
        ))

    n_pre = pred.get("meta_n_sp_pre", 0)
    n_post = pred.get("meta_n_sp_post", 0)
    tau = pred.get("meta_tau", float("nan"))
    title = (
        f"PREDICTION  event={pred.get('meta_event', '?')} "
        f"run={pred.get('meta_run', '?')} "
        f"subrun={pred.get('meta_subrun', '?')}  |  "
        f"n_sp {n_pre} → {n_post}  τ={tau:.3f}  "
        f"matched={pred.get('meta_n_matched', 0)}/"
        f"{pred.get('meta_n_gt_instances', 0)}"
    )
    return go.Figure(data=traces, layout=go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="z (cm)"),
            yaxis=dict(title="x (cm)"),
            zaxis=dict(title="y (cm)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    ))


def metadata_panel(event_data, prediction: "dict | None" = None):
    """HTML metadata table for the current event.

    Args:
        event_data: dict from build_event_gt (GT side).
        prediction: optional dict from _load_slicerpred (model output side).
                     When given, appends a "prediction summary" section
                     with per-pair IoU stats, nu/cosmic confusion breakdown,
                     and per-instance matched-query mapping.
    """
    sample = event_data["sample"]
    rows = [
        html.Tr([html.Td(html.B("run/subrun/event")),
                 html.Td(f"{sample.get('run','?')} / "
                         f"{sample.get('subrun','?')} / "
                         f"{sample.get('event','?')}")]),
        html.Tr([html.Td(html.B("name")),
                 html.Td(sample.get("name", ""))]),
        html.Tr([html.Td(html.B("n_spacepoints")),
                 html.Td(str(sample["n_spacepoints"]))]),
        html.Tr([html.Td(html.B("lm_score_threshold")),
                 html.Td(f"{sample['lm_score_threshold']:.3f}")]),
        html.Tr([html.Td(html.B("n_gt_instances")),
                 html.Td(str(sample["n_gt_instances"]))]),
    ]

    ci = event_data.get("composition_info")
    if ci is not None:
        rows.append(html.Tr([html.Td(colSpan=2,
            children=html.Hr(style={"margin": "4px 0"}))]))
        rows.append(html.Tr([html.Td(html.B("data composition"),
                                     colSpan=2,
                                     style={"fontStyle": "italic"})]))
        rows.append(html.Tr([html.Td("n true-real (hasmatch=1)"),
                             html.Td(str(ci["n_real_pre"]))]))
        rows.append(html.Tr([html.Td("n true-ghost (hasmatch=0)"),
                             html.Td(str(ci["n_ghost_pre"]))]))
        rows.append(html.Tr([html.Td("real fraction (S/(S+N))"),
                             html.Td(f"{ci['real_frac']:.3f}")]))

    di = event_data.get("deghost_info")
    if di is not None:
        rows.append(html.Tr([html.Td(colSpan=2,
            children=html.Hr(style={"margin": "4px 0"}))]))
        rows.append(html.Tr([html.Td(html.B("deghoster filter"),
                                     colSpan=2,
                                     style={"fontStyle": "italic"})]))
        rows.append(html.Tr([html.Td(html.B("deghost τ")),
                             html.Td(f"{di['tau']:.3f}")]))
        rows.append(html.Tr([html.Td(html.B("n_sp pre-filter")),
                             html.Td(str(di["n_sp_pre"]))]))
        rows.append(html.Tr([html.Td(html.B("n_sp post-filter")),
                             html.Td(str(di["n_sp_post"]))]))
        rows.append(html.Tr([html.Td(html.B("keep_frac")),
                             html.Td(f"{di['keep_frac']:.3f}")]))
        if "n_real_pre" in di:
            rr = di["real_recall"]; gr = di["ghost_reject"]
            rows.append(html.Tr([html.Td("real kept / real pre"),
                html.Td(f"{di['n_real_kept']} / {di['n_real_pre']}  "
                        f"({rr:.3f})" if rr == rr else
                        f"{di['n_real_kept']} / {di['n_real_pre']}  (—)")]))
            rows.append(html.Tr([html.Td("ghost kept / ghost pre"),
                html.Td(f"{di['n_ghost_kept']} / {di['n_ghost_pre']}  "
                        f"(reject {gr:.3f})" if gr == gr else
                        f"{di['n_ghost_kept']} / {di['n_ghost_pre']}  (reject —)")]))
        if di.get("warning"):
            rows.append(html.Tr([html.Td(colSpan=2,
                children=html.Div(di["warning"],
                                  style={"color": "#c33", "fontWeight": "bold"}))]))
    md = [html.Table(rows, style={"width": "100%", "fontSize": "13px"})]
    # Per-instance breakdown
    gts = sample["gt_instances"]
    if gts:
        inst_rows = [html.Tr([
            html.Th("k"), html.Th("origin_type"), html.Th("pid"),
            html.Th("n_pts"), html.Th("origin (norm)"), html.Th("extra"),
        ])]
        for k, g in enumerate(gts[:50]):
            extra_keys = [kk for kk in g.keys()
                          if kk not in {"origin_type", "pid",
                                        "n_truth_points",
                                        "origin_coord_norm",
                                        "truth_indices"}]
            extra = ", ".join(f"{kk}={g[kk]}" for kk in extra_keys)
            origin = g.get("origin_coord_norm",
                           np.zeros(3, dtype=np.float32))
            inst_rows.append(html.Tr([
                html.Td(str(k)),
                html.Td(str(int(g.get("origin_type", -1)))),
                html.Td(str(int(g.get("pid", -1)))),
                html.Td(str(int(g.get("n_truth_points", 0)))),
                html.Td(f"({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"),
                html.Td(extra),
            ]))
        md.append(html.Hr())
        md.append(html.Div(html.B(f"gt_instances "
                                  f"(showing first {min(50, len(gts))}):")))
        md.append(html.Table(inst_rows,
                             style={"width": "100%", "fontSize": "12px"}))

    # ---- prediction summary ----------------------------------------------
    if prediction is not None:
        pair_iou = prediction.get("gt/pair_iou", np.zeros(0))
        valid = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        pair_cls_correct = prediction.get("gt/pair_cls_correct", np.zeros(0))
        valid_cls = (pair_cls_correct[pair_cls_correct >= 0]
                     if pair_cls_correct.size else pair_cls_correct)
        origin_type = prediction.get("gt/origin_type", np.zeros(0))
        matched_query = prediction.get("gt/matched_query", np.zeros(0))
        primary_trackid = prediction.get("gt/primary_trackid", np.zeros(0))

        # Per-event summary
        n_matched = int(prediction.get("meta_n_matched", 0))
        n_gt = int(prediction.get("meta_n_gt_instances", 0))
        miou = float(np.mean(valid)) if valid.size else float("nan")
        miou_p25 = float(np.percentile(valid, 25)) if valid.size else float("nan")
        cls_acc = (float(valid_cls.mean()) if valid_cls.size > 0
                   else float("nan"))

        # Per-class breakdown
        nu_mask = (origin_type == 0)
        cosmic_mask = (origin_type == 1)
        nu_iou = (float(np.mean(pair_iou[nu_mask & (pair_iou >= 0)]))
                  if (nu_mask & (pair_iou >= 0)).any() else float("nan"))
        cosmic_iou = (float(np.mean(pair_iou[cosmic_mask & (pair_iou >= 0)]))
                      if (cosmic_mask & (pair_iou >= 0)).any() else float("nan"))

        pred_rows = [
            html.Tr([html.Td(colSpan=2,
                children=html.Hr(style={"margin": "4px 0"}))]),
            html.Tr([html.Td(html.B("prediction summary"),
                             colSpan=2,
                             style={"fontStyle": "italic"})]),
            html.Tr([html.Td("matched / GT"),
                     html.Td(f"{n_matched} / {n_gt}")]),
            html.Tr([html.Td("mean pair IoU"),
                     html.Td(f"{miou:.3f}  (p25={miou_p25:.3f})")]),
            html.Tr([html.Td("nu mean IoU"),
                     html.Td(f"{nu_iou:.3f}")]),
            html.Tr([html.Td("cosmic mean IoU"),
                     html.Td(f"{cosmic_iou:.3f}")]),
            html.Tr([html.Td("class accuracy (matched)"),
                     html.Td(f"{cls_acc:.3f}")]),
        ]
        md.append(html.Table(pred_rows,
                             style={"width": "100%", "fontSize": "13px"}))

        # Per-GT-instance pair table (which query, what IoU, right class?)
        if pair_iou.size > 0:
            # Header. `gt_n` = GT slice size; `pred_n` = matched query's
            # above-threshold mask size (the set pair_iou uses on the
            # pred side). The gap `pred_n - gt_n` shows over-claim:
            # when pred_n >> gt_n, Q is claiming more SPs than the GT
            # actually has — many of those go to OTHER queries in
            # panoptic argmax, so the prediction panel hides them but
            # pair_iou penalizes Q for them. Similarly,
            # `pair_iou - argmax_iou` shows the magnitude of over-claim:
            # when large, Q over-claims SPs other queries win.
            inst_rows = [html.Tr([
                html.Th("k"), html.Th("tid"), html.Th("ot"),
                html.Th("gt_n"), html.Th("pred_n"), html.Th("matched_q"),
                html.Th("pair IoU"), html.Th("argmax IoU"), html.Th("cls ok"),
            ])]
            n_pts = prediction.get("gt/n_truth_points", np.zeros(len(pair_iou)))
            pred_n_pts = prediction.get("gt/pred_n_pts", np.full(len(pair_iou), -1, dtype=np.int64))
            argmax_iou = prediction.get("gt/argmax_iou", np.full(len(pair_iou), -1.0, dtype=np.float32))
            for k in range(min(int(len(pair_iou)), 50)):
                tid = int(primary_trackid[k]) if k < len(primary_trackid) else -1
                ot = int(origin_type[k]) if k < len(origin_type) else -1
                mq = int(matched_query[k]) if k < len(matched_query) else -1
                npp = int(n_pts[k]) if k < len(n_pts) else 0
                ppn = int(pred_n_pts[k]) if k < len(pred_n_pts) else -1
                iou_val = float(pair_iou[k])
                aiou_val = float(argmax_iou[k]) if k < len(argmax_iou) else -1.0
                cls_ok = int(pair_cls_correct[k]) if k < len(pair_cls_correct) else -1
                iou_str = f"{iou_val:.3f}" if iou_val >= 0 else "—"
                aiou_str = f"{aiou_val:.3f}" if aiou_val >= 0 else "—"
                cls_str = ("✓" if cls_ok == 1 else
                           ("✗" if cls_ok == 0 else "—"))
                pred_n_str = str(ppn) if ppn >= 0 else "—"
                inst_rows.append(html.Tr([
                    html.Td(str(k)),
                    html.Td(str(tid)),
                    html.Td(str(ot)),
                    html.Td(str(npp)),
                    html.Td(pred_n_str),
                    html.Td(str(mq)),
                    html.Td(iou_str),
                    html.Td(aiou_str),
                    html.Td(cls_str),
                ]))
            md.append(html.Div(html.B("matched pairs (predicted vs GT):"),
                               style={"marginTop": "4px"}))
            md.append(html.Table(inst_rows,
                                 style={"width": "100%", "fontSize": "12px"}))

    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to a LArFormer training config file. May be "
                         "a plain LArFormer or a CascadedSlicer config "
                         "(auto-detected; cascade pulls deghoster + slicer "
                         "levels from the model subconfig).")
    ap.add_argument("--cascade-config", default=None,
                    help="Optional separate CascadedSlicer config to use "
                         "ONLY for the deghoster + threshold. Use when "
                         "--config is a plain LArFormer but you want to "
                         "visualize against a separately-trained deghoster.")
    ap.add_argument("--no-deghoster", action="store_true",
                    help="Skip the cascade's deghoster filter even when "
                         "--config or --cascade-config is cascade-shaped. "
                         "Falls back to the dataset's lm_score filter only.")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"),
                    help="Which dataset split to read")
    ap.add_argument("--entry", type=int, default=0,
                    help="Initial event index")
    ap.add_argument("--max-spacepoints", type=int, default=None,
                    help="Override dataset's max_spacepoints (for browser perf)")
    ap.add_argument("--slicerpred-dir", default=None,
                    help="Directory of slicerpred_<basename>.h5 files "
                         "produced by tools/run_slicer_inference.py. When "
                         "set, a second 3D scene appears below the GT one "
                         "showing the model's predictions for the same "
                         "event. File matching is by event basename.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    from pointcept.utils.config import Config
    cfg = Config.fromfile(args.config)
    import pointcept.models  # noqa: F401 — register MODELS for cascade build

    # Build the dataset directly (skips MODELS / TRAINERS registration of
    # heavy components). Force `split` and apply CLI overrides.
    from pointcept.datasets import build_dataset
    ds_cfg = dict(cfg.data[args.split])
    if args.max_spacepoints is not None:
        ds_cfg["max_spacepoints"] = args.max_spacepoints
    dataset = build_dataset(ds_cfg)
    print(f"Loaded {args.split} dataset: {len(dataset)} events")

    # Levels source — cascade configs nest them under model.slicer.levels.
    if cfg.model.get("type") == "CascadedSlicer":
        levels_cfg = cfg.model.slicer.levels
        token_dim = int(cfg.model.slicer.token_dim)
        print("Detected CascadedSlicer config; using model.slicer.levels.")
    else:
        levels_cfg = cfg.model.levels
        token_dim = int(cfg.model.token_dim)
    level_names = [L["name"] for L in levels_cfg]
    print(f"Levels declared in config: {level_names}")

    # ---- Optional cascade-deghoster filter --------------------------------
    deghoster_filter = None
    deghost_tau_default = 0.5
    if not args.no_deghoster:
        cascade_cfg = cfg
        if args.cascade_config and args.cascade_config != args.config:
            cascade_cfg = Config.fromfile(args.cascade_config)
        if cascade_cfg.model.get("type") == "CascadedSlicer":
            dgh_cfg = cascade_cfg.model.get("deghoster")
            dgh_w = cascade_cfg.model.get("deghoster_weight")
            cls_real = int(cascade_cfg.model.get(
                "deghoster_class_index_real", 1))
            deghost_tau_default = float(cascade_cfg.model.get(
                "deghost_threshold_val", 0.5))
            if dgh_cfg is None:
                print("[viz] Cascade config has no `deghoster` block; "
                      "skipping deghoster filter.")
            else:
                if dgh_w is None or not os.path.exists(dgh_w):
                    print(f"[viz] WARNING: deghoster_weight {dgh_w!r} "
                          f"not found; building deghoster with RANDOM init "
                          f"(P(real) ≈ 0.5 → keep_frac ≈ τ-dependent).")
                    dgh_w = None
                else:
                    print(f"[viz] Building cascade deghoster from "
                          f"{args.cascade_config or args.config} "
                          f"(class_index_real={cls_real}, "
                          f"τ_default={deghost_tau_default})")
                deghoster_filter = CascadeDeghostFilter(
                    deghoster_cfg=dict(dgh_cfg),
                    deghoster_weight=dgh_w,
                    class_index_real=cls_real,
                )

    app = Dash(__name__)
    app.title = "LArFormer GT visualizer"
    app.layout = html.Div([
        html.Div([
            html.H3("LArFormer GT visualizer", style={"marginBottom": "4px"}),
            html.Div([
                html.Span("Event: ", style={"marginRight": "6px"}),
                dcc.Input(id="entry", type="number", min=0,
                          max=len(dataset) - 1, step=1, value=args.entry,
                          style={"width": "80px", "marginRight": "16px"}),
                html.Span("Level: ", style={"marginRight": "6px"}),
                dcc.Dropdown(id="level",
                             options=[{"label": n, "value": n}
                                      for n in level_names],
                             value=level_names[-1],
                             clearable=False,
                             style={"width": "200px", "display": "inline-block",
                                    "marginRight": "16px"}),
                html.Span("Color by: ", style={"marginRight": "6px"}),
                dcc.Dropdown(id="color",
                             options=[
                                 {"label": "GT instance id", "value": "instance_id"},
                                 {"label": "per-token cls target",
                                  "value": "cls_target"},
                                 {"label": "sp_to_level_id (partition viz)",
                                  "value": "sp_to_level_id"},
                                 {"label": "SPs by level-cluster GT (back-projected)",
                                  "value": "sp_by_level_inst"},
                             ],
                             value="instance_id",
                             clearable=False,
                             style={"width": "320px",
                                    "display": "inline-block",
                                    "marginRight": "16px"}),
                html.Button("Reload event", id="reload", n_clicks=0,
                            style={"marginLeft": "12px"}),
                # τ input — only visible when the cascade deghoster is active.
                html.Span("  |  deghoster τ: ",
                          style={"marginLeft": "12px", "marginRight": "6px",
                                 "display": ("inline-block"
                                             if deghoster_filter is not None
                                             else "none")}),
                dcc.Input(id="tau", type="number", min=0.0, max=1.0,
                          step=0.05, value=deghost_tau_default,
                          style={"width": "80px",
                                 "display": ("inline-block"
                                             if deghoster_filter is not None
                                             else "none")}),
            ], style={"marginBottom": "8px"}),
        ]),
        html.Div([
            html.Div([
                # GT scene (top). 50vh when prediction panel is shown so
                # both scenes fit in one viewport; 85vh otherwise so the
                # GT-only case stays full-height.
                dcc.Graph(
                    id="scene",
                    style={"height": ("50vh"
                                      if args.slicerpred_dir is not None
                                      else "85vh")},
                ),
                # Prediction-scene color-by + level selector (hidden when
                # no --slicerpred-dir was given).
                html.Div([
                    html.Span("Prediction level: ",
                              style={"marginRight": "6px"}),
                    dcc.Dropdown(
                        id="pred-level",
                        options=[{"label": n, "value": n}
                                 for n in level_names],
                        value="spacepoint",
                        clearable=False,
                        style={"width": "180px",
                               "display": "inline-block",
                               "marginRight": "16px"},
                    ),
                    html.Span("Prediction color by: ",
                              style={"marginRight": "6px"}),
                    dcc.Dropdown(
                        id="pred-color",
                        options=[
                            {"label": "correct / wrong (vs GT)",
                             "value": "pred_correct"},
                            {"label": "predicted slice id (panoptic argmax)",
                             "value": "pred_slice_id"},
                            {"label": "raw matched-Q mask (pair_iou view)",
                             "value": "pred_mask_raw"},
                            {"label": "GT slice id (post-filter SPs)",
                             "value": "slice_id_gt"},
                            {"label": "predicted class (nu/cosmic)",
                             "value": "pred_class"},
                            {"label": "deghoster P(real) (all pre-filter SPs)",
                             "value": "p_real"},
                        ],
                        value="pred_correct",
                        clearable=False,
                        style={"width": "340px",
                               "display": "inline-block",
                               "marginRight": "16px"},
                    ),
                    html.Span("min mask prob: ",
                              style={"marginRight": "6px"}),
                    dcc.Input(id="pred-thresh", type="number",
                              min=0.0, max=1.0, step=0.05, value=0.5,
                              style={"width": "80px",
                                     "display": "inline-block"}),
                ], style={
                    "marginTop": "4px",
                    "display": ("block"
                                if args.slicerpred_dir is not None
                                else "none"),
                }),
                # Prediction scene (bottom).
                dcc.Graph(
                    id="pred-scene",
                    style={"height": ("50vh"
                                      if args.slicerpred_dir is not None
                                      else "0"),
                           "display": ("block"
                                       if args.slicerpred_dir is not None
                                       else "none")},
                ),
            ], style={"width": "70%", "display": "inline-block",
                      "verticalAlign": "top"}),
            html.Div(id="meta",
                     style={"width": "28%", "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "8px", "boxSizing": "border-box",
                            "fontFamily": "monospace",
                            "overflowY": "auto",
                            "maxHeight": ("105vh"
                                          if args.slicerpred_dir is not None
                                          else "85vh")}),
        ]),
    ], style={"fontFamily": "Helvetica, Arial, sans-serif",
              "padding": "8px"})

    # Cache last-built event_data, keyed by (entry, tau). Switching level
    # or color reuses the cache; switching entry or τ rebuilds (deghoster
    # forward is expensive but is only re-run on those changes).
    cache = {"key": None, "data": None}
    # Cache last-loaded slicerpred file, keyed by entry only (no tau
    # dependency — the prediction was made by run_slicer_inference at the
    # cascade's val τ, not at the viewer's τ slider).
    pred_cache = {"entry": None, "data": None}

    def get_event(entry, tau):
        key = (int(entry), float(tau) if tau is not None else None)
        if cache["key"] == key and cache["data"] is not None:
            return cache["data"]
        ev = build_event_gt(
            levels_cfg, token_dim, dataset, entry,
            deghoster_filter=deghoster_filter,
            tau=float(tau) if tau is not None else 0.5,
        )
        cache["key"] = key
        cache["data"] = ev
        return ev

    def get_prediction(entry: int) -> "dict | None":
        if args.slicerpred_dir is None:
            return None
        if pred_cache["entry"] == entry and pred_cache["data"] is not None:
            return pred_cache["data"]
        # Look up the slicerpred file by the dataset event's basename.
        # The dataset's sample["name"] is set in LArFormerDataset._load_event.
        sample = dataset[entry]
        in_name = sample.get("name", "")
        stem = os.path.splitext(in_name)[0]
        path = os.path.join(args.slicerpred_dir, f"slicerpred_{stem}.h5")
        pred = _load_slicerpred(path)
        if pred is None:
            print(f"[viz] prediction file not found: {path}")
        pred_cache["entry"] = entry
        pred_cache["data"] = pred
        return pred

    @app.callback(
        Output("scene", "figure"),
        Output("pred-scene", "figure"),
        Output("meta", "children"),
        Input("entry", "value"),
        Input("level", "value"),
        Input("color", "value"),
        Input("reload", "n_clicks"),
        Input("tau", "value"),
        Input("pred-color", "value"),
        Input("pred-level", "value"),
        Input("pred-thresh", "value"),
    )
    def update(entry, level, color, n_clicks, tau, pred_color, pred_level,
               pred_thresh):
        if entry is None:
            entry = args.entry
        entry = max(0, min(int(entry), len(dataset) - 1))
        if tau is None:
            tau = deghost_tau_default
        if pred_thresh is None:
            pred_thresh = 0.5
        # `reload` button invalidates BOTH caches by forcing a re-fetch.
        if n_clicks and cache["key"] is not None and cache["key"][0] == entry:
            cache["key"] = None
            pred_cache["entry"] = None
        ev = get_event(entry, tau)
        fig = figure_for_event(ev, level, color)
        pred = get_prediction(entry)
        pred_fig = figure_for_prediction(
            pred, pred_color or "pred_correct",
            level=pred_level or "spacepoint",
            min_mask_prob=float(pred_thresh),
        )
        meta = metadata_panel(ev, prediction=pred)
        return fig, pred_fig, meta

    print(f"\nStarting Dash on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
