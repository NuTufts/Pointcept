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
                clr = instance_color(k, origin_type=ot)
                pts = coords_plot[m]
                label_bits = [f"k={k}"]
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


def metadata_panel(event_data):
    """HTML metadata table for the current event."""
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
                             ],
                             value="instance_id",
                             clearable=False,
                             style={"width": "260px",
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
                dcc.Graph(id="scene", style={"height": "85vh"}),
            ], style={"width": "70%", "display": "inline-block",
                      "verticalAlign": "top"}),
            html.Div(id="meta",
                     style={"width": "28%", "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "8px", "boxSizing": "border-box",
                            "fontFamily": "monospace",
                            "overflowY": "auto", "maxHeight": "85vh"}),
        ]),
    ], style={"fontFamily": "Helvetica, Arial, sans-serif",
              "padding": "8px"})

    # Cache last-built event_data, keyed by (entry, tau). Switching level
    # or color reuses the cache; switching entry or τ rebuilds (deghoster
    # forward is expensive but is only re-run on those changes).
    cache = {"key": None, "data": None}

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

    @app.callback(
        Output("scene", "figure"),
        Output("meta", "children"),
        Input("entry", "value"),
        Input("level", "value"),
        Input("color", "value"),
        Input("reload", "n_clicks"),
        Input("tau", "value"),
    )
    def update(entry, level, color, n_clicks, tau):
        if entry is None:
            entry = args.entry
        entry = max(0, min(int(entry), len(dataset) - 1))
        if tau is None:
            tau = deghost_tau_default
        # `reload` button invalidates the cache by forcing a re-fetch.
        if n_clicks and cache["key"] is not None and cache["key"][0] == entry:
            cache["key"] = None
        ev = get_event(entry, tau)
        fig = figure_for_event(ev, level, color)
        meta = metadata_panel(ev)
        return fig, meta

    print(f"\nStarting Dash on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
