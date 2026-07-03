"""
muP Coordinate Check for Sonata/PT-v3m2.

Trains models at multiple widths for a few steps and records per-layer
activation statistics. Under correct muP, activations should be O(1)
regardless of width.

Usage:
    python tools/mup_coord_check.py \
        --config-file configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup.py \
        --num-steps 10 \
        --widths 0.25,0.5,1.0,2.0 \
        --output coord_check.png

    # Compare SP vs muP:
    python tools/mup_coord_check.py \
        --config-file configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py \
        --num-steps 10 --widths 0.25,0.5,1.0,2.0 --output coord_check_sp.png
"""

import argparse
import copy
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from pointcept.utils.config import Config
from pointcept.models import build_model


def make_synthetic_batch(
    num_global_views=2,
    num_local_views=6,
    points_per_view=512,
    in_channels=6,
    grid_size=0.0003,
    device="cuda",
):
    """Create a synthetic batch that matches Sonata's expected input format."""
    # Global views
    global_coords = []
    global_origin_coords = []
    global_feats = []
    global_offsets = []
    running_offset = 0
    for _ in range(num_global_views):
        coord = torch.randn(points_per_view, 3, device=device) * 0.5
        global_coords.append(coord)
        global_origin_coords.append(coord.clone())
        global_feats.append(torch.randn(points_per_view, in_channels, device=device))
        running_offset += points_per_view
        global_offsets.append(running_offset)

    # Local views
    local_coords = []
    local_origin_coords = []
    local_feats = []
    local_offsets = []
    running_offset = 0
    for _ in range(num_local_views):
        coord = torch.randn(points_per_view, 3, device=device) * 0.3
        local_coords.append(coord)
        local_origin_coords.append(coord.clone())
        local_feats.append(torch.randn(points_per_view, in_channels, device=device))
        running_offset += points_per_view
        local_offsets.append(running_offset)

    data_dict = {
        "global_coord": torch.cat(global_coords, dim=0),
        "global_origin_coord": torch.cat(global_origin_coords, dim=0),
        "global_feat": torch.cat(global_feats, dim=0),
        "global_offset": torch.tensor(global_offsets, dtype=torch.long, device=device),
        "local_coord": torch.cat(local_coords, dim=0),
        "local_origin_coord": torch.cat(local_origin_coords, dim=0),
        "local_feat": torch.cat(local_feats, dim=0),
        "local_offset": torch.tensor(local_offsets, dtype=torch.long, device=device),
        "grid_size": torch.tensor([grid_size], device=device),
    }
    return data_dict


def register_activation_hooks(model):
    """Register forward hooks on key layers to record activation magnitudes."""
    stats = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                val = output.detach().float().abs().mean().item()
            elif hasattr(output, "feat"):
                val = output.feat.detach().float().abs().mean().item()
            else:
                return
            if name not in stats:
                stats[name] = []
            stats[name].append(val)

        return hook

    for name, module in model.named_modules():
        # Track specific layer types of interest
        should_track = False
        layer_type = None

        if name.endswith(".stem.linear"):
            should_track = True
            layer_type = "embedding"
        elif ".attn.proj" in name and name.endswith(".proj"):
            should_track = True
            layer_type = "attn_proj"
        elif ".attn.qkv" in name:
            should_track = True
            layer_type = "attn_qkv"
        elif ".mlp.0.fc2" in name or (
            ".mlp" in name and name.endswith(".fc2")
        ):
            should_track = True
            layer_type = "mlp_fc2"
        elif ".cpe.0" in name and isinstance(module, nn.Linear):
            should_track = True
            layer_type = "cpe_linear"
        elif "mask_head.mlp" in name and name.endswith("2"):
            # Last linear in head MLP
            should_track = True
            layer_type = "head_mlp"

        if should_track:
            # Use a simplified name for grouping
            hooks.append(module.register_forward_hook(make_hook(f"{layer_type}/{name}")))

    return stats, hooks


def scale_config_width(cfg, width_mult):
    """Scale model width by a multiplier, keeping head_dim constant."""
    cfg = copy.deepcopy(cfg)

    base_channels = list(cfg.model.backbone.enc_channels)
    base_heads = list(cfg.model.backbone.enc_num_head)

    # Compute head_dim from base config
    head_dim = base_channels[0] // base_heads[0]

    # Scale channels (round to nearest multiple of head_dim for clean division)
    new_channels = []
    new_heads = []
    for c in base_channels:
        new_c = max(int(round(c * width_mult / head_dim)) * head_dim, head_dim)
        new_channels.append(new_c)
        new_heads.append(max(new_c // head_dim, 1))

    cfg.model.backbone.enc_channels = tuple(new_channels)
    cfg.model.backbone.enc_num_head = tuple(new_heads)

    # Update head_in_channels based on up_cast_level
    up_cast_level = cfg.model.get("up_cast_level", 2)
    cfg.model.head_in_channels = sum(new_channels[up_cast_level:])

    # Scale head hidden channels proportionally
    if hasattr(cfg.model, "head_hidden_channels"):
        cfg.model.head_hidden_channels = max(
            int(round(cfg.model.head_hidden_channels * width_mult / head_dim)) * head_dim,
            head_dim,
        )

    return cfg


def run_coord_check(config_path, num_steps=10, width_multipliers=None, lr=0.002):
    """Run coord check across multiple widths."""
    if width_multipliers is None:
        width_multipliers = [0.25, 0.5, 1.0, 2.0]

    base_cfg = Config.fromfile(config_path)
    is_mup = base_cfg.model.get("mup_enabled", False)

    print(f"Config: {config_path}")
    print(f"muP enabled: {is_mup}")
    print(f"Width multipliers: {width_multipliers}")
    print(f"Steps: {num_steps}, LR: {lr}")
    print()

    all_results = {}

    for mult in width_multipliers:
        print(f"--- Width multiplier: {mult}x ---")
        cfg = scale_config_width(base_cfg, mult)

        enc_ch = cfg.model.backbone.enc_channels
        enc_nh = cfg.model.backbone.enc_num_head
        print(f"  enc_channels: {enc_ch}")
        print(f"  enc_num_head: {enc_nh}")
        print(f"  head_in_channels: {cfg.model.head_in_channels}")
        if hasattr(cfg.model, "head_hidden_channels"):
            print(f"  head_hidden_channels: {cfg.model.head_hidden_channels}")

        # Build model
        model = build_model(cfg.model).cuda()
        model.train()

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Register hooks on student (the part that trains)
        stats, hooks = register_activation_hooks(model.student)

        # Simple optimizer (no per-param groups for coord check --
        # we use the same LR for all to isolate the effect of init/scaling)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

        # Train for num_steps
        in_channels = cfg.model.backbone.in_channels
        grid_size = base_cfg.get("scaled_grid_size", 0.0003)
        num_global = cfg.model.get("num_global_view", 2)
        num_local = cfg.model.get("num_local_view", 6)

        for step in range(num_steps):
            data_dict = make_synthetic_batch(
                num_global_views=num_global,
                num_local_views=num_local,
                points_per_view=256,
                in_channels=in_channels,
                grid_size=grid_size,
            )
            try:
                output = model(data_dict)
                loss = output["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step == 0 or step == num_steps - 1:
                    print(f"  Step {step}: loss={loss.item():.4f}")
            except Exception as e:
                print(f"  Step {step}: ERROR - {e}")
                break

        # Record final activation stats (last value for each tracked layer)
        results = {}
        for name, values in stats.items():
            if values:
                results[name] = values[-1]
        all_results[mult] = results

        # Cleanup
        for h in hooks:
            h.remove()
        del model, optimizer
        torch.cuda.empty_cache()
        print()

    return all_results


def plot_coord_check(results, output_path="coord_check.png", title=""):
    """Plot activation magnitudes vs width multiplier."""
    widths = sorted(results.keys())
    if not results or not results[widths[0]]:
        print("No activation data to plot.")
        return

    # Group by layer type (first part of name before /)
    layer_groups = defaultdict(dict)
    for name in results[widths[0]].keys():
        layer_type = name.split("/")[0]
        layer_groups[layer_type][name] = True

    num_groups = len(layer_groups)
    if num_groups == 0:
        print("No layer groups found.")
        return

    fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 4.5))
    if num_groups == 1:
        axes = [axes]

    for ax, (group_name, layer_names) in zip(axes, sorted(layer_groups.items())):
        for full_name in sorted(layer_names.keys()):
            values = [results[w].get(full_name, float("nan")) for w in widths]
            short_name = full_name.split("/")[-1].replace("student.", "")
            # Truncate long names
            if len(short_name) > 30:
                short_name = "..." + short_name[-27:]
            ax.plot(widths, values, "o-", markersize=4, label=short_name)

        ax.set_xlabel("Width multiplier")
        ax.set_ylabel("Mean |activation|")
        ax.set_title(group_name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # Only show legend if reasonable number of lines
        if len(layer_names) <= 6:
            ax.legend(fontsize=6)

    suptitle = "muP Coordinate Check"
    if title:
        suptitle += f" ({title})"
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Coord check plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="muP coordinate check: verify activation stability across widths"
    )
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Training steps per width"
    )
    parser.add_argument(
        "--widths",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated width multipliers",
    )
    parser.add_argument(
        "--lr", type=float, default=0.002, help="Learning rate for all param groups"
    )
    parser.add_argument(
        "--output", type=str, default="coord_check.png", help="Output plot path"
    )
    parser.add_argument("--title", type=str, default="", help="Plot title suffix")
    args = parser.parse_args()

    width_mults = [float(x) for x in args.widths.split(",")]
    results = run_coord_check(
        args.config_file, args.num_steps, width_mults, lr=args.lr
    )
    plot_coord_check(results, args.output, title=args.title)

    # Print summary table
    widths = sorted(results.keys())
    if results and results[widths[0]]:
        print("\nSummary: mean |activation| at final step")
        print(f"{'Layer':<40s}", end="")
        for w in widths:
            print(f"  {w:>6.2f}x", end="")
        print("  ratio(max/min)")
        print("-" * (40 + 10 * len(widths) + 16))

        for name in sorted(results[widths[0]].keys()):
            short = name.replace("student.", "")
            if len(short) > 38:
                short = "..." + short[-35:]
            print(f"{short:<40s}", end="")
            vals = []
            for w in widths:
                v = results[w].get(name, float("nan"))
                vals.append(v)
                print(f"  {v:>8.4f}", end="")
            valid = [v for v in vals if v > 0 and not math.isnan(v)]
            if valid:
                ratio = max(valid) / min(valid)
                # Under correct muP, ratio should be close to 1.0
                marker = " OK" if ratio < 2.0 else " !!!" if ratio > 5.0 else " ?"
                print(f"  {ratio:>8.2f}{marker}")
            else:
                print(f"  {'N/A':>8s}")
        print()
        print(
            "Interpretation: Under correct muP, all ratios should be close to 1.0."
        )
        print(
            "Ratios > 2x suggest the parameterization is not width-stable for that layer."
        )


if __name__ == "__main__":
    main()
