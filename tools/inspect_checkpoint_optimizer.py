#!/usr/bin/env python
"""
Inspect optimizer state saved in Pointcept checkpoint files.

This script examines the checkpoint structure, particularly the optimizer
state, to help diagnose training instability issues when resuming.

Usage:
    python inspect_checkpoint_optimizer.py <checkpoint_path>

Example:
    python inspect_checkpoint_optimizer.py exp/lartpc/my-experiment/model/model_last.pth
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np


def format_tensor_stats(tensor):
    """Get statistics for a tensor."""
    if tensor is None:
        return "None"
    if isinstance(tensor, (int, float)):
        return f"{tensor}"
    if not isinstance(tensor, torch.Tensor):
        return f"{type(tensor).__name__}: {tensor}"

    arr = tensor.detach().cpu().numpy().flatten()
    if len(arr) == 0:
        return "empty tensor"
    return (
        f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={arr.min():.6e}, max={arr.max():.6e}, "
        f"mean={arr.mean():.6e}, std={arr.std():.6e}"
    )


def inspect_optimizer_state(checkpoint_path: str, verbose: bool = False):
    """Inspect the optimizer state in a checkpoint file."""

    print(f"\n{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Basic checkpoint info
    print("\n--- Checkpoint Keys ---")
    for key in checkpoint.keys():
        if key == "state_dict":
            print(f"  {key}: {len(checkpoint[key])} parameters")
        elif key == "optimizer":
            print(f"  {key}: present (see below)")
        elif key == "scheduler":
            print(f"  {key}: present (see below)")
        elif key == "scaler":
            if checkpoint[key] is not None:
                print(f"  {key}: present")
            else:
                print(f"  {key}: None (AMP disabled)")
        else:
            print(f"  {key}: {checkpoint[key]}")

    # Optimizer state
    if "optimizer" in checkpoint:
        opt_state = checkpoint["optimizer"]
        print(f"\n{'='*80}")
        print("--- Optimizer State ---")
        print(f"{'='*80}")

        # Param groups
        if "param_groups" in opt_state:
            print(f"\nNumber of param_groups: {len(opt_state['param_groups'])}")
            print("\n--- Parameter Groups (with per-group LR settings) ---")

            for i, group in enumerate(opt_state["param_groups"]):
                print(f"\n  Group {i}:")
                print(f"    lr:           {group.get('lr', 'N/A')}")
                print(f"    initial_lr:   {group.get('initial_lr', 'N/A')}")
                print(f"    max_lr:       {group.get('max_lr', 'N/A')}")
                print(f"    weight_decay: {group.get('weight_decay', 'N/A')}")
                print(f"    betas:        {group.get('betas', 'N/A')}")
                print(f"    eps:          {group.get('eps', 'N/A')}")
                print(f"    amsgrad:      {group.get('amsgrad', 'N/A')}")
                print(f"    num_params:   {len(group.get('params', []))}")

                if verbose:
                    # Show all keys
                    for key, val in group.items():
                        if key not in ['lr', 'initial_lr', 'max_lr', 'weight_decay',
                                      'betas', 'eps', 'amsgrad', 'params']:
                            print(f"    {key}: {val}")

        # State dict (momentum buffers, etc.)
        if "state" in opt_state:
            state = opt_state["state"]
            print(f"\n--- Optimizer State Buffers (momentum, variance, etc.) ---")
            print(f"Number of parameters with state: {len(state)}")

            if len(state) > 0:
                # Sample first few parameters
                sample_keys = list(state.keys())[:5]
                print(f"\nSample of state buffers (first 5 params):")

                for key in sample_keys:
                    param_state = state[key]
                    print(f"\n  Param {key}:")
                    for state_key, state_val in param_state.items():
                        print(f"    {state_key}: {format_tensor_stats(state_val)}")

                # Check for NaN/Inf in all states
                print(f"\n--- Checking for NaN/Inf in optimizer state ---")
                nan_count = 0
                inf_count = 0
                total_tensors = 0

                for param_id, param_state in state.items():
                    for state_key, state_val in param_state.items():
                        if isinstance(state_val, torch.Tensor):
                            total_tensors += 1
                            if torch.isnan(state_val).any():
                                nan_count += 1
                                print(f"  WARNING: NaN found in param {param_id}, {state_key}")
                            if torch.isinf(state_val).any():
                                inf_count += 1
                                print(f"  WARNING: Inf found in param {param_id}, {state_key}")

                if nan_count == 0 and inf_count == 0:
                    print(f"  No NaN/Inf found in {total_tensors} state tensors")
                else:
                    print(f"\n  TOTAL: {nan_count} tensors with NaN, {inf_count} tensors with Inf")

                # Statistics of Adam state buffers
                print(f"\n--- Adam State Statistics (exp_avg and exp_avg_sq) ---")
                exp_avg_stats = []
                exp_avg_sq_stats = []
                step_values = []

                for param_id, param_state in state.items():
                    if "exp_avg" in param_state:
                        t = param_state["exp_avg"]
                        exp_avg_stats.append({
                            "mean": t.mean().item(),
                            "std": t.std().item(),
                            "max_abs": t.abs().max().item(),
                        })
                    if "exp_avg_sq" in param_state:
                        t = param_state["exp_avg_sq"]
                        exp_avg_sq_stats.append({
                            "mean": t.mean().item(),
                            "std": t.std().item(),
                            "max": t.max().item(),
                        })
                    if "step" in param_state:
                        step_values.append(param_state["step"].item() if isinstance(param_state["step"], torch.Tensor) else param_state["step"])

                if exp_avg_stats:
                    means = [s["mean"] for s in exp_avg_stats]
                    stds = [s["std"] for s in exp_avg_stats]
                    max_abs = [s["max_abs"] for s in exp_avg_stats]
                    print(f"\n  exp_avg (momentum buffer) across all params:")
                    print(f"    mean of means:    {np.mean(means):.6e}")
                    print(f"    mean of stds:     {np.mean(stds):.6e}")
                    print(f"    max of max_abs:   {np.max(max_abs):.6e}")

                if exp_avg_sq_stats:
                    means = [s["mean"] for s in exp_avg_sq_stats]
                    maxs = [s["max"] for s in exp_avg_sq_stats]
                    print(f"\n  exp_avg_sq (variance buffer) across all params:")
                    print(f"    mean of means:    {np.mean(means):.6e}")
                    print(f"    max of maxs:      {np.max(maxs):.6e}")

                if step_values:
                    print(f"\n  Optimizer step count: {step_values[0]}")
                    if not all(s == step_values[0] for s in step_values):
                        print(f"    WARNING: Inconsistent step counts across params!")
                        print(f"    Range: [{min(step_values)}, {max(step_values)}]")

    # Scheduler state
    if "scheduler" in checkpoint:
        sched_state = checkpoint["scheduler"]
        print(f"\n{'='*80}")
        print("--- Scheduler State ---")
        print(f"{'='*80}")

        for key, val in sched_state.items():
            if key == "_last_lr":
                print(f"\n  {key}: (showing first 10 of {len(val)})")
                for i, lr in enumerate(val[:10]):
                    print(f"    Group {i}: {lr:.8e}")
                if len(val) > 10:
                    print(f"    ... and {len(val) - 10} more groups")
            else:
                print(f"  {key}: {val}")

    # GradScaler state (AMP)
    if "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler_state = checkpoint["scaler"]
        print(f"\n{'='*80}")
        print("--- GradScaler State (AMP) ---")
        print(f"{'='*80}")

        for key, val in scaler_state.items():
            print(f"  {key}: {val}")

    # Model state summary
    if "state_dict" in checkpoint:
        print(f"\n{'='*80}")
        print("--- Model State Summary ---")
        print(f"{'='*80}")

        state_dict = checkpoint["state_dict"]
        total_params = 0
        nan_params = 0
        inf_params = 0

        for key, val in state_dict.items():
            if isinstance(val, torch.Tensor):
                total_params += 1
                if torch.isnan(val).any():
                    nan_params += 1
                    print(f"  WARNING: NaN in model param: {key}")
                if torch.isinf(val).any():
                    inf_params += 1
                    print(f"  WARNING: Inf in model param: {key}")

        print(f"\n  Total parameters: {total_params}")
        if nan_params > 0 or inf_params > 0:
            print(f"  Parameters with NaN: {nan_params}")
            print(f"  Parameters with Inf: {inf_params}")
        else:
            print(f"  No NaN/Inf found in model parameters")

    print(f"\n{'='*80}")
    print("Inspection complete.")
    print(f"{'='*80}\n")


def compare_checkpoints(ckpt_path1: str, ckpt_path2: str):
    """Compare two checkpoints to see how optimizer state changed."""

    print(f"\n{'='*80}")
    print(f"Comparing checkpoints:")
    print(f"  1: {ckpt_path1}")
    print(f"  2: {ckpt_path2}")
    print(f"{'='*80}\n")

    ckpt1 = torch.load(ckpt_path1, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(ckpt_path2, map_location="cpu", weights_only=False)

    print(f"Epochs: {ckpt1.get('epoch', 'N/A')} -> {ckpt2.get('epoch', 'N/A')}")

    # Compare param groups
    if "optimizer" in ckpt1 and "optimizer" in ckpt2:
        groups1 = ckpt1["optimizer"]["param_groups"]
        groups2 = ckpt2["optimizer"]["param_groups"]

        print(f"\n--- Learning Rate Changes ---")
        for i, (g1, g2) in enumerate(zip(groups1, groups2)):
            lr_change = g2.get('lr', 0) - g1.get('lr', 0)
            print(f"  Group {i}: {g1.get('lr', 'N/A'):.8e} -> {g2.get('lr', 'N/A'):.8e} (delta: {lr_change:+.8e})")

    # Compare scheduler
    if "scheduler" in ckpt1 and "scheduler" in ckpt2:
        s1, s2 = ckpt1["scheduler"], ckpt2["scheduler"]
        print(f"\n--- Scheduler Changes ---")
        print(f"  last_epoch: {s1.get('last_epoch', 'N/A')} -> {s2.get('last_epoch', 'N/A')}")
        print(f"  _step_count: {s1.get('_step_count', 'N/A')} -> {s2.get('_step_count', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect optimizer state in Pointcept checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint file (.pth)"
    )
    parser.add_argument(
        "--compare",
        help="Compare with another checkpoint file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show more details"
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    inspect_optimizer_state(args.checkpoint, verbose=args.verbose)

    if args.compare:
        if not Path(args.compare).exists():
            print(f"Error: Comparison checkpoint not found: {args.compare}")
            sys.exit(1)
        compare_checkpoints(args.checkpoint, args.compare)


if __name__ == "__main__":
    main()
