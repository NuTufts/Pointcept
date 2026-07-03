"""
Test script to visualize learning rate schedules when extending training.

This script helps tune scheduler parameters to minimize the learning rate jump
when resuming training with an extended epoch count.

Usage:
    python tools/test_scheduler_extension.py \
        --checkpoint exp/lartpc/your_exp/model/epoch_90.pth \
        --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py \
        --original-epochs 100 \
        --extended-epochs 200

Or without a checkpoint (uses config defaults):
    python tools/test_scheduler_extension.py \
        --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py \
        --original-epochs 100 \
        --extended-epochs 200 \
        --resume-epoch 90
"""

import argparse
import os
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config


def create_dummy_optimizer(config):
    """Create a dummy optimizer matching the config structure."""
    # Create dummy parameters - one for base LR and one for each param group
    num_param_groups = 1
    if hasattr(config, 'param_dicts') and config.param_dicts:
        num_param_groups += len(config.param_dicts)

    params = [torch.zeros(1, requires_grad=True) for _ in range(num_param_groups)]

    # Create optimizer with param groups
    param_groups = [{'params': [params[0]], 'lr': config.optimizer.lr}]
    if hasattr(config, 'param_dicts') and config.param_dicts:
        for i, pd in enumerate(config.param_dicts):
            param_groups.append({
                'params': [params[i + 1]],
                'lr': pd.get('lr', config.optimizer.lr)
            })

    optimizer = optim.AdamW(param_groups, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    return optimizer


def create_scheduler(optimizer, scheduler_cfg, total_steps):
    """Create OneCycleLR scheduler."""
    # Build max_lr list from config
    max_lr = scheduler_cfg.max_lr if isinstance(scheduler_cfg.max_lr, list) else [scheduler_cfg.max_lr]

    # Ensure max_lr has correct length for all param groups
    if len(max_lr) < len(optimizer.param_groups):
        max_lr = max_lr + [max_lr[0]] * (len(optimizer.param_groups) - len(max_lr))

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=scheduler_cfg.get('pct_start', 0.3),
        anneal_strategy=scheduler_cfg.get('anneal_strategy', 'cos'),
        div_factor=scheduler_cfg.get('div_factor', 25.0),
        final_div_factor=scheduler_cfg.get('final_div_factor', 1e4),
    )
    return scheduler


def compute_onecycle_lr(step, total_steps, max_lr, pct_start=0.3, div_factor=25.0,
                         final_div_factor=1e4, anneal_strategy='cos'):
    """
    Compute OneCycleLR learning rate analytically (much faster than stepping).

    This reimplements the OneCycleLR formula without creating a scheduler.
    """
    initial_lr = max_lr / div_factor
    min_lr = max_lr / final_div_factor

    # Phase 1: warmup from initial_lr to max_lr
    warmup_steps = int(pct_start * total_steps)

    if step < warmup_steps:
        # Linear interpolation during warmup (or cosine if anneal_strategy='cos')
        if anneal_strategy == 'cos':
            # Cosine annealing up
            pct = step / warmup_steps
            return initial_lr + (max_lr - initial_lr) * (1 - np.cos(np.pi * pct)) / 2
        else:
            # Linear
            return initial_lr + (max_lr - initial_lr) * (step / warmup_steps)
    else:
        # Phase 2: anneal from max_lr to min_lr
        pct = (step - warmup_steps) / (total_steps - warmup_steps)
        if anneal_strategy == 'cos':
            return min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * pct)) / 2
        else:
            # Linear
            return max_lr - (max_lr - min_lr) * pct


def _get_scheduler_param(scheduler_cfg, key, default=None):
    """Get a parameter from scheduler config, handling both dict and object access."""
    if isinstance(scheduler_cfg, dict):
        return scheduler_cfg.get(key, default)
    else:
        return getattr(scheduler_cfg, key, default) if hasattr(scheduler_cfg, key) else scheduler_cfg.get(key, default)


def _get_max_lr(scheduler_cfg):
    """Get max_lr from scheduler config, handling both dict and object access."""
    if isinstance(scheduler_cfg, dict):
        max_lr = scheduler_cfg.get('max_lr')
    else:
        max_lr = getattr(scheduler_cfg, 'max_lr', None) or scheduler_cfg.get('max_lr')
    return max_lr[0] if isinstance(max_lr, list) else max_lr


def get_lr_schedule_fast(scheduler_cfg, total_steps, num_samples=1000):
    """Get learning rate schedule using analytical formula (fast)."""
    max_lr = _get_max_lr(scheduler_cfg)
    pct_start = _get_scheduler_param(scheduler_cfg, 'pct_start', 0.3)
    div_factor = _get_scheduler_param(scheduler_cfg, 'div_factor', 25.0)
    final_div_factor = _get_scheduler_param(scheduler_cfg, 'final_div_factor', 1e4)
    anneal_strategy = _get_scheduler_param(scheduler_cfg, 'anneal_strategy', 'cos')

    steps = np.linspace(0, total_steps - 1, num_samples).astype(int)
    lrs = [compute_onecycle_lr(s, total_steps, max_lr, pct_start, div_factor,
                                final_div_factor, anneal_strategy) for s in steps]
    return steps, np.array(lrs)


def get_lr_at_step(scheduler_cfg, total_steps, step):
    """Get LR at a specific step using analytical formula."""
    max_lr = _get_max_lr(scheduler_cfg)
    pct_start = _get_scheduler_param(scheduler_cfg, 'pct_start', 0.3)
    div_factor = _get_scheduler_param(scheduler_cfg, 'div_factor', 25.0)
    final_div_factor = _get_scheduler_param(scheduler_cfg, 'final_div_factor', 1e4)
    anneal_strategy = _get_scheduler_param(scheduler_cfg, 'anneal_strategy', 'cos')

    return compute_onecycle_lr(step, total_steps, max_lr, pct_start, div_factor,
                               final_div_factor, anneal_strategy)


def get_lr_schedule(optimizer, scheduler_cfg, total_steps, start_step=0):
    """Get the full learning rate schedule (legacy, slower method)."""
    # Use fast analytical method instead
    steps, lrs = get_lr_schedule_fast(scheduler_cfg, total_steps, num_samples=total_steps)
    return lrs.tolist()


def main():
    parser = argparse.ArgumentParser(
        description='Test scheduler extension - compare original vs extended LR schedules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare original (from command line) vs extended (from config):
  python tools/test_scheduler_extension.py \\
      --checkpoint epoch_90.pth \\
      --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py \\
      --original-max-lr 0.004 \\
      --original-div-factor 10.0 \\
      --original-final-div-factor 1000.0 \\
      --original-pct-start 0.05

  # Test different extended parameters without modifying config:
  python tools/test_scheduler_extension.py \\
      --checkpoint epoch_90.pth \\
      --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py \\
      --original-max-lr 0.004 \\
      --extended-max-lr 0.001 \\
      --extended-pct-start 0.025
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (optional)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (used for extended scheduler defaults)')
    parser.add_argument('--original-epochs', type=int, default=100,
                        help='Original number of epochs')
    parser.add_argument('--extended-epochs', type=int, default=200,
                        help='Extended number of epochs')
    parser.add_argument('--resume-epoch', type=int, default=90,
                        help='Epoch to resume from (used if no checkpoint provided)')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Steps per epoch (if not provided, inferred from checkpoint)')

    # Original scheduler parameters (what you trained with for the first 100 epochs)
    parser.add_argument('--original-max-lr', type=float, default=None,
                        help='Original max_lr (if not set, uses config value)')
    parser.add_argument('--original-div-factor', type=float, default=None,
                        help='Original div_factor (if not set, uses config value)')
    parser.add_argument('--original-final-div-factor', type=float, default=None,
                        help='Original final_div_factor (if not set, uses config value)')
    parser.add_argument('--original-pct-start', type=float, default=None,
                        help='Original pct_start (if not set, uses config value)')

    # Extended scheduler parameters (overrides config for testing)
    parser.add_argument('--extended-max-lr', type=float, default=None,
                        help='Extended max_lr (overrides config)')
    parser.add_argument('--extended-div-factor', type=float, default=None,
                        help='Extended div_factor (overrides config)')
    parser.add_argument('--extended-final-div-factor', type=float, default=None,
                        help='Extended final_div_factor (overrides config)')
    parser.add_argument('--extended-pct-start', type=float, default=None,
                        help='Extended pct_start (overrides config)')

    parser.add_argument('--output', type=str, default='scheduler_comparison.png',
                        help='Output plot filename')
    parser.add_argument('--show', action='store_true',
                        help='Show plot interactively')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Determine resume epoch and infer steps_per_epoch from checkpoint
    inferred_steps_per_epoch = None
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        resume_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {resume_epoch}")

        # Check if scheduler state exists
        if 'scheduler' in checkpoint:
            old_scheduler_state = checkpoint['scheduler']
            old_total_steps = old_scheduler_state.get('total_steps', None)
            old_last_epoch = old_scheduler_state.get('last_epoch', None)
            print(f"Old scheduler total_steps: {old_total_steps}")
            print(f"Old scheduler last_epoch (step count): {old_last_epoch}")

            # Infer steps_per_epoch from checkpoint
            if old_total_steps and args.original_epochs:
                inferred_steps_per_epoch = old_total_steps // args.original_epochs
                print(f"Inferred steps_per_epoch from checkpoint: {inferred_steps_per_epoch}")
    else:
        resume_epoch = args.resume_epoch
        print(f"No checkpoint provided, using resume_epoch={resume_epoch}")

    # Calculate steps per epoch (optimizer steps, accounting for gradient accumulation)
    # Priority: command line > inferred from checkpoint > default estimate
    gradient_accumulation = getattr(cfg, 'gradient_accumulation_steps', 1)

    if args.steps_per_epoch:
        # Command line value is assumed to be batch iterations, convert to optimizer steps
        steps_per_epoch = args.steps_per_epoch // gradient_accumulation
        print(f"Using command-line steps_per_epoch={args.steps_per_epoch} / {gradient_accumulation} = {steps_per_epoch}")
    elif inferred_steps_per_epoch:
        # Inferred from checkpoint is already optimizer steps (no conversion needed)
        steps_per_epoch = inferred_steps_per_epoch
        print(f"Using inferred steps_per_epoch={steps_per_epoch} (from checkpoint, already accounts for grad accum)")
    else:
        # Default estimate based on typical LArTPC dataset (batch iterations)
        steps_per_epoch = 3324 // gradient_accumulation
        print(f"Using estimated steps_per_epoch={steps_per_epoch}")
        print("(Override with --steps-per-epoch if you know the exact value)")

    original_total_steps = args.original_epochs * steps_per_epoch
    extended_total_steps = args.extended_epochs * steps_per_epoch
    resume_step = resume_epoch * steps_per_epoch

    print(f"\nTraining Configuration:")
    print(f"  Original epochs: {args.original_epochs}")
    print(f"  Extended epochs: {args.extended_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Original total steps: {original_total_steps:,}")
    print(f"  Extended total steps: {extended_total_steps:,}")
    print(f"  Resume epoch: {resume_epoch}")
    print(f"  Resume step: {resume_step:,}")

    # Create dummy optimizer
    optimizer = create_dummy_optimizer(cfg)

    # Get base scheduler config from file
    base_scheduler_cfg = cfg.scheduler
    base_max_lr = base_scheduler_cfg.max_lr[0] if isinstance(base_scheduler_cfg.max_lr, list) else base_scheduler_cfg.max_lr

    # Build ORIGINAL scheduler config (what you trained with for first 100 epochs)
    original_scheduler_cfg = {
        'max_lr': [args.original_max_lr if args.original_max_lr else base_max_lr],
        'pct_start': args.original_pct_start if args.original_pct_start else base_scheduler_cfg.get('pct_start', 0.3),
        'div_factor': args.original_div_factor if args.original_div_factor else base_scheduler_cfg.get('div_factor', 25.0),
        'final_div_factor': args.original_final_div_factor if args.original_final_div_factor else base_scheduler_cfg.get('final_div_factor', 1e4),
        'anneal_strategy': base_scheduler_cfg.get('anneal_strategy', 'cos'),
    }

    # Build EXTENDED scheduler config (what you want to use for epochs 90-200)
    extended_scheduler_cfg = {
        'max_lr': [args.extended_max_lr if args.extended_max_lr else base_max_lr],
        'pct_start': args.extended_pct_start if args.extended_pct_start else base_scheduler_cfg.get('pct_start', 0.3),
        'div_factor': args.extended_div_factor if args.extended_div_factor else base_scheduler_cfg.get('div_factor', 25.0),
        'final_div_factor': args.extended_final_div_factor if args.extended_final_div_factor else base_scheduler_cfg.get('final_div_factor', 1e4),
        'anneal_strategy': base_scheduler_cfg.get('anneal_strategy', 'cos'),
    }

    print(f"\nORIGINAL scheduler parameters (first {args.original_epochs} epochs):")
    print(f"  max_lr: {original_scheduler_cfg['max_lr'][0]}")
    print(f"  pct_start: {original_scheduler_cfg['pct_start']}")
    print(f"  div_factor: {original_scheduler_cfg['div_factor']}")
    print(f"  final_div_factor: {original_scheduler_cfg['final_div_factor']}")

    print(f"\nEXTENDED scheduler parameters ({args.extended_epochs} epochs total):")
    print(f"  max_lr: {extended_scheduler_cfg['max_lr'][0]}")
    print(f"  pct_start: {extended_scheduler_cfg['pct_start']}")
    print(f"  div_factor: {extended_scheduler_cfg['div_factor']}")
    print(f"  final_div_factor: {extended_scheduler_cfg['final_div_factor']}")

    # Generate learning rate schedules using fast analytical method
    print("\nGenerating learning rate schedules (analytical, fast)...")

    # Original schedule (100 epochs) - sample 2000 points for smooth plotting
    original_steps, original_lrs = get_lr_schedule_fast(original_scheduler_cfg, original_total_steps, num_samples=2000)

    # Extended schedule (200 epochs) - sample 2000 points
    extended_steps, extended_lrs = get_lr_schedule_fast(extended_scheduler_cfg, extended_total_steps, num_samples=2000)

    # Calculate LR at resume point for both schedules (exact values)
    original_lr_at_resume = get_lr_at_step(original_scheduler_cfg, original_total_steps, min(resume_step, original_total_steps - 1))
    extended_lr_at_resume = get_lr_at_step(extended_scheduler_cfg, extended_total_steps, resume_step)

    lr_jump = extended_lr_at_resume - original_lr_at_resume
    lr_jump_pct = (lr_jump / original_lr_at_resume) * 100 if original_lr_at_resume > 0 else 0

    print(f"\nLearning rate analysis at resume point (epoch {resume_epoch}):")
    print(f"  Original schedule LR: {original_lr_at_resume:.6f}")
    print(f"  Extended schedule LR: {extended_lr_at_resume:.6f}")
    print(f"  LR jump: {lr_jump:+.6f} ({lr_jump_pct:+.1f}%)")

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Convert steps to epochs for x-axis
    original_epochs_axis = original_steps / steps_per_epoch
    extended_epochs_axis = extended_steps / steps_per_epoch

    # Plot 1: Full schedules comparison
    ax1 = axes[0, 0]
    ax1.plot(original_epochs_axis, original_lrs, 'b-', label=f'Original ({args.original_epochs} epochs)', linewidth=2)
    ax1.plot(extended_epochs_axis, extended_lrs, 'r--', label=f'Extended ({args.extended_epochs} epochs)', linewidth=2)
    ax1.axvline(x=resume_epoch, color='g', linestyle=':', label=f'Resume point (epoch {resume_epoch})', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Full Learning Rate Schedules')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, args.extended_epochs)

    # Plot 2: Zoomed view around resume point
    ax2 = axes[0, 1]
    zoom_start = max(0, resume_epoch - 20)
    zoom_end = min(args.extended_epochs, resume_epoch + 20)

    # Filter data for zoom range
    orig_mask = (original_epochs_axis >= zoom_start) & (original_epochs_axis <= zoom_end)
    ext_mask = (extended_epochs_axis >= zoom_start) & (extended_epochs_axis <= zoom_end)

    ax2.plot(original_epochs_axis[orig_mask], original_lrs[orig_mask],
             'b-', label='Original', linewidth=2)
    ax2.plot(extended_epochs_axis[ext_mask], extended_lrs[ext_mask],
             'r--', label='Extended', linewidth=2)
    ax2.axvline(x=resume_epoch, color='g', linestyle=':', label='Resume point', linewidth=2)
    ax2.scatter([resume_epoch], [original_lr_at_resume], color='b', s=100, zorder=5, marker='o')
    ax2.scatter([resume_epoch], [extended_lr_at_resume], color='r', s=100, zorder=5, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'Zoomed View (epochs {zoom_start}-{zoom_end})\nLR jump: {lr_jump_pct:+.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: What training will actually see (extended schedule from resume point)
    ax3 = axes[1, 0]
    # Original: what we completed
    orig_completed_mask = original_epochs_axis <= resume_epoch
    ax3.plot(original_epochs_axis[orig_completed_mask], original_lrs[orig_completed_mask],
             'b-', label='Completed (original)', linewidth=2)
    # Extended: what we will continue with
    ext_remaining_mask = extended_epochs_axis >= resume_epoch
    ax3.plot(extended_epochs_axis[ext_remaining_mask], extended_lrs[ext_remaining_mask],
             'r-', label='Continuing (extended)', linewidth=2)
    ax3.axvline(x=resume_epoch, color='g', linestyle=':', label='Resume point', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Actual Training Path\n(Completed + Extended)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, args.extended_epochs)

    # Plot 4: LR jump sensitivity analysis (using fast analytical method)
    ax4 = axes[1, 1]
    # Try different pct_start values to find one that minimizes the jump
    pct_start_values = np.linspace(0.01, 0.5, 50)
    lr_jumps = []

    max_lr = extended_scheduler_cfg['max_lr'][0]
    div_factor = extended_scheduler_cfg['div_factor']
    final_div_factor = extended_scheduler_cfg['final_div_factor']

    for pct_start in pct_start_values:
        # Use analytical formula directly - much faster
        test_extended_lr = compute_onecycle_lr(
            resume_step, extended_total_steps, max_lr,
            pct_start=pct_start, div_factor=div_factor,
            final_div_factor=final_div_factor, anneal_strategy='cos'
        )
        test_jump = test_extended_lr - original_lr_at_resume
        lr_jumps.append(test_jump)

    ax4.plot(pct_start_values, lr_jumps, 'b-', linewidth=2)
    ax4.axhline(y=0, color='g', linestyle='--', label='No jump', linewidth=1)
    ax4.axvline(x=extended_scheduler_cfg['pct_start'], color='r', linestyle=':',
                label=f'Current pct_start={extended_scheduler_cfg["pct_start"]}', linewidth=2)

    # Find optimal pct_start
    optimal_idx = np.argmin(np.abs(lr_jumps))
    optimal_pct_start = pct_start_values[optimal_idx]
    ax4.scatter([optimal_pct_start], [lr_jumps[optimal_idx]], color='g', s=100, zorder=5)
    ax4.annotate(f'Optimal: {optimal_pct_start:.3f}',
                 xy=(optimal_pct_start, lr_jumps[optimal_idx]),
                 xytext=(optimal_pct_start + 0.05, lr_jumps[optimal_idx]),
                 fontsize=10)

    ax4.set_xlabel('pct_start')
    ax4.set_ylabel('LR Jump at Resume Point')
    ax4.set_title('pct_start Sensitivity Analysis\n(Lower |jump| is better)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = args.output
    if not output_path.endswith('.png'):
        output_path += '.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if args.show:
        plt.show()

    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)

    if abs(lr_jump_pct) > 50:
        print(f"\nWARNING: Large LR jump detected ({lr_jump_pct:+.1f}%)")
        print("\nTo minimize the LR jump, consider one of these approaches:")
        print(f"\n1. Adjust pct_start to ~{optimal_pct_start:.3f}")
        print("   This will shift the warmup phase to better align with your resume point.")
        print(f"   Add to config: scheduler = dict(..., pct_start={optimal_pct_start:.3f})")

        print("\n2. Use a gradual warmup after resume:")
        print("   Manually reduce the starting LR for the first few epochs after resume")

        print("\n3. Accept the jump but reduce total training:")
        print(f"   The extended schedule has you at step {resume_step:,} of {extended_total_steps:,}")
        print(f"   This is {100*resume_step/extended_total_steps:.1f}% through the schedule")
    else:
        print(f"\nLR jump is reasonable ({lr_jump_pct:+.1f}%)")
        print("You can proceed with the current scheduler settings.")

    # Show what the LR will be like for the rest of training
    print(f"\nLR at key points after resume (using extended scheduler):")
    for pct in [0, 25, 50, 75, 100]:
        step = resume_step + int((extended_total_steps - resume_step) * pct / 100)
        step = min(step, extended_total_steps - 1)
        epoch = step / steps_per_epoch
        lr_at_step = get_lr_at_step(extended_scheduler_cfg, extended_total_steps, step)
        print(f"  {pct:3d}% remaining (epoch {epoch:.0f}): LR = {lr_at_step:.6f}")


if __name__ == '__main__':
    main()
