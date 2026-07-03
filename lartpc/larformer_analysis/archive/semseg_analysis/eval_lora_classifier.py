"""
Confusion Matrix Evaluation for SonataLoRASegmentor (SSNet, 8-class)
=====================================================================

Loads a trained SSNet LoRA checkpoint, runs inference on a random subset
of events drawn from a specified file list, and produces:

  1. A confusion matrix plot (Blues colormap, matching project style)
  2. A printed per-class IoU / classification report
  3. A metrics summary saved as a .json file

The random subset tests whether strong validation performance generalises
beyond the events seen during training.  Run with --n-bootstrap > 1 to
draw multiple independent samples and report mean ± std.

Usage
-----
python tools/eval_ssnet_confusion.py \\
    --config     configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6.py \\
    --checkpoint sonata/lora_finetune_v6_p100_50_epochs_noghost/model/model_best.pth \\
    --data-list  /path/to/holdout_filelist.txt \\
    --output-dir exp/lartpc/sonata/lora_finetune_v6/eval \\
    --num-events 500 \\
    --n-bootstrap 5 \\
    --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.utils.config import Config
from pointcept.models.builder import build_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["electron", "muon", "pion", "proton", "gamma", "michel", "delta", "led"]
IGNORE_INDEX = -1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Confusion matrix evaluation for SonataLoRASegmentor (SSNet 8-class)"
    )
    parser.add_argument("--config", required=True,
        help="Path to lorafinetune-sonata-v1m1-lartpc-v6.py")
    parser.add_argument("--checkpoint", required=True,
        help="Path to model_best.pth or model_last.pth")
    parser.add_argument("--data-list", default=None,
        help="Override val file list from config. Recommended: use a held-out "
             "file list not seen during training or validation.")
    parser.add_argument("--output-dir", default=None,
        help="Directory to save outputs. Defaults to checkpoint directory/eval.")
    parser.add_argument("--num-events", type=int, default=500,
        help="Number of events to randomly sample per bootstrap trial. "
             "Default 500.  Set to -1 to use all events.")
    parser.add_argument("--n-bootstrap", type=int, default=1,
        help="Number of independent random samples. >1 reports mean ± std.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-size", type=float, default=None,
        help="Override voxel grid size from config.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ssnet_model(cfg, checkpoint_path, device):
    """
    Register SonataLoRASegmentor and load checkpoint.
    Mirrors the import / registration logic from eval_deghost_confusion.py.
    """
    import importlib
    import importlib.util
    from collections import OrderedDict

    here           = os.path.dirname(os.path.abspath(__file__))
    repo_root      = os.path.normpath(os.path.join(here, ".."))
    models_dir     = os.path.join(repo_root, "pointcept", "models")
    cluster_models = "/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept/pointcept/models"

    for module_name, filename in [
        ("lora_sonata", "lora_sonata.py"),
    ]:
        try:
            importlib.import_module(module_name)
            print(f"  {module_name} already registered (via __init__.py)")
            continue
        except ModuleNotFoundError:
            pass

        for path in [
            os.path.join(cluster_models, filename),
            os.path.join(here,           filename),
            os.path.join(repo_root,      filename),
            os.path.join(models_dir,     filename),
        ]:
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod  = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    print(f"  Registered {module_name} from {path}")
                except KeyError:
                    print(f"  {module_name} already registered (skipping re-register)")
                break

    cfg.model.backbone.up_cast_level = 4
    model = build_model(cfg.model).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[7:] if k.startswith("module.") else k] = v
    info = model.load_state_dict(new_state, strict=False)

    lora_missing = [k for k in info.missing_keys if "lora_"    in k]
    head_missing = [k for k in info.missing_keys if "seg_head" in k]
    other_missing = [k for k in info.missing_keys
                     if k not in lora_missing and k not in head_missing]
    print(f"  LoRA keys (expected missing)   : {len(lora_missing)}")
    print(f"  seg_head keys (expected missing): {len(head_missing)}")
    if other_missing:
        print(f"  WARNING — unexpected missing keys: {other_missing[:10]}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_single(data, in_channels, device):
    """Collate a single dataset item and move to device."""
    coord = data["coord"]
    feat  = data["feat"][:, :in_channels]
    return {
        "coord":      coord.to(device),
        "feat":       feat.to(device),
        "grid_coord": data["grid_coord"].to(device),
        "offset":     torch.tensor([coord.shape[0]], dtype=torch.long, device=device),
        "segment":    data["segment"].to(device) if "segment" in data else None,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(confusion: np.ndarray):
    """Per-class IoU from an (C, C) confusion matrix."""
    tp    = np.diag(confusion)
    fn    = confusion.sum(axis=1) - tp
    fp    = confusion.sum(axis=0) - tp
    denom = tp + fn + fp
    return np.where(denom > 0, tp / denom, np.nan)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(confusion, output_path, title="Confusion Matrix"):
    """
    Plot the 8-class confusion matrix in Blues colormap, matching project style.
    Rows = true class, columns = predicted class.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=CLASS_NAMES,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix:\n{confusion}")


# ---------------------------------------------------------------------------
# Single evaluation trial
# ---------------------------------------------------------------------------

def run_trial(model, dataset, num_events, in_channels, device, seed):
    rng = np.random.default_rng(seed)

    n_total  = len(dataset)
    indices  = rng.choice(n_total, size=min(num_events, n_total), replace=False) \
               if num_events > 0 else np.arange(n_total)

    n_classes  = len(CLASS_NAMES)
    confusion  = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Event {i+1}/{len(indices)} ...", end="\r")

            data   = dataset[int(idx)]
            batch  = collate_single(data, in_channels, device)
            labels = batch["segment"]                      # (N,) long

            out    = model(batch)
            preds  = out["seg_logits"].argmax(dim=-1)      # (N,)

            valid  = labels != IGNORE_INDEX
            p = preds[valid].cpu().numpy()
            t = labels[valid].cpu().numpy()
            for true, pred in zip(t.tolist(), p.tolist()):
                if 0 <= true < n_classes and 0 <= pred < n_classes:
                    confusion[true, pred] += 1

    print(f"\n  Inference complete over {len(indices)} events.")
    return confusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Config -------------------------------------------------------
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)

    grid_size = args.grid_size if args.grid_size is not None else cfg.get("grid_size", 0.25)

    # ---- Output directory ---------------------------------------------
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- Data list ----------------------------------------------------
    data_list = args.data_list
    if data_list is None:
        if hasattr(cfg, "VAL_FILE_LIST"):
            data_list = cfg.VAL_FILE_LIST
        elif hasattr(cfg, "data") and hasattr(cfg.data, "val"):
            data_list = cfg.data.val.get("data_list_file", None)
    if data_list is None:
        raise ValueError(
            "Cannot determine data list. Pass --data-list or set VAL_FILE_LIST in config."
        )
    print(f"File list: {data_list}")

    # ---- Dataset (re-uses val transform from config exactly) ----------
    from pointcept.datasets.lartpc import LArTPCDataset
    val_cfg = cfg.data.val

    dataset = LArTPCDataset(
        split="val",
        data_list_file=data_list,
        transform=val_cfg.transform,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="ssnet",
        coord_scale=val_cfg.get("coord_scale", 1.0),
        adc_scale=val_cfg.get("adc_scale", 500.0),
        include_ghosts=False,
        true_points_only=True,
        exclude_other=True,
        drop_cosmics=True,
        drop_cosmics_prob=0.9,
        test_mode=False,
        loop=1,
    )
    print(f"Dataset size: {len(dataset)} events")

    # ---- Model --------------------------------------------------------
    model       = load_ssnet_model(cfg, args.checkpoint, args.device)
    in_channels = cfg.model.backbone.backbone.in_channels
    print(f"Model input channels: {in_channels}")

    # ---- Bootstrap trials --------------------------------------------
    ckpt_name   = os.path.splitext(os.path.basename(args.checkpoint))[0]
    all_confusions = []

    for trial in range(args.n_bootstrap):
        seed = args.seed + trial
        print(f"\n[Trial {trial+1}/{args.n_bootstrap}]  seed={seed}")
        confusion = run_trial(
            model, dataset,
            num_events=args.num_events,
            in_channels=in_channels,
            device=args.device,
            seed=seed,
        )
        all_confusions.append(confusion)

        iou = compute_iou(confusion)
        total   = confusion.sum()
        correct = np.diag(confusion).sum()
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n{'='*52}")
        print(f"  SSNet Evaluation — Trial {trial+1}")
        print(f"{'='*52}")
        print(f"  Total points : {total:,}")
        print(f"  Accuracy     : {accuracy:.4f}")
        print(f"  Macro-IoU    : {np.nanmean(iou):.4f}")
        print(f"\n  Per-class IoU:")
        for i, cls in enumerate(CLASS_NAMES):
            val = f"{iou[i]:.4f}" if not np.isnan(iou[i]) else "  N/A "
            bar = "█" * int((iou[i] if not np.isnan(iou[i]) else 0) * 30)
            print(f"    {cls:<10} {val}  {bar}")
        print(f"{'='*52}")

    # ---- Aggregate across bootstrap trials ---------------------------
    sum_confusion = np.sum(all_confusions, axis=0)   # aggregate over all trials
    iou_per_trial = [compute_iou(c) for c in all_confusions]
    iou_arr       = np.array(iou_per_trial)           # (n_bootstrap, n_classes)

    if args.n_bootstrap > 1:
        print(f"\n{'='*52}")
        print(f"  BOOTSTRAP SUMMARY  ({args.n_bootstrap} trials)")
        print(f"{'='*52}")
        macro_per_trial = [float(np.nanmean(compute_iou(c))) for c in all_confusions]
        print(f"  Macro-IoU : {np.mean(macro_per_trial):.4f} ± {np.std(macro_per_trial):.4f}")
        print(f"\n  Per-class IoU (mean ± std):")
        for i, cls in enumerate(CLASS_NAMES):
            vals = iou_arr[:, i]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                print(f"    {cls:<10} {np.mean(vals):.4f} ± {np.std(vals):.4f}")
            else:
                print(f"    {cls:<10}   N/A")

    # ---- Plot (aggregated confusion matrix) --------------------------
    cm_path = os.path.join(out_dir, f"confusion_matrix_ssnet_{ckpt_name}.png")
    n_events_total = args.num_events * args.n_bootstrap if args.num_events > 0 \
                     else len(dataset) * args.n_bootstrap
    plot_confusion_matrix(
        sum_confusion,
        output_path=cm_path,
        title=(f"SSNet Confusion Matrix\n"
               f"{ckpt_name}  |  "
               f"n_trials={args.n_bootstrap}  |  "
               f"n_points={sum_confusion.sum():,}"),
    )
    print(f"\nSaved confusion matrix to: {cm_path}")

    # ---- Save metrics JSON -------------------------------------------
    final_iou = compute_iou(sum_confusion)
    total     = int(sum_confusion.sum())
    correct   = int(np.diag(sum_confusion).sum())

    metrics = {
        "checkpoint":    args.checkpoint,
        "data_list":     data_list,
        "num_events":    args.num_events,
        "n_bootstrap":   args.n_bootstrap,
        "total_points":  total,
        "accuracy":      round(correct / total, 4) if total > 0 else 0.0,
        "macro_iou":     round(float(np.nanmean(final_iou)), 4),
        "iou_per_class": {
            CLASS_NAMES[i]: round(float(final_iou[i]), 4)
                if not np.isnan(final_iou[i]) else None
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": sum_confusion.tolist(),
    }
    if args.n_bootstrap > 1:
        macro_per_trial = [float(np.nanmean(compute_iou(c))) for c in all_confusions]
        metrics["macro_iou_std"] = round(float(np.std(macro_per_trial)), 4)

    json_path = os.path.join(out_dir, f"metrics_ssnet_{ckpt_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to        : {json_path}")


if __name__ == "__main__":
    main()
