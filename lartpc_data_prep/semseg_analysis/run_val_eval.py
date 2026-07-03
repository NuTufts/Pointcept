"""Reproduce the trainer's SemSegEvaluator val-time metric against a saved
checkpoint, without going through tools/train.py.

This calls the model on the val_loader exactly the way SemSegEvaluator.eval()
does, accumulates intersection/union/target across batches, and prints per-
class IoU + accuracy + mIoU. Used to verify whether a saved checkpoint
reproduces the wandb mIoU at training-time eval.

Usage:
    python lartpc_data_prep/semseg_analysis/run_val_eval.py \
        --config-file configs/lartpc/lora_finetune/archive/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
        --weight sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth \
        --val-filelist prod4_valsplit_subset.txt \
        --batch-size 48 \
        --nfiles 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow `from pointcept...` regardless of where this is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
POINTCEPT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if POINTCEPT_ROOT not in sys.path:
    sys.path.insert(0, POINTCEPT_ROOT)

from pointcept.utils.config import Config  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.datasets.utils import point_collate_fn  # noqa: E402
import pointcept.datasets  # noqa: F401,E402
import pointcept.models  # noqa: F401,E402


def intersection_and_union_gpu(output, target, K, ignore_index=-1):
    """Reproduces pointcept/utils/misc.py:intersection_and_union_gpu.

    output: (N,) int prediction
    target: (N,) int truth (-1 = ignore)
    K     : number of classes
    """
    assert output.dim() in [1, 2]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection.float(), bins=K, min=0, max=K - 1
    )
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--val-filelist", required=True,
                   help="Path to text file with one H5 path per line (val pipeline)")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--nfiles", type=int, default=-1,
                   help="Use only first N files from the filelist")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Determinism for the val pipeline's stochastic transforms (drop_cosmics,
    # BiasedSphereCrop, GridSample mode='train').
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config_file)
    cfg.data.val.data_list_file = os.path.abspath(args.val_filelist)

    if args.nfiles > 0:
        # Truncate the val dataset to N files. The dataset loads from a text
        # file at init; the cleanest way to truncate without editing the file
        # is to write a temp truncated list.
        with open(cfg.data.val.data_list_file) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        lines = lines[: args.nfiles]
        tmp_path = f"/tmp/_val_eval_truncated_{os.getpid()}.txt"
        with open(tmp_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        cfg.data.val.data_list_file = tmp_path
        print(f"[main] using truncated val filelist with {len(lines)} files")

    val_ds = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=point_collate_fn,
    )
    print(f"[main] val_loader: {len(val_loader)} batches, batch_size={args.batch_size}")

    model = build_model(cfg.model)
    ckpt = torch.load(args.weight, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[main] load: missing={len(missing)} unexpected={len(unexpected)}  "
          f"(epoch={ckpt.get('epoch')}, best={ckpt.get('best_metric_value')})")
    if missing[:3]:
        print(f"  first 3 missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"  first 3 unexpected: {unexpected[:3]}")
    model.eval().cuda()

    K = int(cfg.data.num_classes)
    ignore_index = int(cfg.data.ignore_index)
    names = list(cfg.data.names)

    total_inter = torch.zeros(K).cuda()
    total_union = torch.zeros(K).cuda()
    total_target = torch.zeros(K).cuda()
    total_loss = 0.0
    n_batches = 0

    t0 = time.time()
    for i, input_dict in enumerate(val_loader):
        for k in input_dict:
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].cuda(non_blocking=True)
        with torch.no_grad():
            output_dict = model(input_dict)
        logits = output_dict["seg_logits"]
        loss = output_dict.get("loss", torch.tensor(0.0))
        pred = logits.argmax(dim=-1)
        segment = input_dict["segment"]
        inter, union, target = intersection_and_union_gpu(
            pred, segment, K, ignore_index
        )
        total_inter += inter
        total_union += union
        total_target += target
        total_loss += float(loss.item()) if hasattr(loss, "item") else float(loss)
        n_batches += 1
        print(f"  batch {i+1}/{len(val_loader)}  "
              f"N={pred.shape[0]:>6d}  loss={float(loss.item()):.4f}  "
              f"batch_acc={(pred[segment != ignore_index] == segment[segment != ignore_index]).float().mean().item():.4f}")

    elapsed = time.time() - t0
    iou_class = total_inter / (total_union + 1e-10)
    acc_class = total_inter / (total_target + 1e-10)
    m_iou = iou_class.mean().item()
    m_acc = acc_class.mean().item()
    all_acc = total_inter.sum().item() / (total_target.sum().item() + 1e-10)
    avg_loss = total_loss / max(n_batches, 1)

    print()
    print(f"Val result: mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}  "
          f"(avg loss {avg_loss:.4f}, {elapsed:.1f}s)")
    for c in range(K):
        print(f"  Class_{c}-{names[c]}: iou/accuracy "
              f"{iou_class[c].item():.4f}/{acc_class[c].item():.4f}")


if __name__ == "__main__":
    main()
