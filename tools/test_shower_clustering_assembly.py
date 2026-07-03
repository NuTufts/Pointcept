"""
Phase 7 assembly test — builds the full ShowerClusteringMask2Former model
from a config and runs forward + backward on one or more events.

Two modes:
    --mock-backbone       : substitute a tiny Linear(6, backbone_out_channels)
                            for the real Sonata. CPU-friendly. Verifies the
                            data flow + shape contracts without needing GPU.

    (default)             : build the real frozen Sonata. Needs:
                              - Pretrained Sonata weights
                              - GPU (P100 / H200) for reasonable performance
                              - --device cuda

Usage examples:

    # CPU smoke test with mock backbone
    python tools/test_shower_clustering_assembly.py \\
        -c configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py \\
        --mock-backbone --num-events 1

    # GPU run with real backbone (P100 / H200)
    python tools/test_shower_clustering_assembly.py \\
        -c configs/lartpc/shower_origin/archive/shower-cluster-sonata-v1.py \\
        --device cuda --num-events 1
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.shower_clustering import shower_clustering_collate
from pointcept.models.builder import MODELS, build_model


# ----------------------------------------------------------------------
# Mock backbone for CPU smoke testing
# ----------------------------------------------------------------------

class _MockSonata(nn.Module):
    """Stand-in for Sonata-v1m1 that mimics the input/output contract.

    The real backbone takes a dict {coord, feat, offset, grid_coord} and
    returns {"point": Point} with `.feat` of shape (N, out_channels).
    This mock just projects feat with a Linear and packages it the same way.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, input_dict, return_point=False):
        feat = self.proj(input_dict["feat"])
        return {"point": SimpleNamespace(feat=feat)}


def _swap_in_mock(model, in_channels, out_channels):
    """Replace the model.backbone with a mock for CPU testing."""
    print("[assembly-test] WARNING: using mock backbone (random init "
          "Linear). Use without --mock-backbone for the real Sonata.")
    model.backbone = _MockSonata(in_channels, out_channels)
    # Mock has trainable params; freeze them since the real backbone is frozen
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.freeze_backbone = True
    return model


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--split", default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--data-list", default=None)
    p.add_argument("--num-events", type=int, default=1,
                   help="Events per batch (single-batch test).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mock-backbone", action="store_true",
                   help="Replace Sonata with a tiny Linear (CPU-friendly).")
    p.add_argument("--load-weights", default="auto",
                   help=("Path to a Sonata pretrained checkpoint to load into "
                         "model.backbone. 'auto' = use cfg.weight; 'none' = "
                         "skip loading (random init Sonata)."))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------
# Pretrained-weight loading into model.backbone
# ----------------------------------------------------------------------

def _strip_ddp_prefix(state_dict):
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _load_backbone_weights(model, checkpoint_path):
    """Load a Sonata pretraining checkpoint into model.backbone.

    The checkpoint typically has keys for the full Sonata wrapper (teacher,
    student, head, etc.) without a 'backbone.' prefix. Our wrapper places
    the Sonata under `model.backbone`, so we need to (a) strip DDP, (b)
    auto-detect whether the checkpoint keys are flat (sonata-relative) or
    already prefixed (model-relative), and (c) call load_state_dict on
    model.backbone with strict=False.
    """
    print(f"[assembly-test] loading backbone weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = _strip_ddp_prefix(sd)

    # Detect whether the keys are sonata-relative (e.g. "teacher.backbone...")
    # or model-relative (e.g. "backbone.teacher.backbone..."). We always load
    # into model.backbone (a Sonata-v1m1 instance).
    sample_key = next(iter(sd.keys()))
    if sample_key.startswith("backbone."):
        # model-relative checkpoint — strip the "backbone." prefix
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}
        print("[assembly-test]   detected model-relative keys; "
              "stripped 'backbone.' prefix")
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    print(f"[assembly-test]   loaded; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[assembly-test]   first missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"[assembly-test]   first unexpected: {unexpected[:3]}")
    n_loaded = sum(p.numel() for n, p in model.backbone.named_parameters()
                   if n not in missing)
    n_total = sum(p.numel() for p in model.backbone.parameters())
    print(f"[assembly-test]   backbone params loaded: "
          f"{n_loaded:,} / {n_total:,} "
          f"({100.0 * n_loaded / max(1, n_total):.1f}%)")
    return missing, unexpected


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)

    # ---- Build dataset ------------------------------------------------
    dataset_cfg = getattr(cfg.data, args.split).copy()
    if args.data_list is not None:
        dataset_cfg["data_list_file"] = os.path.abspath(args.data_list)
    dataset_cfg["transform"] = None
    dataset = DATASETS.build(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — check data_list_file in config")
    print(f"[assembly-test] dataset built ({len(dataset)} events)")

    # ---- Build model --------------------------------------------------
    model = build_model(cfg.model)
    if args.mock_backbone:
        in_ch = cfg.model.backbone.backbone.in_channels
        out_ch = cfg.model.backbone_out_channels
        _swap_in_mock(model, in_channels=in_ch, out_channels=out_ch)
    else:
        # Resolve --load-weights argument
        weights = args.load_weights
        if weights == "auto":
            weights = getattr(cfg, "weight", None)
            if weights is None:
                print("[assembly-test] --load-weights=auto but cfg.weight "
                      "is not set; running with RANDOM-INIT backbone")
        if weights and weights != "none":
            _load_backbone_weights(model, weights)
        else:
            print("[assembly-test] running with RANDOM-INIT backbone "
                  "(--load-weights=none)")
    print(f"[assembly-test] model built; params = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    device = torch.device(args.device)
    model = model.to(device)
    model.train()

    # ---- Build a small batch ------------------------------------------
    n = min(args.num_events, len(dataset))
    samples = [dataset[i] for i in range(n)]
    batch = shower_clustering_collate(samples)
    # Move tensors to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    # The lists-of-tensors fields contain CPU tensors that need moving too
    for k in ("fragment_indices_per_event",
              "fragment_trackid_per_event",
              "fragment_pid_per_event",
              "fragment_type_per_event"):
        batch[k] = [
            ([t.to(device) for t in v] if isinstance(v, list)
             else v.to(device))
            for v in batch[k]
        ]
    # gt_instances are dicts of np arrays — move per-instance arrays
    new_gt = []
    for ev_gts in batch["gt_instances_per_event"]:
        ev_out = []
        for g in ev_gts:
            g2 = dict(g)
            if isinstance(g2.get("truth_indices"), np.ndarray):
                g2["truth_indices"] = torch.as_tensor(
                    g2["truth_indices"], dtype=torch.long, device=device,
                )
            elif isinstance(g2.get("truth_indices"), torch.Tensor):
                g2["truth_indices"] = g2["truth_indices"].to(device)
            ev_out.append(g2)
        new_gt.append(ev_out)
    batch["gt_instances_per_event"] = new_gt

    print(f"[assembly-test] batch built: {n} event(s)")
    print(f"  total spacepoints   : {batch['n_spacepoints'].sum().item():,}")
    print(f"  total voxels        : {batch['n_voxels'].sum().item():,}")
    print(f"  total fragments     : {batch['n_fragments'].sum().item()}")
    print(f"  total gt_instances  : {batch['n_gt_instances'].sum().item()}")

    # ---- Forward + backward ------------------------------------------
    out = model(batch)
    loss = out["loss"]
    print(f"[assembly-test] forward OK; loss = {loss.item():.4f}")

    if loss.requires_grad:
        loss.backward()
        # Count params with non-zero gradient (sanity check)
        nz = 0
        zero = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None or p.grad.abs().sum().item() == 0.0:
                zero += 1
            else:
                nz += 1
        print(f"[assembly-test] backward OK; {nz} params w/ grad, "
              f"{zero} without")
        assert zero == 0, f"{zero} trainable params got no gradient"
    else:
        print("[assembly-test] loss did not require grad (frozen-only path)")

    print("\n[assembly-test] PASS")


if __name__ == "__main__":
    main()
