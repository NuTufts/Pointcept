"""Phase 2 (Option A) integration test: overfit the query keypoint head on the
Stage-3 LArFormer particle segmenter, loaded from a trained checkpoint.

Builds the particle segmenter from the cached config, flips
`enable_keypoint_head=True` (adds end/uncertainty heads + the keypoint loss
terms, reusing the existing query→GT Hungarian match), loads the trained
Stage-3 checkpoint (strict=False → keypoint heads stay random-init), and
overfits a few dev-cache events with the backbone frozen and the PT-v3 decoder
+ keypoint heads trainable.

Acceptance: the matched-query start/end position errors (loss_kpq_start_err_cm
/ loss_kpq_end_err_cm) decrease — i.e. the keypoint head learns particle-
associated keypoints on top of a real Stage 3.

    python lartpc_data_prep/larformer_keypoint/train_phase2_overfit.py \
        --n-events 3 --steps 200 --device cuda
"""

import argparse
import copy
import sys

import torch

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.larformer import larformer_collate

CONFIG = "configs/lartpc/larformer-particle-v1-cached-ptv3crosslevel.py"
CKPT = ("exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4_"
        "bugfixed/epoch_6.pth")
TRAIN_CACHE = "exp/cache_stage12_devdata/train"


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The keypoint heads are the expected "missing" (newly added) keys.
    kp_missing = [m for m in missing if any(
        t in m for t in ("end_head", "logvar", "end_exist"))]
    print(f"  loaded {path}: missing={len(missing)} "
          f"(keypoint-head missing={len(kp_missing)}) unexpected={len(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--refine", action="store_true",
                    help="Enable the Phase-3 (2B) refinement decoder.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cfg = Config.fromfile(CONFIG)

    # ---- enable the keypoint head (Option A) ----
    model_cfg = copy.deepcopy(cfg.model)
    assert model_cfg["type"] == "LArFormer", model_cfg["type"]
    model_cfg["enable_keypoint_head"] = True
    model_cfg.setdefault("loss_kwargs", {})
    model_cfg["loss_kwargs"].update(dict(
        weight_kp_start=1.0, weight_kp_end=1.0, weight_kp_end_exist=0.5,
        kp_reg_kind="betanll", kp_beta=0.5,
        coord_scale=float(cfg.get("coord_scale", 179.55)),
    ))
    if args.refine:
        model_cfg["keypoint_refine"] = dict(
            num_layers=2, num_heads=4, mlp_ratio=4.0, mask_threshold=0.0)
    # Don't auto-load a backbone weight here (we load the full ckpt below).
    model_cfg.pop("backbone_weight", None)

    # ---- dataset with keypoints ----
    ds_cfg = copy.deepcopy(cfg.data["train"])
    ds_cfg["data_root"] = TRAIN_CACHE
    ds_cfg["emit_keypoints"] = True
    ds_cfg.pop("data_list_file", None)
    dataset = build_dataset(ds_cfg)
    n = min(args.n_events, len(dataset))
    samples = [dataset[i] for i in range(n)]
    batch = larformer_collate(samples)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    n_gt = sum(len(s["gt_instances"]) for s in samples)
    n_end = sum(int(g.get("has_end", False))
                for s in samples for g in s["gt_instances"])
    print(f"[p2] {n} events, {n_gt} GT particles ({n_end} with track-end)")

    print("[p2] building model + loading checkpoint ...")
    model = build_model(model_cfg).to(device)
    load_ckpt(model, CKPT)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[p2] trainable params: {sum(p.numel() for p in trainable):,} "
          f"(backbone frozen, PT-v3 decoder + heads trainable)")
    opt = torch.optim.Adam(trainable, lr=args.lr)

    def _get(d, k):
        v = d.get(k)
        return float(v) if v is not None else float("nan")

    first = None
    for step in range(args.steps):
        opt.zero_grad()
        out = model(batch)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if first is None:
            first = (_get(out, "loss_kpq_start_err_cm"),
                     _get(out, "loss_kpq_end_err_cm"))
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={float(loss):.4f}  "
                  f"kpq_start={_get(out,'loss_kpq_start'):.4f}  "
                  f"kpq_end={_get(out,'loss_kpq_end'):.4f}  "
                  f"start_err={_get(out,'loss_kpq_start_err_cm'):.2f}cm  "
                  f"end_err={_get(out,'loss_kpq_end_err_cm'):.2f}cm")

    s0, e0 = first
    s1 = _get(out, "loss_kpq_start_err_cm")
    e1 = _get(out, "loss_kpq_end_err_cm")
    print(f"\n[p2] start_err {s0:.2f} -> {s1:.2f} cm")
    print(f"[p2] end_err   {e0:.2f} -> {e1:.2f} cm")
    ok = torch.isfinite(loss) and s1 < s0 and s1 < 30.0
    print(f"[p2] {'PASS' if ok else 'NEEDS-WORK'}: keypoint head learns "
          f"start/end on a real Stage 3")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
