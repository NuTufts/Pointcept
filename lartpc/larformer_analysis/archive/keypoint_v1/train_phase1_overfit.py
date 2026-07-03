"""Phase 1 milestone: overfit the dense keypoint-score head on a few dev-cache
events and drive the loss toward zero.

Standalone training loop (no Pointcept trainer) so the keypoint data path +
model can be validated quickly. Builds a LArFormerKeypoint (Sonata backbone +
KeypointScoreHead), loads N augmented cache events, and runs Adam.

Note on the metric: most `kpscores` are 0 (points far from any keypoint), so
plain MSE is trivially minimized by predicting zeros. We therefore (a) weight
positives via --pos-weight and (b) report `mse_pos` (MSE on points within the
Gaussian peaks) as the meaningful "are the peaks learned" signal.

    python lartpc/larformer_analysis/archive/keypoint_v1/train_phase1_overfit.py \
        --cache-root exp/cache_stage12_devdata/train \
        --n-events 3 --steps 400 --pos-weight 50 --device cuda
        [--unfreeze-backbone]
"""

import argparse
import sys

import torch

from pointcept.datasets.larformer_stage12_cache import (
    LArFormerStage12CacheDataset,
)
from pointcept.datasets.larformer import larformer_collate
from pointcept.models.builder import build_model

LOCAL_SONATA_PRETRAIN = (
    "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/"
    "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth"
)
FLASH_BACKEND = "flash_attn"


def build_backbone_cfg():
    return dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 9, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend=FLASH_BACKEND,
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=1088, head_hidden_channels=2048,
        head_embed_channels=256, head_num_prototypes=4096,
        num_global_view=2, num_local_view=6,
        up_cast_level=4,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="exp/cache_stage12_devdata/train")
    ap.add_argument("--n-events", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=50.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--weight", default=LOCAL_SONATA_PRETRAIN)
    ap.add_argument("--unfreeze-backbone", action="store_true")
    ap.add_argument("--offset-head", action="store_true",
                    help="Also train the KeypointOffsetHead (vote regression).")
    ap.add_argument("--weight-offset", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    ds = LArFormerStage12CacheDataset(
        split="train", data_root=args.cache_root,
        source_set_filter="union", emit_keypoints=True,
        keypoint_sigma_cm=3.0, min_spacepoints=1,
    )
    n = min(args.n_events, len(ds))
    samples = [ds[i] for i in range(n)]
    batch = larformer_collate(samples)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    n_sp = int(batch["offset"][-1].item())
    pos = (batch["kpscores"] > 0.05)
    print(f"[overfit] {n} events, {n_sp} SPs, "
          f"frac positive kpscore elems = {pos.float().mean().item():.4f}")

    model = build_model(dict(
        type="LArFormerKeypoint",
        backbone=build_backbone_cfg(),
        backbone_out_channels=1232,
        n_keypoint_types=6,
        freeze_backbone=(not args.unfreeze_backbone),
        backbone_weight=args.weight,
        head_hidden_dim=args.hidden,
        loss_kind="mse",
        pos_weight=args.pos_weight,
        pos_threshold=0.05,
        enable_offset_head=args.offset_head,
        weight_offset=args.weight_offset,
        offset_supervision_threshold=0.05,
        coord_scale=179.55,
    )).to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in trainable)
    print(f"[overfit] trainable params: {n_par:,} "
          f"(backbone {'TRAINABLE' if args.unfreeze_backbone else 'frozen'})")
    opt = torch.optim.Adam(trainable, lr=args.lr)

    first = None
    for step in range(args.steps):
        opt.zero_grad()
        out = model(batch)
        loss = out["loss"]
        loss.backward()
        opt.step()
        if first is None:
            first = float(out["loss_kp"])
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            msg = (f"  step {step:4d}  loss={float(loss):.5f}  "
                   f"kp={float(out['loss_kp']):.5f}  "
                   f"mse_pos={float(out['loss_kp_mse_pos']):.6f}")
            if "loss_off" in out:
                msg += (f"  off={float(out['loss_off']):.6f}  "
                        f"off_err_cm={float(out['loss_off_err_cm']):.3f}")
            print(msg)

    last_kp = float(out["loss_kp"])
    last_pos = float(out["loss_kp_mse_pos"])
    if "loss_off_err_cm" in out:
        print(f"[overfit] final offset vote error = "
              f"{float(out['loss_off_err_cm']):.3f} cm "
              f"(over {int(float(out['loss_off_n_sup']))} supervised SP-types)")
    print(f"\n[overfit] kp loss {first:.5f} -> {last_kp:.5f}  "
          f"(reduced {100*(1-last_kp/max(first,1e-9)):.1f}%)")
    print(f"[overfit] final mse_pos (peak fit) = {last_pos:.6f}")
    ok = last_kp < 0.5 * first and last_pos < 0.05
    print(f"[overfit] {'PASS' if ok else 'NEEDS-WORK'}: "
          f"loss decreasing and peaks fitting"
          + ("" if ok else " — consider --unfreeze-backbone or more steps"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
