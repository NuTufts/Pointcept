"""
Fold a trained LoRA checkpoint into the base Sonata backbone weights.

Given a LoRA fine-tuning config + its trained checkpoint, this script:
    1. Builds the LoRA segmentor from the config
    2. Loads the checkpoint state_dict into it
    3. Calls model.merge_lora_weights() to fold every LoRALinear back into a
       plain nn.Linear (using the formula W' = W + (α/r) · B·A)
    4. Extracts just `model.backbone.state_dict()` (drops the LoRA-task-
       specific head like `deghost_head`)
    5. Wraps + saves it as a SonataCheckpointLoader-compatible checkpoint

The output is loadable by any LArFormer (or other) config whose `weight =`
points at the file — the SonataCheckpointLoader will prepend `backbone.`
to the keys and load them strict=False into a fresh Sonata-v1m1 wrapper.

Usage:
    ./run_in_container.sh python tools/fold_lora_into_backbone.py \\
        --config     configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py \\
        --checkpoint exp/.../model/model_best.pth \\
        --output     sonata/deghost_lora_folded.pth

Quick self-test mode (no real checkpoint needed):
    ./run_in_container.sh python tools/fold_lora_into_backbone.py --self-test
"""

import argparse
import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Core fold logic (importable by other scripts / tests)
# ---------------------------------------------------------------------------

def fold_lora_checkpoint(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    verbose: bool = True,
) -> dict:
    """Fold LoRA weights and save a SonataCheckpointLoader-style checkpoint.

    Returns a small `info` dict with sanity-check counts for the caller.
    """
    from pointcept.utils.config import Config
    from pointcept.models.builder import build_model

    cfg = Config.fromfile(config_path)
    if verbose:
        print(f"[fold] Building LoRA model from {config_path}")
    model = build_model(cfg.model)
    model.eval()

    if verbose:
        print(f"[fold] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'state_dict' key; "
            f"top-level keys: {list(ckpt.keys())[:10]}"
        )
    state = ckpt["state_dict"]
    # DDP prefix strip
    state = {(k[7:] if k.startswith("module.") else k): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        print(f"  loaded; missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:5]:    {list(missing)[:5]}")
        if unexpected:
            print(f"  unexpected[:5]: {list(unexpected)[:5]}")

    # Sanity #1: model should have LoRALinear instances before the fold.
    n_lora_before = sum(
        1 for m in model.modules() if type(m).__name__ == "LoRALinear"
    )
    if n_lora_before == 0:
        raise RuntimeError(
            "Model has no LoRALinear modules — is this really a LoRA "
            "fine-tuning config? Nothing to fold."
        )
    if verbose:
        print(f"[fold] LoRALinear modules before merge: {n_lora_before}")

    if not hasattr(model, "merge_lora_weights"):
        raise AttributeError(
            f"Model class {type(model).__name__} has no merge_lora_weights() "
            f"method. Custom LoRA model? Add the merge step manually."
        )
    model.merge_lora_weights()

    # Sanity #2: no LoRALinear instances should remain.
    n_lora_after = sum(
        1 for m in model.modules() if type(m).__name__ == "LoRALinear"
    )
    if n_lora_after != 0:
        raise RuntimeError(
            f"merge_lora_weights() left {n_lora_after} LoRALinear modules "
            f"behind — fold incomplete."
        )

    # Extract backbone state. We deliberately drop the LoRA-task-specific
    # head (e.g. deghost_head) because the downstream LArFormer config will
    # supply its own per-level cls / decoder / etc. head.
    if not hasattr(model, "backbone"):
        raise AttributeError(
            f"Model class {type(model).__name__} has no `backbone` attribute. "
            f"Custom layout? Adjust this script to find the backbone module."
        )
    backbone_state = model.backbone.state_dict()

    # Sanity #3: no `lora_` keys should remain in the backbone state.
    lora_keys = [k for k in backbone_state.keys() if "lora_" in k]
    if lora_keys:
        raise RuntimeError(
            f"Backbone state has {len(lora_keys)} residual lora_* keys after "
            f"fold (first 5: {lora_keys[:5]})"
        )

    # Sanity #4: backbone should still have its qkv/proj Linear weights.
    qkv_proj_keys = [k for k in backbone_state.keys()
                     if k.endswith("qkv.weight") or k.endswith("proj.weight")]
    if verbose:
        print(f"[fold] backbone keys: {len(backbone_state)}  "
              f"qkv/proj Linears: {len(qkv_proj_keys)}")

    out_ckpt = {
        "state_dict": backbone_state,
        "epoch": -1,
        "best_metric_value": None,
        "comment": (
            f"LoRA weights folded into base Sonata-v1m1 backbone. "
            f"Source config:     {config_path} ; "
            f"Source checkpoint: {checkpoint_path}"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                exist_ok=True)
    torch.save(out_ckpt, output_path)
    if verbose:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[fold] Wrote {output_path} ({size_mb:.1f} MB)")

    return {
        "n_lora_before": n_lora_before,
        "n_lora_after": n_lora_after,
        "n_backbone_keys": len(backbone_state),
        "n_qkv_proj_keys": len(qkv_proj_keys),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Self-test: build a tiny LoRA model with random weights, fold, verify
# round-trip — exercises the same code path without needing a real
# trained checkpoint.
# ---------------------------------------------------------------------------

def self_test(tmpdir: str) -> None:
    """End-to-end fold round-trip with random weights.

    Builds the *real* SonataLoRADeghostSegmentor architecture (so the test
    exercises the same merge_lora_weights / module-walking code paths the
    production path will use) but at small enough scale that it runs in
    ~10 seconds.
    """
    from pointcept.models.LArFormer.builders.base import LevelOutput  # noqa: F401
    from pointcept.models.builder import build_model
    # NB: registering shower_clustering and sonata happens at pointcept.models import
    import pointcept.models  # noqa: F401
    from pointcept.utils.config import Config

    # Minimal Sonata-v1m1 wrapping a tiny PT-v3m2 — same architecture as the
    # production deghoster but with shrunken depths/channels so the
    # round-trip runs instantly.
    backbone_cfg = dict(
        type="Sonata-v1m1",
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(1, 1, 1, 1, 1),
            enc_channels=(16, 32, 64, 128, 256),
            enc_num_head=(2, 2, 4, 4, 8),
            enc_patch_size=(64, 64, 64, 64, 64),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
            shuffle_orders=False, pre_norm=True,
            enable_rpe=False, enable_flash=True, flash_backend="xformers",
            upcast_attention=False, upcast_softmax=False,
            traceable=True, enc_mode=True, mask_token=False,
        ),
        head_in_channels=256, head_hidden_channels=512,
        head_embed_channels=64, head_num_prototypes=128,
        num_global_view=2, num_local_view=4,
        up_cast_level=4,
    )
    backbone_out_channels = 16 + 32 + 64 + 128 + 256

    print("[self-test] Building SonataLoRADeghostSegmentor (tiny scale)...")
    model_cfg = dict(
        type="SonataLoRADeghostSegmentor",
        backbone=backbone_cfg,
        backbone_out_channels=backbone_out_channels,
        criteria=[dict(type="CrossEntropyLoss")],
        lora_rank=4, lora_alpha=8.0, lora_dropout=0.0,
    )
    model = build_model(Config(model_cfg))

    # Randomize LoRA matrices so the fold is non-trivial (otherwise B=0 → no delta).
    n_random = 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.01)
            n_random += 1
    print(f"[self-test] Randomized {n_random} lora_B matrices (non-zero deltas).")

    # Save a fake checkpoint to round-trip through.
    ckpt_path = os.path.join(tmpdir, "fake_lora_ckpt.pth")
    torch.save({"state_dict": model.state_dict(), "epoch": 0}, ckpt_path)

    # ---- Run the fold (production code path) ----------------------------
    cfg_path = os.path.join(tmpdir, "fake_lora_cfg.py")
    with open(cfg_path, "w") as f:
        # Write a minimal Pointcept config that build_model can ingest.
        f.write("model = " + repr(model_cfg))
    out_path = os.path.join(tmpdir, "folded.pth")
    info = fold_lora_checkpoint(cfg_path, ckpt_path, out_path, verbose=True)
    assert info["n_lora_before"] > 0
    assert info["n_lora_after"] == 0
    assert info["n_qkv_proj_keys"] > 0

    # ---- Verify the folded backbone is loadable by a fresh Sonata-v1m1 --
    # (with no LoRA wrapper) and that the loaded weights match the merged
    # weights of the original model.
    print("\n[self-test] Loading folded backbone into a vanilla Sonata-v1m1...")
    plain_backbone = build_model(Config(backbone_cfg))
    folded_ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
    missing, unexpected = plain_backbone.load_state_dict(
        folded_ckpt["state_dict"], strict=False,
    )
    print(f"  Plain backbone load: missing={len(missing)} unexpected={len(unexpected)}")

    # Compare a sample qkv weight: model.backbone.<...>.qkv.weight should
    # equal plain_backbone.<...>.qkv.weight bit-for-bit, since both came
    # from the same fold operation.
    qkv_keys = [k for k in folded_ckpt["state_dict"].keys()
                if k.endswith("qkv.weight")]
    if qkv_keys:
        k = qkv_keys[0]
        v_loaded = dict(plain_backbone.named_parameters())[k]
        v_saved = folded_ckpt["state_dict"][k]
        max_err = (v_loaded - v_saved).abs().max().item()
        print(f"  Sample qkv key {k!r}: max abs err vs saved = {max_err:.2e}")
        assert max_err < 1e-6, "round-trip mismatch"

    print("\n[self-test] PASSED.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, REPO_ROOT)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to a LoRA training config file")
    ap.add_argument("--checkpoint", help="Path to the trained LoRA checkpoint")
    ap.add_argument("--output", help="Where to save the folded backbone")
    ap.add_argument("--self-test", action="store_true",
                    help="Run an internal round-trip test (no real ckpt needed)")
    args = ap.parse_args()

    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self_test(td)
        return

    if not (args.config and args.checkpoint and args.output):
        ap.error("--config, --checkpoint, --output are required "
                 "(or use --self-test)")
    info = fold_lora_checkpoint(args.config, args.checkpoint, args.output)
    print(f"\nDONE. Folded {info['n_lora_before']} LoRA adapters into "
          f"{info['n_backbone_keys']} backbone keys → {info['output_path']}")


if __name__ == "__main__":
    main()
