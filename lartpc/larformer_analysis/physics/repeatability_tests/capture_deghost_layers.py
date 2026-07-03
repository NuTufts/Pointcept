"""Tier-B: localize WHERE in the deghoster the membership/cross-GPU divergence
first appears, via forward hooks (NO model edits).

Hooks each deghoster backbone stage (embedding, every enc/dec stage) and the
deghost head, and records an ORDER-INVARIANT fingerprint of each stage's output
features per event:
  - ssq   : sum of squares in float64 (accumulation noise ~1e-15, so this tracks
            feature VALUES, not summation order -> a reordering does NOT look like
            divergence),
  - maxabs: max |feature| (exactly order-invariant),
  - n     : number of points.

Run on two lists (e.g. probe-alone vs probe+padding) and diff the fingerprints
stage-by-stage: the first stage whose ssq differs by >> the fp floor is where the
divergence enters. That sets the minimal FP64 scope.

  python capture_deghost_layers.py --config <cascade.py> --weights <ckpt> \
      --input-list <list.txt> --out-dir <dir> [--max-events N]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # pointcept repo root
sys.path.insert(0, _REPO)                          # for pointcept.*
sys.path.insert(0, os.path.join(_REPO, "tools", "larformer"))   # for run_larformer_stage3_inference
from run_larformer_stage3_inference import (   # noqa: E402
    set_deterministic, _move_batch, _resolve_cascaded_slicer)


def _slicer_stage_modules(cs):
    """(ordered) hook list for the SLICER: backbone (Sonata PTv3) -> tokenizer
    builders -> token_refiner (CrossLevelAttn) -> decoder layers -> heads. Finds
    where the (post-deghost, identical-input) membership divergence enters the
    slicer's per-SP query assignment."""
    sl = getattr(cs, "slicer", cs)
    mods = []
    bb = sl.backbone.teacher.backbone
    mods.append(("bb.embedding", bb.embedding))
    if hasattr(bb, "enc"):
        for n, m in bb.enc.named_children():
            mods.append((f"bb.enc.{n}", m))
    if hasattr(bb, "dec"):
        for n, m in bb.dec.named_children():
            mods.append((f"bb.dec.{n}", m))
    if hasattr(sl, "tokenizer") and hasattr(sl.tokenizer, "builders"):
        for n, m in sl.tokenizer.builders.named_children():
            mods.append((f"tok.{n}", m))
    if hasattr(sl, "token_refiner"):
        mods.append(("token_refiner", sl.token_refiner))
    if hasattr(sl, "query_selector"):
        mods.append(("query_selector", sl.query_selector))
    if hasattr(sl, "decoder"):
        dec = sl.decoder
        if hasattr(dec, "init_heads"):
            mods.append(("dec.init_heads", dec.init_heads))
        if hasattr(dec, "layers"):
            for n, m in dec.layers.named_children():
                mods.append((f"dec.layer{n}", m))
    return mods


def _stage_modules(deghoster, fine_enc0=True):
    """(ordered) list of (name, module) to hook: embedding, enc stages, dec
    stages, deghost head — in rough forward-execution order. When fine_enc0,
    also hook the internals of enc0's first block (the spconv CPE vs attention
    vs MLP) to localize the divergence WITHIN the first encoder stage."""
    bb = deghoster.backbone.teacher.backbone   # PTv3
    mods = [("embedding", bb.embedding)]
    if hasattr(bb, "enc"):
        for n, m in bb.enc.named_children():
            mods.append((f"enc.{n}", m))
            if fine_enc0 and n == "enc0":
                # pooling/down (if present) + block0 sub-ops in execution order
                for sub in ("down",):
                    if hasattr(m, sub):
                        mods.append((f"enc0.{sub}", getattr(m, sub)))
                if hasattr(m, "block0"):
                    b = m.block0
                    if hasattr(b, "cpe"):
                        # the SubMConv3d (spconv) specifically, then the full CPE
                        for cn, cm in b.cpe.named_children():
                            if type(cm).__name__ == "SubMConv3d":
                                mods.append(("enc0.block0.cpe.spconv", cm))
                        mods.append(("enc0.block0.cpe", b.cpe))
                    if hasattr(b, "attn"):
                        mods.append(("enc0.block0.attn", b.attn))
                    if hasattr(b, "mlp"):
                        mods.append(("enc0.block0.mlp", b.mlp))
    if hasattr(bb, "dec"):
        for n, m in bb.dec.named_children():
            mods.append((f"dec.{n}", m))
    head = getattr(deghoster, "deghost_head", None)
    if head is not None:
        mods.append(("deghost_head", head))
    return mods


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--input-list", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--deghost-fp64", action="store_true")
    ap.add_argument("--target", default="deghoster", choices=["deghoster", "slicer"],
                    help="which sub-model's stages to hook")
    args = ap.parse_args()
    if not args.no_deterministic:
        set_deterministic()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models   # noqa: F401
    import pointcept.datasets  # noqa: F401

    cfg = Config.fromfile(args.config)
    ds_cfg = dict(cfg.data[args.split]); ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset) if args.max_events is None else min(args.max_events, len(dataset))

    sd = torch.load(args.weights, map_location="cpu"); sd = sd.get("state_dict", sd)
    model = build_model(cfg.model); model.load_state_dict(sd, strict=False)
    model = model.to(args.device).eval()
    # Disable serialization order-shuffle model-wide (matches production
    # run_full_cascade_mode). Needed so the SLICER Tier-B isolates the SECOND
    # membership source, not the already-understood deghoster shuffle (the
    # SerializedPooling stages default shuffle_orders=True regardless of config).
    _ns = 0
    for _m in model.modules():
        if getattr(_m, "shuffle_orders", False):
            _m.shuffle_orders = False; _ns += 1
    print(f"[layers] disabled serialization order-shuffle on {_ns} modules")

    cs = _resolve_cascaded_slicer(model)
    if args.deghost_fp64:
        cs.enable_fp64_deghoster()

    stages = (_slicer_stage_modules(cs) if args.target == "slicer"
              else _stage_modules(cs.deghoster))
    print(f"[layers] hooking {len(stages)} stages: {[n for n,_ in stages]}")
    store = {}

    def _collect(out, acc):
        """Aggregate (ssq, n_elem) over ALL tensors reachable in `out` (handles
        Point.feat, SparseConvTensor.features, dict/list/tuple). Order-invariant."""
        if torch.is_tensor(out):
            if out.is_floating_point():
                acc[0] += float(out.double().pow(2).sum().item())
                acc[1] += out.numel()
                acc[2] = max(acc[2], float(out.abs().max().item()) if out.numel() else 0.0)
            return
        for attr in ("feat", "features", "tokens", "coords"):
            v = getattr(out, attr, None)
            if torch.is_tensor(v):
                _collect(v, acc)
        if isinstance(out, dict):
            for v in out.values():
                _collect(v, acc)
        elif isinstance(out, (tuple, list)):
            for v in out:
                _collect(v, acc)

    def mk(name):
        def hook(mod, inp, out):
            acc = [0.0, 0, 0.0]   # ssq, n_elem, maxabs
            _collect(out, acc)
            if acc[1] > 0:
                store[name] = (acc[0], acc[2], int(acc[1]))
        return hook

    for name, mod in stages:
        mod.register_forward_hook(mk(name))

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[layers] {n_events} events -> {args.out_dir}  "
          f"GPU={torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu'}")
    for i in range(n_events):
        sample = dataset[i]
        stem = os.path.splitext(sample.get("name", f"event{i:06d}.h5"))[0]
        out_path = os.path.join(args.out_dir, f"layers_{stem}.npz")
        store.clear()
        batched = _move_batch(larformer_collate([sample]), args.device)
        try:
            with torch.no_grad():
                cs(batched)            # runs deghoster (+slicer); hooks fire in the deghoster
        except RuntimeError as ex:
            if "out of memory" not in str(ex).lower():
                raise
            torch.cuda.empty_cache(); print(f"[{i+1}/{n_events}] OOM — skipped"); continue
        rec = {"run": int(sample.get("run", -1)), "subrun": int(sample.get("subrun", -1)),
               "event": int(sample.get("event", -1)), "name": str(sample.get("name", ""))}
        for name, (ssq, mx, n) in store.items():
            rec[f"{name}__ssq"] = np.float64(ssq)
            rec[f"{name}__maxabs"] = np.float64(mx)
            rec[f"{name}__n"] = np.int64(n)
        np.savez(out_path, **rec)
        if (i + 1) % 10 == 0 or i == n_events - 1:
            print(f"[{i+1}/{n_events}] {stem}")
    print(f"[layers] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
