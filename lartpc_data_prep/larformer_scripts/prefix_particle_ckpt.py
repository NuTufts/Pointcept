"""Re-prefix a standalone Stage-3 LArFormer checkpoint to the
`particle_segmenter.*` namespace so it loads into a CascadedParticleSegmenter
via run_larformer_stage3_inference.py --input-mode full-cascade --weights.

The Stage-3 particle segmenter is trained as a standalone `LArFormer` (cached
config), so its state_dict keys are un-prefixed (`decoder.*`, `backbone.*`,
`token_refiner.*`, ...). The full-cascade wrapper expects them under
`particle_segmenter.*`. This script rewrites the keys once (idempotent) and
writes a sibling checkpoint.

Usage:
    python prefix_particle_ckpt.py --in model_iter_98652.pth \
        --out model_iter_98652.particle_segmenter.pth
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="particle_segmenter.")
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    out = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        if not kk.startswith(args.prefix):
            kk = args.prefix + kk
        out[kk] = v
    torch.save({"state_dict": out}, args.out)
    print(f"wrote {args.out}  ({len(out)} tensors, prefix={args.prefix!r})")


if __name__ == "__main__":
    main()
