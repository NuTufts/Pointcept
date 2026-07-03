# LoRA Finetune Configs

Project studying the performance of LoRA finetune training on top of frozen Sonata
backbones (deghosting and semantic-segmentation tasks).

## Current

| Config | Purpose |
|---|---|
| `lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py` | **Production Stage-1 deghoster**: LoRA finetune of the ghost-aware `pretrain-sonata-v7-extbnb-larmatch` backbone. A copy also lives in `../larformer/stage1_deghost/` for stage-chain completeness. |

## archive/

- `lorafinetune-sonata-v1m1-lartpc-v5-deghost.py`, `-v5-seg.py` — v5-backbone
  generation (deghost and semseg tasks).
- `lorafinetune-sonata-v1m1-lartpc-v6-deghost.py` — v6 sim-only-backbone deghoster,
  superseded by the extbnb-larmatch production config.
- `lorafinetune-sonata-v1m1-lartpc-v6-deghost_overfit.py` — overfit sanity check.
- `lorafinetune-sonata-v1m1-lartpc-v6-seg.py` — semantic-segmentation LoRA study.
