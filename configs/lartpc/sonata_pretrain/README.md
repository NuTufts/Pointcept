# Sonata Pre-training Configs

Self-supervised pre-training of the PTv3 backbone via the Sonata method.

## Current production

| Config | Backbone role |
|---|---|
| `pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py` | **Sim-only backbone** used by the LArFormer slicer / particle / keypoint stages. Fixes a data-presentation mistake in `archive/pretrain-sonata-v1m1-lartpc-v6.py`. |
| `pretrain-sonata-v7-extbnb-larmatch.py` | **Ghost-aware backbone** trained on a dataset that includes ghost points and real (extbnb) data; backbone for the Stage-1 deghoster (see `../lora_finetune/`). |

## probes/

Linear-probe evaluations of backbone quality (frozen features + linear classifier):
`linearprobe-sonata-v1m1-lartpc.py` and the v2/v5 no-ghost variants.

## archive/

Earlier pre-training generations, kept for provenance:

- `pretrain-sonata-v1m1-lartpc.py`, `-restart.py`, `-v2` … `-v5` — successive dataset/
  recipe generations leading to v6.
- `pretrain-sonata-v1m1-lartpc-v6.py` — superseded by the production
  `logspace-resume` config (data-presentation bug).
- `pretrain-sonata-v1m1-lartpc-v6-mup.py`, `-mup-proxy.py` — μP (maximal update
  parameterization) study; inherit from `./pretrain-sonata-v1m1-lartpc-v6.py`, so all
  three must stay in this folder together. See `docs/reference/muP_for_Sonata.md`.
- `pretrain-sonata-v6-extbnb.py`, `pretrain-sonata-v8-extbnb-mc-combined-larmatch.py`
  — extbnb-dataset experiments before/after the production v7.
