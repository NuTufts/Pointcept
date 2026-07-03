# Semantic Segmentation Configs (historical project)

Early supervised semantic-segmentation work on LArTPC spacepoints, predating the
LArFormer cascade.

- `semseg-pt-v3m1-0-base.py` — PTv3 supervised-from-scratch baseline (this is the
  config referenced by the README/CLAUDE.md training examples).
- `semseg-pt-v3m1-1-novoxel.py` — no-voxelization variant.
- `archive/` — Sonata-finetune generations (`semseg-sonata-v1m1-lartpc-finetune.py`
  and the v2–v5 decoder-finetune variants) evaluating pre-trained backbones via
  full finetuning. Analysis lived in `lartpc_data_prep/semseg_analysis/`.
