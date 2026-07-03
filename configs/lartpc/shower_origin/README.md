# Shower Origin / Shower Clustering Configs (exploratory project)

Shower-origin prediction (3D origin point + inside/outside classification) and
Mask2Former-style shower clustering on Sonata features. Exploratory work that
informed the LArFormer design; not part of the production cascade.

All configs are in `archive/`:

- `shower-origin-sonata-v1m1-v3.py` and `-p1cmp075` / `-reco-fragments` variants —
  successive shower-origin trainings (see `docs/reference/shower_origin_spec.md`).
- `shower-cluster-sonata-v1.py`, `-h200.py` — shower-clustering trainings
  (see `docs/reference/shower_clustering_design.md`).

Related analysis: `lartpc/larformer_analysis/archive/shower_origin_reco_scripts/`.
