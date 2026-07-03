# Documentation

- [`LArFormer.md`](LArFormer.md) — **the project hub**: cascade overview,
  documentation map, code map. Start here.
- [`Reorganization_Plan.md`](Reorganization_Plan.md) — repo layout plan and the
  record of the 2026-07 reorganization.
- [`reference/`](reference/) — maintained specs and guides (data formats, dataset
  guides, per-stage LArFormer specs, shower-origin/clustering designs, Sonata
  losses and muP, reproducibility, cluster guide). Each begins with a
  `Status: REFERENCE` line.
- [`devlog/`](devlog/) — dated development records (training-stability diagnosis,
  sweep campaigns, resolved debugging trails). Historical: they describe the state
  of the code when written, not necessarily now. Each begins with a
  `Status: WORKLOG/RESOLVED` line.

Component-local docs live next to their code: `configs/lartpc/README.md`
(production config chain), `lartpc/*/README.md`, and
`lartpc/larformer_reco/specs/`.
