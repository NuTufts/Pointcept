# Flash Matching

Shared flash-matching library — used by the slicer evaluation
(`lartpc/larformer_analysis/slicer_eval/`), the single-photon physics studies,
Stage-3 inference (`tools/larformer/run_larformer_stage3_inference.py`), and the
planned unified reco + flash-matching inference (see `docs/Reorganization_Plan.md` §1b).

- `flash_predict.py` — slice → predicted PE per PMT. Thin wrapper around
  `pointcept.models.event_slicer.photonlib.PhotonLibLookup`; charge selection
  (Y-plane with U/V fallback). Import as `from lartpc.flashmatch import flash_predict`.
- `flash_chi2.py` — Neyman χ² with systematic floor + out-of-bounds rejection.
- `prepare_flashinfo_h5.py` — builds the flash-info H5 (step 5 of the
  training-data pipeline; needs the larlite/ubdl env).
- `build_photonlib_cache.py` — converts the MicroBooNE photon-library ROOT file
  to the npz cache (needs ubdl env).
- `data/photonlib_v6_70kV.npz` — the production photon-library cache (55 MB,
  **not in git**; regenerate with `build_photonlib_cache.py` or copy from
  another checkout). `flash_predict.PHOTONLIB_DEFAULT_CACHE` points here.

Related truth categorization (`categorize.py`) stays in
`larformer_analysis/slicer_eval/lib/` — it is analysis-side, not flash machinery.
