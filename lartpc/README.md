# lartpc/ — LArTPC data prep, reconstruction, and analysis

Top-level home for the LArTPC-specific work built around the Pointcept fork,
separated by concern (see `docs/Reorganization_Plan.md` §3):

```
lartpc/
├── data_prep/            # HDF5 production: training data + official MicroBooNE sim/data
├── larformer_analysis/   # model/reco performance analysis (slicer, particle, physics)
└── (planned)
    ├── larformer_reco/   # nu-interaction reconstruction (from lartpc_data_prep/larformer_keypoint_v2)
    ├── flashmatch/       # shared flash-matching library (from larformer_analysis/slicer_eval/lib)
    └── viz/              # shared event-display utilities
```

Everything here runs inside the pointcept apptainer container
(`run_in_tufts_pointcept_container.sh`) with `PYTHONPATH=<repo root>`, except the
LANTERN data-prep steps which use the lantern container (see
`data_prep/training_data/`).

Model code lives in `pointcept/models/LArFormer/`; training configs in
`configs/lartpc/` (see its README for the production configuration chain).
