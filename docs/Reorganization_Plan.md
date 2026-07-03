# Repository Reorganization Plan

**Status:** PLAN — drafted 2026-07-03 on branch `nutufts_larformer_reorganization`.
**Scope:** configs, data prep, reconstruction post-processing, analysis, visualization, docs.
**Non-goals:** no changes to model code in `pointcept/models/LArFormer/` (already well-structured); no algorithmic rewrites — moves, import fixes, and de-duplication only.

---

## 1. Production configuration chain (record of truth)

These configs produced the current production checkpoints. This table is the reason
for the config reorg: it should be obvious from the directory layout alone.

| Stage | Production config | Notes |
|---|---|---|
| Sonata pre-training (sim-only backbone) | `pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py` | Fixes a data-presentation mistake in predecessor `pretrain-sonata-v1m1-lartpc-v6.py` |
| Sonata pre-training (ghost-aware backbone, real data) | `pretrain-sonata-v7-extbnb-larmatch.py` | Backbone used by the deghost stage; trained on a dataset that includes ghost points |
| LArFormer Stage 1 — deghosting | `lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py` | LoRA finetune on the v7 extbnb-larmatch backbone |
| LArFormer Stage 2 — event slicing | `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py` | |
| LArFormer Stage 3 — particle segmentation | `larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py` | Final decay-LR run, warm-started from a flat-LR run of `larformer-particle-v1-cached-ptv3crosslevel.py` |
| LArFormer Stage 4 — keypoints | `larformer-keypoint2-particle-predmask-cached-v1.py` | |
| Full-cascade inference (keypoint2) | `larformer-keypoint2-fullcascade.py` | Used by `tools/run_larformer_keypoint2_cascade_inference.py` |
| Full-cascade inference (stage 3, official data pipeline) | `larformer-particle-fullcascade-ptv3crosslevel.py` | Default in `larformer_scripts/run_stepB_cascade_wconfig.sh` |
| Stage-1+2 cache building | `larformer-particle-v1.py` | Referenced by `tools/build_stage12_cache_{event,shard}.py`, `benchmark_larformer_s3_cascade.py` |

## 1b. Target of the next work stage (context that shapes this reorg)

After the reorg, the next stage is a **unified inference pipeline that processes both
the validation data and the official MicroBooNE sim**, incorporating the nu-interaction
reco *and* flash-matching outputs. It can remain staged while under development, but the
target output is selection-performance understanding for different neutrino interaction
types, with **emphasis on single-photon events**. Two prior threads feed this:

- `lartpc_data_prep/larformer_physics/single_photon` — worked from particle-segmenter
  output only, exploring the flash-matching machinery.
- `lartpc_data_prep/larformer_keypoint_v2` — efficiency post nu-interaction reco, also
  exploring alternate event slices + eventual flash matching to select single-photon
  events efficiently.

Implications baked into this plan:

1. **Flash matching becomes a shared package**, not a private helper of the slicer
   eval. `larformer_analysis/lib/{flash_predict,flash_chi2}.py`, `prepare_flashinfo_h5.py`,
   and `build_photonlib_cache.py` are promoted to `lartpc/flashmatch/` so both
   `larformer_reco` and `larformer_analysis` (and the future unified inference driver)
   import the same implementation.
2. **The single-photon studies stay first-class** (under
   `lartpc/larformer_analysis/physics/single_photon/`), never archived — they are the
   forerunners of the next stage.
3. `lartpc/larformer_reco/` is packaged so a future unified inference entry point can
   import reco + flashmatch cleanly, and **both data-prep scopes (training/validation
   data and official sim) must feed the same inference entry point** — another reason
   the two pipelines in Section 3.1 stay parallel in structure.

## 2. Config directory reorganization (`configs/lartpc/`)

Organize by **project**, with per-stage subfolders under the larformer project.
Within every (sub)division, an `archive/` folder isolates past experiments so the
current production config is the only `.py` at that level. Every division gets a
`README.md` documenting what the current config is for and what each archived
config tested.

```
configs/lartpc/
├── README.md                          # production chain table (Section 1) + pointers
├── sonata_pretrain/
│   ├── README.md
│   ├── pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py     [PRODUCTION sim-only]
│   ├── pretrain-sonata-v7-extbnb-larmatch.py                 [PRODUCTION ghost-aware]
│   ├── probes/                        # backbone-quality evaluation configs
│   │   ├── linearprobe-sonata-v1m1-lartpc.py
│   │   ├── linearprobe-sonata-lartpc-v2-noghost.py
│   │   └── linearprobe-sonata-lartpc-v5-noghost.py
│   └── archive/
│       ├── pretrain-sonata-v1m1-lartpc.py / -restart.py / -v2 / -v3 / -v4 / -v5
│       ├── pretrain-sonata-v1m1-lartpc-v6.py                 # superseded (data-presentation bug)
│       ├── pretrain-sonata-v1m1-lartpc-v6-mup.py / -mup-proxy.py   # μP study; inherit v6 via ./ — keep together
│       ├── pretrain-sonata-v6-extbnb.py
│       └── pretrain-sonata-v8-extbnb-mc-combined-larmatch.py
├── lora_finetune/                     # project: LoRA finetune performance study
│   ├── README.md
│   ├── lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py   [CURRENT]
│   └── archive/
│       ├── lorafinetune-sonata-v1m1-lartpc-v5-deghost.py / -v5-seg.py
│       └── lorafinetune-sonata-v1m1-lartpc-v6-deghost.py / -v6-deghost_overfit.py / -v6-seg.py
├── larformer/
│   ├── README.md                      # stage chain diagram, checkpoint provenance
│   ├── stage1_deghost/
│   │   ├── lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py  [PRODUCTION — duplicate of lora_finetune copy, kept for completeness]
│   │   └── archive/ larformer-deghost-v0.py, larformer-deghost-v0-ptv3decoder.py
│   ├── stage2_slicer/
│   │   ├── larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py         [PRODUCTION]
│   │   └── archive/ larformer-slicer-v0.py, larformer-slicer-v1-cascaded.py,
│   │       -identity.py, -perlevel.py, -crosslevel.py, -ptv3decoder.py,
│   │       -ptv3hybrid_perlevel.py, -loradeghost.py
│   ├── stage3_particle/
│   │   ├── larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py   [PRODUCTION final]
│   │   ├── larformer-particle-v1-cached-ptv3crosslevel.py                [PRODUCTION warm-start]
│   │   ├── larformer-particle-fullcascade-ptv3crosslevel.py              [PRODUCTION inference — stepB default]
│   │   ├── larformer-particle-v1.py                                      [ACTIVE — cache building]
│   │   └── archive/ larformer-particle-v1-cached.py, larformer-particle-v1.1-cached-ptv3crosslevel.py
│   └── stage4_keypoint/
│       ├── larformer-keypoint2-particle-predmask-cached-v1.py            [PRODUCTION training]
│       ├── larformer-keypoint2-fullcascade.py                            [PRODUCTION inference]
│       └── archive/ larformer-keypoint-v1.py (phase 1), larformer-keypoint-query-v1.py (phase 2),
│           larformer-keypoint2-slice-v1.py, larformer-keypoint2-particle-v1.py
├── semseg/                            # earlier semantic-segmentation project (mostly historical)
│   ├── README.md
│   ├── semseg-pt-v3m1-0-base.py, semseg-pt-v3m1-1-novoxel.py
│   └── archive/ semseg-sonata-v1m1-lartpc-finetune.py, -v2-, -v3-decoder-, -v4-decoder-, -v5-decoder-finetune.py
└── shower_origin/                     # shower-origin / shower-clustering project (exploratory)
    ├── README.md
    └── archive/ shower-origin-sonata-v1m1-v3*.py (5), shower-cluster-sonata-v1*.py (2)
```

### Migration mechanics for configs

- `git mv` everything so history follows.
- **`_base_` depth fix:** 52 configs use `_base_ = ["../_base_/default_runtime.py"]`.
  One level deeper → `../../_base_/`, two levels (larformer stages) → `../../../_base_/`.
  The two μP configs inherit `./pretrain-sonata-v1m1-lartpc-v6.py` — move all three into
  the same archive folder so the relative path survives.
- **Hardcoded path sweep:** 78 files under `tools/`, `slurm_scripts/`, `scripts/`,
  `lartpc_data_prep/` reference `configs/lartpc/<name>.py` (mostly docstring examples
  and env-var defaults in submit scripts). Update with a scripted sed pass, then
  `grep -rn "configs/lartpc/[a-z]" --include='*.py' --include='*.sh' --include='*.md'`
  must return only paths that exist.
- `scripts/train.sh` interpolates `configs/${DATASET}/${CONFIG}.py`, so subfolder
  configs work as `-c larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel`.
- Resumed experiments are unaffected: `exp/<...>/config.py` copies are self-contained.
- **Verification:** load every moved config with `pointcept.utils.config.Config.fromfile`
  in a loop; all must parse and resolve `_base_`.

## 3. Top-level `lartpc/` split: data prep / reco / analysis

Create a top-level `lartpc/` area separating the three concerns currently mixed
inside `lartpc_data_prep/`:

```
lartpc/
├── data_prep/
├── larformer_reco/
├── larformer_analysis/
├── flashmatch/             # shared flash-matching library (Section 1b)
└── viz/                    # shared visualization/geometry utilities (Section 6)
```

### 3.1 `lartpc/data_prep/` — two scopes, two pipelines

**Scope A — training data (current pipeline, generation 2).** Newer sim files
generated with MicroBooNE code; richer truth for spacepoint ground-truth labels;
generation 2 added LANTERN truepoint scores for augmentation / curriculum focus.
Pipeline lives in `lartpc_data_prep/lantern_scripts/` (step1 lantern → step234
pointcept conversion → step5 flashinfo, with per-sample corsika configs).

**Scope B — official MicroBooNE sim/data.** Different truth handling; feeds the
exploratory physics analysis (`larformer_physics/single_photon`). Pipeline lives in
`lartpc_data_prep/larformer_scripts/` (stepA convert → stepB cascade inference,
documented in `LARFORMER_DATAPREP.md`).

```
lartpc/data_prep/
├── README.md                  # explains the two scopes and their truth differences
├── training_data/             # ← lantern_scripts/ (gen-2 LANTERN pipeline)
│   ├── lantern_configs/
│   ├── run_step1_lantern_wconfig.sh, run_step234_pointcept_wconfig.sh,
│   │   run_step5_flashinfo_pointcept_wconfig.sh, submit_lantern*.sh, ...
│   └── convert_larlite_to_pointcept_h5.py, write_completion_sentinels.py, ...
├── uboone_official/           # ← larformer_scripts/ (official sim/data pipeline)
│   ├── LARFORMER_DATAPREP.md
│   ├── run_stepA_convert_wconfig.sh, run_stepB_cascade_wconfig.sh
│   └── convert_dlmerged_to_larformer_h5.py, prefix_particle_ckpt.py
├── labels/                    # shared ground-truth label makers imported by pipelines
│   └── slice_labels.py, keypoint_labels.py, shower_fragment_merger.py,
│       prepare_flashinfo_h5.py, build_photonlib_cache.py
├── validation/                # data QA
│   └── validate_hdf5_files.py, audit_particle_labels.py, test_particle_labels.py,
│       dump_h5_keypoints.py
└── archive/gen1/              # generation-1 training prep — VERIFY then archive (likely cruft)
    └── root-level run_corsika_*.sh, submit_*corsika*.sh,
        process_dlmerged_to_hdf5_event_files.py (root + lantern_scripts duplicate — diff, keep newer),
        merge_reco_truth_showerorigin.py duplicate (diff, keep newer),
        inference_sparse_ssnet_uboone.py, recreate_ubspurn.py
```

Open item: confirm which root-level scripts belong to generation 1 before archiving
(candidates listed above are the user's "older scripts are now cruft I think").

### 3.2 `lartpc/larformer_reco/` — from `larformer_keypoint_v2/`

Promote the post-processing to a proper package (imports fixed, no `sys.path` hacks):

```
lartpc/larformer_reco/
├── README.md                  # 1-page: what it does, how to run reco + eval
├── specs/                     # keypoint_reco_spec.md, shower_reco_spec.md,
│   │                          # particle_momentum_spec.md, performance_eval_spec.md,
│   │                          # nu_vertex_eval.md, trajectory_fitting_brief.md
│   └── DEVLOG.md              # the current work-journal content of README.md
├── keypoint/                  # ← reco/: keypoint_reco.py, gaussian_fit.py, io.py, tests
├── trajfit/                   # ← trajfit_dev/ production modules
│   ├── nu_interaction.py, cluster_fit_stitch.py, io.py (← trajfit_io.py)
│   ├── kinematics/            # particle_momentum.py, range_momentum.py, calo.py, make_range2ke_npz.py
│   ├── shower/                # shower_trunk.py, shower_connect.py, shower_truth.py
│   └── data/                  # calo_calib.npz, range2ke_lar.npz (small lookup tables)
├── eval/                      # eval_reco_performance.py (metrics only), eval_keypoint2_inference.py,
│   │                          # eval_nu_vertex_reco.py
│   └── plot_eval.py           # all matplotlib extracted from the eval scripts
├── viz/                       # visualize_cascade_output.py split: loader/GT-matcher module + display CLI
├── studies/                   # spikes, each with header: date / question / conclusion
│   └── run_elpigraph.py, sweep_elpigraph.py, mcs_rdp.py, run_shower_dir.py,
│       scan_shower_attach.py, analyze_photon_recovery.py, analyze_photon_slices.py,
│       select_single_photon_events.py
├── scripts/                   # run_nu_reco.py, run_keypoint_reco.py CLI entry points
├── slurm/                     # submit_*.sh + common.sh (WORKDIR/container defined once)
├── utils.py                   # read_list() etc. — deletes the 4 copy-pasted versions
└── config.py                  # tuned reco parameters (snap_radius, d_vertex, ...) documented in one place
```

Refactor rules: moves + import fixes only; **no algorithmic edits**. Validate by
rerunning `run_nu_reco.py` + `eval_reco_performance.py` on one shard before/after
and diffing outputs.

### 3.3 `lartpc/larformer_analysis/` — active vs archived studies

```
lartpc/larformer_analysis/
├── slicer_eval/               # ← larformer_analysis/ (active, mature 5-stage pipeline)
├── particle_eval/             # ← larformer_particle_analysis/
├── physics/                   # ← larformer_physics/ (single_photon, repeatability_tests)
└── archive/                   # completed one-offs; each gains a 5-line README
    ├── semseg_analysis/       #   (what question, what answer, when closed)
    ├── deghost_analysis/
    ├── shower_origin_reco_scripts/
    ├── extbnb_larmatch/
    └── keypoint_v1/           # ← larformer_keypoint/ (attempt 1, superseded by larformer_reco)
```

Transition aid: leave `lartpc_data_prep/README.md` pointing to the new locations.

## 4. Working-tree hygiene

- Extend `.gitignore`: `output/`, `plots/`, `logs/` (per-directory patterns). Already
  covered: `outputs/`, `*.npz`. (Verified: none of the 1.1 GB under
  `larformer_keypoint_v2/output/` is git-tracked — this is disk clutter, not repo bloat.)
- On-disk artifacts (`output/`, `plots/`, `logs/`, `results*.npz`): do **not** move
  during Phase 1 — the submit scripts default to relative `output/...` paths, so
  relocation happens in Phase 4 when the reco package centralizes its paths.
- Delete `setenv_pointcept_only.sh~`.
- Root-level container/submit scripts (`start_container.sh`, `submit_*.sh`,
  `run_lora_finetune...sh`) → `scripts/` or the relevant project's `slurm/` folder.

## 5. Tools directory

- `tools/larformer/` — inference drivers (`run_larformer_stage3_inference.py`,
  `run_larformer_keypoint2_cascade_inference.py`, `run_larformer_fullcascade_inference.py`,
  `run_slicer_inference.py`) + cache build/augment tools.
- `tools/smoke_tests/` — the 8 `smoke_test_larformer_*` + benchmark scripts.
- `tools/viz/` and `tools/viz_archive/` — see Section 6.
- Upstream `train.py` / `test.py` stay at `tools/` root.

## 6. Shared visualization library

Create `lartpc/viz/` (importable via existing `PYTHONPATH=./`):
`detector.py` (single surviving copy of `detectoroutline.py` — currently duplicated
identically in `tools/` and `lartpc_data_prep/`), `palette.py`, `scatter3d.py`,
`wireplane.py`, `layout.py`.

Convert **lazily**: only the actively used displays (`visualize_lartpc_h5data.py`,
`visualize_larformer_gt.py`, `visualize_keypoint2_cascade_*`, the cascade reco
visualizer) are rewritten against it. The Sonata t-SNE/UMAP/PCA family and other
stage-specific one-offs move untouched to `tools/viz_archive/`. Also move
`pointcept/models/LArFormer/viz_inference.py` out of the model package into this area.

## 7. Documentation restructure

- `docs/reference/` — durable: data formats, dataset guides, LArFormer stage specs,
  `shower_origin` (consolidate the 3 shower-origin docs into one spec + history
  appendix), reproducibility, SLURM guide, Sonata loss functions.
- `docs/devlog/` — dated work logs: `LArFormer_Stage3_TrainingStability.md`,
  `wandb_sweeps_for_mup_proxy.md`, `Sonata_NaN_Gradient_Diagnosis.md` (RESOLVED),
  `semseg_analysis/CLUSTER_DEBUG_BRIEF.md` (RESOLVED), laptop-datasets note.
- `docs/LArFormer.md` stays the hub; its §0 becomes the master index: pipeline diagram
  (deghost → slicer → particle → keypoint → nu-reco → eval) and per-stage rows of
  {spec, config, model file, inference tool, analysis dir}.
- Every doc gets a one-line status header: `REFERENCE` / `WORKLOG` / `RESOLVED` /
  `SUPERSEDED by <doc>`.

## 8. Execution order

| Phase | Content | Risk | Verification |
|---|---|---|---|
| 1 | Hygiene (Section 4) + `configs/lartpc/README.md` manifest written **before** any moves | none | — |
| 2 | Config reorg (Section 2) | low (mechanical, but wide path sweep) | config-load loop over all moved configs; grep sweep clean; one smoke test run |
| 3 | `lartpc/` split — data_prep + analysis moves (3.1, 3.3) | low (scripts are leaf entry points) | spot-run one pipeline step per scope |
| 4 | `larformer_reco` package (3.2) | **medium** — import restructure | shard-level before/after diff of `run_nu_reco.py` + eval outputs |
| 5 | Shared viz library + tools regrouping (5, 6) | low, incremental | render one event display per converted script |
| 6 | Docs restructure (7) | none | link check from LArFormer.md hub |

Each phase is one commit series on `nutufts_larformer_reorganization`, landable
independently.

## 9. Open items to confirm before executing

1. Generation-1 training-prep script list (Section 3.1 archive candidates) — confirm
   none are still referenced by active submit workflows.
2. `semseg/` and `shower_origin/` project folders: confirm both are historical
   (all archived) or whether any config is still in use.
3. Whether `lartpc/` top-level is preferred over keeping everything under
   `lartpc_data_prep/` — this plan assumes the top-level split per 2026-07-03 discussion.
