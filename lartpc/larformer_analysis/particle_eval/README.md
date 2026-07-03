# LArFormer Stage-3 particle-segmenter — validation analysis

Offline efficiency/purity validation for the Stage-3 particle segmenter:
run inference over a full val/test split on SLURM, distill each event to
a small per-pair record, then aggregate into the same scalar metrics the
in-training evaluator (`LArFormerParticleEvaluator`) reports — plus
size-stratified extras the evaluator doesn't compute.

Sibling of [`../larformer_analysis/`](../larformer_analysis/) (the
Stage-2 *slicer* analysis, same pattern). Model/architecture context:
[`docs/LArFormer.md`](../../docs/LArFormer.md) (hub) and
[`docs/reference/LArFormer_particlesegment_stage.md`](../../docs/reference/LArFormer_particlesegment_stage.md)
(Stage-3 design + as-built status).

## Pipeline

```
stage3pred_<stem>.h5            per-event inference output
  (tools/larformer/run_larformer_stage3_inference.py — cached or full-cascade mode)
        │
        ▼  analyze_event.py     one event → perevent_<stem>.h5
                                (or skipped_<stem>.h5: no GT / dropped)
        │
        ▼  aggregate_metrics.py walk perevent_*.h5 →
                                summary_<TAG>_<MODEL_TAG>.json
                                pairs_<TAG>_<MODEL_TAG>.parquet
                                events_<TAG>_<MODEL_TAG>.parquet
```

### `analyze_event.py`

Thin transformer: reads one `stage3pred_*.h5`, extracts the per-GT pair
records the inference file already carries (`stage3_gt/pair_iou`,
`pair_cls_correct`, `pair_origin_l2_cm`, `class_id`, `n_truth_points`,
`matched_query`) plus event metadata, and writes `perevent_<stem>.h5`.
No per-pixel recomputation. Events without GT or dropped by the cascade
get a `skipped_<stem>.h5` marker so the driver's skip-if-exists logic
treats them as done.

### `aggregate_metrics.py`

Walks an analysis dir of `perevent_*.h5` and reproduces the evaluator's
scalars on the full split:

- `val/mask_iou_{mean,median,p25}` and per-class `val/mask_iou_<class>`
- `val/cls_accuracy`, `val/matched_fraction`, `val/n_active_queries_mean`
- `val/origin_l2_cm_{mean,median,p25,p75}` and per-class

plus the **size-stratified stress metrics** not in the in-training
evaluator: `val/mask_iou_smallest25` (+ median), the matched fraction in
the smallest-25%-by-spacepoint-count GT bucket, and the SP-count
threshold defining that bucket — so small-particle failures (the usual
tail) are visible instead of averaged away. The full per-pair table is
kept as parquet for ad-hoc slicing (per-class × per-size, per-event
joins, etc.).

### `slurm/` — split-level driver

- [`slurm/submit_valtest.sh`](slurm/submit_valtest.sh) — self-resubmitting
  sbatch wrapper. Point it at a per-(dataset × model) conf; it sizes the
  array from the input source automatically.
- [`slurm/run_valtest_per_task.py`](slurm/run_valtest_per_task.py) — the
  per-array-task driver. Two input modes, matching the inference CLI:
  - `cached` — enumerate Stage-1+2 cache events (what the in-training
    evaluator sees); each task takes a contiguous slice, runs
    `run_larformer_stage3_inference.py --input-mode cached
    --cache-file-list <task list>` once for the slice, then loops
    `analyze_event.py`.
  - `full-cascade` — read filenos from a rerun-lines file + manifest CSV
    and run deghoster → slicer → particle segmenter end-to-end per event.
- [`slurm/valtest_pi0_ptv3crosslevel.conf`](slurm/valtest_pi0_ptv3crosslevel.conf)
  — sample conf (paths, TAG, MODEL_TAG, SBATCH knobs, INPUT_MODE).

Per-event idempotency: an existing `perevent_*.h5` / `skipped_*.h5`
marker skips that event; delete markers to force re-analysis.

```bash
# 1. copy + edit a conf (model ckpt, cache dir or rerun lines, tags)
# 2. from the repo root on a head node:
sbatch lartpc/larformer_analysis/particle_eval/slurm/submit_valtest.sh \
       lartpc/larformer_analysis/particle_eval/slurm/valtest_pi0_ptv3crosslevel.conf
# 3. after the array completes:
python lartpc/larformer_analysis/particle_eval/aggregate_metrics.py \
       --analysis-dir <OUTPUT_DIR>/analysis/<TAG> --tag <TAG> --model-tag <MODEL_TAG>
```

## Output schema (`perevent_*.h5`)

```
pair/class_id            (K,) int    GT class per instance
pair/n_truth_points      (K,) int
pair/matched_query       (K,) int    -1 = unmatched
pair/pair_iou            (K,) float  -1 where unmatched
pair/pair_cls_correct    (K,) int8   -1 where unmatched
pair/pair_origin_l2_cm   (K,) float  -1 where unmatched / no GT origin
root attrs: run, subrun, event, name, model_tag, n_sp_post,
            n_active_queries, n_gt, n_matched, no_object_class_id,
            class_prob_threshold, stage3pred_path
```

## Notes

- The pair metrics are **raw-model** (Hungarian-matched, full-mask IoU)
  — they are unaffected by the inference-side query dedup
  (`--dedup-iou-threshold`, see
  [`docs/devlog/LArFormer_Stage3_TrainingStability.md`](../../docs/devlog/LArFormer_Stage3_TrainingStability.md)
  §7), which only changes the per-SP panoptic *assignment* keys. A
  post-dedup assigned-IoU metric is a planned follow-up (R8.5 there).
- For metric definitions shared with training-time eval, the source of
  truth is
  [`pointcept/models/LArFormer/particle_evaluator.py`](../../pointcept/models/LArFormer/particle_evaluator.py).
