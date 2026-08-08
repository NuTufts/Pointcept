# Stage-1 (Deghoster) + Stage-2 (Slicer) Retraining — Plan, Iterations, and Results

**Period:** 2026-07-20 → 2026-08-07
**Goal:** recover the soft-photon / soft-electron charge the LArFormer
reconstruction chain was losing upstream of the particle segmenter, and
choose the production stage-1+2 configuration for rebuilding the stage-3
training cache.

**Bottom line:** end-to-end photon charge completeness into the selected nu
slice improved from **0.603 → ~0.79** (label ceiling: 0.815), with
catastrophically-lost photons (<10% coverage) cut from **10.6% → 4.0%**,
using a retrained Mask2Former-recipe slicer, a retrained PTv3-decoder
deghoster, and a re-chosen deghost operating threshold.

---

## 1. Motivation

The pi0-mass analysis found ~25% of true photons effectively lost before
the particle segmenter (`sbnd_photon_loss.py` failure-stage breakdown,
`missSlice` bucket). Two upstream causes were pursued:

1. **Slicer training data was being decimated.** The training dataloader's
   `max_spacepoints=100k` cap (applied pre-deghost, by *random thinning*)
   bit on **66.7% of training events** (median post-dedup SP count 117.5k),
   systematically thinning sparse/soft showers
   (`lartpc/larformer_reco/tools/cap_study_spacepoints.py`).
2. **The deghoster was discarding real soft-shower charge** (quantified
   later in this campaign — see §5).

---

## 2. Slicer retraining

### 2.1 Iteration 0 — cap300k (abandoned)

Config: `configs/lartpc/larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py`
(cap raised 100k→300k train / 450k val; bite → 1.6%). Trained at the legacy
recipe (flat lr 1e-5, full-epoch warmup, plateau LR scheduler). **Outcome:
slow convergence, worse than the previous production model.** Abandoned
after ~2 epochs (~45 h/epoch).

### 2.2 Mask2Former best-practices review

A code review against the standard M2F recipe found the config compliant on
deep supervision, matcher-cost/loss-weight identity (2/5/5), point-sampled
mask loss, empty-attention-mask guard, and MaskDINO denoising — but
deficient on:

| Gap | Fix |
|---|---|
| `base_lr=1e-5` (10× below recipe) + ~45 h warmup | lr 1e-4, OneCycleLR, ~2.5k-step warmup |
| Weight decay on queries/norms/biases | `no_decay_on_1d_and_embeddings=["query_content","query_pos","class_embedding"]`, wd 0.05 |
| `no_object_weight=0.5` | 0.1 (M2F eos) |
| `clip_grad=1.0` | 0.1 |
| Plateau scheduler that could never act at 45 h/epoch | OneCycleLR sized to real wall-clock (5 epochs) |
| Matching-stability diagnostics off | `log_diagnostics=True` (`diag_match_agreement` etc.) |

### 2.3 Iteration 1 — m2frecipe (config-only)

Config: `...-m2frecipe.py`. Losses fell far faster than iteration 0.
**But** `diag_match_agreement` peaked ~0.3 then regressed to ~0.1 —
the query-assignment-churn signature (near-duplicate queries swapping
Hungarian assignments; the shared final-layer match then retargets every
decoder layer's loss each flip).

### 2.4 Iteration 2 — m2frecipe-v2 (production candidate)

Config: `...-m2frecipe-v2.py`. Three changes:

1. **`use_vectorized_pair_loss=True`** — batched per-pair BCE/Dice.
   Benchmarked (job 1809369): **2.04× end-to-end** (3.78 vs 7.70 s/iter),
   exact numerical parity for masks >8,196 positives; small-mask corner
   reads dice ~21% lower at equal quality (fewer sampled negatives —
   per-sample expectation unchanged; `diag_*_rand` scales shift vs
   loop-based runs).
2. **`num_queries` 128 → 48** — removes near-duplicate queries (flip cause).
3. **`match_per_layer=True`** (new opt-in flag in `LArFormerLoss`) —
   standard DETR/M2F per-layer Hungarian re-matching (flip damage removal).

Side experiments: bf16 AMP was benchmarked and **rejected** (4.8× *slower*
+ numerically broken under fp32-sanitizer interactions; job 1808825).

**Training result (5 epochs, ~22 h/epoch, 4×A100):**

| epoch | val loss | nu_mIoU | nu_recall | nu_purity |
|---|---|---|---|---|
| 1 | 22.50 | 0.6310 | 0.868 | 0.908 |
| 2 | 21.14 | 0.6596 | 0.869 | 0.917 |
| 3 | **20.81** | 0.6684 | 0.870 | 0.901 |
| 4 | 21.28 | **0.6740** | 0.869 | 0.904 |
| 5 | 21.63 | 0.6730 | 0.869 | 0.903 |

Val loss bottoms at epoch 3; metric peaks at epoch 4 (=`model_best`).
`diag_match_agreement` stabilized at ~0.5–0.6 (v1: 0.1) — churn fixed.
Note `nu_recall` is flat: aggregate recall saturated early, which is why
per-particle completeness (below) was built as the discriminating metric.

### 2.5 Slicer checkpoint selection (per-particle completeness)

New tool: `lartpc/larformer_analysis/slicer_eval/particle_slice_completeness.py`
— per-TRUE-particle charge completeness at the slicer stage.
Conventions: trackid linkage per `eval_reco_performance.py`
(`mc_particle_tree.trackid == triplet_data.trackid`, no descendant walk);
charge = raw Y-plane pixval sum (NOT de-double-counted — valid for A/B).
Decomposition per particle: `q_kept` (survived deghost) and `q_nuslc`
(kept AND in a nu-classed predicted slice), denominator = pre-deghost SPs.

Pi0-filter val/test set, 1,616 events, 4,098 particles (γ/e⁻/e⁺), all with
the **LoRA deghoster** at τ=0.5:

| species | old slicer (iter_75750) | v2 ep3 | v2 ep4 | frac<0.10: old → ep4 |
|---|---|---|---|---|
| γ (n=3336) | 0.603 | 0.637 | **0.640** | 10.6% → 5.7% |
| e⁻ (n=569) | 0.585 | 0.611 | 0.611 | 18.1% → 14.6% |
| e⁺ (n=193) | 0.554 | 0.588 | 0.588 | 15.0% → 8.8% |

Slicer-attributable photon loss (post-deghost − nu-slice): **0.049 → 0.013**
(~3.7× reduction). Post-deghost column identical across all three models
(same frozen deghoster) — pipeline consistency check.

**Chosen slicer checkpoint: `m2frecipe_v2` `epoch_4` (= model_best).**

---

## 3. Deghoster retraining

### 3.1 Step 0 — the label ceiling (decisive measurement)

The deghoster trains to reproduce `hasmatch`; a perfect one cannot beat the
labels. Charge completeness of the `hasmatch==1` set
(`particle_slice_completeness.py --keep-source hasmatch`):

| species | LoRA deghoster @τ=0.5 | label ceiling | shortfall |
|---|---|---|---|
| γ | 0.652 | **0.815** | −0.163 |
| e⁻ | 0.622 | **0.813** | −0.191 |
| e⁺ | 0.589 | **0.794** | −0.205 |

Two findings: (a) the LoRA model sat 16–21 charge-points below its own
labels → retraining had real headroom; (b) the labels themselves cap at
~0.80–0.82 (LArMatch truth-matching misses soft shower charge) — a
separate, still-open label-generation work item.

### 3.2 Architecture choice

Replaced `SonataLoRADeghostSegmentor` (LoRA rank-16 on the v7 extbnb Sonata
encoder + `Linear(1232→2)` on up-cast skip-concat features) with
**`DefaultSegmentorV2` + bare PT-v3m2 with the decoder enabled**
(`enc_mode=False`, dec channels 64/64/128/256, `Linear(64→2)` head), frozen
encoder (via `param_dicts lr=0.0`; the class's `freeze_backbone` is
all-or-nothing), same v7 extbnb pretrain
(`SonataFinetuneCheckpointLoader(use_teacher=True)`), same
`HasmatchAsGhost` labels (real=0), Focal+Lovász, same crop/augmentation
stack. Config: `configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-v1-frozenenc-extbnb.py`.
(An alternative — LArFormer `num_queries=0` "cls-only" mode — exists and is
production-proven by stage-4, but needs a new evaluator + Focal/Lovász
support in its per-level cls loss; deferred.)

### 3.3 Failures found and fixed on the way

1. **The original training data is gone.** The prod4 `v2_expandedclasses`
   dataset (LoRA's training data, 390k files) was deleted from
   `ub_on_tufts/hdf5`. Switched to the v3 LANTERN merged-h5 lists (the same
   files the slicer trains on; verified to carry `hasmatch`, `ssnet_label`,
   `nu_vertices`). The LoRA baseline can never be retrained on its
   original data.
2. **`segment_counts` label-space clash**: the dataset builds it in the
   9-class ssnet space before `HasmatchAsGhost` rewrites labels to 2-class;
   FocalLoss's (newer) count-weighting asserts. Fix: drop `segment_counts`
   from `Collect` (plain Focal α/γ path — matches what the LoRA model
   actually trained with).
3. **`flash_attn` produces NaN in fp32** — the first production launch
   NaN'd on *every* batch from iter 0, silently (the trainer's NaN-skip
   guard hides it; smoke health checks must grep loss *finiteness*).
   This is the same landmine `LArFormer._ensure_decoder_fp32_forward()`
   patches. Fix: `flash_backend="xformers"` (also −45% memory).
   The same fix is mandatory in any fp32 inference config.
4. Stale post-reorganization paths in `run_valtest_per_fileno.py` and the
   `REPO_ROOT` computation of all three `tools/larformer/run_*_inference.py`
   CLIs were repaired along the way (new-layout-first fallbacks).

### 3.4 v1 (crop-trained) results — and the key methodological finding

20 epochs, nearest-10,240-point crops, 2×A100, ~13 h. Crop-level val mIoU
**0.7813** vs LoRA's 0.7777 — **statistically tied**. But in the cascade
(with the ep4 slicer, τ=0.5):

| species | LoRA | PTv3-dec v1 | ceiling |
|---|---|---|---|
| γ post-deghost | 0.653 | **0.686** | 0.815 |
| e⁻ | 0.623 | **0.670** | 0.813 |
| e⁺ | 0.599 | **0.645** | 0.794 |

**Crop-level metrics were blind to a 3–5 point full-event difference.**
The per-particle completeness pipeline is therefore the standard deghoster
metric going forward. The slicer-attributable gap stayed at 0.013 → no
deghoster→slicer domain shift (the slicer's τ∈[0.4,0.6]-randomized
training generalized).

### 3.5 The τ-sweep — the operating point was the real lever

Sweeping the deghost threshold offline from the saved `p_real`
(`--keep-source tau_sweep`): the v1 model reaches **near-ceiling** shower
completeness at low τ — the p_real *ranking* is nearly perfect; soft
shower points sit at p_real ∈ [0.2, 0.5] and were being cut by τ=0.5.

Slicer-response reruns at τ=0.20 and τ=0.35 confirmed the **slicer gap is
flat (0.013–0.016) across the whole range**. Full operating curve
(v1 deghoster + ep4 slicer, end-to-end nu-slice values):

| τ | γ | e⁻ | e⁺ | nu-slice ghost-charge | γ frac<0.10 |
|---|---|---|---|---|---|
| 0.50 | 0.673 | 0.657 | 0.632 | 26.4% | 4.9% |
| 0.35 | 0.728 | 0.711 | 0.688 | 30.0% | 4.3% |
| 0.20 | **0.790** | **0.781** | **0.756** | 34.2% | 4.0% |

Trade ≈ 1.5 points photon completeness per point of contamination; no knee
— the operating point is a physics choice.

### 3.6 v2 — full-event fine-tune

Targeting the crop(10k)→full-event(100–300k) train/inference mismatch:
warm-start from v1, `BiasedSphereCrop point_max` 10,240 → 300,000
(nearest-k guard only), 2 epochs × 80k events, lr 1e-4.
Config: `deghost-ptv3decoder-v2-fullevent-ft.py`. Full-event val mIoU
0.7813 → **0.7880**. τ-sweep comparison (γ completeness / kept purity):

| τ | v1 (crops) | v2 (full-event ft) |
|---|---|---|
| 0.20 | 0.805 / 0.657 | **0.807 / 0.659** |
| 0.35 | 0.742 / 0.697 | **0.751 / 0.695** |
| 0.50 | 0.686 / 0.731 | **0.700 / 0.727** |

The v2 curve **weakly dominates** (+1–2 pts mid-range at matched purity;
converged at the ceiling-pinned low-τ end). Catastrophic photons at τ=0.20:
1.6% (ceiling: 0.8%). **Chosen deghoster: v2 full-event-ft `model_best`**
(`exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth`).

---

## 4. Flash-χ² selection study (pre-adoption check for low τ)

Question: does τ=0.20 contamination degrade flash-based nu-slice
selection, and does a *post-slicer* deghost cut on the flash charge
(loose slicing / tight light-prediction) help?

Implementation: `analyze_event.py --flash-charge-preal-min` (cut SPs are
excluded from `pe_pred`, pixel-share normalization, and χ² OOB positions;
slice membership untouched). Comparison tool:
`flashchi2_selection_compare.py` — per event, the min-χ² slice among
nu-classed candidates (argmax class = nu; no probability threshold), scored
vs GT nu. Full statistics (~1,610 events, τ=0.20, v1 deghoster + ep4
slicer):

**Slice-selection level (SP-count-weighted, post-deghost denominators):**

| variant | recall | purity | IoU | picked-best | no-candidate |
|---|---|---|---|---|---|
| (a) no cut | 0.967 | 0.650 | 0.638 | 0.993 | 1.3% |
| (b) flash charge p_real≥0.5 | 0.967 | 0.650 | 0.638 | 0.993 | 1.4% |

**Event level (charge-weighted, END-TO-END: denominator = pre-deghost
true-nu raw-Y charge; fully-deghosted / no-candidate events = failures):**

| variant | ⟨q-recall⟩ | median | eff>0.5 | eff>0.7 | eff>0.9 | ⟨q-purity⟩ | purity>0.7 |
|---|---|---|---|---|---|---|---|
| (a) | 0.856 | 0.898 | 96.4% | 93.7% | 48.3% | 0.626 | 28.0% |
| (b) | 0.855 | 0.898 | 96.3% | 93.6% | 48.3% | 0.626 | 28.0% |

Verdicts: (1) **τ=0.20 does not degrade flash selection** — the nu-class
prefilter + χ² ranking absorb the contamination (94% of nu events recover
>70% of total true-nu charge in the selected slice); (2) the flash-charge
cut is **selection-neutral** — retained in the analyzer for
charge-integrating quantities (energy proxies, absolute χ² in cuts), not
needed for slice picking. (In the non-production "min-χ² over ALL slices"
mode the cut is slightly harmful — stripping ghost charge makes some
cosmic slices more flash-competitive — so the nu-class prefilter should
stay in the selection path.)

---

## 5. Production configuration for the stage-3 cache

| component | choice |
|---|---|
| Stage-1 deghoster | `exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth` (`DefaultSegmentorV2` + PT-v3m2 decoder, **xformers** backend mandatory in fp32) |
| Deghost threshold τ | **physics choice on the v2 curve** (0.20 = max soft-shower recovery at 0.66 purity; 0.35 = middle; measured curve in §3.5–3.6) |
| Stage-2 slicer | `exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_v2/model/epoch_4.pth` (48 queries — reconcile `num_queries` in the full-cascade inference config, which inlines 128) |
| Cascade consumption | deghoster emits `seg_logits` → existing branch, `deghoster_class_index_real=0` unchanged |

**Cumulative campaign result** (old production chain → new chain @ τ=0.20):
γ 0.603 → **0.790**, e⁻ 0.585 → **0.781**, e⁺ 0.554 → **0.756**;
catastrophic photons 10.6% → 4.0%. Remaining gap to unity is dominated by
the `hasmatch` label quality (~0.18) — the next upstream work item.

**Cache-rebuild plan:** build new stage-3 cache
(`build_stage12_cache_shard.py` + `augment_stage12_cache_particle_class_id.py`)
with the pair above → validate (stage-3 smoke against it) → then delete the
old 87 GB cache (`larformer_cache_stage12__ptv3crosslevelslicer_iter_75750`).
Keep the 152 GB stage-4 `highermaxsp` cache until the stage-4 rebuild.
Filesystem `/cluster/tufts/wongjiradlab` is at 97% (1.1 TB free) — the new
cache (est. 100–150+ GB; larger at lower τ) fits without pre-deletion.
Stage-3 retrain config is ready:
`configs/lartpc/larformer/stage3_particle/larformer-particle-v2-cached-ptv3crosslevel-m2frecipe.py`.

---

## 6. Artifact index

**Configs** (`configs/lartpc/larformer/`):
- `stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel-m2frecipe.py` (+ `-v2` overlay; `-bf16amp` = rejected, documented)
- `stage2_slicer/larformer-slicer-m2frecipe-v2-ptv3deghost{,-tau020,-tau035,-ftfull}.py` — cascade eval configs (new deghoster swapped via `_delete_`)
- `stage1_deghost/deghost-ptv3decoder-v1-frozenenc-extbnb.py`, `-v2-fullevent-ft.py`

**Checkpoints:**
- Slicer: `exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_v2/model/epoch_4.pth`
- Deghoster: `exp/deghost_ptv3decoder_v2_fullevent_ft/model/model_best.pth` (v1 crops: `exp/deghost_ptv3decoder_v1_frozenenc_extbnb/model/model_best.pth`)
- Old production (baselines): slicer `…nonzeroinit_maskdn_noamp/model/model_ptv3crosslevel_iter_75750.pth`, deghoster `sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth` (old checkout)

**Tools** (`lartpc/larformer_analysis/slicer_eval/`):
- `particle_slice_completeness.py` — per-particle completeness; modes `pred` / `hasmatch` (label ceiling) / `tau_sweep`
- `flashchi2_selection_compare.py` — flash-selection quality incl. event-level charge-weighted efficiency/purity
- `analyze_event.py --flash-charge-preal-min` — post-slicer deghost cut on flash charge

**Result files** (under the m2frecipe_v2 exp dir unless noted):
- `valtest_epoch{3,4}/particle_completeness_m2fv2_ep{3,4}.npz` (LoRA deghost)
- old-checkout `…/valtest_iter_75750/particle_completeness_iter75750.npz` + `particle_completeness_hasmatch_ceiling.npz`
- `valtest_ep4_ptv3deghost{,_tau020,_tau035}/particle_completeness_*.npz` + `tau_sweep_ptv3deghost.npz`
- `valtest_ep4_ptv3dgftfull/particle_completeness_ftfull.npz` + `tau_sweep_ftfull.npz`
- `valtest_ep4_ptv3dg_tau020_prealcut05/` (flash-charge-cut analysis variant)

**Model/loss code changes** (all additive/opt-in):
- `LArFormerLoss`: `match_per_layer`, `use_vectorized_pair_loss`, `log_diagnostics`, `cost_origin` housekeeping
- `LArFormerParticleEvaluator`: val-probe kwargs forwarded
- `pointcept/utils/optimizer.py`: `no_decay_on_1d_and_embeddings`
- `tools/larformer/run_*_inference.py` + `slicer_eval/slurm/run_valtest_per_fileno.py`: reorg-path fixes; driver `EXTRA_ANALYZE_ARGS` passthrough

**Known caveats recorded:**
- Charge metrics use raw Y-plane pixval sums (not de-double-counted) — A/B valid, absolute values not comparable to reco metric C.
- `diag_*_rand` training diagnostics changed measurement scale with the vectorized loss.
- Crop-level deghoster val metrics do not predict full-event performance.
- `flash_attn` must not run in fp32 (xformers for trainable/inference fp32 paths).
- Perevent H5 schema no longer carries `sp_level_nu_recall/precision`/`m1_iou` datasets (metrics groups m3/m4/overclaim only).
