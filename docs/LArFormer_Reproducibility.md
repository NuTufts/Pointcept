# LArFormer Cascade Inference Reproducibility

How to make LArFormer cascade inference repeatable for physics, what makes it
*not* repeatable by default, and the limits of that repeatability across
different GPUs. Written from the single-photon (1γ+0X) study, where a small
selection effect was being swamped by run-to-run output churn.

**TL;DR**

- By default, inference is **non-deterministic**: the same input produces
  different outputs run-to-run. The instability starts at the **deghoster**
  (the surviving point cloud changes every run) and is amplified by downstream
  hard cuts into ~9% event-level label flips.
- A `--deterministic` mode makes inference **bit-exact on a fixed GPU**
  (validated: 0 / 1,079,839 spacepoints differ across two runs).
- **Bit-exactness does NOT survive a change of GPU model.** The same
  deterministic config on two different A100 SKUs diverged on **376 / 378
  events**. For heterogeneous large-scale deployment this must be treated as a
  **systematic**, not eliminated.

---

## 1. Why this matters

A physics measurement must be repeatable: re-running the selection on the same
data must give the same events, or the result isn't well-defined. The
single-photon study exposed the problem — the 1γ+0X selection efficiency jumped
~9% (event-level) between identical re-runs, which is larger than the effects
(flash recovery, deghost-threshold changes) we were trying to measure.

There are **two distinct levels** of reproducibility, and they have very
different difficulty:

| Level | Question | Status |
|-------|----------|--------|
| **Run-to-run, same machine** | same input + same GPU → same output? | **Solved** — bit-exact with `--deterministic` |
| **Cross-hardware** | same input, *different* GPU model → same output? | **Not solved** — large divergence; treat as systematic |

---

## 2. Sources of variability (what we found in the code)

### 2.1 Input pipeline — NOT a source (in this path)

The data loader is deterministic for full-cascade inference:

- `LArFormerDataset` has a random spacepoint subsample
  (`np.random.permutation`, `pointcept/datasets/larformer.py:440`) but it only
  fires when `max_spacepoints` is set **and** exceeded. The full-cascade
  inference forces `ds_cfg["max_spacepoints"] = None`
  (`tools/run_larformer_stage3_inference.py`, `run_full_cascade_mode`), so it
  never runs. **(If you reuse the dataset with a `max_spacepoints` cap, this
  becomes a real per-run input-level source — seed it.)**
- The larmatch-score pre-filter uses a *fixed* `lm_score_val_threshold` for the
  test split (`_sample_threshold`, `larformer.py:328`); only the `train` split
  randomizes it.
- Voxel dedup is `np.unique(...)` — deterministic.

So identical bytes enter the model each run. The variation is entirely in the
**GPU forward pass**.

### 2.2 GPU forward — the actual sources

By default the inference sets **none** of the PyTorch determinism controls (no
seed, no TF32 disable, no deterministic algorithms, no `CUBLAS_WORKSPACE_CONFIG`,
no cuDNN flags). That leaves several jitter generators active:

1. **TF32 matmul (Ampere+ default) — the dominant source.** With TF32, every
   linear / attention projection truncates inputs to a 10-bit mantissa
   (~1e-3 relative), and cuBLAS is free to pick a *different split-K reduction
   order* run-to-run. This produces logit differences far larger than typical
   argmax margins.

2. **Non-stable pooling reduction in the PTv3 backbone.** `SerializedPooling`
   runs at every downsampling stage
   (`pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py:509-526`):

   ```python
   _, indices = torch.sort(cluster)        # CUDA sort: NOT stable for ties
   feat = torch_scatter.segment_csr(self.proj(feat)[indices], idx_ptr,
                                    reduce=self.reduce)   # mean: non-associative
   coord = torch_scatter.segment_csr(coord[indices], idx_ptr, reduce="mean")
   ```

   Points sharing a parent voxel (ties in `cluster`) receive a
   nondeterministic intra-segment order, and `mean`/`sum` reduction is
   non-associative in floating point. Each pooled feature differs by ~1e-7,
   and the error propagates and amplifies through depth.

3. **Atomic-accumulation kernels** (`torch_scatter` segment ops; any
   scatter/`index_add` in mask assembly) accumulate in hardware-nondeterministic
   order when deterministic algorithms are off.

4. **cuDNN benchmark autotuner** can select different conv algorithms run-to-run
   (minor here — the model is attention-dominated).

### 2.3 The amplifier: hard cuts turn ~1e-4 jitter into label flips

The per-spacepoint particle label is an `argmax` over query-mask logits,
followed by a **hard no_object threshold**, then fragment grouping, then the
1γ+0X **X-veto** (a discrete cut on particle counts). Every stage is a step
function. Spacepoints whose top-two logits sit within the numerical jitter band
flip assignment; flips across the no_object boundary add/remove points from a
fragment; occasionally that tips the X-veto and changes the event's label.

This is why the instability presents as **event-level label churn** rather than
visibly broken output — and why it starts at the deghoster: in default mode the
deghoster's `P(real) > τ` decision flips for boundary spacepoints, so the
*surviving point cloud itself* differs every run (measured: 24/24 events had a
different `N_post` between two default-mode runs), which then cascades into
different slicer and segmenter inputs.

---

## 3. The fix: bit-exact on a fixed GPU

`tools/run_larformer_stage3_inference.py` provides `--deterministic`, which calls
`set_deterministic()` **before the model is built**:

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic cuBLAS (read at init)
np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False        # the big lever on Ampere
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True, warn_only=True)
```

What each piece addresses:

| Control | Removes |
|---------|---------|
| `allow_tf32 = False` + `matmul_precision("highest")` | TF32 mantissa truncation + variable split-K order (§2.2-1) |
| `use_deterministic_algorithms(True)` | forces deterministic `torch.sort` → fixes `SerializedPooling` input order → reproducible `segment_csr` (§2.2-2); pins atomic-kernel paths (§2.2-3) |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | required for cuBLAS to be deterministic; must be set **before** the first cuBLAS call (so also exported at the shell level in `run_stageB_capped.sh`) |
| `cudnn.deterministic/benchmark` | conv autotuner variation (§2.2-4) |
| seeds | any residual RNG (e.g. the dataset subsample if `max_spacepoints` is ever set) |

`warn_only=True` is deliberate: instead of hard-failing if some op lacks a
deterministic CUDA kernel, it prints a warning naming the op. In our validation
**no op fell back** — once TF32 is off and `torch.sort` is pinned, every kernel
in the cascade path has a deterministic implementation.

**Usage**

```bash
# direct
python tools/run_larformer_stage3_inference.py ... --deterministic

# via the single-photon Stage-B wrapper (also exports CUBLAS_WORKSPACE_CONFIG)
DETERMINISTIC=1 source run_stageB_capped.sh <config> <list> <outdir>
```

Cost: ~1.3–2× forward latency on A100 (TF32 off). Keep the fast
non-deterministic mode for exploration; use `--deterministic` for any
measurement.

### 3.1 Validation (same GPU)

`slurm/submit_determinism_test.sh` runs the baseline cascade on the same 24
events **4×** (2 default, 2 deterministic, all on one A100) and diffs each pair
with `determinism_diff.py`:

- **Default mode:** every event differs — 24/24 had a different deghosted
  `N_post`, and the drop decision flipped for 6/24 (25%).
- **Deterministic mode:** **bit-exact** — `0 / 1,079,839` post spacepoints
  differ, `0 / 3,664` stage-3 spacepoints differ, 0 drop-flag changes, 0
  coord mismatches across two independent runs. No deterministic-algorithm
  fallback warnings.

So on a fixed GPU + driver + library stack, inference is fully repeatable.

---

## 4. The hard limit: cross-hardware variation

**Determinism is per-GPU-architecture, not portable.** `torch`'s deterministic
algorithms guarantee a *fixed reduction order within a chosen kernel* — but the
kernel itself is selected per GPU architecture (SM count, tensor-core
generation, tiling), so two different GPUs run *different* deterministic
kernels with *different* float accumulation, giving ~1e-6 differences that the
hard cuts (§2.3) then amplify.

We measured this directly. The **same** `base` config (deghost τ=0.5,
`--deterministic`) on two different A100 SKUs:

- `pax050` (`a100:8`, 80 GB SXM) vs `pax003` (`a100:2`, 40 GB PCIe)
- **376 / 378 events** differed in deghosted `N_post`
- **36 / 378 events** flipped their drop/keep decision
- selection efficiency moved **0.138 → 0.161** (~9 events in the numerator)

That cross-hardware spread is **larger than the physics effects we were trying
to measure** (deghost-threshold and flash-recovery changes of a few events). A
multi-arm comparison submitted as a plain SLURM array is therefore **invalid** —
tasks scatter across nodes and SKUs, and the arm-to-arm differences are
dominated by hardware, not the variable under study.

### 4.1 Consequence for comparisons

For any A/B measurement, **pin all arms to one node / one GPU SKU**:

```bash
#SBATCH --nodelist=pax052        # or a --constraint that selects one SKU
```

On the Tufts `gpu` partition the `a100:8` nodes
(`pax049/050/051/052/105/106/007`) are the same SKU (A100 80 GB SXM); `pax003`
(`a100:2`) is a *different* SKU. `submit_deghost_compare_3k_pin.sh` pins the
whole sweep to one node and adds a `base_dup` arm (identical config, different
physical GPU on the same node) to confirm same-node reproducibility before
trusting the comparison.

---

## 5. Deploying across heterogeneous GPUs (the open problem)

Large-scale production will span GPU types (A100 / H100 / L40S / …). Bit-exact
output across them is **not achievable** with stock PyTorch. Strategies, roughly
in order of practicality:

1. **Treat cross-GPU variation as a systematic uncertainty.** Physics results
   are statistical. Run a fixed **control sample on each GPU type** and measure
   the label-flip rate / efficiency shift vs a designated reference GPU; fold
   that into the systematic budget. This is the realistic path for a large
   dataset and is the recommended default.

2. **Pin a GPU SKU per *measurement*, not per *production pass*.** Production
   (making the ntuples) can span GPUs; but any number that enters a *result*
   (efficiency, purity, a tuned threshold) should be measured on a single pinned
   SKU so it is itself reproducible. Re-derive such numbers if the reference
   hardware changes.

3. **Reduce the amplifier — the highest-leverage code change.** The fragility is
   not the ~1e-6 numerics, it is the **hard cuts** stacked on top of them
   (boundary `argmax`, no_object threshold, X-veto). Options:
   - compute the final decision logits / `argmax` in **fp32 or fp64** even if
     the backbone runs in lower precision, to widen decision margins;
   - propagate **soft scores** (probabilities) instead of hard per-SP labels
     where the downstream allows, and apply cuts as late as possible;
   - add **hysteresis / dead-bands** around knife-edge thresholds (e.g. the
     X-veto), so a single flipped spacepoint can't tip an event.
   These shrink sensitivity to *any* numerical perturbation — TF32, cross-GPU,
   or driver upgrades — and are worth pursuing independently of determinism.

4. **Golden reference for spot-checks.** A CPU (or single pinned GPU) run on a
   small fixed subset gives a stable ground truth to regression-test production
   GPUs against, even though it is far too slow for the full dataset.

**Recommended operating procedure**

- *Exploration:* default (fast) mode.
- *Any measurement:* `--deterministic` **and** pin to one GPU SKU.
- *Production at scale:* `--deterministic` ON (removes within-node run-to-run
  churn) + accept cross-GPU spread as a measured systematic + pursue §5.3 to
  shrink it.

---

## 6. Reproducing these checks

| What | Where |
|------|-------|
| `--deterministic` flag + `set_deterministic()` | `tools/run_larformer_stage3_inference.py` |
| Stage-B env knob `DETERMINISTIC=1` | `lartpc_data_prep/larformer_physics/single_photon/run_stageB_capped.sh` |
| Same-GPU bit-exact validation (4× run + diff) | `.../single_photon/slurm/submit_determinism_test.sh`, `determinism_diff.py` |
| Pinned single-node sweep (+ `base_dup` control) | `.../single_photon/slurm/submit_deghost_compare_3k_pin.sh` |

Related: `docs/LArFormer.md`, `docs/LArFormer_Stage3_TrainingStability.md`.
