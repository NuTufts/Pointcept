# LArFormer Cascade Inference Reproducibility

> **Status: REFERENCE** — Training-run reproducibility (seeds, RNG, checkpoints).

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
- **Bit-exactness extends across a uniform driver+library pool, regardless of
  GPU model.** Four A100 nodes (mixed 80 GB / 40 GB, same driver) are
  **bit-identical** — 0 differences across 88 M spacepoints. Determinism is a
  property of the **driver + CUDA/torch stack**, not the GPU SKU.
- The 9% event-level churn that started this came from **one non-conforming node
  (pax003, a different driver/state) silently mixed into the pool** — not an
  intrinsic per-GPU systematic. The control is a **driver allowlist + a cheap
  per-node conformance test** (the capture harness), not a widened error bar.

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
  (`tools/larformer/run_larformer_stage3_inference.py`, `run_full_cascade_mode`), so it
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

`tools/larformer/run_larformer_stage3_inference.py` provides `--deterministic`, which calls
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
python tools/larformer/run_larformer_stage3_inference.py ... --deterministic

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

So on a fixed GPU + driver + library stack, inference is fully repeatable
**when the events are processed in the same order** (see §3.2).

### 3.2 Event-order invariance (same-order repeatability is not enough)

`set_deterministic()` seeds the RNG **once at startup**. That is sufficient for
the §3.1 validation, which re-runs the *same event list in the same order* — both
runs consume the identical RNG sequence, so they match. It does **not** by itself
guarantee that a given event produces the same output independent of its
**position** in the batch sequence.

The forward consumes a small amount of per-event RNG. With a single startup seed,
the Nth event sees RNG state already advanced by the N−1 events before it — so the
*same* physical event processed at a different position (a reordered input list,
or a subset) can give a different result. The dataset itself is not the source
(test split: fixed lm-score threshold, `max_spacepoints=None` → no subsample;
note the loader also `sorted()`s the file list, so input-list order is
canonicalized — reordering must be probed via subsets or the per-event identity,
not list order alone).

Measured on the attempt-2 keypoint cascade (which exercises the full
deghoster→slicer→particle→keypoint stack): the same event processed alone vs 2nd
in a trio differed by **up to 985 cm with an endpoint-existence flip** — a
discrete amplifier flip (§2.3), purely from processing position.

**Fix — re-seed before every event** (`reseed_per_event()` in
`tools/larformer/run_larformer_stage3_inference.py`, called at the top of each event loop
in `--deterministic` mode):

```python
np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

so each event's RNG is reset and its output depends only on the event, not its
position. Wired into all three inference paths (cached-mode + full-cascade-mode
in `run_larformer_stage3_inference.py`, and
`run_larformer_keypoint2_cascade_inference.py`). After the fix (deterministic,
A100): single-event run-to-run **bit-exact (Δ=0)**; the same event processed
alone vs in a set, or a set processed in a different order, agrees to
**≤2e-3 cm (21 µm) with 0 particle-count / class / endpoint-existence flips**
(residual is float-level cuBLAS accumulation order, no deterministic-algorithm
fallbacks — physically negligible vs the cm-scale targets). Use `--deterministic`
for any measurement that re-processes events in a different order or as subsets.

---

## 4. Cross-node variation — it tracks the driver/library stack, NOT the GPU model

This is the part that matters for large-scale deployment, and the first answer
("treat cross-GPU as a systematic") turned out to be **too pessimistic** once we
measured it properly with the Tier-A harness (§6, `capture_cascade_tensors.py` +
`cross_gpu_diff.py`).

**What we first saw (and how it misled us).** The same `base` config
(`--deterministic`) on `pax050` vs `pax003` differed on 36/378 events
(drop-flag) with selection efficiency 0.14 ↔ 0.16 — a ~9 event swing, larger
than the physics effects under study. The natural inference was "different GPU
→ different result, treat as systematic."

**What is actually true (measured per-spacepoint, coordinate-aligned).** We
captured the 378-event decision tensors on **four** nodes and diffed them
spacepoint-by-spacepoint:

| Node | GPU | Driver | vs pax050 |
|------|-----|--------|-----------|
| pax050 | A100 **80 GB** PCIe | 575.57.08 | reference |
| pax052 | A100-PCIE-**40 GB** | 575.57.08 | **0 differences** |
| pax105 | A100 **80 GB** PCIe | 575.57.08 | **0 differences** |
| pax051 | A100-PCIE-**40 GB** | 575.57.08 | **0 differences** |

Zero divergence across **88 M** Stage-1 spacepoints, **17 M** Stage-2, **191 k**
Stage-3, and 0/378 drop-flips — **bit-identical across different GPU memory sizes
(80 GB vs 40 GB)** as long as the **driver + library stack** matches. Determinism
is a property of the **driver + CUDA/torch libraries**, not the GPU model.

**So what was pax003?** The one node that ever diverged. All *currently
available* nodes run the same driver (575.57.08) and agree to the bit; pax003
went **down** before we could read its driver, but the evidence points to it
having been on a **different driver/runtime** (the in-container CUDA/torch libs
are identical across all jobs, leaving the host driver as the variable). The
~9% was **a single non-conforming node silently mixed into the pool** — not an
intrinsic per-GPU systematic.

> Why a plain multi-arm SLURM array was still invalid: the deghost sweep
> scattered its arms across nodes, and *one* of them (pax003) happened to be the
> non-conforming node — so that arm's difference was the node, not the variable.
> The fix is the same either way: pin the comparison (§4.1).

### 4.1 Consequences

1. **Reproducibility within a uniform-driver pool is bit-exact** — across GPU
   memory sizes, physical chips, and nodes. The cross-GPU systematic inside such
   a pool is **zero**, not 9%.
2. **A non-conforming node (different driver) is the real hazard**, and it is
   *silent* — it just produces a different-but-internally-reproducible answer.
   Guard against it with the conformance test below, not by widening error bars.
3. **`Gres` is not a hardware identifier.** The `a100:8` tag spans both 80 GB and
   40 GB A100s; `a100:2` is 40 GB. It doesn't matter for *correctness* (same
   driver → same result regardless), but never use it to reason about hardware —
   read `nvidia-smi --query-gpu=name,driver_version`.
4. **For a pinned measurement**, `#SBATCH --nodelist=<node>` is the simplest
   guarantee; `submit_deghost_compare_3k_pin.sh` does this and adds a `base_dup`
   control arm (came back bit-identical, validating the pinned sweep).

### 4.2 The conformance test (the operational control)

The capture harness *is* a node-conformance gate. To clear a node for
production:

```bash
# capture a fixed reference event set on the candidate node, diff vs the reference node
sbatch --nodelist=<candidate> --export=ALL,CAPTURE_DIR=<dir>/<candidate> submit_capture.sh
python tools/cross_gpu_diff.py <dir>/<reference> <dir>/<candidate>   # must report 0 everywhere
```

Nodes that return 0 join the allowlist; a non-zero node is quarantined until its
driver/stack is brought into line. This is cheap (~10 min/node) and auditable —
a standard HPC practice, not a per-event uncertainty.

### 4.3 Cross-architecture conformance map (measured)

Same 378-event reference set, all driver 575.57.08, deterministic, diffed
spacepoint-by-spacepoint vs the A100-80GB reference (pax050):

| Pair | arch | keep-flips (of 88M) | slice-query flips | Stage-3 class flips | **event drop-flips** | verdict |
|------|------|---------------------|-------------------|---------------------|----------------------|---------|
| A100-80 vs A100-40 | Ampere↔Ampere | 0 | 0 | 0 | **0 / 378** | identical |
| A100 vs L40S | Ampere↔Ada | 3 (1e-5%) | 0.78% | 0.69% | **0 / 378** | **conforms** |
| A100 vs H100 | Ampere↔Hopper | 235 k (0.27%) | 20.2% | 3.2% | **7 / 378 (1.9%)** | **does NOT conform** |
| A100 vs H200 | Ampere↔Hopper | 235 k (0.27%) | 20.2% | 3.8% | **7 / 378 (1.9%)** | **does NOT conform** |
| H100 vs H200 | Hopper↔Hopper | 1 | 0.03% | 1.3% | **0 / 378** | conforms (event-level) |

**Conformance families:** `{A100 (any memory size), L40S}` are mutually
bit/event-identical → one allowlist, zero systematic. `{H100, H200}` form a
separate Hopper family (event-level consistent with each other) that diverges
from the Ampere/Ada family by ~1.9% at the event level.

**Source of the Hopper divergence — it's the GEMMs, NOT attention (tested).** We
re-ran A100 and H100 with the `xformers` attention backend instead of
`flash_attn` and re-diffed:

| A100 vs H100 | keep-flips | slice churn | event drop-flips |
|--------------|-----------|-------------|------------------|
| flash_attn backend   | 0.27% | 20.2% | 7/378 |
| xformers backend     | 0.27% | 19.0% | 7/378 |

Identical — swapping the attention backend does **not** change the cross-arch
gap. The clincher: the **deghoster backbone runs `enable_flash=False`**
(config L161), so its `P(real)` is *bit-identical* between flash and xformers on
a fixed GPU — yet the deghoster is exactly where the cross-arch divergence
enters (the keep-flips that drive everything). Therefore the divergence comes
from the **architecture-specialized cuBLAS GEMMs (linear/QKV/MLP projections)
and `SerializedPooling`**, not the attention kernel. flash-attn's Hopper rewrite
was a red herring. (Swapping flash↔xformers *does* perturb Stage 2/3 by ~1-2%
per-SP on a fixed GPU with 0 event flips — so the backend is a minor
reproducibility knob to pin, but not the cross-arch driver.)

**Anatomy of the Hopper divergence** (from `cross_gpu_diff`'s margin analysis):
`|ΔP(real)|` is bimodal — per-event *median* max ≈ 4e-6 (most events nearly
identical) but *p95* ≈ 0.8 (a tail of events has points that flip hard). Of the
235 k Stage-1 keep-flips, **64% sit within 0.05 of τ and 89% within 0.1** — so a
dead-band / hysteresis at the deghoster (§5.4) would remove most, but not all,
of the Hopper-vs-Ampere divergence (the ~11% large-margin tail is genuinely
different output, not knife-edge). This quantifies the amplifier-reduction payoff
**before** any model change.

---

## 5. Deploying across many nodes

Given §4, large-scale deployment is **operationally** tractable — the controls are
about standardizing and gating the software stack, not inflating error bars.

1. **Standardize the driver + library stack.** The in-container CUDA/torch is
   already fixed (one `.sif`); pin production to a host **driver allowlist**.
   Within a uniform-driver A100 pool, output is bit-identical regardless of GPU
   memory size — the cross-node systematic is **zero**.

2. **Gate every node with the conformance test (§4.2).** Capture a fixed
   reference event set on each candidate node and `cross_gpu_diff` vs the
   reference; admit only nodes that return 0. This catches a non-conforming node
   (the pax003 failure mode) *before* it contaminates a production pass — the
   gate is the safeguard, run it whenever the pool or driver changes.

3. **Different GPU *architecture* splits into conformance families — MEASURED
   (§4.3).** Ampere (A100) and Ada (L40S) **conform** (0 event-level
   differences). Hopper (H100 / H200) does **not** conform with them (~1.9%
   event-level drop-flips, 20% slice-assignment churn). So the allowlist is
   **per architecture-family**: deploy a given measurement within one family, or
   quantify the cross-family shift as a measured systematic. Always run the
   conformance test before adding an architecture.

4. **Reduce the amplifier (defense-in-depth, highest-leverage code change).**
   Whatever the stack, the fragility is the **hard cuts** stacked on ~1e-6
   numerics (boundary `argmax`, no_object floor, X-veto). Worth pursuing
   independently because it shrinks sensitivity to *any* perturbation — a driver
   upgrade, a new GPU architecture, or a future kernel change:
   - compute the final decision logits / `argmax` in **fp32/fp64** even if the
     backbone runs lower precision, to widen decision margins;
   - propagate **soft scores** instead of hard per-SP labels where possible, and
     cut as late as possible;
   - add **hysteresis / dead-bands** around knife-edge thresholds (esp. the
     X-veto) so a single flipped spacepoint can't tip an event.
   The Tier-A captures let you estimate the payoff **offline** (the margin
   analysis in `cross_gpu_diff.py`) before changing any model code.

5. **Golden reference for spot-checks.** A CPU (or single pinned node) run on a
   small fixed subset is a stable ground truth to regression-test production
   nodes against.

**Recommended operating procedure**

- *Exploration:* default (fast) mode.
- *Any measurement:* `--deterministic`, pinned to a conforming node
  (`--nodelist`).
- *Production at scale:* `--deterministic` ON + a **driver allowlist** + the
  **node conformance gate** (§4.2) run whenever the pool/driver changes. Within
  a conforming pool the cross-node systematic is zero; pursue §5.4 (amplifier
  reduction) as defense-in-depth against future stack changes.

---

## 6. Reproducing these checks

| What | Where |
|------|-------|
| `--deterministic` flag + `set_deterministic()` | `tools/larformer/run_larformer_stage3_inference.py` |
| Stage-B env knob `DETERMINISTIC=1` | `lartpc/larformer_analysis/physics/single_photon/run_stageB_capped.sh` |
| Same-GPU bit-exact validation (4× run + diff) | `.../single_photon/slurm/submit_determinism_test.sh`, `determinism_diff.py` |
| Pinned single-node sweep (+ `base_dup` control) | `.../single_photon/slurm/submit_deghost_compare_3k_pin.sh` |
| **Tier-A stage-by-stage tensor capture** (no model edits) | `tools/capture_cascade_tensors.py` |
| **Cross-node attribution + margin analysis** | `tools/cross_gpu_diff.py` |
| **Node conformance capture job** | `.../single_photon/slurm/submit_capture.sh` |

**Conformance families (driver 575.57.08, full-cascade, 378 events) — see §4.3:**
`{A100 80/40 GB, L40S}` mutually identical at event level (one allowlist, zero
systematic); `{H100, H200}` a separate Hopper family that diverges ~1.9% at
event level from Ampere/Ada. pax003 diverged historically (driver unconfirmed —
node down at re-check).

Related: `docs/LArFormer.md`, `docs/devlog/LArFormer_Stage3_TrainingStability.md`.
