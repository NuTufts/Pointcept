# P05B.5 asinh input-scaling — implementation handoff (for a Claude session at Tufts)

**Read this whole file before editing anything.** It specifies the code
changes, the tests that gate them, and the branch workflow. The P05F study
verdict and transform parameters are in `README.md` in this directory
(short version: add an `asinh(scale=50, xmax=1000)` input-scaling variant;
do NOT touch the 1000 ADC clip; do NOT use quantile or asinh25).

## 0. Hard constraints

1. **The Isambard repo working tree is LIVE code for running jobs.** Eight
   Wave A runs resubmit themselves via a SIGUSR1 chain, and each resubmit
   re-imports from the working tree. Therefore ALL changes must be
   **default-preserving**: with existing config arguments, every transform
   must produce *bit-identical* outputs to current behavior. New behavior
   only behind new argument values.
2. **Develop on a branch at Tufts** (suggested: `p05b5-asinh-input` off
   `nutufts_isambard`). Do NOT merge into `nutufts_isambard` from Tufts.
   Push the branch; the Isambard session merges after review, at a moment
   of its choosing, and generates/launches the new configs there.
3. Tufts has the MC data (source files + remapped lists) — use it for the
   pipeline-level tests. CPU is fine for everything below
   (`apptainer exec --bind /cluster:/cluster
   /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif ...`).

## 1. Exact current pipeline semantics (verified 2026-07-16; replicate in tests)

Strength chain for P05 configs, in order:

1. Loader (`pointcept/datasets/lartpc.py` `get_data`):
   `strength = clip(pixval, 0, 1000)/adc_scale + add_min_pixval`
   with `adc_scale=1.0`, `add_min_pixval=0.01` in all P05 configs.
2. Pipeline `LogTransform(min_val=0.01, max_val=1000.0, log=True, keys=("strength",))`
   (`pointcept/datasets/transform.py:1893`):
   `y = 2*(log10(x + 0.01) - log10(0.01)) / (log10(1000.01) - log10(0.01)) - 1`
   **Note the double offset:** the loader already added 0.01, and LogTransform
   adds its `min_val` again inside the log10. The value entering log10 is
   `clip(pixval,0,1000) + 0.02`. This is existing behavior — do not "fix" it;
   model it faithfully in tests.
3. Inside `MultiViewGenerator.global_shared_transform` (SSL configs only):
   `MultiplicativeRandomJitter(sigma=0.05, clip=0.05, keys=("strength"), p=0.8, log_space=True)`
   (`transform.py:553`): adds `log10(1+n)`, `n ~ clipped N(0, 0.05)`, directly
   to the LogTransform-rescaled `y`.

**Known inconsistency (measure it, codify it, do not silently change it):**
`log_space=True` assumes the stored value is plain `log10(x)`, but `y` is an
*affine rescale* with slope `a = 2/(log10(1000.01)-log10(0.01)) ≈ 0.4000`.
Adding `log10(1+n)` to `y` therefore multiplies the underlying `(x+0.02)` by
`(1+n)^(1/a) = (1+n)^2.5` — a nominal ±5% jitter is actually **±13%**
(`1.05^2.5 = 1.1297`). All current Wave A runs share this, so their internal
comparisons are unaffected; new code paths must not alter it for existing
argument combinations.

## 2. Required code changes

### 2.1 `LogTransform`: add an asinh mode (transform.py)

Add a `mode` parameter, e.g. `mode=None` (default) → derive from the legacy
`log` flag ("log" / "linear"), or explicit `mode="asinh"` with
`asinh_scale=50.0`. Asinh branch (input `x` is the loader output, i.e.
already offset by +0.01 — do NOT add `min_val` again in this branch):

```python
y = 2 * arcsinh(clip(x, 0, max_val) / asinh_scale) / arcsinh(max_val / asinh_scale) - 1
```

with `max_val=1000.0`, `asinh_scale=50.0` for P05B.5. Keep the existing
`log=True/False` API untouched (default-preserving requirement); `mode`
overrides it only when explicitly set. Keep the class name (all six
generator slots stay interchangeable).

### 2.2 `MultiplicativeRandomJitter`: exact value-space-aware jitter

Add `value_space=None` (default; preserves current behavior exactly for
both `log_space` settings). New values:

- `value_space="scaled_log"` with params `(min_val, max_val)`: exact
  multiplicative jitter through the current LogTransform, i.e.
  `y' = T_log(T_log_inv(y) * (1+n))` in closed form.
- `value_space="asinh"` with params `(asinh_scale, max_val)`: exact
  multiplicative jitter through the asinh transform:
  `y' = T_asinh(T_asinh_inv(y) * (1+n))` where
  `T_asinh_inv(y) = asinh_scale * sinh((y+1)/2 * arcsinh(max_val/asinh_scale))`.
  Re-clip the multiplied value to `[0, max_val]` before re-transforming.

### 2.3 Jitter strength for P05B.5 — decision needed, with a default

The sibling runs' *effective* multiplicative jitter is `(1+n)^2.5`,
first-order ±12.5%. For one-delta comparability against P05B.1, the default
choice is **sigma=0.125, clip=0.125** with `value_space="asinh"` (first-order
match to the siblings' effective strength). The alternative — "clean"
sigma=0.05 — changes two things at once (transform AND augmentation
strength) and should only be picked if the PI prefers it; record the choice
in the config header and the registry either way.

### 2.4 Config generator (`lartpc/pretraining_studies/gen_p05_configs.py`)

- Parameterize the strength transform across **all six LogTransform slots**
  (SSL template: transform + val_transform; supervised template: train +
  val; probe template: train + val) — a `@STRENGTH_TRANSFORM@` block
  substitution is the natural mechanism, defaulting to the current log dict.
- Add `P05B.5-mc_noghost-s0` to `SSL_RUNS`: identical to P05B.1 (free-rot
  augs, prototypes 4096) except (a) the asinh strength transform in both
  pipelines and (b) the value-space-aware jitter per 2.2/2.3.
- Add `linearprobe-sonata-p05-mc-noghost-asinh-tufts.py` to `PROBE_RUNS`:
  identical to the existing probe but with the asinh transform — **B.5
  snapshots must be probed with matching input scaling** or the probe
  numbers are meaningless.
- Optional (generate but do not schedule): `P05A.5` supervised ceiling with
  asinh, for later if B.5 shows an effect.
- Regenerate and confirm via `git diff` that **only** the new configs appear
  and no existing config changed by even one byte (this is the
  default-preservation check at the config level).

## 3. Tests (new file: `lartpc_tests/test_strength_transforms.py`)

Gate the merge on all of these passing (CPU, uses Tufts MC files via the
remapped diag1k list `lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt`):

1. **Regression / default-preservation:** with fixed seeds, run
   `LogTransform(log=True/False)` and
   `MultiplicativeRandomJitter(log_space=True/False)` on stored reference
   arrays and compare against golden outputs computed from an inline
   *reference reimplementation of the current formulas* (copy the exact
   expressions from §1, not from the refactored code). Bit-identical
   (`np.array_equal`), not allclose.
2. **Asinh correctness:** monotonic; maps [0, 1000]→[-1, 1] exactly at the
   endpoints; matches the closed form from §2.1 at random points; values
   above max_val clip to +1.
3. **Exact-jitter round trip:** for both `value_space="scaled_log"` and
   `"asinh"`, verify `y' == T(clip(T_inv(y)*(1+n)))` to ≤1e-6 for random y
   and n, including at the clip boundaries.
4. **Codify the legacy amplification:** assert that legacy
   `log_space=True` jitter on LogTransform-scaled values equals multiplying
   the underlying `(x+0.02)` by `(1+n)^(1/a)` with `a = 2/(log10(1000.01)-log10(0.01))`
   — this documents the ±13% behavior as intentional-for-compatibility.
5. **Pipeline-level (dataset + MultiViewGenerator):** build the generated
   P05B.5 config and its P05B.1 sibling with `Config.fromfile`, run
   `dataset[0]` through the full transform pipeline for each; assert
   `global_feat` strength channels lie in [-1, 1] for both, and that the
   B.1 output is bit-identical to the same seed's output on the
   `nutufts_isambard` base commit (checkout comparison or golden file) —
   the end-to-end default-preservation proof.
6. **Probe-side:** parse the asinh probe config; assert its strength
   transform parameters match the P05B.5 SSL config's exactly (single
   source of truth through the generator).

## 4. Workflow / acceptance checklist

- [ ] branch `p05b5-asinh-input` off latest `nutufts_isambard`
- [ ] §2 changes implemented, `python3 -m compileall` clean
- [ ] §3 tests all pass in the container at Tufts
- [ ] `python3 lartpc/pretraining_studies/gen_p05_configs.py` → git diff
      shows ONLY new files (p05b5 SSL config, asinh probe config, optional
      p05a5)
- [ ] `lartpc_tests/validate_p05_configs.py` still passes for the old
      configs (run with the Tufts lists or parse-only)
- [ ] jitter-strength decision (§2.3) recorded in the B.5 config header
- [ ] branch pushed; summary comment for the Isambard session: what changed,
      test evidence, and the exact launch command
      (`./launch_p05_run.sh configs/lartpc/p05/<b5 config>.py`)

**What NOT to do:** don't modify the loader clip or `add_min_pixval`; don't
change any existing config file; don't rename `LogTransform`; don't merge to
`nutufts_isambard`; don't launch anything from Tufts.
