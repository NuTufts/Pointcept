# P05B.5 asinh input-scaling — implementation summary (Tufts → Isambard)

Branch **`p05b5-asinh-input`** (off `nutufts_isambard`), implemented and
tested at Tufts on 2026-07-16 per `P05B5_IMPLEMENTATION_HANDOFF.md`. Ready
for Isambard review + merge. Nothing was launched from Tufts; nothing was
merged to `nutufts_isambard`.

## What changed

**`pointcept/datasets/transform.py`** (the only live-code file touched; both
changes are default-preserving — proof below):

1. `LogTransform` gained `mode=None|"log"|"linear"|"asinh"` and
   `asinh_scale=50.0`. `mode=None` (every existing config) derives log/linear
   from the legacy `log` flag through the *same* code paths. `mode="asinh"`:
   `y = 2*asinh(clip(x, 0, max_val)/asinh_scale)/asinh(max_val/asinh_scale) - 1`,
   applied to the loader output (no second `min_val` offset). Class name kept.
2. `MultiplicativeRandomJitter` gained `value_space=None|"scaled_log"|"asinh"`
   (+ `min_val`, `max_val`, `asinh_scale`). `value_space=None` preserves both
   legacy `log_space` behaviors bit-for-bit, including the documented
   `(1+n)^(1/a)` ≈ ±13% amplification. The new modes apply the exact
   composition `y' = T(clip(T_inv(y)*(1+n), 0, max_val))`, so outputs stay in
   [-1, 1]. The RNG draw sequence is identical in every mode (one
   `random.random()` p-gate, one `randn` per key).

**`lartpc/pretraining_studies/gen_p05_configs.py`**: all six strength-
transform slots (SSL/supervised/probe × train/val) now render from a single
`@STRENGTH_TRANSFORM@` constant and the SSL charge jitter from
`@STRENGTH_JITTER@`; defaults reproduce every pre-existing config
**byte-identically** (verified — see below). New runs:

| run | file | delta |
|---|---|---|
| P05B.5-mc_noghost-s0 | `pretrain-sonata-p05b5-mc-noghost-asinh.py` | asinh(50, 1000) + exact jitter σ=clip=**0.125** (first-order match to siblings' effective ±12.5%) — one delta (transform) vs P05B.1 |
| P05B.6-mc_noghost-s0 | `pretrain-sonata-p05b6-mc-noghost-asinh-jitter005.py` | as B.5 but σ=clip=**0.05** — one delta (jitter magnitude) vs B.5 |
| probe | `linearprobe-sonata-p05-mc-noghost-asinh-tufts.py` | base probe + asinh scaling; serves both B.5 and B.6 (probes carry no strength jitter) |
| P05A.5-mc_noghost-s0 | `supervised-ceiling-p05a5-mc-noghost-asinh.py` | **generated, DO NOT SCHEDULE** unless B.5 shows an effect |

The 4th factorial cell (log + exact 0.05 jitter) was intentionally not
generated (handoff §2.3); noted in the B.6 config header.

**`lartpc_tests/`**: new `test_strength_transforms.py` (the §3 gate);
`validate_p05_configs.py` gained optional `--train-list/--val-list` overrides
(no args ⇒ behavior unchanged on Isambard) so the full pipeline check runs at
Tufts against the remapped diag1k list.

Also on the branch, as its own first commit (`688c76b`): the pre-existing
uncommitted Tufts filelist remap (probe template lists, `.gitignore`
exceptions, `remap_filelists_tufts.py`, remapped diag1k list). One local
hand-edit of the probe config (batch_size 288→64, num_worker 22→4, contrary
to its DO-NOT-HAND-EDIT banner) was **dropped** in favor of the generator
output; use `--options batch_size=64 num_worker=4` at probe launch if those
were wanted.

## Test evidence (all run in the container at Tufts, CPU)

`python3 lartpc_tests/test_strength_transforms.py`:

```
PASS  1. legacy LogTransform + MultiplicativeRandomJitter bit-identical
PASS  2. asinh mode: endpoints, clip, monotonicity, closed form
PASS  3. value_space jitter == T(clip(T_inv(y)*(1+n))) to <=1e-6
PASS  4. legacy log_space jitter == x*(1+n)^(1/a), a=0.400000 (nominal 5% -> 12.97%)
PASS  5. P05B.1 bit-identical to golden (688c76b71); B.5/B.6 strengths in [-1,1], geometry unchanged
PASS  6. asinh probe/B.6 transforms match P05B.5; jitter sigmas 0.125/0.05 as decided
```

Test 5 is the end-to-end default-preservation proof: the generated P05B.1
config's full `dataset[0]` pipeline (diag1k data, fixed seeds) is
bit-identical across all 11 output arrays to a golden produced at the
pre-change tree. The golden is gitignored; regenerate at your merge-base with
`python3 lartpc_tests/test_strength_transforms.py --make-golden`, then run
the suite on the branch (procedure in the test's docstring). Determinism of
the golden itself was verified by rebuilding twice.

Config-level: regeneration on the branch changes **zero bytes** of the 12
pre-existing configs (`git diff` empty) and adds exactly the 4 new files.

`python3 -m compileall pointcept/datasets lartpc_tests lartpc/pretraining_studies`
clean; `validate_p05_configs.py --train-list ... --val-list ...` (diag1k):
**16/16 PASS**, including full dataset builds of all four new configs.

One deviation from the handoff's literal test text: §3.5 says strengths "lie
in [-1,1] for both". For B.1 the *legacy* jitter can overshoot to
±|log10(1∓0.05)| ≈ 0.022 beyond ±1 by construction, so the B.1 bound is
widened by exactly that amount (derived from the config's own jitter params);
B.5/B.6 are held to strict [-1, 1], which their exact jitter guarantees.

## Launch (from Isambard, after merge, at a moment of your choosing)

```bash
./slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh \
    configs/lartpc/p05/pretrain-sonata-p05b5-mc-noghost-asinh.py   # primary
./slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh \
    configs/lartpc/p05/pretrain-sonata-p05b6-mc-noghost-asinh-jitter005.py
```

B.5/B.6 snapshots must be probed with
`linearprobe-sonata-p05-mc-noghost-asinh-tufts.py` only — probing them with
the log-scaled base probe silently mismatches the input scaling.
