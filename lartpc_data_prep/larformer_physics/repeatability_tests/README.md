# LArFormer Cascade Repeatability Tests

Tools to verify that **LArFormer full-cascade inference is reproducible** — the same
inputs give the same outputs — and to **localize the source** when they don't. Run
these whenever the model/inference path changes or when moving to a **new cluster**
(e.g. ALCF **Polaris**, A100-SXM4-40GB).

They use the **1g+0X selection** as the test analysis (`../single_photon/`,
`run_stageB_capped.sh` + `analyze_1g0X.py`) rather than duplicating it. The full
story of what these tests found is in `pointcept/docs/LArFormer_Reproducibility.md`.

## Background: three layers of reproducibility

| Level | Question | Control |
|-------|----------|---------|
| **Same run** | same input list + same GPU → same output? | `--deterministic` (env `DETERMINISTIC=1`): TF32 off, deterministic cuBLAS/cuDNN, seeds, `CUBLAS_WORKSPACE_CONFIG`. **Required.** |
| **Cross-membership** | does an event's output depend on which *other* events are in the run? | Two train-time RNG augmentations were leaking into inference and broke this — **now fixed** (see below). |
| **Cross-GPU** | different GPU model → same output? | Bit-exact within a **driver + library + GPU-architecture family**; Hopper diverged from Ampere/Ada. Treat as a conformance allowlist, gated by these tests. |

**The two membership bugs (fixed, for reference):** both were train-time augmentations
that call `torch.randperm` per event and were **not gated by eval mode**, so at
inference each consumed the global RNG and made an event's output depend on its
predecessors:
1. `shuffle_orders=True` on the deghoster PTv3 — fixed in the config + a defensive
   disable in `tools/run_larformer_stage3_inference.py` (`run_full_cascade_mode`).
2. `CrossLevelAttn.max_source_tokens_per_level` token subsample — fixed by gating
   `_maybe_subsample` on `not self.training` (`pointcept/models/LArFormer/refiners/cross_level.py`).

A fresh checkout already has these fixes; these tests **confirm** them and catch any
regression or new cluster-specific issue.

## The tests

All SLURM scripts pin/parametrize via `--nodelist` and env vars; outputs land under
`workdir/` (git-ignored) and logs under `slurm/logs/`.

### 1. Same-GPU determinism (`slurm/submit_determinism_test.sh`)
Runs the cascade 4× on one GPU (2× default, 2× `--deterministic`) and diffs each pair.
Confirms `--deterministic` gives **bit-exact** run-to-run.
```bash
sbatch --nodelist=<node> slurm/submit_determinism_test.sh
# PASS: "DET a vs b" reports 0 everywhere. (Default mode will show flips — that's expected.)
```

### 2. Membership / list (`slurm/submit_membership_test.sh`) — the key one
Runs the cascade on a small "probe" list and a larger superset, diffs the common probe
events. An event's output must **not** depend on its batch-mates.
```bash
sbatch --nodelist=<node> slurm/submit_membership_test.sh      # default 30 vs 200 events
# PASS: drop-flag / 1g0X-label / coord-mismatch / per-SP all 0.
# Stronger: pad with LARGE events (high spacepoint count) — the bug showed up worst
#   with ~800k-point padding. Build a custom padded.txt and set WORKDIR.
```
The diff matches events by **(run,subrun,event)** (`determinism_diff.py --by-rse`,
read from `meta/*`), pairing probe/padded events by identity rather than list
position. Besides the two config-level train-RNG leaks this first caught
(`shuffle_orders`, `max_source_tokens_per_level`), the **forward** also consumes
per-event RNG — a single startup seed only gives same-*order* repeatability, so
the inference tools (`run_larformer_stage3_inference.py` cached + full-cascade
modes, and `run_larformer_keypoint2_cascade_inference.py`) now **re-seed before
each event** (`reseed_per_event()`, `--deterministic` only) so an event is
independent of its batch-mates. This membership test is the regression guard for
that. See `docs/LArFormer_Reproducibility.md` §3.2.

### 3. Cross-GPU conformance (`slurm/submit_capture.sh` + `cross_gpu_diff.py`)
Capture the per-event decision tensors on two nodes/SKUs, then diff. Use to clear a
new GPU type for production.
```bash
D=$(pwd)/workdir/xgpu
sbatch --nodelist=<nodeA> --export=ALL,CAPTURE_DIR=$D/A slurm/submit_capture.sh
sbatch --nodelist=<nodeB> --export=ALL,CAPTURE_DIR=$D/B slurm/submit_capture.sh
# after both finish (in the container):
python3 cross_gpu_diff.py $D/A $D/B
# PASS (conforming): keep-flips 0, slice/class flips 0, 0 drop-flips.
# Reports per-stage divergence + margin analysis if not.
```
> On Polaris (A100-SXM4-40GB): A100 was bit-identical to our A100/L40S family in
> testing, so expect a pass. If it diverges, it's a different driver/architecture —
> pin a conforming family and/or quote the shift as a systematic.

### 4. Localize a failure (Tier-B) — `slurm/submit_capture_layers.sh` + `layers_diff.py`
If a membership or cross-GPU test fails, capture per-stage feature fingerprints on the
two lists/GPUs and find the **first diverging stage**.
```bash
D=$(pwd)/workdir/layers
sbatch --nodelist=<node> --export=ALL,CAPTURE_DIR=$D/probe,EVENT_LIST=<probe>,TARGET=deghoster slurm/submit_capture_layers.sh
sbatch --nodelist=<node> --export=ALL,CAPTURE_DIR=$D/padded,EVENT_LIST=<padded>,TARGET=deghoster slurm/submit_capture_layers.sh
python3 layers_diff.py $D/probe $D/padded        # TARGET=slicer to hook the slicer chain
# The first stage with a large rel-diff is the divergence onset (everything upstream is clean).
```
This is how both membership bugs were found: divergence entered at the deghoster
attention (→ `shuffle_orders`) and the slicer `token_refiner` (→ `max_source_tokens`).

## Files
| File | Purpose |
|------|---------|
| `capture_cascade_tensors.py` | dump per-event decision tensors (P(real), slice/particle preds) |
| `capture_deghost_layers.py` | dump per-stage order-invariant fingerprints (Tier-B); `--target deghoster|slicer` |
| `cross_gpu_diff.py` | diff two capture dirs: keep-flips, slice/class flips, margin analysis |
| `determinism_diff.py` | diff two stage3pred dirs (drop-flag / 1g0X-label / per-SP); used by determinism + membership tests. `--by-rse` matches events by (run,subrun,event) from `meta/*` instead of filename (order/subset robust) |
| `layers_diff.py` | diff two layer-fingerprint dirs → divergence onset stage |
| `deadband_offline.py` | (investigation record) offline dead-band evaluation — a fix that was *ruled out* |
| `slurm/submit_determinism_test.sh` | test 1 (same-GPU) |
| `slurm/submit_membership_test.sh` | test 2 (membership/list) |
| `slurm/submit_capture.sh` | test 3 capture (cross-GPU) — env: `CAPTURE_DIR`, `EVENT_LIST`, `DETERM`, `DEGHOST_FP64`, `DEGHOST_NOSHUFFLE` |
| `slurm/submit_capture_layers.sh` | test 4 capture (Tier-B) — env adds `TARGET` |

## Dependencies (not duplicated here)
- `../single_photon/run_stageB_capped.sh` — the cascade runner (test analysis).
- `../single_photon/analyze_1g0X.py` — `reco_label` (used by `determinism_diff.py`).
- `../single_photon/workdir_scale/cascade_inputs_1g0X.txt` — the test event list.
- `../../larformer_scripts/larformer_configs/single_photon_scale1500.conf` — checkpoints/paths.
- `pointcept/tools/run_larformer_stage3_inference.py` — `set_deterministic` + cascade helpers (imported by the capture scripts).
- Container `pointcept_cuml.sif`; the `ubdl` env. **Edit the hard-coded paths at the top of each `slurm/*.sh` for a new cluster.**

## New-cluster checklist (e.g. Polaris)
1. Update paths/partition/`--nodelist` in `slurm/*.sh` and the `.conf`; confirm the container + `ubdl` env work.
2. **Test 1** (determinism) on one node → must be bit-exact in `--deterministic`.
3. **Test 2** (membership) → must be all-0 (confirms the RNG-gating fixes survived the move).
4. **Test 3** (cross-GPU) between the GPU types you'll deploy on → clear each into the allowlist.
5. If 2 or 3 fail → **Test 4** to localize, then check for any new train-time augmentation not gated by eval.
