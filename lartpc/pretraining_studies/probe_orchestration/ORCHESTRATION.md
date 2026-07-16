# P05 probe sweep — orchestration plan (for a Claude Code session at Tufts)

**Goal:** turn the Isambard SSL fleet's weights-only snapshots into
representation-quality curves — linear-probe val mIoU (and per-class IoU,
esp. pion/proton) vs images seen, per training variant — measured against
the P05A.1 supervised ceiling (mIoU 0.8576, pion 0.773, proton 0.933) on the
same matched train/val split.

**State on the Isambard side (2026-07-17):** 8 SSL runs training
(P05B.1, B.2, B.4, C.1, C.3, C.4, C.5, C.6), each writing ~12 log-spaced
weights-only snapshots (751 MB each) to `sonata/p05/<run_id>/snapshot/`,
completing ~July 18–19. ~50 snapshots exist already. B.5/B.6 (asinh) join
later from the `p05b5-asinh-input` branch.

**Connectivity:** Tufts is behind a VPN — ALL transfers originate at Tufts
via ssh/rsync to Isambard (alias assumed `isambard`; override with
`ISAMBARD_SSH=<alias>`). Nothing here pushes to Isambard.

## Files in this directory

| file | runs where | purpose |
|---|---|---|
| `sync_from_isambard.sh` | Tufts | pull new snapshots + run configs + registry (idempotent) |
| `submit_probe_tufts.sbatch` | Tufts SLURM | one probe job (GPU, container) |
| `launch_probe_sweep.py` | Tufts | submit probes for all unprobed (run, snapshot) pairs |
| `harvest_probe_results.py` | Tufts | logs → probe_curves.csv + plots + summary.md |

## Step 0 — prerequisites (once)

1. `git pull` (this directory plus updated probe configs must be present).
2. Verify the matched lists exist:
   `lartpc/filelists/h5list_v3_mc_only_{train,val}_tufts.txt`. The probe
   configs now point at them — probe numbers are only ceiling-comparable on
   this split. If the val list is missing, stop and run the remap task first.
3. GPU + batch size are SETTLED (PI decision 2026-07-16, measured on an
   A100-80G, SLURM job 1632758): all probes run at **batch 64 on
   `--constraint=a100`** (already set in `submit_probe_tufts.sbatch` and the
   generated probe configs). Rationale: B=64 needs 7.8/14.2 GiB
   alloc/reserved so it fits BOTH A100 variants (40G ~18 GPUs + 80G 40
   GPUs), and throughput is nearly flat in batch size (103 vs 125
   samples/s at B=288 — a probe epoch is ~1.1 h either way) because even
   64-crop batches are ~0.5M points. The probe head is a pure nn.Linear
   (no BatchNorm), so batch size cannot affect model statistics — but the
   step-2 LR calibration MUST be done at batch 64 and frozen with it.
   Do NOT mix architectures within the comparison set: L40S (Ada) or
   H100/H200 (Hopper) runs are fine only as a wholesale replacement, never
   as overflow alongside A100 probes. `mkdir -p logs` here BEFORE the
   first sbatch (SLURM opens --output pre-script; missing dir = exit 53).
4. Confirm the ssh alias works: `ssh isambard hostname`.

## Step 1 — sync (repeat daily / per loop)

`./sync_from_isambard.sh` — pulls every run's `snapshot/*.pth` and frozen
`config.py`, plus the Isambard registry to `exp/registry_isambard.csv`.
First invocation moves ~40 GB; later ones only the new points.

## Step 2 — probe-budget calibration (WP3.5, once, BEFORE the mass sweep)

The budget must be frozen once and reused for every probe, or the curves
are not comparable. Grid (12 jobs, use `--runs P05B.1 --max-submit 12` with
hand-edited budget tokens, or submit directly):

- snapshots: B.1 at img96000 (early), img768000 (mid), the latest synced
- budgets: `epoch=1 eval_epoch=1` vs `epoch=2 eval_epoch=2`
- LR: default 2e-4 vs 1e-3
  (`optimizer.lr=0.001 "scheduler.max_lr=[0.001]"` — BOTH must be overridden;
  top-level `base_lr` is already resolved at parse time and does nothing)

Selection rule (from the implementation plan WP3.5): smallest budget where
each snapshot's best mIoU is within ~0.5 points of its 2-epoch value AND the
early/mid/late *ranking* is stable. The eval_freq=100 traces in each
train.log show convergence directly. Record the frozen choice in
`PROBE_BUDGET.txt` in this directory (git-ignored *.txt is fine — also put
it in the summary.md and the final report).

## Step 3 — the sweep (repeat after each sync)

```bash
python3 launch_probe_sweep.py --repo $LOCAL_REPO \
    --budget-opts "<frozen tokens from step 2>" --max-submit 50
```

- Idempotent (`.submitted` markers). ~96 probes total when the fleet
  finishes; pace with --max-submit to be polite on the GPU partition.
- The launcher maps run → probe config automatically:
  default 6-channel config; **P05C.1/C.3 get head_num_prototypes overrides**
  (checkpoint head sizes must match or loading raises size-mismatch);
  **P05B.4 uses the sumcharge config** (in_channels=4, ChannelReduce, no u/v
  swap). B.5/B.6 lines are present but commented until the asinh branch
  merges — uncomment then.
- Failure triage: a probe dir with `.submitted` but no `.done` and a dead
  job → check `logs/p05probe.<jobid>.log`; delete the `.submitted` marker to
  resubmit after fixing.

## Step 4 — harvest + report (repeat after probes land)

```bash
python3 harvest_probe_results.py --repo $LOCAL_REPO
```

Produces `exp/probes/analysis/{probe_curves.csv,summary.md}` and two PNGs
(mIoU curves; pion/proton panels with ceiling lines). Commit the CSV and
summary.md (not PNGs — gitignored) so the Isambard session can read them;
push to `nutufts_isambard` (analysis files only — no code changes on this
branch from Tufts while the asinh work lives on its own branch).

## What the curves are for (analysis framing)

- **Selection metric** for Wave C per the experiment plan: per-class probe
  IoU with special weight on pion/proton (+ prototype-label MI when the
  M1/M2 tooling lands) — not mIoU alone, not t-SNE.
- **Fraction of ceiling (M6):** probe mIoU / 0.8576. PANDA-analog
  expectations: ~0.90–0.95 for probe+FT, LOWER for a pure linear probe —
  don't be alarmed by absolute values; the comparison across variants is
  the point.
- **Key contrasts:** B.2 vs B.1 (detector-sym vs free rotations — the
  charge-discounting hypothesis); B.4 vs B.1 (plane-summed charge); C.1/C.3
  vs B.1 (prototype count); C.4/C.5/C.6 vs B.1 (batch composition); later
  B.5 vs B.1 (asinh) and B.6 vs B.5 (jitter strength).
- Check the **pion panel** specifically: the supervised charge delta was
  +0.114 pion IoU — the variant that recovers more of it is the v9 candidate.

## Cost estimate

~96 probes × (budget from step 2; expect 1–3 GPU-h each at epoch=1) — well
within the ~1000 node-hour reserve. Calibration adds ~12 jobs.
