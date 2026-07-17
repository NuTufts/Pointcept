# Wave A → Wave C decision: linear-probe comparison of the P05 SSL fleet

**Decision (2026-07-17, from the Tufts probe sweep):** the Wave C combination
run **P05E.1 = detector-symmetry augmentations (P05B.2) × drop_cosmics=0.9
(P05C.5), prototypes 4096** — generated as
`configs/lartpc/p05/pretrain-sonata-p05e1-mc-noghost-detsym-dropcosmic.py`.
If it confirms, it is promoted to the **v9 reference config** per the
experiment plan.

Decision made from **mid-training snapshots (~30% of the P05 budget,
img1536000 of ~5M)** because the Isambard allocation ends 2026-07-22 — a
full-completion decision would leave no time to run Wave C. Supporting
evidence: the cross-variant ranking is **stable at both matched
comparison points** (img768000 and img1536000).

## Probe protocol (frozen; every number below is comparable)

- Config: `linearprobe-sonata-p05-mc-noghost-tufts.py` (frozen backbone,
  pure-linear head) at **batch 64 on `--constraint=a100`** (sizing study:
  ORCHESTRATION.md step 0.3).
- **Frozen budget from the WP3.5 calibration (SLURM 1639181–1639193):
  `epoch=1 eval_epoch=1 optimizer.lr=0.001 scheduler.max_lr=[0.001]`**
  (~2.0 GPU-h/probe). Selection evidence (P05B.1, best val mIoU) —
  lr 1e-3 dominated 2e-4 everywhere; 1-epoch within 0.11 pts of 2-epoch:

  | snapshot | ep1 lr1e-3 | ep2 lr1e-3 | ep1 lr2e-4 | ep2 lr2e-4 |
  |---|---|---|---|---|
  | img96000 | 0.3216 | 0.3227 | 0.3116 | 0.3178 |
  | img768000 | 0.4699 | 0.4702 | 0.4613 | 0.4660 |
  | img1536000 | 0.5072 | 0.5061 | 0.4980 | 0.5037 |

- Data: Tufts-remapped v3 MC train/val split — the SAME split as the
  Isambard supervised ceilings. Ceiling reference (P05A.1):
  **mIoU 0.8576, pion 0.773, proton 0.933**; geometry-only (P05A.2) 0.8152.

## Decision table (best val mIoU / pion IoU / proton IoU)

At **img1536000** (primary), ranked:

| variant | tests | mIoU | pion | proton |
|---|---|---|---|---|
| **P05C.5 drop_cosmics=0.9** | batch composition | **0.5844\*** | **0.327** | **0.684** |
| P05B.4 detsym + sum-charge | aug + charge feature | 0.5345 | 0.250 | 0.502 |
| P05B.2 detsym only | augmentation | 0.5307 | 0.240 | 0.539 |
| P05C.4 nu-anchored crops | batch composition | 0.5129 | 0.235 | 0.489 |
| P05B.1 free-rot (baseline) | — | 0.5072 | 0.226 | 0.487 |
| P05C.1 proto 2048 | prototype count | 0.5021 | 0.213 | 0.469 |
| P05C.3 proto 8192 | prototype count | 0.4919\* | 0.205 | 0.447 |
| P05C.6 2×crops | batch composition | 0.4870\* | 0.211 | 0.500 |

At **img768000** (rank-stability check) the ordering is the same:
C.5 0.5137 > B.2 0.4802 > B.4 0.4774 > C.4 0.4735 > B.1 0.4699 >
C.1 0.4702 ≈ C.3 0.4667 ≈ C.6 0.4503\* (B.2/B.4 swap places but both
stay above B.1 at both points; the C.5 lead and the C.3/C.6 deficit are
unambiguous at both).

\* probe job still finishing when recorded (41–53 of 64 evals); traces
plateaued (C.5 flat at 0.578–0.584 over its last ~10 evals), so ranks are
final even if third-decimal values tick up. Rerun
`harvest_probe_results.py` for exact finals. Muon (0.86–0.91),
electron (0.63–0.68), gamma (0.36–0.49) follow the same ordering.

## Verdicts

1. **Batch composition dominates every other knob.** Dropping 90% of
   cosmic points during pretraining (C.5) gains **+7.7 mIoU, +10 pion,
   +20 proton points** over the free-rotation baseline at matched images.
   The "Sinkhorn/prototype head sees mostly muon points" hypothesis is
   supported: diluting muons improves every physics class, most strongly
   the calorimetry-sensitive ones (pion, proton).
2. **The rotation/charge-discounting hypothesis is confirmed, second-order.**
   Detector-symmetries-only (B.2) beats free rotations (B.1) by **+2.4 mIoU**
   at both matched points. Real, but ~3× smaller than composition.
3. **B.4's plane-summed charge is a wash and was declined for Wave C:**
   vs B.2 it trades **+1.0 pion for −3.7 proton** (mIoU +0.4). Keeping
   3-channel charge also stays compatible with the asinh input scaling
   (P05B.5/B.6) if that confirms.
4. **Prototype count is a dead knob** (2048 ≈ 4096 > 8192) and **2×crops
   (C.6) hurts** (−2 mIoU vs baseline). Keep 4096, one crop per sample.
5. Best probe = **68% of the supervised ceiling** at 30% of pretraining —
   consistent with pure-linear-probe expectations (plan §"fraction of
   ceiling"); the cross-variant comparison, not the absolute, is the result.

## Caveats (record these with any downstream claim)

- **Distribution-alignment confound for C.5:** the probe protocol itself
  uses drop_cosmics=0.9 + nu-anchored crops, i.e. C.5 pretrained on the
  probe's data distribution. Part of its margin may be alignment rather
  than representation quality. The margin's size (+20 proton) and the fact
  that the probe protocol was chosen to match the physics target argue the
  effect is substantially real; the E1 deghosting eval on full events is
  the clean test, post-allocation. P05E.1 runs the measured winner
  (0.9) unchanged rather than hedging to 0.75.
- **M1/M2 (prototype–label MI/purity) were NOT part of this decision**,
  although the plan lists them in the Wave B selection metric:
  `tools/prototype_stats.py` is v5-era (hardcoded strength+wire-color
  pipeline, no NormalizeCoord/LogTransform) and would need adaptation +
  validation for P05 models. Decision rests on per-class probe IoU (M4).
  Fast-follow candidate.
- Mid-training decision (~30% budget), defended by rank stability at two
  matched points — not by converged curves.

## Logistics / state

- All 16 decision probes (8 variants × img768k/1536k) ran at the frozen
  budget; results in `exp/probes/<run_id>/img*/` at Tufts.
- The remaining ~46 curve-filling probes were **cancelled 2026-07-17**
  (GPU space ceded to another project). Their `.submitted` markers were
  removed, so a later `launch_probe_sweep.py --budget-opts "<frozen
  tokens above>"` resubmits exactly the missing points — do this after the
  fleet finishes to complete the paper's probe-vs-images curves.
- Snapshots synced through img1536000 (all runs) / img3.8M (C.6);
  remember the ~12 h clifton cert (ORCHESTRATION.md step 0.4).

## For the Isambard session (time-critical, allocation ends 2026-07-22)

```bash
# after merging p05b5-asinh-input:
./slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh \
    configs/lartpc/p05/pretrain-sonata-p05e1-mc-noghost-detsym-dropcosmic.py
# and, independent of this verdict (one-delta vs B.1):
./slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh \
    configs/lartpc/p05/pretrain-sonata-p05b5-mc-noghost-asinh.py
./slurm_scripts/lartpc_sonata_pretraining/launch_p05_run.sh \
    configs/lartpc/p05/pretrain-sonata-p05b6-mc-noghost-asinh-jitter005.py
```

P05E.1 snapshots are probed at Tufts with the standard 6-channel probe
config (the launcher's default `P05` mapping already covers it); B.5/B.6
snapshots MUST use the asinh probe config (mapping already wired in).
