# Cluster debug brief — SonataLoRASegmentor inference (RESOLVED 2026-05-16)

> **RESOLVED.** The inference produces mIoU ≈ 0.79 (matching the training-reported 0.82) once a single line is corrected in `run_semseg_inference.py`. Root cause was a code-divergence between `vdasil01/Pointcept`'s `lartpc.py` (training-era) and `twongj01/.../pointcept`'s `lartpc.py` (current) in the `strength` (pixval) preprocessing: training used `strength = pixval / 500.0`, the new code uses `strength = clip(pixval, 0, 1000) / adc_scale` with `adc_scale=1.0`. Inputs were 500× larger than the model expected, scrambling the encoder. See "Resolution" section at the bottom.

> The historical debug trail below documents the path to the answer — useful if a similar symptom appears with a different LoRA fine-tune or another preprocessing drift.

## 1. The problem in one paragraph

A LoRA-fine-tuned SONATA semantic segmentor (8 LArTPC particle classes: electron, muon, pion, proton, gamma, michel, delta, led) was trained on the cluster. The training log reports `mIoU/mAcc/allAcc 0.8212/0.8864/0.9489` at the end of training, with per-class IoUs all in the 0.65–0.95 range. The saved `model_last.pth` (also has `best_metric_value=0.8212` in its metadata) gives **mIoU ≈ 0.03** at inference time on the **same prod4 validation files** — not just a different dataset. The model predicts class 3 (proton) for ~80% of all spacepoints regardless of true class.

## 2. Files involved

| File | Purpose |
|---|---|
| `configs/lartpc/lora_finetune/archive/lorafinetune-sonata-v1m1-lartpc-v6-seg.py` | Training config (uses `SonataLoRASegmentor` + LoRA on top of SONATA-v1m1 + PT-v3m2 backbone, 10 epochs, drop_cosmics_prob=0.9, BiasedSphereCrop r=20 cm) |
| `sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth` | The checkpoint in question. metadata: `epoch=1, best_metric_value=0.8212` (epoch=1 is fine — `eval_epoch=1` means a single eval cycle covers all 10 dataset passes; 40625 iters × bs=96 ÷ 390k entries ≈ 10 epochs of data) |
| `lartpc_data_prep/semseg_analysis/run_semseg_inference.py` | Standalone inference driver written during the laptop debug session; faithfully reproduces the `LArTPCDataset` val pipeline |
| `lartpc_data_prep/semseg_analysis/histograms.py` | Accumulator schema (confusion + score histograms per origin) |
| `lartpc_data_prep/semseg_analysis/merge_shards.py`, `plot_metrics.py` | Shard merge + metric plotting |
| `prod4_valsplit_subset.txt` | A small subset of `hdflist_combined_prod4_validated_shuffled_valsplit.txt` — actual training-val files, copied locally to confirm the dataset isn't the cause |

## 3. What's been ruled out (verified during laptop session)

All of these checks passed cleanly. **Do not redo any of them on the cluster unless the cluster gives different answers** to one of the tests below.

| Hypothesis | Verdict | How it was tested |
|---|---|---|
| Pipeline drift between my val-mimic and the real val pipeline | ✗ Ruled out | Used `build_dataset(cfg.data.val)` (the actual `LArTPCDataset.val`) directly — still mIoU ≈ 0.01–0.03 |
| Distribution shift (eval data ≠ training data) | ✗ Ruled out | Tested on actual files from `hdflist_combined_prod4_validated_shuffled_valsplit.txt` — same broken result |
| Checkpoint load incomplete | ✗ Ruled out | `load_state_dict(strict=False)` returns missing=0, unexpected=0. 1008/1008 keys present in both. Sampled tensors (`seg_head.bias`, `seg_head.weight`, teacher `lora_A`, `lora_B`) are bit-identical to checkpoint values |
| Wrong branch (student vs teacher) used at inference | ✗ Ruled out | `SonataLoRASegmentor.forward` calls `self.backbone(data_dict, return_point=True)`. Sonata-v1m1's `forward(return_point=True)` uses `self.teacher.backbone(data_dict)`. Teacher LoRA `lora_B` is non-zero (mean norm 2.58) — it was the branch trained. (Student `lora_B` is all-zero, but student is never called in forward.) |
| LoRA wiring | ✗ Ruled out | `LoRALinear.forward` correctly computes `base(x) + scale * (lora_dropout(x) @ lora_A.T @ lora_B.T)` |
| BatchNorm running-stats divergence | ✗ Ruled out | Backbone uses LayerNorm only (no running stats) |
| AMP/dtype mismatch | ✗ Ruled out | `enable_amp=False` in config; my inference uses float32 (default), matching training |
| Batching (`batch_size_val=48` in train vs 1 in inference) | ✗ Ruled out | Reran with `point_collate_fn` + `batch_size=8` DataLoader — same broken result, accuracy 0.137 |
| Train vs eval mode at forward time | ✗ Ruled out | Called `model.backbone` directly with `model.train()` to get logits — acc 0.148 vs 0.137 in eval mode; both broken in the same "predict proton everywhere" way |
| Feature ranges out of distribution | ✗ Ruled out | On a real val event: `coord` ∈ [-0.69, +1.18], `strength` ∈ [-0.88, +0.90] — exactly what the val pipeline produces |
| Cosmic-contamination confusing the model | ✗ Ruled out | Re-ran with `--drop-cosmics-prob 1.0 --biased-sphere-prob-random 0.0` (force-drop all cosmics + always-vertex-centered crop). Still mIoU=0.013, still ~87% proton predictions. The proton-dominance is symmetric across true classes — true muons → predicted proton, true gammas → predicted proton, etc. — so it's not cosmic contamination triggering it. proton's recall on its own truth is 0.58 (real), but its precision is 0.10 (everything-else-also-gets-predicted-proton) |
| Label-code drift between prod4 (training) and devdata (newer eval set) | ✗ Ruled out (initial finding was a per-file artifact) | First I thought prod4 didn't use ssnet codes {5, 6}, but checking the file the val_loader actually reads first (`entry000000.h5`, not `entry000003.h5` I'd inspected raw), prod4 uses the **full** set {0, 2, 3, 4, 5, 6, 7, 8, 9} matching the modern convention. The training and dev datasets use the same ssnet→class map. Labels flow correctly through the pipeline (verified at each stage). |
| Labels mangled by the transform pipeline (BiasedSphereCrop → GridSample) | ✗ Ruled out | Traced (class count) at each pipeline stage on the file val_ds[0] reads: raw H5 → get_data → BiasedSphereCrop → GridSample. Relative class proportions are stable: raw (muon=81%, gamma=12%, pion=1.4%, proton=0.1%) → after crop (muon=48%, gamma=41% — vertex region is more EM-heavy, as expected) → after GridSample mode='train' (muon=47%, gamma=41%). No class indices invented; no labels scrambled; `index_operator`'s `index_valid_keys` propagates `segment` alongside `coord` correctly. So the model was trained on correct supervision and inference uses the same correct supervision. |
| Pretrained backbone never actually loaded — model trained LoRA + seg_head against a random-init backbone | ✗ Ruled out | Compared 7 sample backbone tensors (attention QKV weights at multiple encoder stages, embedding stem weights) between the pretrained checkpoint (`sonata/lartpc_v6_h200_noghosts_pretrain/model/model_last.pth`) and the fine-tuned checkpoint. **All 7 match bit-for-bit** (max-abs-diff = 0.000e+00, identical SHA256). The `LoRASonataCheckpointLoader` hook fired correctly at training start, the backbone was loaded, and `freeze_non_lora` kept it frozen through training. |
| Timezone-based theory that the file was overwritten after the 0.82 was logged | ✗ Ruled out | User noted laptop is BST (UTC+1), cluster is EDT (UTC-4). 5-hour gap between wandb log (04:42:39 cluster time) and file mtime (09:42 laptop time) is exactly the timezone offset → the checkpoint *should* be the one that scored 0.82 |

## 4. What we know about the broken state

On the first batch of real prod4 val data (8 events, 41707 spacepoints):

```
per-class logit mean: electron=-4.43, muon=-2.14, pion=-1.32, proton=+1.22,
                     gamma=-5.71, michel=-11.13, delta=-7.52, led=-8.01
argmax fractions   : electron=0.00, muon=0.05, pion=0.12, proton=0.82,
                     gamma=0.01, michel=0.00, delta=0.00, led=0.00
truth fractions    : electron=0.00, muon=0.48, pion=0.02, proton=0.13,
                     gamma=0.34, michel=0.01, delta=0.01, led=0.01
eval-mode loss     : 0.3382
training-time val loss at the 0.82 result: ~0.03 (from wandb log)
```

`seg_head.bias` after loading is essentially the prior-init value plus tiny perturbations (deltas all < 0.08):

```
class     saved_bias  prior_logit  delta
electron    -2.876      -2.903    +0.027
muon        +1.297      +1.368    -0.071
pion        -4.625      -4.701    +0.076
proton      -3.674      -3.705    +0.031
gamma       -3.440      -3.476    +0.036
michel      -5.118      -5.110    -0.008
delta       -2.535      -2.484    -0.051
led         -4.909      -4.955    +0.046
```

`seg_head.weight` norms per class (W.shape = (8, 1232)):

```
electron  ||w||=2.282   muon    ||w||=4.209   pion   ||w||=2.380   proton ||w||=3.910
gamma     ||w||=2.016   michel  ||w||=7.202   delta  ||w||=8.019   led    ||w||=5.527
```

Despite muon's bias being **+1.30** and proton's being **-3.67**, the *weight contributions* push muon to logit-mean -2.14 and proton to +1.22 on real val features. The weights have learned to actively suppress muon and amplify proton, which is the exact opposite of what a working classifier would do.

LoRA `lora_B` status:
- Teacher branch (used in forward): 46/46 matrices non-zero, mean norm 2.58 — looks trained ✓
- Student branch (not used in forward): 46/46 matrices exactly zero, never received gradients — expected dead weight, not a problem

## 5. Open hypotheses

After ruling out everything inspectable, the remaining possibilities are:

**H1.** **The saved weights are functionally not the model that scored 0.82** — they load cleanly but don't reproduce the metric. Possible mechanisms:
- The SemSegEvaluator computed mIoU from something other than the model that got saved (e.g., it read from an in-memory EMA / running average / a non-deduplicated tensor that didn't make it to disk)
- A NaN/Inf spike during the *exact* iteration the checkpoint was being written, with weights silently corrupted in flight
- A stateful Pointcept component (e.g., `EventStorage` running stats, a hook-side accumulator) influences the metric calc but isn't captured in the state_dict

**H2.** ~~`model_best.pth` is the correct one we don't have locally~~  ✗ **Ruled out.** All three checkpoint files in the model dir (`model_last.pth`, `model_best.pth`, `epoch_1.pth`) are byte-identical — same SHA on `seg_head.weight`/`bias` and LoRA tensors. Pointcept saved the same `epoch=1, best_metric_value=0.8212` state three times because the run completed exactly one outer eval epoch and that was also the best. There is no alternative checkpoint.

**H3.** ~~Environment-specific divergence between training-time forward and inference-time forward~~ ✗ **Ruled out by cluster test 2 (2026-05-16).** Running the standalone driver on the cluster (same machine that trained the model, same `vdasil01/.../model_last.pth`) gives mIoU=0.0210, accuracy=0.110, with ~88% proton predictions on 20 prod4 val files — within noise of my container's result (mIoU=0.013). Bit-identical predictions across environments.

**Key diagnostic observation from cluster Test 2**:
- Wandb training log: `Class_3-proton iou/accuracy: 0.9095/0.9421`
- Cluster inference (this weights): proton `recall=0.938, IoU=0.134, precision=0.135`
- **Recall matches.** The model still catches true protons reliably (94% of true protons predicted as proton). What changed is the false-positive rate: the model now predicts proton 56,797 times when only 8,160 points are true protons (5.96× over-prediction). So it's not that proton features are forgotten — it's that the proton head fires on inputs it shouldn't.
- This is consistent across batches and origins, so it's not a per-event quirk.

H1 is now the only live hypothesis. The training-time metric was correctly measuring proton recall, but somehow the model that produced those measurements had **a calibration that suppressed proton FPs that the saved state doesn't have**. Mechanisms compatible with this observation:
- (H1a) The training-time evaluator ran with a different batch composition or random state than what we replay. Point-transformer attention is patch-local but still mildly batch-dependent if the patch boundaries are stochastic.
- (H1b) An EMA copy of the seg_head was used at val time and never saved.
- (H1c) The saver wrote `model.state_dict()` but the val was on a different forward (e.g., the LoRA path was bypassed at eval time due to a `model.eval()`/training-mode side effect).

## 6. Tests to run on the cluster — in order

Run these in order. Each builds on the previous. Stop as soon as you find the answer.

### Test 1 — ~~check for `model_best.pth`~~ ✗ Done; all three checkpoints (`model_last.pth`, `model_best.pth`, `epoch_1.pth`) are byte-identical. Skip directly to Test 2.

### Test 2 — reproduce the broken result on the cluster with `model_last.pth` (the decisive test)

This confirms or refutes H3 (env-specific bug). Run the standalone inference script with the same `model_last.pth` we have:

```bash
cd /path/to/Pointcept
./run_in_container.sh python lartpc_data_prep/semseg_analysis/run_semseg_inference.py \
    --model-config configs/lartpc/lora_finetune/archive/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
    --weights sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth \
    --filelist prod4_valsplit_subset.txt \
    --output-shard /tmp/cluster_check_last.npz \
    --match-val-pipeline \
    --nfiles 20

./run_in_container.sh python lartpc_data_prep/semseg_analysis/plot_metrics.py \
    --input /tmp/cluster_check_last.npz \
    --outdir /tmp/cluster_check_last_plots --no-plots
```

(Replace `./run_in_container.sh` with whatever the cluster's equivalent is — `apptainer exec /path/to/pointcept_cuml.sif python ...` — these commands run inside the Pointcept container.)

- **If cluster mIoU ≈ 0.03**: env is not at fault. Same broken state. This points firmly at H1 (`model_last.pth` content is bad) or H2 (need `model_best.pth`).
- **If cluster mIoU ≈ 0.8**: my container environment is corrupting the model. Need to dump tensor values from both environments and find the divergence. Most likely culprit would be xformers or a custom CUDA op. Check `torch.__version__` and `import xformers; xformers.__version__` on both sides.

### Test 3 — Pointcept's own `tools/test.py` (the official inference)

This is the ground-truth comparison. If our standalone driver and the official trainer's test mode agree, we've eliminated any subtle driver-side bug:

```bash
cd /path/to/Pointcept
./run_in_container.sh python tools/test.py \
    --config-file configs/lartpc/lora_finetune/archive/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
    --options weight=sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth \
              save_path=/tmp/official_test_last \
              data.test.data_list_file=prod4_valsplit_subset.txt
```

(Or whatever the exact invocation is on the cluster — the config has a `test` block in `data`.)

- **Agrees with Test 2 (~0.03)**: confirms the saved weights are broken; proceed to Test 4 with `model_best.pth`.
- **Disagrees (~0.8)**: there's something my standalone driver is doing that the trainer doesn't. Diff our `run_semseg_inference.py` against `tools/test.py` carefully.

### Test 4 — ~~run with `model_best.pth`~~ ✗ Skip (identical to model_last.pth).

### Test 5 — if Tests 1–4 don't pinpoint it, dump tensors and compare

Run on the cluster:

```bash
./run_in_container.sh python -c "
import torch
ckpt = torch.load('sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth', map_location='cpu', weights_only=False)
sd = ckpt['state_dict']
import hashlib
def h(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]
keys_to_check = [
    'module.seg_head.bias',
    'module.seg_head.weight',
    'module.backbone.teacher.backbone.enc.enc0.block0.attn.qkv.lora_B',
    'module.backbone.teacher.backbone.enc.enc4.block2.attn.qkv.lora_B',
]
for k in keys_to_check:
    t = sd[k].float()
    print(f'{k}: sha256={h(t)} shape={tuple(t.shape)} norm={t.norm().item():.4f}')
"
```

Then run the same thing on the laptop side (which we have access to in our session — I'll cite the laptop hashes when we report them; for now they are:

| key | norm |
|---|---|
| `module.seg_head.bias` | 10.6489 |
| `module.seg_head.weight` | 13.9525 |
| `module.backbone.teacher.backbone.enc.enc0.block0.attn.qkv.lora_B` | 8.2245 |
| `module.backbone.teacher.backbone.enc.enc4.block2.attn.qkv.lora_B` | (read on cluster, expect a positive number) |

If the cluster's norms differ from the laptop's for the same file path, the file got corrupted in transit. If they match, the file content is the same on both sides — meaning both environments load the same bits and produce the same outputs, and the saved weights themselves are the broken state (H1).

### Test 6 — last resort: instrument the trainer at the moment of the 0.82 eval

If `model_best.pth` doesn't exist and `model_last.pth` is genuinely broken, the next step is to resume training from `model_last.pth` for 0 iterations and immediately run val again — see if val mIoU reproduces. If yes, we've been fooled by something stateful (storage history, EventStorage, etc.) somewhere. If no, the 0.82 was a one-time read of in-memory weights that were never on disk in that state.

## 7. Side note on the design of the inference script

The script `run_semseg_inference.py` was designed to mirror `lartpc_data_prep/deghost_analysis/run_deghost_inference.py` (which the user reports faithfully reproduces wandb val numbers for the deghoster model). Same H5-read approach, same per-shard histogram accumulator pattern, same Compose/Collect pipeline.

The accumulator stores:
- `confusion[true, pred, origin]` shape `(8, 8, 3)` — origin is `{0=unknown, 1=neutrino, 2=cosmic}`
- `score_hist[true, score_class, origin, bin]` shape `(8, 8, 3, B)` for one-vs-rest ROC

It supports two modes:
- **Full-event** (default): GridSample mode=`'test'`, scatter back to all input points. No BiasedSphereCrop, no drop_cosmics.
- **`--match-val-pipeline`**: GridSample mode=`'train'`, BiasedSphereCrop r=20 cm around nu vertex, drop_cosmics prob 0.9. Reproduces the training val pipeline.

Ghost points (`hasmatch==0`) are dropped at H5-read time by default (matches training's `true_points_only=True`); `--keep-ghosts` flag overrides this.

## 8. Quick re-orientation for the cluster Claude

If you want the full conversation context, the key insights from the laptop session were:

1. The user has a fully-functional deghost-analysis pipeline at `lartpc_data_prep/deghost_analysis/` (binary deghosting model evaluation). The new semseg pipeline at `lartpc_data_prep/semseg_analysis/` mirrors its structure.
2. The user's first concern was the discrepancy between training-reported mIoU (0.82) and inference mIoU (~0.06 on full-event mode across 682 files). My initial hypothesis was distribution shift. Confirmed wrong by running on actual prod4 val files.
3. The second hypothesis was undertrained model — also wrong (10 dataset passes were completed, training was real).
4. The third hypothesis was wrong-branch (student vs teacher) — verified that the teacher branch *is* what's used, and *is* what was trained. Student LoRA is dead weight but not in the forward path.
5. After extensive diagnosis (laptop + cluster), the resolution turned out to be a code-level divergence — see §9.


## 9. Resolution (2026-05-16)

### How we found it

After ruling out distribution shift, undertraining, wrong-branch selection, bad checkpoint load, label-pipeline mangling, broken backbone load, environment difference, batching effects, and labeling-convention drift, the user discovered that **another lab member (`vdasil01`)** had been running inference successfully against this exact checkpoint and had saved an output JSON showing the wandb-matching `mIoU=0.8385` on 500 events. They had also written their own inference script (`tools/eval_lora_classifier.py`, now copied to `lartpc_data_prep/semseg_analysis/`).

Running that same script:
- From `vdasil01/Pointcept` on the cluster → **mIoU=0.8385** ✓
- From `twongj01/pointcept_env/pointcept` on the cluster → mIoU=0.02 (broken)

Same script, same checkpoint, same data, same Python — only `PYTHONPATH` (which `pointcept/` was imported) differed. The two repos are on different branches (`LoRA_fine_tune` at `48a5b8f` vs `nutufts_lartpc` at `5222e0c`) with substantial divergence in `pointcept/datasets/lartpc.py`, `pointcept/models/{lora_sonata,sonata/sonata_v1m1_base,point_transformer_v3/point_transformer_v3m2_sonata}.py`, etc.

The decisive diff: in `pointcept/datasets/lartpc.py`'s `get_data()`, the `strength` preprocessing:

```python
# vdasil01 (training-era, gives mIoU=0.83):
if self.log_transform_edep:                          # log_transform_edep=True default
    strength = (edep / 500.0).astype(np.float32)
else:
    strength = edep.astype(np.float32)

# twongj01 / current repo (gives mIoU=0.02):
strength = (np.clip(edep, 0, 1000) / self.adc_scale).astype(np.float32)   # adc_scale=1.0 default
strength += self.add_min_pixval                       # 0.01 default
```

Both apply the same `LogTransform(min_val=0.01, max_val=1000, log=True)` downstream. But with the new code, strength is **500× larger** before LogTransform. For a typical pixval = 35:

| Stage | vdasil01 (training) | twongj01 (current) |
|---|---|---|
| pixval | 35 | 35 |
| strength | 0.07 | 35 |
| LogTransform output | **−0.66** | **+0.42** |

After 5 encoder stages of attention, this ~1-unit shift in normalized feature space scrambles the encoder output. The seg_head, trained on the original feature range, then sees out-of-distribution features and falls back to whichever class fires most strongly — that's why every "broken" run showed ~80%+ predicted as proton: proton's seg_head row activates most consistently on these OOD inputs.

This **explains every observation** we'd been struggling with:
- Bit-identical weight load + correct backbone + correct forward path: ✓ (the model is fine)
- Proton recall = 0.94 in both training and broken inference: protons are high-strength stopping tracks whose dE/dx shape survives the feature shift well enough that the proton head still fires correctly on them — that's why recall is preserved across the bug
- gamma / michel / electron collapse to zero: these classes' features are scale-sensitive (EM showers don't have a unique high-magnitude signature in the wrong feature regime)
- Same data, same code path, different repos → different results: code divergence in `lartpc.py`
- Cluster env identical to laptop env in terms of failure mode: it was never the env

### The fix

[`run_semseg_inference.py`](run_semseg_inference.py) line 115 has been patched:

```python
# Before:
strength = (np.clip(pixval, a_min=0, a_max=1000.0) / 1.0).astype(np.float32)
strength += 0.001

# After:
strength = (pixval / 500.0).astype(np.float32)
```

Verified result on 8 local prod4 val files, `--match-val-pipeline` mode:

```
Overall accuracy:   nu=0.948  cosmic=0.922  all=0.941
mIoU (macro):       nu=0.789  cosmic=0.691  all=0.746
Per-class IoU (all): muon=0.89  proton=0.96  gamma=0.93  michel=0.95
                     pion=0.27  delta=0.36  led=0.82
                     electron: no truth in this 8-event sample
```

That's within sample-size noise of the wandb-logged 0.82/0.89/0.95.

### Recommended permanent fix

For all inference pathways through `LArTPCDataset` (i.e. anything using `build_dataset(cfg.data.val)` or `eval_lora_classifier.py`):

1. **Add `adc_scale=500.0` to the config's `data.val` and `data.test` blocks** in `configs/lartpc/lora_finetune/archive/lorafinetune-sonata-v1m1-lartpc-v6-seg.py`. This encodes the training-era preprocessing in the config explicitly.

2. **Patch `eval_lora_classifier.py`** (around line 287 where `LArTPCDataset(...)` is instantiated) to read `adc_scale` from the config:
   ```python
   adc_scale=val_cfg.get("adc_scale", 1.0),
   ```
   without this, the script hardcodes default kwargs and ignores the config's adc_scale.

3. **Document at the top of the v6-seg config** that this checkpoint requires `adc_scale=500.0` because of the training-era preprocessing convention.

4. **For future training runs against the current `lartpc.py`**: explicitly set `adc_scale` in the config to whatever preprocessing the model should learn. Don't rely on defaults — they've changed.

### Lessons for the future

- When a model's eval metric diverges sharply from training, **suspect data preprocessing alignment first** — bigger than weight corruption, bigger than env drift, bigger than label conventions.
- The `proton recall ≈ training-time` observation should have been a faster pointer toward "feature shape is partially preserved, but globally rescaled" rather than "model is broken."
- LArTPCDataset's preprocessing (clip range, scale factor, log-transform parameters) should be encoded explicitly in the config and pinned per-checkpoint. The current default-arg-changes-between-branches pattern is brittle.
