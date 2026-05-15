# Cluster debug brief — SonataLoRASegmentor inference produces mIoU ≈ 0.03, training reported 0.82

> **For the Claude session that picks this up on the cluster:** this doc is the complete handoff. Everything you need is below. Don't waste time re-running diagnostics that are already marked ✗ — go straight to the "Tests to run on the cluster" section. The user is on a GPU node with the same training data + checkpoint, which lets you do things I couldn't from a laptop.

## 1. The problem in one paragraph

A LoRA-fine-tuned SONATA semantic segmentor (8 LArTPC particle classes: electron, muon, pion, proton, gamma, michel, delta, led) was trained on the cluster. The training log reports `mIoU/mAcc/allAcc 0.8212/0.8864/0.9489` at the end of training, with per-class IoUs all in the 0.65–0.95 range. The saved `model_last.pth` (also has `best_metric_value=0.8212` in its metadata) gives **mIoU ≈ 0.03** at inference time on the **same prod4 validation files** — not just a different dataset. The model predicts class 3 (proton) for ~80% of all spacepoints regardless of true class.

## 2. Files involved

| File | Purpose |
|---|---|
| `configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-seg.py` | Training config (uses `SonataLoRASegmentor` + LoRA on top of SONATA-v1m1 + PT-v3m2 backbone, 10 epochs, drop_cosmics_prob=0.9, BiasedSphereCrop r=20 cm) |
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

**H1.** **`model_last.pth` is functionally not the model that scored 0.82.** It loads cleanly but the weight values themselves correspond to a different (broken) state. Possible mechanisms:
- A half-flushed/partial save that completed without erroring
- A NaN/Inf spike late in training that destroyed weights, with the saver overwriting after the 0.82 was already logged to wandb
- An EMA / student-teacher swap that happened during a final save

**H2.** **`model_best.pth` exists on the cluster and is the correct one** — Pointcept's default `CheckpointSaver` writes both `_last` and `_best`. We only have `_last` locally. The 0.82 result was the *best* val mIoU during training, so `model_best.pth` (if it exists) is the file we actually want.

**H3.** **Environment-specific bug in my container** — torch version, xformers presence, CUDA version, or some custom op behaves differently from the cluster's training environment, silently corrupting forward pass. Unlikely (load is bit-identical) but possible if something runs at first-forward time.

H2 is the most likely. H1 is second. H3 is a long shot.

## 6. Tests to run on the cluster — in order

Run these in order. Each builds on the previous. Stop as soon as you find the answer.

### Test 1 — does `model_best.pth` exist?

```bash
ls -la /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/
```

If `model_best.pth` is present, **that's almost certainly the answer**. Skip ahead to Test 4 with it.

### Test 2 — reproduce the broken result on the cluster with `model_last.pth`

This confirms or refutes H3 (env-specific bug). Run the standalone inference script with the same `model_last.pth` we have:

```bash
cd /path/to/Pointcept
./run_in_container.sh python lartpc_data_prep/semseg_analysis/run_semseg_inference.py \
    --model-config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
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
    --config-file configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
    --options weight=sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth \
              save_path=/tmp/official_test_last \
              data.test.data_list_file=prod4_valsplit_subset.txt
```

(Or whatever the exact invocation is on the cluster — the config has a `test` block in `data`.)

- **Agrees with Test 2 (~0.03)**: confirms the saved weights are broken; proceed to Test 4 with `model_best.pth`.
- **Disagrees (~0.8)**: there's something my standalone driver is doing that the trainer doesn't. Diff our `run_semseg_inference.py` against `tools/test.py` carefully.

### Test 4 — same as Test 2 but with `model_best.pth`

```bash
./run_in_container.sh python lartpc_data_prep/semseg_analysis/run_semseg_inference.py \
    --model-config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-seg.py \
    --weights sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_best.pth \
    --filelist prod4_valsplit_subset.txt \
    --output-shard /tmp/cluster_check_best.npz \
    --match-val-pipeline \
    --nfiles 20
```

Expect mIoU ≈ 0.8 here if H2 is correct.

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
5. Where we left off: the inference is correct, the load is correct, the wiring is correct, but the resulting predictions are still uniformly proton. Either the saved weights themselves are bad, or there's an environment-specific divergence we can't see from the laptop.

The remaining work is on the cluster: identify whether `model_best.pth` exists, or instrument the actual training environment to find the divergence.
