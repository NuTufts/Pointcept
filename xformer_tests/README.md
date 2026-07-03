# XFormers Backend Comparison Tests

This directory contains scripts to compare encoder outputs between different flash attention backends:
- `flash_attn` (original, requires A100/H100 GPUs)
- `xformers` (alternative, works on P100/V100 GPUs)

## Scripts

### 1. `extract_encoder_vectors.py`
Extracts encoder features from a PointTransformerV3 backbone for a single data file.

### 2. `compare_encoder_vectors.py`
Compares encoder outputs from two different runs and quantifies differences.

## Usage

### Step 1: Extract vectors on A100 (with flash_attn)

On your A100 machine:
```bash
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/xformers/pointcept_dev

python xformer_tests/extract_encoder_vectors.py \
    --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py \
    --checkpoint /path/to/your/checkpoint.pth \
    --data_file /path/to/single_event.h5 \
    --output xformer_tests/encoder_a100_flash_attn.pt \
    --flash_backend flash_attn \
    --device cuda
```

### Step 2: Extract vectors on P100 (with xformers)

On your P100 machine:
```bash
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/xformers/pointcept_dev

python xformer_tests/extract_encoder_vectors.py \
    --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py \
    --checkpoint /path/to/your/checkpoint.pth \
    --data_file /path/to/single_event.h5 \
    --output xformer_tests/encoder_p100_xformers.pt \
    --flash_backend xformers \
    --device cuda
```

### Step 3: Compare the outputs

On either machine:
```bash
python xformer_tests/compare_encoder_vectors.py \
    --file1 xformer_tests/encoder_a100_flash_attn.pt \
    --file2 xformer_tests/encoder_p100_xformers.pt \
    --output xformer_tests/comparison_report.txt \
    --tolerance 1e-3
```

## Expected Results

Due to numerical differences between float16 (xformers on P100) and bfloat16 (flash_attn on A100), you should expect:

- **Cosine similarity > 0.999**: Outputs are very similar
- **Mean relative error < 1%**: Differences are within acceptable numerical precision
- **Per-point cosine similarity > 0.99**: Each point's features are well-preserved

Small differences are expected because:
1. **Different precision**: flash_attn uses bfloat16, xformers uses float16
2. **Different implementations**: Kernel implementations may have slight numerical differences
3. **Accumulation order**: Parallel reductions may accumulate in different orders

## Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| MAE | Mean Absolute Error | < 1e-3 |
| RMSE | Root Mean Squared Error | < 1e-3 |
| Cosine Similarity | Direction similarity (1.0 = identical) | > 0.999 |
| Pearson Correlation | Linear correlation | > 0.999 |
| Mean Relative Error | Percentage difference | < 1% |

## Troubleshooting

### "AssertionError: Make sure xformers is installed"
Install xformers:
```bash
pip install xformers
```

### "AssertionError: Make sure flash_attn is installed"
On A100, install flash-attn:
```bash
pip install flash-attn
```

### Shape mismatch errors
Ensure you're using the same:
- Data file
- Config file
- Grid size (transforms may produce different point counts)

### Large differences (cosine sim < 0.99)
Check:
1. Same checkpoint file
2. Same data file
3. Model is in eval mode (deterministic)
4. No random augmentations in transforms
