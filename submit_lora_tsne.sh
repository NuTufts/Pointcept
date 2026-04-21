#!/bin/bash
#
# submit_lora_tsne.sh
# -------------------
# SLURM submission script for t-SNE visualization of a LoRA fine-tuned
# SONATA model (SonataLoRASegmentor).
#
# Uses visualize_lora_tsne.py which handles the LoRA model natively:
#   - routes up_cast_level to cfg.model.backbone (not cfg.model)
#   - extracts features via model.backbone.forward() so LoRA adapters are used
#   - resolves in_channels from the doubly-nested backbone config
#
# Usage:
#   sbatch submit_lora_tsne.sh
# or override paths at submission time:
#   CHECKPOINT=/path/to/ckpt.pth sbatch submit_lora_tsne.sh
#
#SBATCH --job-name=lora_tsne
#SBATCH --output=logs/lora_tsne_%j.%N.log
#SBATCH --error=logs/lora_tsne_%j.%N.err
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:1

# ============================================================================
# Paths – edit these or override via environment variables before submitting
# ============================================================================
WORKDIR=/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept

# LoRA fine-tune config (the lorafinetune-sonata-v1m1-lartpc.py config)
CONFIG=${CONFIG:-${WORKDIR}/configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-fixed.py}

# LoRA checkpoint to visualise (model_best.pth or a specific epoch ckpt)
CHECKPOINT=${CHECKPOINT:-${WORKDIR}/sonata/lora_finetune_v6_p100_50_epochs_noghost_logspacefix/model/model_last.pth}

# Validation file list (overrides config value if set)
VAL_FILE_LIST=${VAL_FILE_LIST:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt}

# Output paths
OUTDIR=${WORKDIR}/exp/lartpc/sonata/lora_finetune_v6_fixed/tsne
OUTPUT_IMG=${OUTDIR}/tsne_lora_features_v6_fixed.png
SAVE_FEATURES=${OUTDIR}/lora_features_v6_fixed.npz   # optional: set to "" to skip saving

# Singularity container (same as training)
CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# ============================================================================
# t-SNE / sampling hyper-parameters
# Tune these for speed vs quality:
#   fewer events + lower max-points → fast debug run
#   more events + balanced-sampling → publication-quality figure
# ============================================================================
NUM_EVENTS=10
POINTS_PER_EVENT=10000
MAX_POINTS=25000
PERPLEXITY=50
LEARNING_RATE=200
N_ITER=1000
EARLY_EXAGGERATION=12
SEED=42

# ============================================================================
# Setup
# ============================================================================
module load apptainer/1.2.4-suid
mkdir -p logs
mkdir -p "${OUTDIR}"

# ============================================================================
# Build the command
# ============================================================================
TSNE_CMD="python3 tools/visualize_sonata_tsne_LoRa_v6.py \
    --config        ${CONFIG} \
    --checkpoint    ${CHECKPOINT} \
    --data-list     ${VAL_FILE_LIST} \
    --output        ${OUTPUT_IMG} \
    --num-events    ${NUM_EVENTS} \
    --points-per-event ${POINTS_PER_EVENT} \
    --max-points    ${MAX_POINTS} \
    --perplexity    ${PERPLEXITY} \
    --learning-rate ${LEARNING_RATE} \
    --n-iter        ${N_ITER} \
    --early-exaggeration ${EARLY_EXAGGERATION} \
    --seed          ${SEED} \
    --true-points-only \
    --balanced-sampling"

# Append --save-features only if SAVE_FEATURES is non-empty
if [ -n "${SAVE_FEATURES}" ]; then
    TSNE_CMD="${TSNE_CMD} --save-features ${SAVE_FEATURES}"
fi

# ============================================================================
# Run inside singularity container
# ============================================================================
echo "==========================================="
echo "  LoRA t-SNE Visualization"
echo "  Config     : ${CONFIG}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Output     : ${OUTPUT_IMG}"
echo "  Events     : ${NUM_EVENTS}  x  ${POINTS_PER_EVENT} pts/event"
echo "  t-SNE      : perplexity=${PERPLEXITY}, n_iter=${N_ITER}"
echo "==========================================="

apptainer exec --nv --bind /cluster:/cluster \
    --env WANDB_MODE=disabled \
    "${CONTAINER}" bash -c "
        set -e
        cd ${WORKDIR}

        pip install scikit-learn --quiet --user
        echo 'Running t-SNE extraction ...'
        ${TSNE_CMD}
        echo 'Done. Output: ${OUTPUT_IMG}'
    "
