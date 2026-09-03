#!/bin/bash
#SBATCH --job-name=complete_labels
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/complete.%A_%a.log
#SBATCH --error=logs/overlay_train/complete.%A_%a.err
# In-place label completion over an h5 file list, chunked by array task.
#   sbatch --export=ALL,FILELIST=<list>,NCHUNK=<tasks> --array=1-<tasks>%40 ...
set -u -o pipefail
FILELIST=${FILELIST:?}
NCHUNK=${NCHUNK:?}
I=${SLURM_ARRAY_TASK_ID:?}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
N=$(wc -l < "$FILELIST")
PER=$(( (N + NCHUNK - 1) / NCHUNK ))
FIRST=$(( (I-1)*PER + 1 ))
LAST=$(( I*PER < N ? I*PER : N ))
[ $FIRST -le $N ] || { echo "chunk $I empty"; exit 0; }
module load apptainer 2>/dev/null || true
# chunk list on SHARED storage: node-local /tmp filled up on some nodes in
# the 3089802 campaign and 39 chunks silently no-op'd (empty xargs input)
CHUNKDIR=$KPV2/logs/overlay_train/chunks; mkdir -p "$CHUNKDIR"
CHUNKF=$CHUNKDIR/chunk_${SLURM_JOB_ID:-x}_${I}.txt
sed -n "${FIRST},${LAST}p" "$FILELIST" > "$CHUNKF"
NW=$(wc -l < "$CHUNKF")
[ "$NW" -eq $((LAST-FIRST+1)) ] || { echo "CHUNK FILE SHORT ($NW != $((LAST-FIRST+1))) — aborting"; exit 1; }
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd $KPV2 && export PYTHONPATH=$KPV2 && \
  xargs -a $CHUNKF python3 lartpc/data_prep/uboone_official/complete_labels.py --h5"
rm -f "$CHUNKF"
echo "chunk $I ($FIRST-$LAST) done"
