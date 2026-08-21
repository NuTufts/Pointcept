#!/bin/bash
#SBATCH --job-name=repack_h5
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/repack.%A_%a.log
#SBATCH --error=logs/overlay_train/repack.%A_%a.err
#   sbatch --export=ALL,FILELIST=<list>,NCHUNK=<tasks> --array=1-<tasks>%50 ...
set -u -o pipefail
FILELIST=${FILELIST:?}; NCHUNK=${NCHUNK:?}
I=${SLURM_ARRAY_TASK_ID:?}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
N=$(wc -l < "$FILELIST"); PER=$(( (N + NCHUNK - 1) / NCHUNK ))
FIRST=$(( (I-1)*PER + 1 )); LAST=$(( I*PER < N ? I*PER : N ))
[ $FIRST -le $N ] || { echo "chunk $I empty"; exit 0; }
sed -n "${FIRST},${LAST}p" "$FILELIST" > /tmp/rp_$$.txt
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster,/tmp:/tmp \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd $KPV2 && export PYTHONPATH=$KPV2 && \
  xargs -a /tmp/rp_$$.txt python3 lartpc/data_prep/uboone_official/repack_h5.py --h5" \
  | tail -2
RC=$?
rm -f /tmp/rp_$$.txt
echo "chunk $I ($FIRST-$LAST) rc=$RC"
exit $RC
