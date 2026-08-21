#!/bin/bash
#SBATCH --job-name=scan_labels
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/scan.%A_%a.log
#SBATCH --error=logs/overlay_train/scan.%A_%a.err
# Per-file labeled-point counts for the list-builder dirt filter.
#   sbatch --export=ALL,FILELIST=..,OUTDIR=..,NCHUNK=N --array=1-N ...
set -u
FILELIST=${FILELIST:?}; OUTDIR=${OUTDIR:?}; NCHUNK=${NCHUNK:?}
I=${SLURM_ARRAY_TASK_ID:?}
KPV2=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
mkdir -p "$OUTDIR"
N=$(wc -l < "$FILELIST"); PER=$(( (N + NCHUNK - 1) / NCHUNK ))
FIRST=$(( (I-1)*PER + 1 )); LAST=$(( I*PER < N ? I*PER : N ))
[ $FIRST -le $N ] || exit 0
sed -n "${FIRST},${LAST}p" "$FILELIST" > /tmp/scan_$$.txt
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster,/tmp:/tmp \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif python3 - <<PYEOF > "$OUTDIR/chunk_$(printf %03d $I).csv"
import h5py
for line in open('/tmp/scan_$$.txt'):
    p = line.strip()
    try:
        with h5py.File(p, 'r') as f:
            tid = f['entry_0/triplet_data/trackid'][()]
            print(f"{p},{len(tid)},{(tid > 0).sum()}")
    except Exception as e:
        print(f"{p},-1,-1")
PYEOF
rm -f /tmp/scan_$$.txt
echo "chunk $I done"
