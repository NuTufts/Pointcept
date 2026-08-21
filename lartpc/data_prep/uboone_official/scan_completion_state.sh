#!/bin/bash
#SBATCH --job-name=scan_compl
#SBATCH --mem=8G --cpus-per-task=1 --time=4:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/overlay_train/scanc.%A_%a.log
#SBATCH --error=logs/overlay_train/scanc.%A_%a.err
set -u
FILELIST=${FILELIST:?}; OUTDIR=${OUTDIR:?}; NCHUNK=${NCHUNK:?}
I=${SLURM_ARRAY_TASK_ID:?}
mkdir -p "$OUTDIR"
N=$(wc -l < "$FILELIST"); PER=$(( (N + NCHUNK - 1) / NCHUNK ))
FIRST=$(( (I-1)*PER + 1 )); LAST=$(( I*PER < N ? I*PER : N ))
[ $FIRST -le $N ] || exit 0
sed -n "${FIRST},${LAST}p" "$FILELIST" > /tmp/sc_$$.txt
module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster,/tmp:/tmp \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif python3 - <<PYEOF > "$OUTDIR/chunk_$(printf %03d $I).csv"
import h5py
KEYS = ("trackid", "pid", "origin", "hasmatch")
for line in open('/tmp/sc_$$.txt'):
    p = line.strip()
    try:
        with h5py.File(p, 'r') as f:
            td = f['entry_0/triplet_data']
            missing = [k for k in KEYS if k not in td]
            haspre = all(f'{k}_precomplete' in td for k in KEYS)
            done = 'label_completed' in td
            if missing:
                print(f"{p},BROKEN,{'|'.join(missing)},{int(haspre)}")
            elif done:
                print(f"{p},DONE,,{int(haspre)}")
            else:
                print(f"{p},PENDING,,{int(haspre)}")
    except Exception as e:
        print(f"{p},UNREADABLE,{repr(e)[:50]},0")
PYEOF
rm -f /tmp/sc_$$.txt
