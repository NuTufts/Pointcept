#!/bin/bash
# SMOKE for the v2 tau=0.20 cache build: ~8 val events on 1 GPU, into a
# scratch cache root, then verify the output is readable by
# LArFormerStage12CacheDataset and that the cascade ran at tau=0.2 with the
# new deghoster (attrs deghost_tau / keep_frac).
#   sbatch smoke_build_stage12_cache_v2_tau020.sh

#SBATCH --job-name=lf-cache-smoke
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-cache-smoke.%j.%N.log
#SBATCH --error=logs/lf-cache-smoke.%j.%N.err

set -u
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage3_particle/larformer-fullcascade-production-v2-tau020.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
LISTDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists
SAVE=${WORKDIR}/exp/_smoke_cache_v2_tau020
rm -rf ${SAVE}; mkdir -p ${SAVE}
head -8 ${LISTDIR}/h5list_mcall_lantern_val.txt > ${SAVE}/smoke_list.txt

module load apptainer
apptainer exec --nv --bind /cluster:/cluster $container bash -c "\
  cd ${WORKDIR} && source setenv_pointcept_only.sh && \
  python3 tools/larformer/build_stage12_cache_shard.py \
    --config ${CONFIG} \
    --inputlist ${SAVE}/smoke_list.txt \
    --cache-root ${SAVE}/cache \
    --split val \
    --shard-id 0 --n-shards 1 \
    --max-spacepoints 300000 \
    --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2 && \
  python3 - <<'PYEOF'
import glob, h5py, sys
sys.path.insert(0, '.')
files = sorted(glob.glob('${SAVE}/cache/val/**/*.h5', recursive=True))
print(f'cache files written: {len(files)}')
assert files, 'no cache files produced'
for f in files[:3]:
    with h5py.File(f, 'r') as h:
        a = dict(h.attrs)
        print(f\"  {f.split('/')[-1][:50]}: tau={a.get('deghost_tau')} \"
              f\"keep_frac={a.get('deghost_keep_frac'):.3f} \"
              f\"n_cache={a.get('n_cached_spacepoints', a.get('n_cache','?'))} \"
              f\"loose_floor={a.get('tau_loose_floor')}\")
        assert abs(float(a['deghost_tau']) - 0.2) < 1e-6, 'deghost tau != 0.2!'
# Read-back through the stage-3 dataset class
from pointcept.datasets import build_dataset
ds = build_dataset(dict(
    type='LArFormerStage12CacheDataset', split='val',
    data_root='${SAVE}/cache/val', source_set_filter='stage2_pass',
    recenter_to_centroid=True, min_spacepoints=20, loop=1))
print(f'LArFormerStage12CacheDataset: {len(ds)} samples')
s = ds[0]
print('sample keys:', sorted(k for k in s.keys())[:12])
print('coord shape:', s['coord'].shape)
print('DATASET READ-BACK OK')
PYEOF"
RC=$?
echo "=============================================="
echo "cache smoke exit code: ${RC}"
echo "=============================================="
