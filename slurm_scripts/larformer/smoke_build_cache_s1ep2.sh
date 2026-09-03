#!/bin/bash
# SMOKE for the S1-ep2 cache build: 8 completed-val files + 8 OVERLAY
# files (the genuinely new path: GT-instance build through the stepA
# overlay schema with partial truth), then readback via the cache dataset.
#SBATCH --job-name=lf-cache-s1-smoke
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:50:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-cache-s1-smoke.%j.log
#SBATCH --error=logs/lf-cache-s1-smoke.%j.err
set -u
W=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${W}/configs/lartpc/larformer/stage3_particle/larformer-fullcascade-s1ep2-tau020.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
LED=${W}/lartpc/data_prep/uboone_official/training_data_ledger
SAVE=${W}/exp/_smoke_cache_s1ep2
rm -rf ${SAVE}; mkdir -p ${SAVE}
head -8 ${LED}/h5list_lantern_val_completed_1500.txt > ${SAVE}/val_list.txt
grep overlay_train ${LED}/h5list_mix_enriched_train_v1.txt | head -8 > ${SAVE}/ovl_list.txt
module load apptainer
apptainer exec --nv --bind /cluster:/cluster $container bash -c "\
  cd ${W} && source setenv_pointcept_only.sh && \
  python3 tools/larformer/build_stage12_cache_shard.py \
    --config ${CONFIG} --inputlist ${SAVE}/val_list.txt \
    --cache-root ${SAVE}/cache --split val --shard-id 0 --n-shards 1 \
    --max-spacepoints 300000 \
    --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2 && \
  python3 tools/larformer/build_stage12_cache_shard.py \
    --config ${CONFIG} --inputlist ${SAVE}/ovl_list.txt \
    --cache-root ${SAVE}/cache --split train --shard-id 0 --n-shards 1 \
    --max-spacepoints 300000 \
    --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2 && \
  python3 - <<'PYEOF'
import glob, h5py, sys, numpy as np
sys.path.insert(0, '.')
for split in ('val', 'train'):
    files = sorted(glob.glob(f'${SAVE}/cache/{split}/**/*.h5', recursive=True))
    print(f'[{split}] cache files: {len(files)}')
    assert files, f'no cache files for {split}'
    for f in files[:3]:
        with h5py.File(f, 'r') as h:
            a = dict(h.attrs)
            ninst = sum(1 for k in h.keys() if k.startswith('instance_')) \
                    if not any('particle' in k for k in h.keys()) else -1
            insts = [k for k in h.keys()]
            print(f\"  {f.split('/')[-1][:58]}: tau={a.get('deghost_tau')} \"
                  f\"keep={float(a.get('deghost_keep_frac',-1)):.3f} groups={len(insts)}\")
            assert abs(float(a['deghost_tau']) - 0.2) < 1e-6
from pointcept.datasets import build_dataset
for split in ('val', 'train'):
    ds = build_dataset(dict(type='LArFormerStage12CacheDataset', split=split,
        data_root=f'${SAVE}/cache/{split}', source_set_filter='stage2_pass',
        recenter_to_centroid=True, min_spacepoints=20, loop=1))
    s = ds[0]
    ng = int(s.get('n_gt_instances', len(s.get('gt_instances', []))) or 0)
    print(f'[{split}] dataset: {len(ds)} samples; sample0 n_gt_instances={ng}, '
          f'n_points={len(s[\"coord\"])}')
print('SMOKE OK')
PYEOF"
echo "smoke exit: $?"
