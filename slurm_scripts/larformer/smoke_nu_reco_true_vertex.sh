#!/bin/bash
#
# SMOKE for the --true-vertex reco seed (run_nu_reco.py, 2026-08-11).
#
# Runs the nu-interaction reco twice over the SAME 20 events of the old-chain
# bnb-nu satfix keypoint2 output -- (a) baseline pred-vertex seeding and
# (b) --true-vertex seeding -- then checks:
#   1. both shards produced reco events without errors
#   2. vertex_seed provenance attr = pred / true respectively
#   3. every true-mode PRIMARY vertex equals gt_nu_vertex_cm exactly
#      (i.e. the truth seed was used verbatim -- no snap moved it)
#   4. prints pred-mode vertex->truth distances for reference
#
# Submit:  sbatch slurm_scripts/larformer/smoke_nu_reco_true_vertex.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=smk_truevtx
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/nu_reco/smoke_true_vertex.%j.log
#SBATCH --error=logs/nu_reco/smoke_true_vertex.%j.err

set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

KP2_LIST=${RECODIR}/outputlists/keypoint2_out_mcc9_bnbnu_overlay_1500_full_satfix_nu.txt
MSP_LIST=${RECODIR}/inputlists/merged_sp_mcc9_bnbnu_overlay_1500_full_satfix.txt
OUTBASE=${RECODIR}/output/smoke_true_vertex
N_EVENTS=20

mkdir -p "${WORKDIR}/logs/nu_reco" "${OUTBASE}/pred" "${OUTBASE}/true"

module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 ${RECODIR}/scripts/run_nu_reco.py \
    --keypoint2-list ${KP2_LIST} --merged-sp-list ${MSP_LIST} \
    --output-dir ${OUTBASE}/pred --start 0 --n ${N_EVENTS} && \
  python3 ${RECODIR}/scripts/run_nu_reco.py \
    --keypoint2-list ${KP2_LIST} --merged-sp-list ${MSP_LIST} \
    --output-dir ${OUTBASE}/true --start 0 --n ${N_EVENTS} --true-vertex && \
  python3 - <<'PYEOF'
import numpy as np, h5py

base = '${OUTBASE}'
shards = {m: h5py.File(f'{base}/{m}/nu_reco_shard0000000.h5', 'r')
          for m in ('pred', 'true')}
fails = []
for mode, f in shards.items():
    seed = f.attrs['vertex_seed']
    seed = seed.decode() if isinstance(seed, bytes) else seed
    print(f'[{mode}] vertex_seed={seed}  n_reco={f.attrs[\"n_reco\"]}  '
          f'n_skip={f.attrs[\"n_skip\"]}  n_err={f.attrs[\"n_err\"]}')
    if seed != mode:
        fails.append(f'{mode}: provenance attr is {seed}')
    if int(f.attrs['n_err']) != 0:
        fails.append(f'{mode}: {f.attrs[\"n_err\"]} errors')
    if int(f.attrs['n_reco']) == 0:
        fails.append(f'{mode}: zero reco events')

for ev in sorted(shards['true'].keys()):
    g = shards['true'][ev]
    gt = np.asarray(g.attrs['gt_nu_vertex_cm'], np.float64)
    # the truth-seeded interaction is the top-score one (score 1.0); with a
    # single candidate every interaction beyond it comes from the leftover
    # shower-only seeding, so check the FIRST primary vertex
    v0 = np.asarray(g['vertices_cm'][0], np.float64)
    d_true = float(np.linalg.norm(v0 - gt))
    line = f'  {ev}: |v_true - gt| = {d_true:.4f} cm'
    if ev in shards['pred']:
        vp = np.asarray(shards['pred'][ev]['vertices_cm'][0], np.float64)
        line += f'   (pred-mode |v - gt| = {np.linalg.norm(vp - gt):.2f} cm)'
    print(line)
    if d_true > 1e-3:
        fails.append(f'{ev}: true-mode vertex moved {d_true:.4f} cm off truth')

n_true = len([k for k in shards['true'] if k.startswith('event_')])
n_pred = len([k for k in shards['pred'] if k.startswith('event_')])
print(f'events reconstructed: true-mode {n_true}, pred-mode {n_pred}')
if fails:
    print('SMOKE FAILED:')
    for x in fails:
        print('  -', x)
    raise SystemExit(1)
print('SMOKE PASSED')
PYEOF
"
echo "DONE smoke_nu_reco_true_vertex"
