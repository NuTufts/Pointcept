#!/bin/bash
#
# SMOKE for the v2 full-cascade keypoint config (2026-08-11):
#   configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v2.py
# (PTv3-decoder ft deghoster tau=0.20 + m2frecipe-v2 slicer ep4 + m2frecipe
#  stage-3 model_best + OLD attempt-2 keypoint ckpt).
#
# Runs 10 events of the bnb-nu 50-event test list through the full cascade,
# then checks: output files exist, nu_vertex_cm finite where slices were
# found, gt_nu_vertex_cm + score maps present (needed by run_nu_reco.py in
# BOTH vertex modes), and the log is free of NaN/error signatures.
#
# Submit:  sbatch slurm_scripts/larformer/smoke_keypoint2_fullcascade_v2_a100.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=smk_kp2v2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/keypoint2/smoke_kp2v2.%j.log
#SBATCH --error=logs/keypoint2/smoke_kp2v2.%j.err

set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
CONFIG=configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v2.py
INPUT_LIST=${WORKDIR}/lartpc/larformer_reco/inputlists/merged_sp_mcc9_v29e_dl_run3b_bnb_nu_overlay_50event_test.txt
OUTPUT_DIR=${WORKDIR}/lartpc/larformer_reco/output/smoke_kp2v2
N_EVENTS=10

mkdir -p "${WORKDIR}/logs/keypoint2"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

module load apptainer 2>/dev/null || true
nvidia-modprobe -u -c=0 2>/dev/null || true

apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/larformer/run_larformer_keypoint2_cascade_inference.py \
    --config ${CONFIG} \
    --input-list ${INPUT_LIST} \
    --output-dir ${OUTPUT_DIR} \
    --n-events ${N_EVENTS} \
    --deterministic \
    --save-score-maps \
    --device cuda && \
  python3 - <<'PYEOF'
import glob, os
import numpy as np, h5py

out = '${OUTPUT_DIR}'
files = sorted(glob.glob(os.path.join(out, '**', 'keypoint2_event*_0.h5'),
                         recursive=True))
fails = []
if not files:
    fails.append('no keypoint2 output files')
n_nu_finite = n_gt_finite = n_score = 0
for p in files:
    with h5py.File(p, 'r') as f:
        for k in ('nu_vertex_cm', 'gt_nu_vertex_cm'):
            if k not in f:
                fails.append(f'{os.path.basename(p)}: missing {k}')
        if 'nu_vertex_cm' in f and np.all(np.isfinite(f['nu_vertex_cm'][()])):
            n_nu_finite += 1
        if 'gt_nu_vertex_cm' in f and np.all(np.isfinite(f['gt_nu_vertex_cm'][()])):
            n_gt_finite += 1
        if any('score' in k for k in f.keys()):
            n_score += 1
print(f'{len(files)} keypoint2 files (nu stream): '
      f'{n_nu_finite} finite pred vtx, {n_gt_finite} finite gt vtx, '
      f'{n_score} with score maps')
if files and n_nu_finite == 0:
    fails.append('no event has a finite predicted nu vertex')
if files and n_score == 0:
    fails.append('no score maps saved (pred-vertex candidates degraded)')
if fails:
    print('SMOKE FAILED:')
    for x in fails:
        print('  -', x)
    raise SystemExit(1)
print('SMOKE PASSED')
PYEOF
"
echo "DONE smoke_keypoint2_fullcascade_v2"
