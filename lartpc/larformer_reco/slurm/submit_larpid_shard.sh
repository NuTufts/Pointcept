#!/bin/bash
#
# Apply the LArPID CNN to nu_reco shards (one array task per shard file),
# writing nu_reco_larpid_shard*.h5 alongside. Run per stream with the
# matching kp2 list (gidx linkage).
#
# Submit (nu stream example):
#   NU_RECO_DIR=.../nu_reco_streams_nu KP2_LIST=.../keypoint2_out_<tag>_nu.txt \
#   OUTPUT_DIR=.../nu_reco_larpid_nu sbatch --array=0-9 submit_larpid_shard.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=larpid
#SBATCH --output=logs/export/larpid.%A_%a.%N.log
#SBATCH --error=logs/export/larpid.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9

set -eu
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco
TAG=mcc9_bnbnu_overlay_1500

NU_RECO_DIR=${NU_RECO_DIR:-${RECODIR}/output/${TAG}/nu_reco_streams_nu}
KP2_LIST=${KP2_LIST:-${RECODIR}/outputlists/keypoint2_out_${TAG}_2400_nu.txt}
MERGED_SP_LIST=${MERGED_SP_LIST:-${RECODIR}/inputlists/merged_sp_${TAG}_2400.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/${TAG}/nu_reco_larpid_nu}
SAMPLE_TAG=${SAMPLE_TAG:-mcc9_v29e_dl_run3b_bnb_nu_overlay}

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/export"
SHARDS=($(ls ${NU_RECO_DIR}/nu_reco_shard*.h5 | sort))
[ ${SLURM_ARRAY_TASK_ID} -ge ${#SHARDS[@]} ] && { echo "no shard"; exit 0; }
IN=${SHARDS[$SLURM_ARRAY_TASK_ID]}
OUT=${OUTPUT_DIR}/nu_reco_larpid_$(basename ${IN} | sed 's/nu_reco_//')
echo ">>> ${IN} -> ${OUT}"

module load apptainer 2>/dev/null || true
apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && source setenv_pointcept_only.sh && \
  PYTHONPATH=./ python3 lartpc/larformer_reco/larpid/apply_larpid.py \
    --nu-reco-shard ${IN} --kp2-list ${KP2_LIST} \
    --merged-sp-list ${MERGED_SP_LIST} --out ${OUT} \
    --sample-tag ${SAMPLE_TAG} --device cuda"
echo "DONE"
