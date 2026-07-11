#!/bin/bash
#
# Sharded gen2ntuple EXPORT (export_gen2ntuple.py). The merged_sp list (the
# event universe) is split into NSHARDS contiguous chunks via --start/--n;
# each array task writes ${OUT%.root}_shardNN.root. A follow-up hadd job
# (submit_export_merge.sh) concatenates them into the final ntuple.
#
# potTree correctness under sharding: the exporter writes a fileno's POT
# entry only in the shard holding that fileno's FIRST event, so the hadd
# union counts each fileno exactly once even when its events straddle a
# shard boundary.
#
# Submit:
#   JID=$(NSHARDS=16 OUT=.../dlgen2_larformer_ntuple.root ... \
#         sbatch --parsable --array=0-15 submit_export_shard.sh)
#   OUT=... sbatch --dependency=afterok:${JID} submit_export_merge.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=ntupshard
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/export/ntupshard.%A_%a.%N.log
#SBATCH --error=logs/export/ntupshard.%A_%a.%N.err
#SBATCH --array=0-15

set -eu

NSHARDS=${NSHARDS:-16}
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco
TAG=${TAG:-mcc9_bnbnu_overlay_1500_full}
D=${RECODIR}/output/${TAG}

MERGED_SP_LIST=${MERGED_SP_LIST:-${RECODIR}/inputlists/merged_sp_mcc9_v29e_dl_run3b_bnb_nu_overlay_1500files.txt}
TRUTH_DIR=${TRUTH_DIR:-${RECODIR}/output/mcc9_bnbnu_overlay_1500/truth_sidecar}
KP2_NU_LIST=${KP2_NU_LIST:-${RECODIR}/outputlists/keypoint2_out_${TAG}_nu.txt}
KP2_FM_LIST=${KP2_FM_LIST:-${RECODIR}/outputlists/keypoint2_out_${TAG}_fm.txt}
NU_RECO_NU_DIR=${NU_RECO_NU_DIR:-${D}/nu_reco_larpid_nu}
NU_RECO_FM_DIR=${NU_RECO_FM_DIR:-${D}/nu_reco_larpid_fm}
WEIGHTS_PKL=${WEIGHTS_PKL:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple/event_weighting/weights_forCV_v48_Sep24_bnb_nu_run3.pkl}
OUT=${OUT:-${D}/dlgen2_larformer_ntuple_${TAG}_67k.root}

NLINES=$(grep -c . "${MERGED_SP_LIST}")
PER_SHARD=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
SHARD_OUT=$(printf "%s_shard%02d.root" "${OUT%.root}" "${SLURM_ARRAY_TASK_ID}")

echo ">>> shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: ${NLINES} events; "
echo "    start=${START} n=${PER_SHARD} -> ${SHARD_OUT}"
mkdir -p "${WORKDIR}/logs/export"
if [ "${START}" -ge "${NLINES}" ]; then
  echo ">>> start ${START} >= ${NLINES}; nothing to do."; exit 0
fi

module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  PYTHONPATH=./ python3 lartpc/larformer_reco/export/export_gen2ntuple.py \
    --merged-sp-list ${MERGED_SP_LIST} \
    --truth-dir ${TRUTH_DIR} \
    --kp2-nu-list ${KP2_NU_LIST} \
    --kp2-fm-list ${KP2_FM_LIST} \
    --nu-reco-nu-dir ${NU_RECO_NU_DIR} \
    --nu-reco-fm-dir ${NU_RECO_FM_DIR} \
    --weights-pkl ${WEIGHTS_PKL} \
    --start ${START} --n ${PER_SHARD} \
    --out ${SHARD_OUT}
"
echo "DONE shard ${SLURM_ARRAY_TASK_ID} -> ${SHARD_OUT}"
