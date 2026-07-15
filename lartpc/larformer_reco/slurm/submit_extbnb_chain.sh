#!/bin/bash
#
# EXT-BNB (beam-off cosmic) full downstream chain orchestrator.
# Submits the dependency-chained SLURM jobs that turn a directory of
# merged_sp H5 (produced by stepA in --is-data mode) into a gen2ntuple ROOT
# file, mirroring the proven bnb5e19 data campaign:
#
#   prep (build+clean lists/dirs)
#     -> inference  (GPU array, keypoint2_streams: nu + fm kp2)
#     -> regen      (split keypoint2_streams -> nu / fm lists)
#     -> nu_reco nu + nu_reco fm   (CPU-ish arrays, LLR attachment)
#     -> larpid  nu + larpid  fm   (CPU arrays)
#     -> export   (data-mode, both streams -> ntuple shards)
#     -> hadd     (merge shards)
#
# Everything runs truth-free (data mode); the 5M-triplet noise veto was already
# applied in stepA so capped events simply have no merged_sp (no exclude-pattern
# surgery needed, unlike the bnb5e19 straggler cleanup).
#
# Usage (run on the login node, inside the repo):
#   DEP=<stepA_jobid> TAG=extbnb_val \
#   DATADIR=/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9_v29e_dl_run3_G1_extbnb \
#   NINF=8 NNR=8 NEXP=4 bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
#
# Prints the job ids of every stage; the ntuple lands at
#   ${DATADIR}/dlgen2_larformer_ntuple_${TAG}.root
# ---------------------------------------------------------------------------
set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
RECODIR=${WORKDIR}/lartpc/larformer_reco
SLURMDIR=${RECODIR}/slurm
CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
cd "${WORKDIR}"

TAG=${TAG:?set TAG (e.g. extbnb_val)}
DATADIR=${DATADIR:?set DATADIR (the EXT dataset dir with merged_sp/)}
MSP_DIR=${DATADIR}/merged_sp
NINF=${NINF:-8}          # inference (GPU) shards
NNR=${NNR:-8}            # nu_reco shards per stream
NEXP=${NEXP:-4}          # export shards
DEP=${DEP:-}             # optional afterok stepA job id gating the chain

MSP_LIST=${RECODIR}/inputlists/merged_sp_${TAG}.txt
KP2_NU=${RECODIR}/outputlists/keypoint2_out_${TAG}_nu.txt
KP2_FM=${RECODIR}/outputlists/keypoint2_out_${TAG}_fm.txt
KP2_STREAMS=${DATADIR}/keypoint2_streams
NR_NU=${DATADIR}/nu_reco_streams_nu
NR_FM=${DATADIR}/nu_reco_streams_fm
LP_NU=${DATADIR}/nu_reco_larpid_nu
LP_FM=${DATADIR}/nu_reco_larpid_fm
OUT_NTUPLE=${DATADIR}/dlgen2_larformer_ntuple_${TAG}.root
mkdir -p "${WORKDIR}/logs/export" "${WORKDIR}/logs/data_prep"

# stepA gate uses afterany: individual stepA tasks may fail on a bad dlreco
# file, and prep just globs whatever merged_sp succeeded. Internal stage deps
# below stay afterok (halt the chain on a genuine stage failure).
dep_arg() { [ -n "$1" ] && echo "--dependency=afterany:$1" || echo ""; }

# ---- 0) prep: build the merged_sp list + clean downstream dirs --------------
# (find, not ls -- E2BIG at scale; clean keypoint2_streams + nu_reco dirs so
#  stale shard files can't poison the glob consumers.)
# stable (fileno,entry) sort so the cascade index<->event linkage is invariant
# to the merged_sp flat-vs-tree layout (list_merged_sp.py; NOT path-sorted find)
PREP=$(sbatch --parsable $(dep_arg "${DEP}") \
  --partition=batch --time=1:00:00 --mem=4G --job-name=${TAG}_prep \
  --output=logs/data_prep/${TAG}_prep.%j.log \
  --error=logs/data_prep/${TAG}_prep.%j.err \
  --wrap="apptainer exec --bind /cluster:/cluster ${CONTAINER} python3 \
            ${WORKDIR}/lartpc/data_prep/uboone_official/list_merged_sp.py \
            --dir ${MSP_DIR} --out ${MSP_LIST}; \
          rm -rf ${KP2_STREAMS} ${NR_NU} ${NR_FM} ${LP_NU} ${LP_FM}; \
          mkdir -p ${KP2_STREAMS}; \
          echo \"merged_sp events: \$(wc -l < ${MSP_LIST})\"")
echo "prep      : ${PREP}  -> ${MSP_LIST}"

# ---- 1) inference (GPU) : merged_sp -> keypoint2_streams (nu + fm) ----------
# --output-tree: write cascade files into an index tree (avoid a 1M-file dir)
INF=$(INPUT_LIST=${MSP_LIST} OUTPUT_DIR=${KP2_STREAMS}/ NSHARDS=${NINF} \
  EXTRA_INF_ARGS="--output-tree" \
  sbatch --parsable --export=ALL --dependency=afterok:${PREP} \
  --array=0-$((NINF-1)) --time=8:00:00 \
  ${SLURMDIR}/submit_inference_shard.sh)
echo "inference : ${INF}  (${NINF} GPU shards) -> ${KP2_STREAMS}"

# ---- 2) regen: split keypoint2_streams into nu / fm lists -------------------
REGEN=$(sbatch --parsable --dependency=afterok:${INF} \
  --partition=batch --time=0:30:00 --mem=4G --job-name=${TAG}_regen \
  --output=logs/export/${TAG}_regen.%j.log \
  --error=logs/export/${TAG}_regen.%j.err \
  --wrap="find ${KP2_STREAMS} -name 'keypoint2_event*_0.h5' ! -name '*_fm_0.h5' | sort > ${KP2_NU}; \
          find ${KP2_STREAMS} -name 'keypoint2_event*_fm_0.h5' | sort > ${KP2_FM}; \
          wc -l ${KP2_NU} ${KP2_FM}")
echo "regen     : ${REGEN}  -> ${KP2_NU} , ${KP2_FM}"

# ---- 3) nu_reco : nu + fm streams (LLR attachment) -------------------------
NRNU=$(KEYPOINT2_LIST=${KP2_NU} MERGED_SP_LIST=${MSP_LIST} OUTPUT_DIR=${NR_NU}/ \
  NSHARDS=${NNR} sbatch --parsable --export=ALL --dependency=afterok:${REGEN} \
  --array=0-$((NNR-1)) ${SLURMDIR}/submit_nu_reco_shard.sh)
echo "nu_reco nu: ${NRNU}  (${NNR} shards) -> ${NR_NU}"

NRFM=$(KEYPOINT2_LIST=${KP2_FM} MERGED_SP_LIST=${MSP_LIST} OUTPUT_DIR=${NR_FM}/ \
  NSHARDS=${NNR} sbatch --parsable --export=ALL --dependency=afterok:${REGEN} \
  --array=0-$((NNR-1)) ${SLURMDIR}/submit_nu_reco_shard.sh)
echo "nu_reco fm: ${NRFM}  (${NNR} shards) -> ${NR_FM}"

# ---- 4) larpid : nu + fm (CPU) ---------------------------------------------
LPNU=$(NU_RECO_DIR=${NR_NU} KP2_LIST=${KP2_NU} MERGED_SP_LIST=${MSP_LIST} \
  OUTPUT_DIR=${LP_NU} SAMPLE_TAG=${TAG} DEVICE=cpu TAG=${TAG} \
  sbatch --parsable --export=ALL --partition=batch --gres=gpu:0 \
  --dependency=afterok:${NRNU} --array=0-$((NNR-1)) \
  ${SLURMDIR}/submit_larpid_shard.sh)
echo "larpid  nu: ${LPNU}  -> ${LP_NU}"

LPFM=$(NU_RECO_DIR=${NR_FM} KP2_LIST=${KP2_FM} MERGED_SP_LIST=${MSP_LIST} \
  OUTPUT_DIR=${LP_FM} SAMPLE_TAG=${TAG} DEVICE=cpu TAG=${TAG} \
  sbatch --parsable --export=ALL --partition=batch --gres=gpu:0 \
  --dependency=afterok:${NRFM} --array=0-$((NNR-1)) \
  ${SLURMDIR}/submit_larpid_shard.sh)
echo "larpid  fm: ${LPFM}  -> ${LP_FM}"

# ---- 5) export (data mode, both streams) -----------------------------------
EXP=$(TAG=${TAG} MERGED_SP_LIST=${MSP_LIST} NSHARDS=${NEXP} \
  TRUTH_DIR=${DATADIR}/truth_sidecar_absent \
  KP2_NU_LIST=${KP2_NU} KP2_FM_LIST=${KP2_FM} \
  NU_RECO_NU_DIR=${LP_NU} NU_RECO_FM_DIR=${LP_FM} \
  OUT=${OUT_NTUPLE} \
  sbatch --parsable --export=ALL --dependency=afterok:${LPNU}:${LPFM} \
  --array=0-$((NEXP-1)) ${SLURMDIR}/submit_export_shard.sh)
echo "export    : ${EXP}  (${NEXP} shards) -> ${OUT_NTUPLE%.root}_shard*.root"

# ---- 6) hadd merge ---------------------------------------------------------
HADD=$(TAG=${TAG} OUT=${OUT_NTUPLE} \
  sbatch --parsable --export=ALL --dependency=afterok:${EXP} \
  ${SLURMDIR}/submit_export_merge.sh)
echo "hadd      : ${HADD}  -> ${OUT_NTUPLE}"

echo
echo "chain submitted for TAG=${TAG}. final ntuple: ${OUT_NTUPLE}"
echo "watch: squeue -j ${PREP},${INF},${REGEN},${NRNU},${NRFM},${LPNU},${LPFM},${EXP},${HADD}"
