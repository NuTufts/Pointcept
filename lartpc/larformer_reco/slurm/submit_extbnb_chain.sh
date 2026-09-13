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
# CPU stages (nu_reco, larpid, export): MANY shards on the non-preemptable CPU
# partitions and >= 24 h per job, so a production never needs relaunch cycles
# (user rule 2026-09-12; the per-user CPU job limit on batch is >= 60).
NNR=${NNR:-60}           # nu_reco / larpid shards per stream
CPU_PARTITION=${CPU_PARTITION:-batch,wongjiradlab}
JOB_TIME=${JOB_TIME:-24:00:00}
# extra run_nu_reco.py flags for BOTH streams, e.g. "--true-vertex" (dedicated
# knob -- a bare EXTRA_ARGS would leak into the other EXTRA_ARGS-consuming
# stages through --export=ALL)
NU_RECO_EXTRA_ARGS=${NU_RECO_EXTRA_ARGS:-}
NEXP=${NEXP:-60}         # export shards
DEP=${DEP:-}             # optional afterok stepA job id gating the chain
# --- MC-vs-data knobs (defaults = data mode) --------------------------------
# TRUTH_DIR: real truth-sidecar dir for MC (fills truth branches + potTree +
#   xsecWeight); default is a non-existent dir -> data mode (exporter tolerant).
# LARPID_SAMPLE_TAG: controls the LArPID checkpoint via select_checkpoint
#   ('run3' in the tag -> alternate/run3 weights, else default/run1). MC used
#   alternate; bnb5e19/EXT use default. Independent of the dir-naming TAG.
TRUTH_DIR_IN=${TRUTH_DIR:-${DATADIR}/truth_sidecar_absent}
LARPID_TAG=${LARPID_SAMPLE_TAG:-${TAG}}
# EXCLUDE_NODES: comma-sep nodes to avoid (e.g. ones with stale LDAP that give
# apptainer "Couldn't determine user account information"). Applied to all steps.
EXCL=${EXCLUDE_NODES:+--exclude=${EXCLUDE_NODES}}

# ---- frozen chain version (v2_s1ep2p8cew6, see
# lartpc/larformer_analysis/model_and_output_file_versions.md) -----------------
# The stage scripts read these from the environment (--export=ALL). They used to
# be exported by hand before launching; a launch from a clean shell then fell
# back to the base cascade config + default checkpoints (caught 2026-09-12 on
# the first run-1 EXT launch: cosmic slice rows, 311 fm vs 66 nu files). Every
# value is a default here, overridable, and checked for existence.
export CONFIG=${CONFIG:-configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py}
export LARFORMER_BATTERY_SLICER_CKPT=${LARFORMER_BATTERY_SLICER_CKPT:-exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth}
export LARFORMER_KP_PARTICLE_CKPT=${LARFORMER_KP_PARTICLE_CKPT:-exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth}
export LARFORMER_SHOWER_BDT=${LARFORMER_SHOWER_BDT:-lartpc/larformer_reco/export/data/shower_cosmic_bdt_cew6.joblib}
export LARFORMER_SHOWER_BDT_NOVTX=${LARFORMER_SHOWER_BDT_NOVTX:-lartpc/larformer_reco/export/data/shower_novtx_bdt_cew6.joblib}
ATTACH_LLR=${ATTACH_LLR:-lartpc/larformer_reco/trajfit/data/attachment_llr_tables_s1ep2p8.npz}
if [[ "${NU_RECO_EXTRA_ARGS}" != *"--attach-llr"* ]]; then
  NU_RECO_EXTRA_ARGS="--attach-llr-tables ${ATTACH_LLR} --attach-llr-thr ${ATTACH_LLR_THR:-4.0} ${NU_RECO_EXTRA_ARGS}"
fi
for f in "${CONFIG}" "${LARFORMER_BATTERY_SLICER_CKPT}" "${LARFORMER_KP_PARTICLE_CKPT}" \
         "${LARFORMER_SHOWER_BDT}" "${LARFORMER_SHOWER_BDT_NOVTX}" "${ATTACH_LLR}"; do
  [ -f "${WORKDIR}/${f}" ] || [ -f "${f}" ] || { echo "ERROR: chain input missing: ${f}" >&2; exit 2; }
done
echo "chain version : config=${CONFIG}"
echo "                slicer=${LARFORMER_BATTERY_SLICER_CKPT}"
echo "                segmenter=${LARFORMER_KP_PARTICLE_CKPT}"
echo "                shower BDTs=${LARFORMER_SHOWER_BDT} , ${LARFORMER_SHOWER_BDT_NOVTX}"
echo "                nu_reco args=${NU_RECO_EXTRA_ARGS}"

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

# ---- resume mode: RESUME_AFTER_INF=<inference job id> skips prep + inference
# (keeps the existing MSP_LIST and keypoint2_streams) and hangs regen on that
# job -- for re-running a single failed inference shard by hand and then the
# CPU tail without redoing the GPU pass.
if [ -n "${RESUME_AFTER_INF:-}" ]; then
  [ -s "${MSP_LIST}" ] || { echo "ERROR: resume needs the existing ${MSP_LIST}" >&2; exit 2; }
  INF=${RESUME_AFTER_INF}
  echo "RESUME: skipping prep + inference; regen waits for job ${INF}"
else
# ---- 0) prep: build the merged_sp list + clean downstream dirs --------------
# (find, not ls -- E2BIG at scale; clean keypoint2_streams + nu_reco dirs so
#  stale shard files can't poison the glob consumers.)
# stable (fileno,entry) sort so the cascade index<->event linkage is invariant
# to the merged_sp flat-vs-tree layout (list_merged_sp.py; NOT path-sorted find).
# MSP_LIST_SRC (optional): use a pre-built subset list (e.g. a pilot head -N)
# instead of listing the whole dir.
if [ -n "${MSP_LIST_SRC:-}" ]; then
  BUILD_LIST="cp ${MSP_LIST_SRC} ${MSP_LIST}"
else
  BUILD_LIST="apptainer exec --bind /cluster:/cluster ${CONTAINER} python3 \
    ${WORKDIR}/lartpc/data_prep/uboone_official/list_merged_sp.py \
    --dir ${MSP_DIR} --out ${MSP_LIST}"
fi
PREP=$(sbatch --parsable ${EXCL} $(dep_arg "${DEP}") \
  --partition=${CPU_PARTITION} --time=${JOB_TIME} --mem=4G --job-name=${TAG}_prep \
  --output=logs/data_prep/${TAG}_prep.%j.log \
  --error=logs/data_prep/${TAG}_prep.%j.err \
  --wrap="${BUILD_LIST}; \
          rm -rf ${KP2_STREAMS} ${NR_NU} ${NR_FM} ${LP_NU} ${LP_FM}; \
          mkdir -p ${KP2_STREAMS}; \
          echo \"merged_sp events: \$(wc -l < ${MSP_LIST})\"")
echo "prep      : ${PREP}  -> ${MSP_LIST}"

# ---- 1) inference (GPU) : merged_sp -> keypoint2_streams (nu + fm) ----------
# --output-tree: write cascade files into an index tree (avoid a 1M-file dir)
# GAMMA_SPEC (REQUIRED): the flash light-yield scale spec for this sample --
#   "auto"       legacy run-period table (dead_channels.GAMMA_SCALE_BY_PERIOD:
#                run1 -> 0.80, else 1.0; keyed on run number only, so run-1 MC
#                would inherit the value measured on run-1 DATA)
#   "auto:data" / "auto:mc" / "table"  calibrated (kind, period) cell in
#                lartpc/flashmatch/flash_calib.py (error if unmeasured)
#   "<float>"    explicit multiplier on gamma_beam (5.25)
# SAMPLE_KIND: data | mc | auto (default auto = from the merged_sp truth content)
# FLASH_WINDOW: off (default, legacy) | auto | lo,hi [us]
# INF_EXTRA_ARGS: any further inference flags. The chain refuses to launch
# without an explicit GAMMA_SPEC (or a --gamma-run-scale inside INF_EXTRA_ARGS)
# because the scale is baked into the GPU pass and decides the nu/fm streams.
# See lartpc/larformer_analysis/flashmodel_calib/PROTOCOL.md.
# Defaults (2026-09-12): the calibrated (kind, period) table, kind detected from
# the merged_sp truth content, in-window flash choice. EXT samples MUST pass
# FLASH_WINDOW=lo,hi (beam-off windows differ from beam-on: run-1 EXT 3.2,5.4
# vs bnb5e19 2.8,5.0; see flash_calib.FLASH_WINDOW_US). GAMMA_SPEC=auto
# reproduces the legacy run-period table.
GAMMA_SPEC=${GAMMA_SPEC:-table}
if [[ "${INF_EXTRA_ARGS:-}" == *"--gamma-run-scale"* ]]; then GAMMA_SPEC=""; fi
GAMMA_ARGS=""
[ -n "${GAMMA_SPEC}" ] && GAMMA_ARGS="--gamma-run-scale ${GAMMA_SPEC}"
GAMMA_ARGS="${GAMMA_ARGS} --sample-kind ${SAMPLE_KIND:-auto} --flash-window ${FLASH_WINDOW:-auto}"
echo "flash gamma: ${GAMMA_ARGS} ${INF_EXTRA_ARGS:-}"
INF=$(INPUT_LIST=${MSP_LIST} OUTPUT_DIR=${KP2_STREAMS}/ NSHARDS=${NINF} \
  EXTRA_INF_ARGS="--output-tree ${GAMMA_ARGS} ${INF_EXTRA_ARGS:-}" \
  sbatch --parsable ${EXCL} --export=ALL --dependency=afterok:${PREP} \
  --array=0-$((NINF-1)) --time=${JOB_TIME} \
  ${SLURMDIR}/submit_inference_shard.sh)
echo "inference : ${INF}  (${NINF} GPU shards) -> ${KP2_STREAMS}"
fi   # end of the non-resume (prep + inference) block

# ---- 2) regen: split keypoint2_streams into nu / fm lists -------------------
REGEN=$(sbatch --parsable ${EXCL} --dependency=afterok:${INF} \
  --partition=${CPU_PARTITION} --time=${JOB_TIME} --mem=4G --job-name=${TAG}_regen \
  --output=logs/export/${TAG}_regen.%j.log \
  --error=logs/export/${TAG}_regen.%j.err \
  --wrap="find ${KP2_STREAMS} -name 'keypoint2_event*_0.h5' ! -name '*_fm_0.h5' | sort > ${KP2_NU}; \
          find ${KP2_STREAMS} -name 'keypoint2_event*_fm_0.h5' | sort > ${KP2_FM}; \
          wc -l ${KP2_NU} ${KP2_FM}")
echo "regen     : ${REGEN}  -> ${KP2_NU} , ${KP2_FM}"

# ---- 3) nu_reco : nu + fm streams (LLR attachment) -------------------------
NRNU=$(KEYPOINT2_LIST=${KP2_NU} MERGED_SP_LIST=${MSP_LIST} OUTPUT_DIR=${NR_NU}/ \
  EXTRA_ARGS="${NU_RECO_EXTRA_ARGS}" \
  NSHARDS=${NNR} sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} --dependency=afterok:${REGEN} \
  --array=0-$((NNR-1)) ${SLURMDIR}/submit_nu_reco_shard.sh)
echo "nu_reco nu: ${NRNU}  (${NNR} shards) -> ${NR_NU}"

NRFM=$(KEYPOINT2_LIST=${KP2_FM} MERGED_SP_LIST=${MSP_LIST} OUTPUT_DIR=${NR_FM}/ \
  EXTRA_ARGS="${NU_RECO_EXTRA_ARGS}" \
  NSHARDS=${NNR} sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} --dependency=afterok:${REGEN} \
  --array=0-$((NNR-1)) ${SLURMDIR}/submit_nu_reco_shard.sh)
echo "nu_reco fm: ${NRFM}  (${NNR} shards) -> ${NR_FM}"

# ---- 4) larpid : nu + fm (CPU) ---------------------------------------------
LPNU=$(NU_RECO_DIR=${NR_NU} KP2_LIST=${KP2_NU} MERGED_SP_LIST=${MSP_LIST} \
  OUTPUT_DIR=${LP_NU} SAMPLE_TAG=${LARPID_TAG} DEVICE=cpu TAG=${TAG} \
  sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} --gres=gpu:0 \
  --dependency=afterok:${NRNU} --array=0-$((NNR-1)) \
  ${SLURMDIR}/submit_larpid_shard_cpu.sh)
echo "larpid  nu: ${LPNU}  -> ${LP_NU}"

LPFM=$(NU_RECO_DIR=${NR_FM} KP2_LIST=${KP2_FM} MERGED_SP_LIST=${MSP_LIST} \
  OUTPUT_DIR=${LP_FM} SAMPLE_TAG=${LARPID_TAG} DEVICE=cpu TAG=${TAG} \
  sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} --gres=gpu:0 \
  --dependency=afterok:${NRFM} --array=0-$((NNR-1)) \
  ${SLURMDIR}/submit_larpid_shard_cpu.sh)
echo "larpid  fm: ${LPFM}  -> ${LP_FM}"

# ---- 5) export (TRUTH_DIR set -> MC truth mode; else data mode) -------------
EXP=$(TAG=${TAG} MERGED_SP_LIST=${MSP_LIST} NSHARDS=${NEXP} \
  TRUTH_DIR=${TRUTH_DIR_IN} \
  KP2_NU_LIST=${KP2_NU} KP2_FM_LIST=${KP2_FM} \
  NU_RECO_NU_DIR=${LP_NU} NU_RECO_FM_DIR=${LP_FM} \
  OUT=${OUT_NTUPLE} \
  sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} --dependency=afterok:${LPNU}:${LPFM} \
  --array=0-$((NEXP-1)) ${SLURMDIR}/submit_export_shard.sh)
echo "export    : ${EXP}  (${NEXP} shards) -> ${OUT_NTUPLE%.root}_shard*.root"

# ---- 6) hadd merge ---------------------------------------------------------
HADD=$(TAG=${TAG} OUT=${OUT_NTUPLE} \
  sbatch --parsable ${EXCL} --export=ALL --partition=${CPU_PARTITION} --time=${JOB_TIME} \
  --dependency=afterok:${EXP} ${SLURMDIR}/submit_export_merge.sh)
echo "hadd      : ${HADD}  -> ${OUT_NTUPLE}"

echo
echo "chain submitted for TAG=${TAG}. final ntuple: ${OUT_NTUPLE}"
echo "watch: squeue -j ${PREP},${INF},${REGEN},${NRNU},${NRFM},${LPNU},${LPFM},${EXP},${HADD}"
