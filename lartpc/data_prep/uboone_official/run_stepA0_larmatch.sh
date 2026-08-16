#!/bin/bash
#
# run_stepA0_larmatch.sh — LArMatch deploy on OFFICIAL uboone dlmerged files
# (Stage A0 of the overlay data path; 2026-08-16).
#
# The official production already contains SSNet (ubspurn) products, so unlike
# the training-data step1 this runs ONLY the LArMatch deploy. Output: a
# larmatchme larlite ROOT per input file, consumed by attach_larmatch_scores.py
# to replace the dummy lm_score==1.0 in the stepA merged_sp H5s with real
# scores — closing the preprocessing gap vs the LANTERN training production
# (which hard-cut spacepoints at lm_score < 0.15 at deploy).
#
# Deploy min-score deliberately LOWER than the training 0.15 (default 0.05)
# so the cut can be applied (and varied) downstream at the dataset level;
# triplets absent from the deploy output get score 0.0 at attach time.
#
# The container payload is written to a script file and exec'd (clean $0 /
# positional args — ROOT's thisroot.sh shell detection breaks under an
# inline bash -c with a long command string).
#
# Env vars (sbatch --export or wrapper):
#   INPUTLIST  : dlmerged list (fileno = line number, 1-based)  [required]
#   FILENO     : line to process (or SLURM_ARRAY_TASK_ID)       [required]
#   OUTDIR     : where larmatchme_fileno<NNNNN>*.root goes      [required]
#   LM_MIN_SCORE (0.05), ADCNAME (wire), TBFLAG (-tb), MAX_EVENTS (-1)
#
#   sbatch --export=ALL,INPUTLIST=...,OUTDIR=... --array=1-N run_stepA0_larmatch.sh

#SBATCH --job-name=stepA0_lm
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/stepA0/larmatch.%A_%a.log
#SBATCH --error=logs/stepA0/larmatch.%A_%a.err

set -u

FILENO=${FILENO:-${SLURM_ARRAY_TASK_ID:?set FILENO or run as array}}
INPUTLIST=${INPUTLIST:?}
OUTDIR=${OUTDIR:?}
LM_MIN_SCORE=${LM_MIN_SCORE:-0.05}
ADCNAME=${ADCNAME:-wire}
TBFLAG=${TBFLAG:--tb}
MAX_EVENTS=${MAX_EVENTS:--1}
ZFILENO=$(printf "%05d" ${FILENO})

INPUT_ROOT=$(sed -n "${FILENO}p" "${INPUTLIST}")
[ -n "${INPUT_ROOT}" ] && [ -f "${INPUT_ROOT}" ] || {
    echo "ERROR: line ${FILENO} of ${INPUTLIST} missing: '${INPUT_ROOT}'"; exit 1; }
OUTFILE=${OUTDIR}/larmatchme_fileno${ZFILENO}.root
mkdir -p "${OUTDIR}"
if ls "${OUTDIR}"/larmatchme_fileno${ZFILENO}*larlite.root >/dev/null 2>&1; then
    echo "exists, skip: fileno ${ZFILENO}"; exit 0
fi

LANTERN_CONTAINER=${LANTERN_CONTAINER:-/cluster/tufts/wongjiradlab/larbys/cvmfs_containers/lantern_v2_me_06_03_prod/}
module load apptainer 2>/dev/null || true

NENTFLAG=""
if [ "${MAX_EVENTS}" -gt 0 ] 2>/dev/null; then NENTFLAG="-n ${MAX_EVENTS}"; fi

PAYLOAD=${OUTDIR}/.stepA0_payload_${ZFILENO}.sh
cat > "${PAYLOAD}" <<PEOF
#!/bin/bash
# NO set -e: the ubdl env scripts return benign nonzero statuses (step1
# never used it either). The payload's exit status is the deploy's (last
# command).
# the ubdl setup scripts (and their child processes) call bare python;
# only python3 exists in the container. A PATH shim covers children too.
# NOTE: no backticks in this heredoc -- it is UNQUOTED, the outer shell
# would command-substitute them.
shopt -s expand_aliases
alias python=python3
SHIM=\$(mktemp -d)
ln -s \$(command -v python3) \${SHIM}/python
export PATH=\${SHIM}:\${PATH}
echo '[payload] shim ready:' \$(command -v python)
export LANTERN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/lantern_ubdl/
export UBDL_DIR=\${LANTERN_DIR}/ubdl/
export LARMATCH_DIR=\${UBDL_DIR}/larflow/larmatchnet/larmatch/
export LANTERN_SCRIPTS=\${LANTERN_DIR}/lantern_scripts/
cd \${UBDL_DIR}
source setenv_py3_container.sh
source configure_container.sh
echo '[payload] base env done'
cd \${UBDL_DIR}/larflow/larmatchnet
source set_pythonpath.sh
export PYTHONPATH=\${LARMATCH_DIR}:\${PYTHONPATH}
echo '[payload] pythonpath done; starting deploy'
cd ${OUTDIR}
python3 \${LARMATCH_DIR}/deploy_larmatchme.py \\
    --config-file \${LANTERN_SCRIPTS}/config_larmatchme_deploycpu.yaml \\
    --supera ${INPUT_ROOT} \\
    --weights \${LARMATCH_DIR}/larmatch_ckpt78k.pt \\
    --output ${OUTFILE} \\
    --min-score ${LM_MIN_SCORE} \\
    --adc-name ${ADCNAME} --chstatus-name ${ADCNAME} \\
    --device-name cpu --use-skip-limit ${TBFLAG} ${NENTFLAG}
PEOF
chmod +x "${PAYLOAD}"

apptainer exec --bind /cluster:/cluster "${LANTERN_CONTAINER}" bash "${PAYLOAD}"
rc=$?
[ ${rc} -eq 0 ] && rm -f "${PAYLOAD}" || echo "payload kept: ${PAYLOAD}"
echo "stepA0 fileno ${ZFILENO} exit ${rc}"
ls -lh "${OUTDIR}"/larmatchme_fileno${ZFILENO}* 2>/dev/null
exit ${rc}
