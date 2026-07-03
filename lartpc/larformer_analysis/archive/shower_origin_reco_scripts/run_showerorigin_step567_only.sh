#!/bin/bash
#
# run_showerorigin_step567_only.sh
# Standalone job script for Steps 5-7 only (shower origin inference + merge + ROOT).
# Designed for GPU jobs that process H5 files already produced by Steps 1-4 on CPU.
#
# For each input file line, locates the merged H5 event files in H5_INPUT_DIR,
# runs Steps 5-7, and writes one ROOT file per original input file.
#
# Runs on the bare node (NOT inside a container). Calls apptainer for the
# pointcept container.
#
# Usage:
#   sbatch --array=0-N run_showerorigin_step567_only.sh
# Or for testing:
#   bash run_showerorigin_step567_only.sh
#

# ==================== CONFIGURATION ====================
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/archive/shower_origin_reco_scripts
POINTCEPT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
POINTCEPT_CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# Input file list (same list used for Steps 1-4 so line numbers match)
INPUTLIST=${POINTCEPT_DIR}/lartpc_data_prep/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge_devlist.txt

# Directory where Steps 1-4 wrote the merged H5 files
H5_INPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/shower_origin_reco/mcc9_v29e_dl_run3b_bnb_nu_overlay_test

# ROOT output directory
ROOT_OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/root/shower_reco/mcc9_v29e_dl_run3b_bnb_nu_overlay_test_showerrecoonly

# Tag used by Steps 1-4 to name the merged H5 files
TAG=showerorigin_reco_mcc9_v29e_dl_run3b_bnb_nu_overlay_test

# Model configuration
SHOWER_ORIGIN_CONFIG=${POINTCEPT_DIR}/configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py
SHOWER_ORIGIN_CKPT=${POINTCEPT_DIR}/shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter_recofragments/model/model_epoch165.pth
DEVICE=cuda

# Merger parameters
ORIGIN_PROXIMITY_RADIUS=5.0
CONE_HALF_ANGLE=20.0
CONE_MAX_LENGTH=300.0
MIN_POINTS_FRACTION_IN_CONE=0.5

# Job stride: number of input file lines to process per array task
stride=1
OFFSET=0

# Set variable below when testing locally. Comment out when running on cluster.
SLURM_ARRAY_TASK_ID=0
# =======================================================

jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${OFFSET}+${stride}*${jobid}")

# Create persistent log directory
jobworkdir=$(printf "%s/workdir/${TAG}_step567_jobid_%04d" ${WORKDIR} ${jobid})
mkdir -p ${jobworkdir}
mkdir -p ${ROOT_OUTPUT_DIR}

# Main log file
local_logfile=${jobworkdir}/log_step567_${TAG}_jobid${jobid}.txt
echo "======================================" > ${local_logfile}
echo "Job started: $(date)" >> ${local_logfile}
echo "SLURM_ARRAY_TASK_ID: ${jobid}" >> ${local_logfile}
echo "startline: ${startline}" >> ${local_logfile}
echo "stride: ${stride}" >> ${local_logfile}
echo "H5_INPUT_DIR: ${H5_INPUT_DIR}" >> ${local_logfile}
echo "ROOT_OUTPUT_DIR: ${ROOT_OUTPUT_DIR}" >> ${local_logfile}
echo "======================================" >> ${local_logfile}

for (( i=1; i<=stride; i++ )); do

    let lineno=${startline}+${i}

    # Get the input file path (used only for identification, not read directly)
    inputfile=$(sed -n ${lineno}p ${INPUTLIST})

    if [ -z "${inputfile}" ]; then
        echo "LINE ${lineno}: empty, skipping" >> ${local_logfile}
        continue
    fi

    echo "" >> ${local_logfile}
    echo "============================================" >> ${local_logfile}
    echo "Processing line ${lineno}: $(basename ${inputfile})" >> ${local_logfile}
    echo "Start time: $(date)" >> ${local_logfile}
    echo "============================================" >> ${local_logfile}

    # Locate merged H5 files for this line number
    # Steps 1-4 store them in H5_INPUT_DIR/<NNN>/<NNN>/merged_<TAG>_fileno<NNNNN>_entry*.h5
    ZFILENO=$(printf "%05d" ${lineno})
    let nsubdir1=$(expr "${lineno}/1000")
    zsubdir1=$(printf "%03d" ${nsubdir1})
    let nsubdir2=$(expr "${lineno}/100")
    zsubdir2=$(printf "%03d" ${nsubdir2})

    h5_subdir=${H5_INPUT_DIR}/${zsubdir1}/${zsubdir2}
    h5_pattern="merged_${TAG}_fileno${ZFILENO}_entry*.h5"

    h5_files=$(ls ${h5_subdir}/${h5_pattern} 2>/dev/null)
    if [ -z "${h5_files}" ]; then
        echo "No H5 files found matching ${h5_subdir}/${h5_pattern}, skipping" >> ${local_logfile}
        continue
    fi

    nfiles=$(echo "${h5_files}" | wc -l)
    echo "Found ${nfiles} merged H5 file(s) for fileno ${ZFILENO}" >> ${local_logfile}

    # Create per-file working directory in /tmp
    local_jobdir=$(printf "/tmp/step567_${TAG}_jobid%04d_line%05d" ${jobid} ${lineno})
    rm -rf ${local_jobdir}
    mkdir -p ${local_jobdir}

    # Copy H5 files to local workdir
    echo "Copying H5 files to ${local_jobdir}" >> ${local_logfile}
    for f in ${h5_files}; do
        cp ${f} ${local_jobdir}/
    done
    echo "Copy done: $(date)" >> ${local_logfile}

    # Create file list for the pipeline script
    FILELIST=${local_jobdir}/step567_filelist.txt
    ls ${local_jobdir}/merged_*.h5 > ${FILELIST} 2>/dev/null

    echo "File list contents:" >> ${local_logfile}
    cat ${FILELIST} >> ${local_logfile}

    # Output: one ROOT file per input file, named by tag and file number
    COMBINED_ROOT=${local_jobdir}/showerreco_${TAG}_fileno${ZFILENO}.root

    # ---- STEPS 5-7: Run shower origin pipeline inside pointcept container ----
    echo "--- STEPS 5-7: shower origin pipeline ---" >> ${local_logfile}

    apptainer exec --nv --bind /cluster:/cluster,/tmp:/tmp \
        ${POINTCEPT_CONTAINER} \
        bash -c "
cd ${local_jobdir}
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl && source setenv_pointcept_container.sh && cd ${local_jobdir}
python3 ${POINTCEPT_DIR}/tools/run_shower_origin_pipeline_step567.py \
    -c ${SHOWER_ORIGIN_CONFIG} \
    --checkpoint ${SHOWER_ORIGIN_CKPT} \
    --input-list ${FILELIST} \
    --output-dir ${local_jobdir}/root_output \
    --device ${DEVICE} \
    --origin-proximity-radius ${ORIGIN_PROXIMITY_RADIUS} \
    --cone-half-angle ${CONE_HALF_ANGLE} \
    --cone-max-length ${CONE_MAX_LENGTH} \
    --min-points-fraction-in-cone ${MIN_POINTS_FRACTION_IN_CONE} \
    --combined-output ${COMBINED_ROOT}
" >> ${local_logfile} 2>&1
    step567_status=$?
    echo "Steps 5-7 exit status: ${step567_status}" >> ${local_logfile}

    if [ ${step567_status} -ne 0 ]; then
        echo "Steps 5-7 FAILED for line ${lineno}" >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi

    # Copy combined ROOT file to output dir
    if [ -f "${COMBINED_ROOT}" ]; then
        mkdir -p ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
        cp ${COMBINED_ROOT} ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
        echo "Copied combined ROOT file to ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/" >> ${local_logfile}
        ls -lh ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/$(basename ${COMBINED_ROOT}) >> ${local_logfile} 2>/dev/null
    else
        echo "WARNING: No ROOT file produced for line ${lineno}" >> ${local_logfile}
    fi

    echo "Finish time: $(date)" >> ${local_logfile}

    # Clean up
    rm -rf ${local_jobdir}

done

echo "" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
echo "Job finished: $(date)" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
