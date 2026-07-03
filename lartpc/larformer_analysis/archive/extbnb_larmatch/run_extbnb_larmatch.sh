#!/bin/bash
#
# run_extbnb_larmatch.sh
# Main per-job run script. Runs on the bare node (NOT inside a container).
# Processes a stride of input files determined by SLURM_ARRAY_TASK_ID.
# Uses apptainer exec twice:
#   Step 1 (lantern container): SSNet + LArMatch
#   Step 2 (pointcept container): Convert to HDF5 with larmatch scores
#

# ==================== CONFIGURATION ====================
WORKDIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc/larformer_analysis/archive/extbnb_larmatch
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl

# Input list of merged_dlreco ROOT files
INPUTLIST=/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/inputlists/inputlist_run3_G1_extbnb_dlreco.txt

# Output directory for final HDF5 files
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/

# Containers
LANTERN_CONTAINER=/cvmfs/uboone.opensciencegrid.org/containers/lantern_v2_me_06_03_prod
POINTCEPT_CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# Data format flags
ADCNAME=wire
TBFLAG="-tb"

TAG=extbnb_larmatch
stride=100
OFFSET=0

# Set variable below when testing locally. Comment out when running on cluster.
#SLURM_ARRAY_TASK_ID=0
# =======================================================

jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${OFFSET}+${stride}*${jobid}")

# Create persistent log directory
jobworkdir=$(printf "%s/workdir/${TAG}_jobid_%04d" ${WORKDIR} ${jobid})
mkdir -p ${jobworkdir}
mkdir -p ${OUTPUT_DIR}

# Main log file
local_logfile=${jobworkdir}/log_${TAG}_jobid${jobid}.txt
echo "======================================" > ${local_logfile}
echo "Job started: $(date)" >> ${local_logfile}
echo "SLURM_ARRAY_TASK_ID: ${jobid}" >> ${local_logfile}
echo "startline: ${startline}" >> ${local_logfile}
echo "stride: ${stride}" >> ${local_logfile}
echo "======================================" >> ${local_logfile}

for (( i=1; i<=stride; i++ )); do

    let lineno=${startline}+${i}

    # Get the input file path
    inputfile=$(sed -n ${lineno}p ${INPUTLIST})

    if [ -z "${inputfile}" ]; then
        echo "LINE ${lineno}: empty, skipping" >> ${local_logfile}
        continue
    fi

    if [ ! -f "${inputfile}" ]; then
        echo "LINE ${lineno}: file not found: ${inputfile}" >> ${local_logfile}
        continue
    fi

    inputbasename=$(basename ${inputfile})
    echo "" >> ${local_logfile}
    echo "============================================" >> ${local_logfile}
    echo "Processing line ${lineno}: ${inputbasename}" >> ${local_logfile}
    echo "Start time: $(date)" >> ${local_logfile}
    echo "============================================" >> ${local_logfile}

    # Create per-file working directory in /tmp
    local_jobdir=$(printf "/tmp/extbnb_larmatch_${TAG}_jobid%04d_line%05d" ${jobid} ${lineno})
    rm -rf ${local_jobdir}
    mkdir -p ${local_jobdir}

    # Copy input ROOT file to workdir
    echo "Copying input file to ${local_jobdir}" >> ${local_logfile}
    cp ${inputfile} ${local_jobdir}/
    echo "Copy done: $(date)" >> ${local_logfile}

    # ---- STEP 1: Run SSNet + LArMatch inside lantern container ----
    echo "--- STEP 1: lantern (SSNet + LArMatch) ---" >> ${local_logfile}
    apptainer exec --bind /cluster/tufts:/cluster/tufts,/tmp:/tmp \
        ${LANTERN_CONTAINER} \
        bash -c "cd ${local_jobdir} && source ${WORKDIR}/run_step1_lantern.sh ${local_jobdir} ${ADCNAME} ${TBFLAG}" \
        >> ${local_logfile} 2>&1
    step1_status=$?
    echo "Step 1 exit status: ${step1_status}" >> ${local_logfile}

    if [ ${step1_status} -ne 0 ]; then
        echo "Step 1 FAILED for line ${lineno}, skipping Step 2" >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi

    # Verify Step 1 outputs exist
    if [ ! -f "${local_jobdir}/larmatchme_larlite.root" ] || \
       [ ! -f "${local_jobdir}/merged_dlreco_with_ssnet.root" ]; then
        echo "Step 1 output files missing for line ${lineno}, skipping" >> ${local_logfile}
        ls -la ${local_jobdir}/ >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi
    echo "Step 1 outputs verified" >> ${local_logfile}

    # ---- STEP 2: Convert to HDF5 inside pointcept container ----
    echo "--- STEP 2: pointcept (convert to HDF5) ---" >> ${local_logfile}
    apptainer exec --nv --bind /cluster:/cluster,/tmp:/tmp \
        ${POINTCEPT_CONTAINER} \
        bash -c "cd ${local_jobdir} && source ${WORKDIR}/run_step2_pointcept.sh ${local_jobdir} ${lineno} ${TAG} ${ADCNAME} ${TBFLAG} ${inputfile}" \
        >> ${local_logfile} 2>&1
    step2_status=$?
    echo "Step 2 exit status: ${step2_status}" >> ${local_logfile}

    if [ ${step2_status} -ne 0 ]; then
        echo "Step 2 FAILED for line ${lineno}" >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi

    # Copy final H5 files to output dir
    let nsubdir1=$(expr "${lineno}/1000")
    zsubdir1=$(printf %03d ${nsubdir1})
    let nsubdir2=$(expr "${lineno}/100")
    zsubdir2=$(printf %03d ${nsubdir2})

    outfolder=${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
    mkdir -p ${outfolder}

    echo "Copying H5 files to ${outfolder}" >> ${local_logfile}
    nh5=$(ls ${local_jobdir}/h5_output/pointceptdata_*.h5 2>/dev/null | wc -l)
    echo "Number of H5 files: ${nh5}" >> ${local_logfile}

    if [ ${nh5} -gt 0 ]; then
        cp ${local_jobdir}/h5_output/pointceptdata_*.h5 ${outfolder}/
        echo "Copy complete" >> ${local_logfile}
        ls -lh ${outfolder}/pointceptdata_*.h5 >> ${local_logfile} 2>/dev/null
    else
        echo "WARNING: No H5 files produced for line ${lineno}" >> ${local_logfile}
    fi

    echo "Finish time: $(date)" >> ${local_logfile}

    # Clean up workdir
    rm -rf ${local_jobdir}

done

echo "" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
echo "Job finished: $(date)" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
