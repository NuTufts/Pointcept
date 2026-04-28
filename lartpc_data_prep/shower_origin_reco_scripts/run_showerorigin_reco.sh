#!/bin/bash
#
# run_showerorigin_reco.sh
# Main per-job run script. Runs on the bare node (NOT inside a container).
# Processes a stride of input files determined by SLURM_ARRAY_TASK_ID.
# Uses apptainer exec twice: once for Step 1 (lantern) and once for Steps 2-4 (pointcept).
#

# ==================== CONFIGURATION ====================
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/shower_origin_reco_scripts
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl
POINTCEPT_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
LANTERN_CONTAINER=/cvmfs/uboone.opensciencegrid.org/containers/lantern_v2_me_06_03_prod
POINTCEPT_CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# ==================== DATASET CONFIGURATION ====================
#INPUTLIST=${POINTCEPT_DIR}/lartpc_data_prep/inputlists/bnb_nu_pi0filter_corsika.txt

#INPUTLIST=${POINTCEPT_DIR}/lartpc_data_prep/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge_devlist.txt
#TAG=showerorigin_reco_mcc9_v29e_dl_run3b_bnb_nu_overlay_test

INPUTLIST=${POINTCEPT_DIR}/lartpc_data_prep/inputlists/mcc9_v40_run3b_ncpi0_dlana.txt
TAG=showerorigin_reco_mcc9_v40_run3b_ncpi0_dlana

OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/shower_origin_reco/${TAG}_test

ADCNAME=wire
TBFLAG="-tb"
MCC9FLAG="--mcc9"

stride=1
OFFSET=0

# Set variable below when testing. Comment out when running on cluster.
SLURM_ARRAY_TASK_ID=0

# Step 5-7 configuration (shower origin inference + merger + ROOT output)
SHOWER_ORIGIN_CONFIG=${POINTCEPT_DIR}/configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py
SHOWER_ORIGIN_CKPT=${POINTCEPT_DIR}/shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter_recofragments/model/model_epoch165.pth
DEVICE=cuda  # Set to "cuda" if using GPU partition (cpu-mode does not work)
ROOT_OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/root/shower_reco/mcc9_v29e_dl_run3b_bnb_nu_overlay_test
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
    local_jobdir=$(printf "/tmp/shower_origin_reco_${TAG}_jobid%04d_line%05d" ${jobid} ${lineno})
    rm -rf ${local_jobdir}
    mkdir -p ${local_jobdir}

    # Copy input ROOT file to workdir
    echo "Copying input file to ${local_jobdir}" >> ${local_logfile}
    cp ${inputfile} ${local_jobdir}/
    echo "Copy done: $(date)" >> ${local_logfile}

    # ---- STEP 1: Run lantern workflow inside lantern container ----
    echo "--- STEP 1: lantern workflow ---" >> ${local_logfile}
    apptainer exec --bind /cluster/tufts:/cluster/tufts,/tmp:/tmp \
        ${LANTERN_CONTAINER} \
        bash -c "cd ${local_jobdir} && source ${WORKDIR}/run_step1_lantern.sh ${local_jobdir} ${ADCNAME} ${TBFLAG}" \
        >> ${local_logfile} 2>&1
    step1_status=$?
    echo "Step 1 exit status: ${step1_status}" >> ${local_logfile}

    if [ ${step1_status} -ne 0 ]; then
        echo "Step 1 FAILED for line ${lineno}, skipping remaining steps" >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi

    # Verify Step 1 outputs exist
    if [ ! -f "${local_jobdir}/larmatchme_larlite.root" ] || [ ! -f "${local_jobdir}/merged_dlreco_with_ssnet.root" ]; then
        echo "Step 1 output files missing for line ${lineno}, skipping" >> ${local_logfile}
        ls -la ${local_jobdir}/ >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi
    echo "Step 1 outputs verified" >> ${local_logfile}

    # ---- STEPS 2-4: Run pointcept processing inside pointcept container ----
    echo "--- STEPS 2-4: pointcept processing ---" >> ${local_logfile}
    apptainer exec --nv --bind /cluster:/cluster,/tmp:/tmp \
        ${POINTCEPT_CONTAINER} \
        bash -c "cd ${local_jobdir} && source ${WORKDIR}/run_step234_pointcept.sh ${local_jobdir} ${lineno} ${TAG} ${ADCNAME} ${TBFLAG} ${MCC9FLAG}" \
        >> ${local_logfile} 2>&1
    step234_status=$?
    echo "Steps 2-4 exit status: ${step234_status}" >> ${local_logfile}

    if [ ${step234_status} -ne 0 ]; then
        echo "Steps 2-4 FAILED for line ${lineno}" >> ${local_logfile}
        rm -rf ${local_jobdir}
        continue
    fi

    # Compute output subdirectories
    let nsubdir1=$(expr "${lineno}/1000")
    zsubdir1=$(printf %03d ${nsubdir1})
    let nsubdir2=$(expr "${lineno}/100")
    zsubdir2=$(printf %03d ${nsubdir2})

    # ---- STEPS 5-7: Run shower origin inference + merging + ROOT output ----
    echo "--- STEPS 5-7: shower origin pipeline ---" >> ${local_logfile}
    export SHOWER_ORIGIN_CONFIG SHOWER_ORIGIN_CKPT DEVICE
    apptainer exec --nv --bind /cluster:/cluster,/tmp:/tmp \
        ${POINTCEPT_CONTAINER} \
        bash -c "cd ${local_jobdir} && source ${WORKDIR}/run_step567_pointcept.sh ${local_jobdir} ${lineno} ${TAG}" \
        >> ${local_logfile} 2>&1
    step567_status=$?
    echo "Steps 5-7 exit status: ${step567_status}" >> ${local_logfile}

    if [ ${step567_status} -ne 0 ]; then
        echo "Steps 5-7 FAILED for line ${lineno} (H5 output still available)" >> ${local_logfile}
    fi

    # Copy ROOT files to output dir
    nroot=$(ls ${local_jobdir}/showerreco_*.root 2>/dev/null | wc -l)
    if [ ${nroot} -gt 0 ]; then
        mkdir -p ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
        cp ${local_jobdir}/showerreco_*.root ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
        echo "Copied ${nroot} ROOT file(s) to ${ROOT_OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/" >> ${local_logfile}
    fi

    # Copy final merged H5 files to output dir
    outfolder=${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
    mkdir -p ${outfolder}

    echo "Copying merged H5 files to ${outfolder}" >> ${local_logfile}
    nmerged=$(ls ${local_jobdir}/merged_*.h5 2>/dev/null | wc -l)
    echo "Number of merged H5 files: ${nmerged}" >> ${local_logfile}

    if [ ${nmerged} -gt 0 ]; then
        cp ${local_jobdir}/merged_*.h5 ${outfolder}/
        echo "Copy complete" >> ${local_logfile}
        ls -lh ${outfolder}/merged_*.h5 >> ${local_logfile} 2>/dev/null
    else
        echo "WARNING: No merged H5 files produced for line ${lineno}" >> ${local_logfile}
    fi

    echo "Finish time: $(date)" >> ${local_logfile}

    # Clean up workdir
    rm -rf ${local_jobdir}

done

echo "" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
echo "Job finished: $(date)" >> ${local_logfile}
echo "======================================" >> ${local_logfile}
