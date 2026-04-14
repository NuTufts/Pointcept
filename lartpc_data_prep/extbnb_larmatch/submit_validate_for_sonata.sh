#!/bin/bash

#SBATCH --job-name=val_sonata
#SBATCH --output=logs/gridlog_val_sonata.%j.%A_%a.%N.log
#SBATCH --error=logs/griderr_val_sonata.%j.%A_%a.%N.err
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --array=0-54

# ==================== CONFIGURATION ====================
WORKDIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch
POINTCEPT_DIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept
POINTCEPT_CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

FILELIST=${WORKDIR}/hdlist_extbnb_larmatch_run3_g1.txt
OUTDIR=${WORKDIR}/validation_sonata_results

# Files per array job (544723 files / 55 jobs ~ 9904 per job)
STRIDE=10000
# =======================================================

mkdir -p ${WORKDIR}/logs
mkdir -p ${OUTDIR}

module load apptainer/1.2.4-suid

jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${STRIDE}*${jobid}+1")
let endline=$(expr "${STRIDE}*(${jobid}+1)")

# Extract this job's chunk of the file list
chunk_file=${OUTDIR}/chunk_${jobid}.txt
sed -n "${startline},${endline}p" ${FILELIST} > ${chunk_file}

nfiles=$(wc -l < ${chunk_file})
echo "Job ${jobid}: lines ${startline}-${endline}, ${nfiles} files"

if [ ${nfiles} -eq 0 ]; then
    echo "No files in chunk, exiting"
    exit 0
fi

# Run validation inside pointcept container
apptainer exec --bind /cluster:/cluster,/tmp:/tmp \
    ${POINTCEPT_CONTAINER} \
    python3 ${WORKDIR}/validate_for_sonata.py \
        ${chunk_file} \
        --workers 4 \
        --min-points 50 \
        --output ${OUTDIR}/bad_files_${jobid}.txt

echo "Job ${jobid} finished: $(date)"
