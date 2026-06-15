#!/bin/bash
#
# submit_deghost_compare_3k_pin.sh — CORRECTED deghost/rescue sweep.
# The unpinned 3k sweep (job 1178191) was CONFOUNDED: each array task landed on a
# different A100 SKU (pax003=a100:2 40GB PCIe vs pax049/105=a100:8 80GB SXM) and
# deterministic mode is only bit-exact WITHIN a GPU SKU/driver. Cross-node
# variation (~9 events TP, 36/378 drop-flag flips) exceeded the tau effect.
#
# Fix: pin ALL arms to ONE node (same SKU + driver) so differences are real tau /
# rescue effects. Extra arm `base_dup` = identical config to `base` on a different
# physical GPU of the same node — validates that same-node IS reproducible (expect
# base == base_dup bit-for-bit).
#   sbatch submit_deghost_compare_3k_pin.sh

#SBATCH --job-name=sp_dgfix
#SBATCH --partition=gpu
#SBATCH --nodelist=pax052
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/dgfix/arm.%A_%a.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon/slurm/logs/dgfix/arm.%A_%a.err

SP=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/single_photon
CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_scripts/larformer_configs/single_photon_scale1500.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
DATADIR=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v29e_dl_run3b_bnb_nu_overlay
INLIST=${SP}/workdir_scale/cascade_inputs.txt
OUTBASE=${DATADIR}/sp_compare_3k_fixed

NAMES=(base    dg0p4  dg0p3  rescue  dg0p3_resc  base_dup)
DEGHOST=(""    "0.4"  "0.3"  ""      "0.3"       "")
RESCUE=( "0"   "0"    "0"    "1"     "1"         "0")

i=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$i]}
OUTDIR=${OUTBASE}/${NAME}
mkdir -p ${OUTDIR} ${SP}/slurm/logs/dgfix

ENVS="DETERMINISTIC=1"
[ -n "${DEGHOST[$i]}" ] && ENVS="${ENVS} DEGHOST_THRESHOLD_VAL=${DEGHOST[$i]}"
if [ "${RESCUE[$i]}" = "1" ]; then
    ENVS="${ENVS} FLASH_RECOVER_K=1 FLASH_RECOVER_CHI2_MAX=500 FLASH_RECOVER_OOB_MAX=0.05 FLASH_RECOVER_GAMMA=5.25"
fi
echo "ARM ${NAME}  (task ${i})  node=$(hostname)"
echo "  envs : ${ENVS}"
echo "  out  : ${OUTDIR}  ($(wc -l < ${INLIST}) events)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

module load apptainer/1.4.0 2>/dev/null || true
apptainer exec --nv --bind /cluster:/cluster ${SIF} \
    bash -c "export ${ENVS}; source ${SP}/run_stageB_capped.sh ${CONFIG} ${INLIST} ${OUTDIR}"
echo "ARM ${NAME} DONE"
