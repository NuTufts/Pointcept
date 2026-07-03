#!/bin/bash
#
# submit_determinism_test.sh — confirm the run-to-run reproducibility diagnosis.
# Runs the baseline cascade (no flash-recover, no dedup) on the SAME ~24 1g0X
# events FOUR times:
#   nondet_a, nondet_b  — default (current) settings
#   det_a,   det_b      — DETERMINISTIC=1 (--deterministic)
# then diffs a-vs-b in each mode. Expectation: nondet shows per-SP / label
# flips (~the 9% effect); det shows ZERO.
#   sbatch submit_determinism_test.sh

#SBATCH --job-name=sp_determ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/repeatability_tests/slurm/logs/determinism/run.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc/larformer_analysis/physics/repeatability_tests/slurm/logs/determinism/run.%j.err

POINTCEPT=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
REPEAT=${POINTCEPT}/lartpc/larformer_analysis/physics/repeatability_tests
SP=${POINTCEPT}/lartpc/larformer_analysis/physics/single_photon   # run_stageB_capped + input lists
CONFIG=${POINTCEPT}/lartpc/data_prep/uboone_official/larformer_configs/single_photon_scale1500.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

NEV=${NEV:-24}
WORK=${WORKDIR:-${REPEAT}/workdir/determinism_test}
mkdir -p ${WORK} ${REPEAT}/slurm/logs/determinism
SLICE=${WORK}/events.txt
head -n ${NEV} ${SP}/workdir_scale/cascade_inputs_1g0X.txt > ${SLICE}
echo "determinism test on $(wc -l < ${SLICE}) events"
nvidia-smi --query-gpu=name --format=csv,noheader

module load apptainer/1.4.0 2>/dev/null || true

run() {  # $1=mode(0/1) $2=outdir
    rm -rf "$2"; mkdir -p "$2"
    apptainer exec --nv --bind /cluster:/cluster ${SIF} \
        bash -c "export DETERMINISTIC=$1; source ${SP}/run_stageB_capped.sh ${CONFIG} ${SLICE} $2"
}

echo "########## NONDET run a ##########"; run 0 ${WORK}/nondet_a
echo "########## NONDET run b ##########"; run 0 ${WORK}/nondet_b
echo "########## DET    run a ##########"; run 1 ${WORK}/det_a
echo "########## DET    run b ##########"; run 1 ${WORK}/det_b

echo; echo "################ DIFF: NONDET a vs b (expect flips) ################"
apptainer exec --nv --bind /cluster:/cluster ${SIF} bash -c \
    "source ${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}/setenv_pointcept_container.sh >/dev/null 2>&1; \
     export PYTHONPATH=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept:\$PYTHONPATH; \
     python3 ${REPEAT}/determinism_diff.py ${WORK}/nondet_a ${WORK}/nondet_b --thr 20"

echo; echo "################ DIFF: DET a vs b (expect ZERO) ###################"
apptainer exec --nv --bind /cluster:/cluster ${SIF} bash -c \
    "source ${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}/setenv_pointcept_container.sh >/dev/null 2>&1; \
     export PYTHONPATH=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept:\$PYTHONPATH; \
     python3 ${REPEAT}/determinism_diff.py ${WORK}/det_a ${WORK}/det_b --thr 20"
echo; echo "DONE"
