#!/bin/bash
#
# submit_membership_test.sh — test PER-EVENT (membership / list / batch) reproducibility.
#
# Runs the 1g0X cascade (run_stageB_capped.sh, the test analysis) on a small "probe"
# list and on a larger "padded" superset that contains the probe events, then diffs
# the COMMON probe events. An event's output MUST NOT depend on which OTHER events are
# in the run, so with the fixes in place the diff must be 0 everywhere.
#
# This is the test that caught the two train-time RNG augmentations leaking into
# inference (shuffle_orders, max_source_tokens_per_level). Run it after any change to
# the model / inference path, or when moving to a new cluster.
#
#   sbatch --nodelist=<node> submit_membership_test.sh        # default 30 probe vs 200 padded
# Env knobs: PROBE_N (default 30), PAD_N (default 200), WORKDIR.
#
# Stronger test: make the padded list's extra events LARGE (high spacepoint count) to
# maximize the membership perturbation (see the README — the original bug showed up
# strongest with ~800k-point padding events).

#SBATCH --job-name=rt_member
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/repeatability_tests/slurm/logs/membership/run.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/repeatability_tests/slurm/logs/membership/run.%j.err

POINTCEPT=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
REPEAT=${POINTCEPT}/lartpc_data_prep/larformer_physics/repeatability_tests
SP=${POINTCEPT}/lartpc_data_prep/larformer_physics/single_photon       # run_stageB_capped + input lists
CONFIG=${POINTCEPT}/lartpc_data_prep/larformer_scripts/larformer_configs/single_photon_scale1500.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
UBDL=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl

PROBE_N=${PROBE_N:-30}; PAD_N=${PAD_N:-200}
WORK=${WORKDIR:-${REPEAT}/workdir/membership_test}
mkdir -p ${WORK} ${REPEAT}/slurm/logs/membership
LIST=${SP}/workdir_scale/cascade_inputs_1g0X.txt
head -n ${PROBE_N} ${LIST} > ${WORK}/probe.txt
head -n ${PAD_N}   ${LIST} > ${WORK}/padded.txt
echo "membership test: ${PROBE_N}-event probe vs ${PAD_N}-event padded (common = ${PROBE_N})"
nvidia-smi --query-gpu=name --format=csv,noheader

module load apptainer/1.4.0 2>/dev/null || true
run() {  # $1=input-list $2=outdir
    rm -rf "$2"; mkdir -p "$2"
    apptainer exec --nv --bind /cluster:/cluster ${SIF} \
        bash -c "export DETERMINISTIC=1; source ${SP}/run_stageB_capped.sh ${CONFIG} $1 $2"
}
echo "########## cascade on probe list ##########";  run ${WORK}/probe.txt  ${WORK}/probe_out
echo "########## cascade on padded list ##########"; run ${WORK}/padded.txt ${WORK}/padded_out

echo; echo "######## MEMBERSHIP DIFF: probe vs padded on the common ${PROBE_N} events (expect ALL 0) ########"
apptainer exec --nv --bind /cluster:/cluster ${SIF} bash -c \
    "source ${UBDL}/setenv_pointcept_container.sh >/dev/null 2>&1; \
     export PYTHONPATH=${POINTCEPT}:\$PYTHONPATH; \
     python3 ${REPEAT}/determinism_diff.py ${WORK}/probe_out ${WORK}/padded_out --thr 20 --by-rse"
echo; echo "DONE  (drop-flag / LABEL flips / coord mismatch / per-SP must all be 0)"
