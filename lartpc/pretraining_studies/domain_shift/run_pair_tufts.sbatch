#!/bin/bash
# One compute_metrics.py battery on a single feature pair (CPU). Usage:
#   sbatch run_pair_tufts.sbatch <a.npz> <b.npz> <label> <out.json>
#
#SBATCH --job-name=dspair
#SBATCH --partition=batch
#SBATCH --time=47:59:00
                          # early-snapshot pairs exceed 2h: weakly separated
                          # features make the PAD logreg fits converge slowly
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dspair.%j.log
#SBATCH --error=logs/dspair.%j.err

set -u
REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
A=${1:?a.npz}; B=${2:?b.npz}; LABEL=${3:?label}; OUT=${4:?out.json}

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --bind /cluster:/cluster "$CONTAINER" bash -c "
set -eu
cd ${REPO}/lartpc/pretraining_studies/domain_shift
python3 compute_metrics.py --a ${A} --b ${B} --label '${LABEL}' --out ${OUT}
echo PAIR-COMPLETE
"
