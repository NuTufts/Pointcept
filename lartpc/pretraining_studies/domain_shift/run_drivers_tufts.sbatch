#!/bin/bash
# Discriminant-driver diagnostic (CPU; the h5 observable loop reads ~10 GB).
#   sbatch run_drivers_tufts.sbatch <feature-prefix>
# e.g.  sbatch run_drivers_tufts.sbatch features/P5B.1_img6144000
#
#SBATCH --job-name=dsdrivers
#SBATCH --partition=batch
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/dsdrivers.%j.log
#SBATCH --error=logs/dsdrivers.%j.err

set -u
REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
PREFIX=${1:?feature prefix, e.g. features/P5B.1_img6144000}

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --bind /cluster:/cluster "$CONTAINER" bash -c "
set -eu
cd ${REPO}/lartpc/pretraining_studies/domain_shift
TAG=\$(basename ${PREFIX})
python3 diagnose_drivers.py \
  --feat-a ${PREFIX}_mc_cosmic.npz --feat-b ${PREFIX}_data_raw.npz \
  --clean-a ${PREFIX}_mc_cosmicclean.npz --clean-b ${PREFIX}_data_cosmicclean.npz \
  --mc-list ${REPO}/lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt \
  --data-list ${REPO}/lartpc/filelists/h5list_v3_extbnb_diag1k_tufts.txt \
  --fig-dir figures \
  --out results/\${TAG}_drivers.json
echo DRIVERS-COMPLETE
"
