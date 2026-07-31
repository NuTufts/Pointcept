#!/bin/bash
# DCTR reweighting demo (CPU). Usage:
#   sbatch run_dctr_tufts.sbatch <feature-prefix>
# Uses the method-symmetric clean pair of that prefix.
#
#SBATCH --job-name=dsdctr
#SBATCH --partition=batch
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dsdctr.%j.log
#SBATCH --error=logs/dsdctr.%j.err

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
python3 dctr_demo.py \
  --a ${PREFIX}_mc_cosmiclmclean.npz --b ${PREFIX}_data_cosmicclean.npz \
  --mc-list ${REPO}/lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt \
  --data-list ${REPO}/lartpc/filelists/h5list_v3_extbnb_diag1k_tufts.txt \
  --fig-dir figures --out results/\${TAG}_dctr.json
echo DCTR-COMPLETE
"
