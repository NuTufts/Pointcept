#!/bin/bash
# Week-1 acceptance test for the extraction pipeline (plan section 6):
# 20 events per diag set through the latest P5B.1 snapshot, tiers exercised,
# then verify_acceptance.py checks shapes, masking, and runs a smoke PAD.
#
#   cd lartpc/pretraining_studies/domain_shift && mkdir -p logs features
#   sbatch acceptance_test.sbatch
#
#SBATCH --job-name=dsaccept
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/dsaccept.%j.log
#SBATCH --error=logs/dsaccept.%j.err

set -u

REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}

CFG=configs/lartpc/p05/pretrain-sonata-p5b1-mix-raw-detsym.py
CKPT=sonata/p5b/P5B.1-mix_raw-s0/snapshot/snapshot_iter0128000_img6144000.pth
MC_LIST=lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt
DATA_LIST=lartpc/filelists/h5list_v3_extbnb_diag1k_tufts.txt
OUT=lartpc/pretraining_studies/domain_shift/features/acceptance

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --nv --bind /cluster:/cluster "$CONTAINER" bash -c "
set -eu
cd ${REPO} && export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
EX=lartpc/pretraining_studies/domain_shift/extract_features.py
for spec in \
    'mc    ${MC_LIST}   raw          mc_raw' \
    'mc    ${MC_LIST}   cosmic       mc_cosmic' \
    'mc    ${MC_LIST}   cosmic-clean mc_cosmicclean' \
    'data  ${DATA_LIST} raw          data_raw' \
    'data  ${DATA_LIST} cosmic-clean data_cosmicclean' ; do
  set -- \$spec
  echo \"=== extract: domain=\$1 tier=\$3 ===\"
  python3 \$EX --config ${CFG} --checkpoint ${CKPT} \
      --data-list \$2 --tier \$3 --num-events 20 \
      --points-per-event 256 \
      --out ${OUT}/\$4.npz
done
python3 lartpc/pretraining_studies/domain_shift/verify_acceptance.py ${OUT}
"
RC=$?
echo "acceptance exited rc=$RC"
exit $RC
