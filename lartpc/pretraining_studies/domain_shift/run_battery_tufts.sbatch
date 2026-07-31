#!/bin/bash
# CPU metric battery for one snapshot's tier comparisons. Submitted from
# the domain_shift dir:  sbatch run_battery_tufts.sbatch <feature-prefix>
# e.g.  sbatch run_battery_tufts.sbatch features/P5B.1_img6144000
# Runs as a SLURM job (not a login-shell child) so it survives disconnects.
#
#SBATCH --job-name=dsbattery
#SBATCH --partition=batch
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dsbattery.%j.log
#SBATCH --error=logs/dsbattery.%j.err

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
python3 compute_metrics.py --a ${PREFIX}_mc_raw.npz --b ${PREFIX}_data_raw.npz \
  --label \"\${TAG} tier0-raw\" --out results/\${TAG}_tier0.json
python3 compute_metrics.py --a ${PREFIX}_mc_cosmic.npz --b ${PREFIX}_data_raw.npz \
  --label \"\${TAG} tier1-cosmic\" --out results/\${TAG}_tier1.json
python3 compute_metrics.py --a ${PREFIX}_mc_cosmicclean.npz --b ${PREFIX}_data_cosmicclean.npz \
  --label \"\${TAG} tier1-clean\" --out results/\${TAG}_tier1clean.json
echo BATTERY-COMPLETE
"
RC=$?
echo "battery exited rc=$RC"
exit $RC
