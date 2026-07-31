#!/bin/bash
# Point-level battery (CPU). Usage:
#   sbatch run_point_battery_tufts.sbatch <a_pts.npz> <b_pts.npz> <label> <out.json>
#
#SBATCH --job-name=dspoint
#SBATCH --partition=batch
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=logs/dspoint.%j.log
#SBATCH --error=logs/dspoint.%j.err

set -u
REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
CONTAINER=${CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
A=${1:?a.npz}; B=${2:?b.npz}; LABEL=${3:?label}; OUT=${4:?out.json}

module load modtree/deprecated
module load apptainer/1.2.4-suid

apptainer exec --bind /cluster:/cluster "$CONTAINER" bash -c "
set -eu
cd ${REPO}/lartpc/pretraining_studies/domain_shift
python3 point_level_battery.py --a ${A} --b ${B} --label '${LABEL}' --out ${OUT}
echo POINT-BATTERY-COMPLETE
"
