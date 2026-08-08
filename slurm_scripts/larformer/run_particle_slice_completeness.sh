#!/bin/bash
# Per-true-particle slicer-stage completeness over one valtest inference dir.
# Usage: sbatch [--dependency=...] run_particle_slice_completeness.sh <inference-dir> <out-npz>
# CPU-only (KDTree joins + h5 reads).

#SBATCH --job-name=lf-slicer-pcomp
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/lf-slicer-pcomp.%j.log
#SBATCH --error=logs/lf-slicer-pcomp.%j.err

set -eu
INFDIR=${1:?usage: run_particle_slice_completeness.sh <inference-dir> <out-npz> [extra flags...]}
OUT=${2:?usage: run_particle_slice_completeness.sh <inference-dir> <out-npz> [extra flags...]}
shift 2
EXTRA_ARGS=("$@")   # e.g. --keep-source hasmatch

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
MANIFEST=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_analysis/valtest/manifest/bnb_nu_pi0filter_corsika_valtest.csv
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif

module load apptainer
apptainer exec --bind /cluster:/cluster $container python3 \
  $K/lartpc/larformer_analysis/slicer_eval/particle_slice_completeness.py \
  --inference-dir "$INFDIR" \
  --manifest-csv "$MANIFEST" \
  --out "$OUT" \
  "${EXTRA_ARGS[@]}"
