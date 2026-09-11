#!/bin/bash
#SBATCH --job-name=bulkgamma_ref
#SBATCH --mem=24G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/bulkgamma_ref.%j.log --error=logs/pilot_matrix/bulkgamma_ref.%j.err
set -u
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
F=$K/lartpc/larformer_analysis/flashmodel_calib/archive/july2026_gamma
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
# tag : ntuple : cascade dir : gamma the sample was PRODUCED at (auto table)
run_one () {
  echo "######## $1  (produced at gamma_scale=$4) ########"
  apptainer exec --bind /cluster:/cluster $C bash -c "
    cd $K && export PYTHONPATH=$K && \
    python3 $F/measure_bulk_gamma.py --ntuple $2 --cascade-dir $3 \
      --sample-tag $1 --out $F/out/bulk_$1.npz"
}
run_one run3_mc_cew6 \
  $D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root \
  $D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams 1.0
run_one run3_ext_cew6 \
  $D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root \
  $D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams 1.0
run_one run1_bnb5e19_cew6 \
  $D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root \
  $D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams 0.80
echo DONE_BULK_REFS
