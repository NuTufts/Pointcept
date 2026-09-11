#!/bin/bash
#SBATCH --job-name=gamma2x2
#SBATCH --mem=32G --cpus-per-task=4 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/gamma2x2.%j.log --error=logs/pilot_matrix/gamma2x2.%j.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
F=$K/lartpc/larformer_analysis/flashmodel_calib
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
mkdir -p $F/plots
# tag | ntuple | cascade | gamma_eff baked into pred_pe (MUST match the production)
run_one () {
  TAG=$1; NT=$2; CASC=$3; GEFF=$4
  echo "######## $TAG (pred_pe at gamma_eff=$GEFF) ########"
  apptainer exec --bind /cluster:/cluster $C bash -c "
    cd $K && export PYTHONPATH=$K && \
    python3 $F/fit_gamma_run.py --ntuple $NT --cascade-dir $CASC \
      --sample-tag $TAG --gamma-beam $GEFF \
      --plots $F/plots --out $F/gamma_${TAG}.npz" 2>&1 | grep -E "calibration muons|spatial|median obs/pred|plots ->|Error|Traceback"
}
run_one run1_data_bnb5e19 \
  $D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root \
  $D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams 4.20
run_one run3_data_extbnb \
  $D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root \
  $D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams 5.25
run_one run3_mc_overlay \
  $D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root \
  $D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams 5.25
echo DONE_GAMMA_2X2
