#!/bin/bash
#SBATCH --job-name=sbdt_cew6
#SBATCH --mem=48G --cpus-per-task=8 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/sbdt_cew6.%j.log --error=logs/pilot_matrix/sbdt_cew6.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
apptainer exec --bind /cluster:/cluster $C bash -c "
  cd $K && export PYTHONPATH=$K:$P && \
  python3 $P/shower_cosmic_bdt.py \
    --mc-ntuple $D/larformer_overlaytrain_pi0bdt_cew6/dlgen2_larformer_ntuple_overlaytrain_pi0bdt_cew6_run3.root \
    --no-mc-split \
    --ext-ntuple $D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root \
    --ext-row-max 100000 \
    --data-ntuple $D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --plots $P/plots_cew6_sbdt_train \
    --save-model $K/lartpc/larformer_reco/export/data/shower_cosmic_bdt_cew6.joblib"
echo DONE_SBDT_CEW6
