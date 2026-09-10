#!/bin/bash
#SBATCH --job-name=cew6_reval
#SBATCH --mem=32G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/cew6_reval.%j.log --error=logs/pilot_matrix/cew6_reval.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts

echo "######## Part A: calib provenance + shower-BDT curve ########"
apptainer exec --bind /cluster:/cluster $C bash -c "cd $K && export PYTHONPATH=$K:$P && python3 $P/cew6_bdt_revalidation.py A"
MODE=$(cat $P/cew6_calib_mode.txt)
if [ "$MODE" = "baked" ]; then RECAL=""; else RECAL="--recal-gamma-a 0.01553 --recal-gamma-b -12.80"; fi
echo "calib mode: $MODE | table recal flags: '$RECAL'"

echo "######## tables ########"
for LEG in mc data ext; do
  case $LEG in
    mc)  NT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root; CASC=$D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams; DF="";;
    data) NT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root; CASC=$D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams; DF="--data";;
    ext) NT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root; CASC=$D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams; DF="--data";;
  esac
  apptainer exec --bind /cluster:/cluster $C bash -c "
    cd $K && export PYTHONPATH=$K:$P && \
    python3 $P/pi0_mass_analysis.py --ntuple $NT $DF $RECAL \
      --cascade-dir $CASC --saturation-mask \
      --mu-ke-min 50 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --cpi-ke-min 60 \
      --out $P/${LEG}_s1ep2p8cew6_table.npz --plots $P/plots_${LEG}_cew6_base" | grep -E "CUTFLOW|selected|efficiency|>>>" | head -12
done

echo "######## Part B: event-BDT staged point ########"
if [ "$MODE" = "baked" ]; then BGA=0.020101; BGB=-15.49; else BGA=0.01553; BGB=-12.80; fi
apptainer exec --bind /cluster:/cluster $C bash -c "cd $K && export PYTHONPATH=$K:$P && python3 $P/cew6_bdt_revalidation.py B $BGA $BGB"
echo DONE_CEW6_REVAL
