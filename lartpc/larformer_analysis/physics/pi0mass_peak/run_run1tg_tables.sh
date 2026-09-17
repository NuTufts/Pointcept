#!/bin/bash
# Run-1 table-gamma pi0 remake, stage 1: selection tables for the three
# run-1 productions (versions doc s3c). Array task 0/1/2 = mc/data/ext.
# Submit from $K:  sbatch lartpc/larformer_analysis/physics/pi0mass_peak/run_run1tg_tables.sh
#SBATCH --job-name=run1tg_tab
#SBATCH --array=0-2
#SBATCH --mem=64G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/run1tg_tables.%A_%a.log --error=logs/pilot_matrix/run1tg_tables.%A_%a.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
TS=0.164; PRE=run1tg_ts0164          # talk WP: shower cosmicScore >= 0.164
LEGS=(mc data ext); LEG=${LEGS[$SLURM_ARRAY_TASK_ID]}
case $LEG in
  mc)   NT=$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root; DF="";
        EXTRA="--exclude-rows $P/run1ovl_trainpool_exclusion.npz";;
  data) NT=$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root; DF="--data"; EXTRA="";;
  ext)  NT=$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root; DF="--data"; EXTRA="";;
esac
echo ">>> $LEG: $NT"
if [ "$LEG" = mc ]; then
  echo "######## MC: drop the segmenter-training filenos (whole files, POT re-summed) ########"
  apptainer exec --bind /cluster:/cluster $C bash -c "cd $K && export PYTHONPATH=$K:$P && \
    python3 $P/run1ovl_trainpool_exclusion.py --ntuple $NT \
      --sidecar-dir $L/run1_bnboverlay_half/truth_sidecar \
      --out $P/run1ovl_trainpool_exclusion.npz"
fi
echo "######## table: $LEG ########"
apptainer exec --bind /cluster:/cluster $C bash -c "cd $K && export PYTHONPATH=$K:$P && \
  python3 $P/pi0_mass_analysis.py --ntuple $NT $DF $EXTRA \
    --flashchi2-from-ntuple \
    --shower-bdt-min $TS --muon-finder union \
    --mu-ke-min 50 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --cpi-ke-min 60 \
    --out $P/${LEG}_${PRE}_table.npz --plots $P/plots_${LEG}_${PRE}"
echo DONE_RUN1TG_TABLE_$LEG
