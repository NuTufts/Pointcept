#!/bin/bash
# Run-1 table-gamma pi0 remake, stage 2 (after run_run1tg_tables.sh):
# event-BDT filters, data/MC+EXT overlays, step tables, flash-chi2 spectra,
# before/after comparison against the cew6 talk tables, SBND-style suite.
# Normalisation (versions doc s3c / data_prep README): MC POT-scaled to
# 4.4e19 inside the tables; EXT half = bnb5e19 spills / EXT-half spills
# = 94,414,115 / 5,772,737. No classifier hygiene: no BDT trained on these
# run-1 samples. Submit from $K with --dependency=afterok:<tables job>.
#SBATCH --job-name=run1tg_suite
#SBATCH --mem=48G --cpus-per-task=4 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/run1tg_suite.%j.log --error=logs/pilot_matrix/run1tg_suite.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
TS=0.164; TE=0.21; PRE=run1tg_ts0164
# bnb5e19 spill count: the July normalisation (memory extbnb-cosmic-normalization,
# beam/EXT-run3 = 10375708/58677653) used 10,375,708, which also matches
# 4.4e19 POT at ~4.2e12 POT/spill. data_prep/README.md lists 94,414,115 for the
# same sample; that value makes the cosmic-only prediction 10x the beam trigger
# count and puts the EXT band 9x above the data in the pure-cosmic flash-chi2
# sideband, so it cannot be the spill count. Override with BEAM_SPILLS=... .
BEAM_SPILLS=${BEAM_SPILLS:-10375708}; EXT_SPILLS=5772737
EXT_SCALE=$(python3 -c "print($BEAM_SPILLS/$EXT_SPILLS)")
echo ">>> EXT scale = $BEAM_SPILLS / $EXT_SPILLS = $EXT_SCALE"
declare -A NT
NT[mc]=$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root
NT[data]=$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root
NT[ext]=$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root
for LEG in mc data ext; do test -s $P/${LEG}_${PRE}_table.npz || { echo "missing ${LEG} table"; exit 1; }; done
RUN="apptainer exec --bind /cluster:/cluster $C bash -c"
echo "######## event-BDT filters (cew6 model, hygiene none) ########"
for LEG in mc data ext; do
  $RUN "cd $K && export PYTHONPATH=$K:$P && \
    python3 $P/apply_event_bdt.py --model $P/ext_bdt_model_flashblind_cew6.joblib --sample $LEG \
      --ntuple ${NT[$LEG]} --table $P/${LEG}_${PRE}_table.npz --out $P/${LEG}_${PRE}_bdt_table.npz \
      --shower-bdt-min $TS --te $TE --hygiene none && \
    python3 $P/apply_event_bdt.py --model $P/ext_bdt_model_flashblind_cew6.joblib --sample $LEG \
      --ntuple ${NT[$LEG]} --table $P/${LEG}_${PRE}_table.npz --out $P/${LEG}_${PRE}_nochi2bdt_table.npz \
      --shower-bdt-min $TS --te $TE --hygiene none --chi2-open"
done
$RUN "cd $K && export PYTHONPATH=$K:$P && \
  echo '######## data vs MC+EXT overlays ########' && \
  python3 $P/datamc_ext_overlay.py --mc-npz $P/mc_${PRE}_bdt_table.npz \
    --data-npz $P/data_${PRE}_bdt_table.npz --ext-npz $P/ext_${PRE}_bdt_table.npz \
    --ext-scale $EXT_SCALE --ext-weight-from-table \
    --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --combined-chi2-cut 3162.3 \
    --plots $P/plots_run1tg_overlay_ts0164 && \
  echo '######## step tables ########' && \
  python3 $P/pi0_step_tables.py --prefix $PRE --mc-ntuple ${NT[mc]} --ext-ntuple ${NT[ext]} \
    --data-ntuple ${NT[data]} --ts $TS --te $TE --ext-scale $EXT_SCALE --hygiene none \
    --out $P/plots_run1tg_overlay_ts0164/step_tables.txt && \
  echo '######## flash-chi2 spectra (event-BDT applied, chi2 open) ########' && \
  python3 $P/flashchi2_from_tables.py \
    --mc-npz $P/mc_${PRE}_nochi2bdt_table.npz --data-npz $P/data_${PRE}_nochi2bdt_table.npz \
    --ext-npz $P/ext_${PRE}_nochi2bdt_table.npz --ext-scale $EXT_SCALE --ext-weight-from-table \
    --chi2-cut 1e4 --chi2-cut-nc 1778 --chi2-cut-combined 3162.3 \
    --plots $P/plots_run1tg_flashchi2_ts0164 && \
  echo '######## before/after vs cew6 talk tables ########' && \
  python3 $P/flashchi2_before_after.py --old-prefix cew6_ts0164 --old-ext-scale 1.1818 \
    --new-prefix $PRE --new-ext-scale $EXT_SCALE --plots $P/plots_run1tg_flashchi2_ts0164 && \
  echo '######## SBND-style suite ########' && \
  python3 $P/sbnd_cutflow.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_table.npz \
    --ext-scale $EXT_SCALE --ext-row-min 0 && \
  python3 $P/sbnd_mgg_sbndflow.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_table.npz \
    --ext-scale $EXT_SCALE --ext-row-min 0 --plots $P/plots_run1tg_sbnd_sbndflow_ts0164 && \
  python3 $P/sbnd_cc1pi0.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_table.npz \
    --ext-scale $EXT_SCALE --plots $P/plots_run1tg_sbnd_ts0164"
echo DONE_RUN1TG_SUITE
