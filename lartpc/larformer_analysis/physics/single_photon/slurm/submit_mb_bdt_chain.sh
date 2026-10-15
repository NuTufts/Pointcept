#!/bin/bash
# WC-style MB BDT chain: train on the TRAINPOOL overlay half + EXT first half, evaluate on the RESERVED halves + data.
#   sbatch slurm/submit_single_photon_path_tables.sh [extra args]
#   sbatch slurm/submit_single_photon_diagnostics.sh [extra args]
#SBATCH --job-name=sp_mbchain
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_mbchain.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_mbchain.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
# ---- RUN-1 TABLE-GAMMA samples (default since 2026-09-23) -------------------
# Both MC halves and both EXT halves (comma-separated -> concatenated, POT
# summed, duplicate (run,subrun,event) dropped). Calibrated flash gamma:
# mc,1 4.5024 / data,1 2.8607, so data and MC flash chi2 are comparable.
# EXT_SCALE: SIDEBAND-MEASURED 0.601 (default). The pure-cosmic flash-chi2
#   sideband (log10 chi2 > 3.5/4.0/4.5, MC-nu subtracted) gives 0.612/0.602/0.573
#   and makes data/prediction 1.00-1.03 at EVERY chi2 cut. The spill-count value
#   (bnb5e19 10,375,708 / both EXT halves 11,545,474 = 0.8987) leaves data 10-25%
#   below prediction -- consistent with the pi0 suite's note that the EXT spill
#   count is ~25-30% low (versions doc section 3c). One EXT half: double these.
# No BDT-hygiene halves here: both cew6 BDTs trained on RUN-3 samples, so
# run-1 MC/EXT are fully held out (--ext-row-min 0, no --mc-odd-only).
MCD=${MCD:-$L/run1_bnboverlay_half}
EXTD=${EXTD:-$L/run1_C1_extbnb_half}
DAD=${DAD:-$L/bnb5e19_run1_table}
TRAIN_MC=${TRAIN_MC:-$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root}      # TRAINPOOL half
MC=${MC:-$L/run1_bnboverlay_half2/dlgen2_larformer_ntuple_bnbovl_run1_half2.root}               # RESERVED half (evaluation)
TRAIN_EXT=${TRAIN_EXT:-$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root}
EXT=${EXT:-$L/run1_C1_extbnb_half2/dlgen2_larformer_ntuple_extbnb_run1_half2.root}
DATA=${DATA:-$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root}
EXT_SCALE=${EXT_SCALE:-1.202}      # one EXT half
EXT_ROW_MIN=${EXT_ROW_MIN:-0}
PLOTS=${PLOTS:-plots_run1_mbchain}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 single_photon_mb_bdt_chain.py \
    --train-mc ${TRAIN_MC} --train-ext ${TRAIN_EXT} --mc-ntuple ${MC} --ext-ntuple ${EXT} --data-ntuple ${DATA} --ext-scale ${EXT_SCALE} --ext-row-min ${EXT_ROW_MIN} --plots ${PLOTS} "$@"
