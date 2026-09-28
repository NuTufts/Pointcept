#!/bin/bash
# Train the vertex-free per-shower cosmic BDT (shower_novtx_bdt.py) on the
# novtx MC + EXT ntuples and save the exporter model.
#   sbatch slurm/submit_shower_novtx_bdt.sh [extra args]
#SBATCH --job-name=sp_novtxbdt
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_novtxbdt.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_novtxbdt.%j.err

K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
SP=${K}/lartpc/larformer_analysis/physics/single_photon
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
# ---- RUN-1 TABLE-GAMMA samples (default since 2026-09-23) -------------------
# Both MC halves and both EXT halves (comma-separated -> concatenated, POT
# summed, duplicate (run,subrun,event) dropped). Calibrated flash gamma:
# mc,1 4.5024 / data,1 2.8607, so data and MC flash chi2 are comparable.
# EXT_SCALE: bnb5e19 10,375,708 spills / both EXT halves 11,545,474 = 0.8987
#   (the pi0 suite's convention; the versions doc's 94.4M spill count gives
#   16.355 for one half and is believed ~10x high -- see section 3c NOTE).
# No BDT-hygiene halves here: both cew6 BDTs trained on RUN-3 samples, so
# run-1 MC/EXT are fully held out (--ext-row-min 0, no --mc-odd-only).
MCD=${MCD:-$L/run1_bnboverlay_half}
EXTD=${EXTD:-$L/run1_C1_extbnb_half}
DAD=${DAD:-$L/bnb5e19_run1_table}
# TRAINER: single files only. Default = run-1 half 1 for both MC and EXT;
# EXT_ROW_MAX=0 keeps the whole EXT file as background (train on half 1,
# evaluate/apply on half 2).
MC=${MC:-$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root}
EXT=${EXT:-$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root}
DATA=${DATA:-$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root}
EXT_SCALE=${EXT_SCALE:-1.202}   # one EXT half, sideband-measured
EXT_ROW_MAX=${EXT_ROW_MAX:-52258}   # half of the 104,516-row run-1 EXT half -> held-out tail
PLOTS=${PLOTS:-plots_v2_s1ep2p8/novtx_bdt}
MODEL=${MODEL:-${K}/lartpc/larformer_reco/export/data/shower_novtx_bdt.joblib}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 shower_novtx_bdt.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --ext-row-max ${EXT_ROW_MAX} --plots ${PLOTS} --save-model ${MODEL} "$@"
