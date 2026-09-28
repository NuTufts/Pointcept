#!/bin/bash
# Per-shower cosmic-BDT performance study (photon_bdt_study.py) on batch.
#   sbatch slurm/submit_photon_bdt_study.sh
#   sbatch slurm/submit_single_photon_selection.sh [extra args]
#SBATCH --job-name=sp_bdt
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24000
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_bdt.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/single_photon/slurm/logs/sp_bdt.%j.err

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
# STUDY: single files only -- run-1 half 1 of MC and EXT.
MC=${MC:-$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root}
EXT=${EXT:-$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root}
DATA=${DATA:-$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root}
EXT_SCALE=${EXT_SCALE:-1.202}   # one EXT half, sideband-measured
EXT_ROW_MIN=${EXT_ROW_MIN:-0}
PLOTS=${PLOTS:-plots_v2_s1ep2p8/bdt_study}
OUTDIR=${OUTDIR:-workdir}

cd ${SP}
${K}/run_in_tufts_pointcept_container.sh python3 photon_bdt_study.py \
    --mc-ntuple ${MC} --ext-ntuple ${EXT} --ext-scale ${EXT_SCALE} --ext-row-min ${EXT_ROW_MIN} \
    --plots ${PLOTS} "$@"
