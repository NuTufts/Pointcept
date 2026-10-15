#!/bin/bash
# Submit the full single-photon plot suite for one convention.
#   slurm/run_suite.sh <PLOT_ROOT> [extra selection args, e.g. --flashpe-ratio-max 0]
# Writes <PLOT_ROOT>/ (selection), <PLOT_ROOT>_chi2_225/, <PLOT_ROOT>_base/ and
# <PLOT_ROOT>/{paths6,diagnostics_*,truth,missed_photon,ccmu_anatomy}/.
set -eu
ROOT=$1; shift
X="$@"
A="--novtx --novtx-bdt-min 0.5 --shower-bdt-min 0.164 $X"
cd "$(dirname "$0")/.."
J=""
J="$J $(PLOTS=$ROOT OUTDIR=workdir_$(basename $ROOT) sbatch --parsable slurm/submit_single_photon_selection.sh $A)"
J="$J $(PLOTS=${ROOT}_chi2_225 OUTDIR=workdir_$(basename $ROOT)_chi2_225 sbatch --parsable slurm/submit_single_photon_selection.sh $A --flashchi2-cut 177.8)"
J="$J $(PLOTS=${ROOT}_base OUTDIR=workdir_$(basename $ROOT)_base sbatch --parsable slurm/submit_single_photon_selection.sh --shower-bdt-min 0.164 $X)"
J="$J $(PLOTS=$ROOT/paths6 sbatch --parsable slurm/submit_single_photon_path_tables.sh $A)"
# "tightbdt" (redefined 2026-10-02, the cosmic BDT >= 0.6 being standard now): in-FV-like cell = BDT_enter > 0.3 (re-derived 2026-10-02 after the retrain)
# & BDT_ext > 0.2 & BDT_numu > 0.1 (MB-chain flattened scores); failing in-FV-like events land in no path
J="$J $(PLOTS=$ROOT/paths6_tightbdt sbatch --parsable slurm/submit_single_photon_path_tables.sh $A --fc-split enter --enter-cut 0.3 --chain-ext-min 0.2 --chain-numu-min 0.1)"
J="$J $(PLOTS=$ROOT/diagnostics_all sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_pass sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1 --veto2g pass)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex_fail sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_fail sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g fail)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_in sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --dfw in)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_ent sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --dfw ent)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_in_tightbdt sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --enter in --enter-cut 0.3 --chain-ext-min 0.2 --chain-numu-min 0.1)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_PC sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain pc)"
# alternative FC split by the MB-chain BDT_enter score (0.3; no-PE-ratio model) instead of dwall
J="$J $(PLOTS=$ROOT/paths6_enter sbatch --parsable slurm/submit_single_photon_path_tables.sh $A --fc-split enter --enter-cut 0.3)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_enter_in sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --enter in --enter-cut 0.3)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_enter_ent sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --enter ent --enter-cut 0.3)"
J="$J $(PLOTS=$ROOT/truth sbatch --parsable slurm/submit_single_photon_truth.sh $A --no-selected)"
J="$J $(PLOTS=$ROOT/missed_photon sbatch --parsable slurm/submit_single_photon_missed_photon.sh $A)"
J="$J $(PLOTS=$ROOT/ccmu_anatomy sbatch --parsable slurm/submit_single_photon_ccmu_anatomy.sh $A)"
echo "$J"
