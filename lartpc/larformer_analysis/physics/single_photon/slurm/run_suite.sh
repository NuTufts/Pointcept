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
J="$J $(PLOTS=$ROOT/diagnostics_vertex sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0)"
J="$J $(PLOTS=$ROOT/diagnostics_novertex sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 1)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_in sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --dfw in)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_FC_ent sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain fc --dfw ent)"
J="$J $(PLOTS=$ROOT/diagnostics_vertex_pass_PC sbatch --parsable slurm/submit_single_photon_diagnostics.sh $A --path 0 --veto2g pass --contain pc)"
J="$J $(PLOTS=$ROOT/truth sbatch --parsable slurm/submit_single_photon_truth.sh $A)"
J="$J $(PLOTS=$ROOT/missed_photon sbatch --parsable slurm/submit_single_photon_missed_photon.sh $A)"
J="$J $(PLOTS=$ROOT/ccmu_anatomy sbatch --parsable slurm/submit_single_photon_ccmu_anatomy.sh $A)"
echo "$J"
