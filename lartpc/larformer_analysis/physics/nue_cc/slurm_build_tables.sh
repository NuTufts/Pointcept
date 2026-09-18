#!/bin/bash
#SBATCH --job-name=nuecc_tables
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3
#SBATCH --output=logs/nuecc/tables.%A_%a.log
#SBATCH --error=logs/nuecc/tables.%A_%a.err
# Build the four per-sample nue-CC tables, one per array task.
#   sbatch --export=ALL,SET=run1_tg   slurm_build_tables.sh   -> tables_run1/
#   sbatch --export=ALL,SET=run3_cew6  slurm_build_tables.sh   -> tables/
# The sample set is recorded in every table; the plotters take the EXT weight
# and hygiene rules from it (nue_cc_common.SAMPLE_SETS).
set -eu
SET=${SET:-run3_cew6}
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
N=$K/lartpc/larformer_analysis/physics/nue_cc
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
L=/cluster/tufts/wongjiradlab/larbys/data/larformer
EXCL=$K/lartpc/larformer_analysis/physics/pi0mass_peak/run1ovl_trainpool_exclusion.npz

if [ "$SET" = "run1_tg" ]; then
  OUTD=$N/tables_run1
  case ${SLURM_ARRAY_TASK_ID} in
    0) TAG=nue;  NT=$L/run1_nueintrinsics_half/dlgen2_larformer_ntuple_nue_run1_half.root;   EXTRA="" ;;
    # overlay half: drop the filenos that fed segmenter training, re-sum POT
    1) TAG=bnb;  NT=$L/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root;   EXTRA="--exclude-rows $EXCL" ;;
    2) TAG=ext;  NT=$L/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root;    EXTRA="--data" ;;
    3) TAG=data; NT=$L/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root;   EXTRA="--data" ;;
  esac
else
  OUTD=$N/tables
  case ${SLURM_ARRAY_TASK_ID} in
    0) TAG=nue;  NT=$D/larformer_nueoverlay79k_s1ep2p8cew6/dlgen2_larformer_ntuple_nue_overlay_s1ep2p8cew6_run3.root; EXTRA="" ;;
    1) TAG=bnb;  NT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root;   EXTRA="" ;;
    2) TAG=ext;  NT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root;          EXTRA="--data" ;;
    3) TAG=data; NT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root;                EXTRA="--data" ;;
  esac
fi
mkdir -p $OUTD
echo ">>> [$SET] $TAG <- $NT"
apptainer exec --bind /cluster:/cluster \
  /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif bash -c "
  cd $K && export PYTHONPATH=$K && \
  python3 $N/nue_cc_analysis.py --ntuple $NT --out $OUTD/${TAG}.npz \
    --sample-set $SET $EXTRA"
echo ">>> [$SET] $TAG done"
