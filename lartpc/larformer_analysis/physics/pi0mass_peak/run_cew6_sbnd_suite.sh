#!/bin/bash
#SBATCH --job-name=cew6_sbnd
#SBATCH --mem=24G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/cew6_sbnd.%j.log --error=logs/pilot_matrix/cew6_sbnd.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCNT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root
DANT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root
EXNT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root
# identity recal (cew6 ntuples are recal3-baked; pass the NTUP constants)
IDGA=0.020101; IDGB=-15.49
apptainer exec --bind /cluster:/cluster $C bash -c "
  cd $K && export PYTHONPATH=$K:$P && \
  python3 - <<'PYEOF'
import numpy as np
tb='$P'+'/ext_cew6_ts0075_table.npz'
t2=dict(np.load(tb,allow_pickle=True))
t2['flash_chi2']=np.asarray(t2['flash_chi2'],float).copy(); t2['flash_chi2'][:100000]=np.inf
np.savez(tb.replace('_table.npz','_analysishalf_chi2inf_table.npz'),**t2)
print('SBND ext half table written')
PYEOF
  echo '######## SBND ordered cutflow (cew6, ts=0.075, union) ########' && \
  python3 $P/sbnd_cutflow.py --muon-finder union --shower-bdt-min 0.075 \
    --recal-gamma-a $IDGA --recal-gamma-b $IDGB \
    --mc-ntuple $MCNT --mc-table $P/mc_cew6_ts0075_table.npz \
    --data-ntuple $DANT --data-table $P/data_cew6_ts0075_table.npz \
    --ext-ntuple $EXNT --ext-table $P/ext_cew6_ts0075_table.npz \
    --ext-scale 1.1818 && \
  echo '######## SBND-flow mass plots ########' && \
  python3 $P/sbnd_mgg_sbndflow.py --muon-finder union --shower-bdt-min 0.075 \
    --recal-gamma-a $IDGA --recal-gamma-b $IDGB \
    --mc-ntuple $MCNT --mc-table $P/mc_cew6_ts0075_table.npz \
    --data-ntuple $DANT --data-table $P/data_cew6_ts0075_table.npz \
    --ext-ntuple $EXNT --ext-table $P/ext_cew6_ts0075_table.npz \
    --plots $P/plots_cew6_sbnd_sbndflow && \
  echo '######## SBND-style selection (standing format) ########' && \
  python3 $P/sbnd_cc1pi0.py --muon-finder union --shower-bdt-min 0.075 \
    --recal-gamma-a $IDGA --recal-gamma-b $IDGB \
    --mc-ntuple $MCNT --mc-table $P/mc_cew6_ts0075_table.npz \
    --data-ntuple $DANT --data-table $P/data_cew6_ts0075_table.npz \
    --ext-ntuple $EXNT --ext-table $P/ext_cew6_ts0075_analysishalf_chi2inf_table.npz \
    --ext-scale 1.1818 --plots $P/plots_cew6_sbnd"
echo DONE_CEW6_SBND
