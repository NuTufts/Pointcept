#!/bin/bash
#SBATCH --job-name=extbdt_cew6
#SBATCH --mem=32G --cpus-per-task=4 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/extbdt_cew6.%j.log --error=logs/pilot_matrix/extbdt_cew6.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCNT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root
DANT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root
EXNT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root
apptainer exec --bind /cluster:/cluster $C bash -c "
  cd $K && export PYTHONPATH=$K:$P && \
  echo '######## event-BDT training (flash-blind, cew6, ep8 recipe) ########' && \
  python3 $P/ext_bdt.py \
    --mc-ntuple $MCNT --mc-table $P/mc_s1ep2p8cew6_table.npz --mc-cascade $D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams \
    --data-ntuple $DANT --data-table $P/data_s1ep2p8cew6_table.npz --data-cascade $D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams \
    --ext-ntuple $EXNT --ext-table $P/ext_s1ep2p8cew6_table.npz --ext-cascade $D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams \
    --ext-scale 0.5909 \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --drop-feats logchi2,flashPE --holdout-plots --tag cew6 \
    --plots $P/plots_cew6_extbdt_train \
    --save-model $P/ext_bdt_model_flashblind_cew6.joblib && \
  echo '######## staged-context threshold scan (shower 0.075 + NEW model) ########' && \
  python3 - <<PYEOF
import numpy as np, joblib, sys
sys.path.insert(0,'$P')
from datamc_diagnostics import load
M=joblib.load('$P/ext_bdt_model_flashblind_cew6.joblib'); clf,FEATS=M['clf'],M['feats']
smp={}
for s,nt,tb in (('mc','$MCNT','mc_s1ep2p8cew6_table.npz'),('ext','$EXNT','ext_s1ep2p8cew6_table.npz')):
    d=load(nt,'$P/'+tb,0.020101,-15.49,50.0,1e12,1e12,shower_bdt_min=0.075,muon_finder='union')
    d['flashPE']=np.full(len(d['run']),np.nan)
    d['escore']=clf.predict_proba(np.column_stack([d[f].astype(float) for f in FEATS]))[:,1]
    smp[s]=d
cat=np.load('$P/mc_s1ep2p8cew6_table.npz')['cat'][smp['mc']['row']]
sig=cat<2; odd=smp['mc']['event']%2==1
ws=smp['mc']['w'][sig&odd]*2; ss=smp['mc']['escore'][sig&odd]
ek=(smp['ext']['row']>=100000)&(smp['ext']['event']%2==1)
se=smp['ext']['escore'][ek]
print(f'staged holdout: sig N {int((sig&odd).sum())} | ext N {len(se)}')
for te in (0.10,0.15,0.21,0.28,0.35,0.45):
    print(f'  te={te}: sig eff {ws[ss>=te].sum()/ws.sum():.3f} | EXT rej {1-(se>=te).mean():.3f}')
PYEOF"
echo DONE_EXTBDT_CEW6
