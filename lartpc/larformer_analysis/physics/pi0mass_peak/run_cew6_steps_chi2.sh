#!/bin/bash
#SBATCH --job-name=cew6_steps
#SBATCH --mem=32G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/cew6_steps.%j.log --error=logs/pilot_matrix/cew6_steps.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCNT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root
DANT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root
EXNT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root
echo "######## nochi2bdt tables (chi2 open at event-BDT scoring) ########"
apptainer exec --bind /cluster:/cluster $C bash -c "
  cd $K && export PYTHONPATH=$K:$P && python3 - <<PYEOF
import numpy as np, joblib, sys
sys.path.insert(0,'$P')
from datamc_diagnostics import load
M=joblib.load('$P/ext_bdt_model_flashblind.joblib'); clf,FEATS=M['clf'],M['feats']
for name,nt in (('mc','$MCNT'),('data','$DANT'),('ext','$EXNT')):
    tb=f'$P/{name}_cew6_ts0075_table.npz'
    d=load(nt,tb,0.020101,-15.49,50.0,1e12,1e12,shower_bdt_min=0.075,muon_finder='union')
    sc=clf.predict_proba(np.column_stack([d[f].astype(float) for f in FEATS]))[:,1]
    tab=dict(np.load(tb,allow_pickle=True)); n=len(tab['w'])
    fail=np.zeros(n,bool); train=np.zeros(n,bool); held=np.zeros(n,bool)
    cat=tab.get('cat')
    for r,s in zip(d['row'],sc):
        if s<0.21: fail[r]=True
    if name=='mc':
        for r,ev in zip(d['row'],d['event']):
            if cat[r]<2 and ev%2==0: train[r]=True
            if cat[r]<2 and ev%2==1: held[r]=True
    if name=='ext':
        for r,ev in zip(d['row'],d['event']):
            if r<100000 or ev%2==0: train[r]=True
            else: held[r]=True
    for k in ('sel_ge2','sel_eq2'):
        tab[k]=np.asarray(tab[k]).copy(); tab[k][fail|train]=0
    tab['w']=np.asarray(tab['w'],float).copy(); tab['w'][held]*=2.0
    np.savez(tb.replace('_table.npz','_nochi2bdt_table.npz'),**tab)
    print(f'{name}: scored {len(d[\"row\"])} | fail {int(fail.sum())} | train-excluded {int(train.sum())}')
PYEOF
  echo '######## step tables ########' && \
  export PYTHONPATH=$K:$P && python3 $P/cew6_step_tables.py && \
  echo '######## flashchi2 spectra (staged cuts, chi2 uncut) ########' && \
  python3 $P/flashchi2_from_tables.py \
    --mc-npz $P/mc_cew6_ts0075_nochi2bdt_table.npz \
    --data-npz $P/data_cew6_ts0075_nochi2bdt_table.npz \
    --ext-npz $P/ext_cew6_ts0075_nochi2bdt_table.npz \
    --ext-scale 1.1818 --chi2-cut 1e4 --chi2-cut-nc 1778 --chi2-cut-combined 3162.3 \
    --plots $P/plots_cew6_flashchi2_ts0075"
echo DONE_CEW6_STEPS
