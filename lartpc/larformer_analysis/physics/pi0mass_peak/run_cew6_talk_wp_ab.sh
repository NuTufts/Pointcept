#!/bin/bash
#SBATCH --job-name=cew6_wp_ab
#SBATCH --mem=32G --cpus-per-task=2 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/cew6_wp_ab.%j.log --error=logs/pilot_matrix/cew6_wp_ab.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
MCNT=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root
DANT=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root
EXNT=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root
for TS in 0.075 0.192; do
  SFX=$(echo $TS | tr -d '.')
  echo "######## WP variant: shower>=$TS, event>=0.21 ########"
  for LEG in mc data ext; do
    case $LEG in
      mc)  NT=$MCNT; CASC=$D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams; DF="";;
      data) NT=$DANT; CASC=$D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams; DF="--data";;
      ext) NT=$EXNT; CASC=$D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams; DF="--data";;
    esac
    apptainer exec --bind /cluster:/cluster $C bash -c "
      cd $K && export PYTHONPATH=$K:$P && \
      python3 $P/pi0_mass_analysis.py --ntuple $NT $DF \
        --shower-bdt-min $TS --muon-finder union \
        --cascade-dir $CASC --saturation-mask \
        --mu-ke-min 50 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --cpi-ke-min 60 \
        --out $P/${LEG}_cew6_ts${SFX}_table.npz --plots $P/plots_${LEG}_cew6_ts${SFX}" | grep -E "efficiency|composition|reco-" | head -6
  done
  apptainer exec --bind /cluster:/cluster $C bash -c "
    cd $K && export PYTHONPATH=$K:$P && python3 - <<PYEOF
import numpy as np, joblib, sys
sys.path.insert(0,'$P')
from datamc_diagnostics import load
M=joblib.load('$P/ext_bdt_model_flashblind.joblib'); clf,FEATS=M['clf'],M['feats']
for name,nt in (('mc','$MCNT'),('data','$DANT'),('ext','$EXNT')):
    tb=f'$P/{name}_cew6_ts$SFX'+'_table.npz'
    d=load(nt,tb,0.020101,-15.49,50.0,1e12,1e12,shower_bdt_min=$TS,muon_finder='union')
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
    np.savez(tb.replace('_table.npz','_bdt_table.npz'),**tab)
    print(f'{name}: event-bdt fail {int(fail.sum())}/{len(d[\"row\"])}')
PYEOF
    export PYTHONPATH=$K:$P && \
    python3 $P/datamc_ext_overlay.py --mc-npz $P/mc_cew6_ts${SFX}_bdt_table.npz \
      --data-npz $P/data_cew6_ts${SFX}_bdt_table.npz --ext-npz $P/ext_cew6_ts${SFX}_bdt_table.npz \
      --ext-scale 1.1818 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --combined-chi2-cut 3162.3 \
      --plots $P/plots_cew6_overlay_ts${SFX}"
done
echo DONE_WP_AB
