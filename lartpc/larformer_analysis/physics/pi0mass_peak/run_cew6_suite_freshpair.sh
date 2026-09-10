#!/bin/bash
#SBATCH --job-name=cew6_fresh
#SBATCH --mem=32G --cpus-per-task=4 --time=24:00:00
#SBATCH --partition=batch,preempt,wongjiradlab
#SBATCH --output=logs/pilot_matrix/cew6_fresh.%j.log --error=logs/pilot_matrix/cew6_fresh.%j.err
set -eu
module load apptainer 2>/dev/null || true
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
P=$K/lartpc/larformer_analysis/physics/pi0mass_peak
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
D=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts
TS=0.164; TE=0.21; PRE=cew6_ts0164
declare -A NT
NT[mc]=$D/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root
NT[data]=$D/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root
NT[ext]=$D/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root

echo "######## verify + promote sbdt2 ntuples ########"
apptainer exec --bind /cluster:/cluster $C python3 - <<PYEOF
import uproot, numpy as np, awkward as ak, os, sys
D="$D"
pairs={"mc":("larformer_mcoverlay67k_s1ep2p8cew6","dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3"),
       "data":("larformer_bnb5e19_s1ep2p8cew6","dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6"),
       "ext":("larformer_extbnb200k_s1ep2p8cew6","dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6")}
for tag,(dd,base) in pairs.items():
    old=f"{D}/{dd}/{base}.root"; new=f"{D}/{dd}/{base}_sbdt2.root"
    to=uproot.open(old)["EventTree"]; tn=uproot.open(new)["EventTree"]
    ao=to.arrays(["run","event"]); an=tn.arrays(["run","event"])
    ok=(to.num_entries==tn.num_entries and np.array_equal(np.asarray(ao["run"]),np.asarray(an["run"]))
        and np.array_equal(np.asarray(ao["event"]),np.asarray(an["event"])))
    sc=ak.flatten(tn.arrays(["showerCosmicScore"])["showerCosmicScore"]).to_numpy()
    pid=ak.flatten(tn.arrays(["showerLArFormerPID"])["showerLArFormerPID"]).to_numpy()
    ph=(pid==22)&(sc>=0)
    print(f"{tag}: rows {to.num_entries}/{tn.num_entries} aligned {ok} | scored photons {int(ph.sum())} median {np.median(sc[ph]):.3f}")
    assert ok and ph.sum()>1000
    os.replace(old, f"{D}/{dd}/{base}_ep8score.root")
    os.replace(new, old)
print("PROMOTED (old scores archived as *_ep8score.root)")
PYEOF

echo "######## WP tables (shower>=TS, union) ########"
for LEG in mc data ext; do
  case $LEG in
    mc) CASC=$D/larformer_mcoverlay67k_s1ep2p8cew6/keypoint2_streams; DF="";;
    data) CASC=$D/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams; DF="--data";;
    ext) CASC=$D/larformer_extbnb200k_s1ep2p8cew6/keypoint2_streams; DF="--data";;
  esac
  apptainer exec --bind /cluster:/cluster $C bash -c "
    cd $K && export PYTHONPATH=$K:$P && \
    python3 $P/pi0_mass_analysis.py --ntuple ${NT[$LEG]} $DF \
      --shower-bdt-min $TS --muon-finder union \
      --cascade-dir $CASC --saturation-mask \
      --mu-ke-min 50 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --cpi-ke-min 60 \
      --out $P/${LEG}_${PRE}_table.npz --plots $P/plots_${LEG}_${PRE}" | grep -E "efficiency|composition|reco-" | head -6
done

echo "######## event-BDT (NEW model) filters: bdt + nochi2bdt + sbnd-ext ########"
apptainer exec --bind /cluster:/cluster $C bash -c "
  cd $K && export PYTHONPATH=$K:$P && python3 - <<PYEOF
import numpy as np, joblib, sys
sys.path.insert(0,'$P')
from datamc_diagnostics import load
M=joblib.load('$P/ext_bdt_model_flashblind_cew6.joblib'); clf,FEATS=M['clf'],M['feats']
NT={'mc':'${NT[mc]}','data':'${NT[data]}','ext':'${NT[ext]}'}
for chi2open,sfx in ((False,'_bdt'),(True,'_nochi2bdt')):
    cuts=(1e12,1e12) if chi2open else (1e4,1778.0)
    for name in ('mc','data','ext'):
        tb=f'$P/{name}_${PRE}_table.npz'
        d=load(NT[name],tb,0.020101,-15.49,50.0,cuts[0],cuts[1],shower_bdt_min=$TS,muon_finder='union')
        d['flashPE']=np.full(len(d['run']),np.nan)
        sc=clf.predict_proba(np.column_stack([d[f].astype(float) for f in FEATS]))[:,1]
        tab=dict(np.load(tb,allow_pickle=True)); n=len(tab['w'])
        fail=np.zeros(n,bool); train=np.zeros(n,bool); held=np.zeros(n,bool)
        cat=tab.get('cat')
        for r,s in zip(d['row'],sc):
            if s<$TE: fail[r]=True
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
        np.savez(tb.replace('_table.npz',sfx+'_table.npz'),**tab)
        print(f'{name}{sfx}: fail {int(fail.sum())}/{len(d[\"row\"])}')
t2=dict(np.load('$P/ext_${PRE}_table.npz',allow_pickle=True))
t2['flash_chi2']=np.asarray(t2['flash_chi2'],float).copy(); t2['flash_chi2'][:100000]=np.inf
np.savez('$P/ext_${PRE}_analysishalf_chi2inf_table.npz',**t2)
print('sbnd ext gate written')
PYEOF
  echo '######## overlay ########' && export PYTHONPATH=$K:$P && \
  python3 $P/datamc_ext_overlay.py --mc-npz $P/mc_${PRE}_bdt_table.npz \
    --data-npz $P/data_${PRE}_bdt_table.npz --ext-npz $P/ext_${PRE}_bdt_table.npz \
    --ext-scale 1.1818 --flashchi2-cut 1e4 --flashchi2-cut-nc 1778 --combined-chi2-cut 3162.3 \
    --plots $P/plots_cew6_overlay_${PRE} && \
  echo '######## step tables ########' && \
  STEP_TS=$TS STEP_PREFIX=$PRE python3 $P/cew6_step_tables.py && \
  echo '######## flashchi2 spectra ########' && \
  python3 $P/flashchi2_from_tables.py \
    --mc-npz $P/mc_${PRE}_nochi2bdt_table.npz --data-npz $P/data_${PRE}_nochi2bdt_table.npz \
    --ext-npz $P/ext_${PRE}_nochi2bdt_table.npz \
    --ext-scale 1.1818 --chi2-cut 1e4 --chi2-cut-nc 1778 --chi2-cut-combined 3162.3 \
    --plots $P/plots_cew6_flashchi2_${PRE} && \
  echo '######## SBND suite ########' && \
  python3 $P/sbnd_cutflow.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_table.npz --ext-scale 1.1818 && \
  python3 $P/sbnd_mgg_sbndflow.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_table.npz \
    --plots $P/plots_cew6_sbnd_sbndflow_${PRE} && \
  python3 $P/sbnd_cc1pi0.py --muon-finder union --shower-bdt-min $TS \
    --recal-gamma-a 0.020101 --recal-gamma-b -15.49 \
    --mc-ntuple ${NT[mc]} --mc-table $P/mc_${PRE}_table.npz \
    --data-ntuple ${NT[data]} --data-table $P/data_${PRE}_table.npz \
    --ext-ntuple ${NT[ext]} --ext-table $P/ext_${PRE}_analysishalf_chi2inf_table.npz \
    --ext-scale 1.1818 --plots $P/plots_cew6_sbnd_${PRE}"
echo DONE_CEW6_FRESH
