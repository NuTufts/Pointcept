"""Step-by-step eff/purity tables at the cew6 talk WP (shower>=0.075,
event BDT>=0.21, union muon finder; cew6 ntuples = recal3-baked, so the
recal transform is identity). Combined >=2-photon table + per-stream
exactly-2 tables (CC: signal=true CC pi0; NC: signal=true NC pi0).
Signal weights use the odd-event x2 estimator (event-BDT hygiene); EXT =
rows>=100k AND odd, x2.3636. Event-BDT pass masks come from the
*_nochi2bdt tables (chi2 gates open at scoring time).

    PYTHONPATH=... python3 cew6_step_tables.py
"""
import os, sys
import numpy as np, uproot, awkward as ak

HERE=os.path.dirname(os.path.abspath(__file__))
D="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts"
MC=f"{D}/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root"
EX=f"{D}/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root"
NGA,NGB=0.020100999623537064,-15.489999771118164   # identity: baked recal3
TS=float(os.environ.get("STEP_TS","0.075"))
PRE=os.environ.get("STEP_PREFIX","cew6_ts0075")

def steps(nt):
    t=uproot.open(nt)["EventTree"]
    a=t.arrays(["run","event","foundVertex","primaryVtxStream","vtxIsFiducial",
                "showerLArFormerPID","showerRecoE","showerCosmicScore",
                "trackLArFormerPID","trackIsSecondary","trackRecoE",
                "trackClassified","trackMuScore","trackElScore","trackPhScore",
                "trackPiScore","trackPrScore"])
    E=a["showerRecoE"]                       # baked: no transform
    vtx=((np.asarray(a["foundVertex"])==1)&(np.asarray(a["primaryVtxStream"])==0)
         &(np.asarray(a["vtxIsFiducial"])==1))
    g_nos=(a["showerLArFormerPID"]==22)&(E>20.0)
    g_sc=g_nos&(a["showerCosmicScore"]>=TS)
    seg=(a["trackLArFormerPID"]==13)&(a["trackIsSecondary"]==0)&(a["trackRecoE"]>50.0)
    lp=((a["trackClassified"]==1)&(a["trackMuScore"]>a["trackElScore"])
        &(a["trackMuScore"]>a["trackPhScore"])&(a["trackMuScore"]>a["trackPiScore"])
        &(a["trackMuScore"]>a["trackPrScore"])&(a["trackIsSecondary"]==0)
        &(a["trackRecoE"]>50.0))
    cc=ak.to_numpy(ak.any(seg|lp,axis=1))
    return dict(vtx=vtx,
                ge2_nos=vtx&(ak.to_numpy(ak.sum(g_nos,axis=1))>=2),
                ge2=vtx&(ak.to_numpy(ak.sum(g_sc,axis=1))>=2),
                eq2_nos=vtx&(ak.to_numpy(ak.sum(g_nos,axis=1))==2),
                eq2=vtx&(ak.to_numpy(ak.sum(g_sc,axis=1))==2),
                cc=cc, event=np.asarray(a["event"]))

mc=steps(MC); ex=steps(EX)
mt=np.load(f"{HERE}/mc_{PRE}_table.npz")
mn=np.load(f"{HERE}/mc_{PRE}_nochi2bdt_table.npz")
et=np.load(f"{HERE}/ext_{PRE}_table.npz")
en=np.load(f"{HERE}/ext_{PRE}_nochi2bdt_table.npz")
cat=mt["cat"]; wmc=np.asarray(mt["w"],float)
odd=mc["event"]%2==1
e_keep=(np.arange(len(ex["vtx"]))>=100000)&(ex["event"]%2==1)
we=np.where(e_keep,1.1818*2.0,0.0)
chi=np.asarray(mn["flash_chi2"],float); echi=np.asarray(en["flash_chi2"],float)
m_chi=np.where(mc["cc"],chi<1e4,chi<1778.0)
e_chi=np.where(ex["cc"],echi<1e4,echi<1778.0)

def table(title, sel_key, strm=None, truecat=None, cut=None):
    if truecat is None:
        sigm=cat<2
    else:
        sigm=cat==truecat
    ws=np.where(sigm&odd,wmc*2.0,0.0); wb=np.where(~sigm,wmc,0.0)
    sm=np.ones(len(wmc),bool) if strm is None else (mc["cc"]==strm)
    se=np.ones(len(we),bool) if strm is None else (ex["cc"]==strm)
    DEN=ws.sum()
    key_nos="ge2_nos" if sel_key=="ge2" else "eq2_nos"
    bdt_m=np.asarray(mn["sel_"+sel_key],bool); bdt_e=np.asarray(en["sel_"+sel_key],bool)
    base_m=np.asarray(mt["sel_"+sel_key],bool); base_e=np.asarray(et["sel_"+sel_key],bool)
    chi_m=m_chi if cut is None else (chi<cut)
    chi_e=e_chi if cut is None else (echi<cut)
    rows=[("all events (stream-tagged)", sm, se),
          ("reco vtx (nu-stream, FV)", sm&mc["vtx"], se&ex["vtx"]),
          (f"{'>=2' if sel_key=='ge2' else 'exactly 2'} photons >20 MeV", sm&mc[key_nos], se&ex[key_nos]),
          (f"+ shower score >={TS}", sm&mc[sel_key], se&ex[sel_key]),
          ("+ mass ok (pair found)", sm&base_m, se&base_e),
          ("+ event BDT >=0.21", sm&bdt_m, se&bdt_e),
          ("+ flash chi2", sm&bdt_m&chi_m, se&bdt_e&chi_e)]
    print(f"\n== {title} | signal denominator (POT-wtd, odd x2): {DEN:.1f} ==")
    print(f"{'cut':<30}{'signal':>8}{'eff':>7}{'purity':>8}{'MC other':>10}{'EXT':>8}")
    for nm,mm,em in rows:
        s=ws[mm].sum(); b=wb[mm].sum(); e=we[em].sum()
        print(f"{nm:<30}{s:8.1f}{s/DEN:7.3f}{s/max(s+b+e,1e-9):8.3f}{b:10.1f}{e:8.1f}")

table("COMBINED CC+NC, >=2 photons (signal = true CC+NC pi0; chi2 per-stream 1e4/1778)","ge2")
table("reco-CC stream, exactly 2 (signal = true CC pi0; chi2<1e4)","eq2",strm=True,truecat=0,cut=1e4)
table("reco-NC stream, exactly 2 (signal = true NC pi0; chi2<1778)","eq2",strm=False,truecat=1,cut=1778.0)
