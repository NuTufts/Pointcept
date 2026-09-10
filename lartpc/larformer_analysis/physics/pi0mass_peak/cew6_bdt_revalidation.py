"""cew6 BDT working-point revalidation (talk prep, 2026-09-09).

Part A0: calibration provenance — median showerRecoE/E_true for
truth-matched photons on the cew6 MC ntuple (fresh nu_reco => deployed
recal3 calib may be BAKED IN, unlike the old ntuples). Writes
cew6_calib_mode.txt = 'baked' or 'needs_recal'.
Part A : per-shower cosmic-BDT curve on cew6 (MC nu-photons vs EXT
analysis-half photons) vs the same on the old chain; threshold check
at 0.192.
    python3 cew6_bdt_revalidation.py A
Part B (after tables exist): event-BDT staged point on cew6.
    python3 cew6_bdt_revalidation.py B <ga> <gb>
"""
import sys, os
import numpy as np, uproot, awkward as ak

D="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts"
HERE=os.path.dirname(os.path.abspath(__file__))
NT={"mc_old":f"{D}/larformer_mcoverlay67k_s1ep2p8/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3.root",
    "ext_old":f"{D}/larformer_extbnb200k_s1ep2p8_flash/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f.root",
    "mc":f"{D}/larformer_mcoverlay67k_s1ep2p8cew6/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root",
    "ext":f"{D}/larformer_extbnb200k_s1ep2p8cew6/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root",
    "data":f"{D}/larformer_bnb5e19_s1ep2p8cew6/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root"}

def photons(path, is_mc, row_min=None):
    t=uproot.open(path)["EventTree"]
    br=["showerLArFormerPID","showerCosmicScore","showerRecoE",
        "showerTrueTID","showerTruePhPurity","showerTrueE","showerTruePID"]
    a=t.arrays([b for b in br if b in set(t.keys())])
    pid=ak.flatten(a["showerLArFormerPID"]).to_numpy()
    sc=ak.flatten(a["showerCosmicScore"]).to_numpy()
    E=ak.flatten(a["showerRecoE"]).to_numpy()
    n_per=ak.num(a["showerLArFormerPID"]).to_numpy()
    row=np.repeat(np.arange(len(n_per)),n_per)
    g=(pid==22)&(sc>=0)
    if row_min is not None: g&=(row>=row_min)
    out=dict(sc=sc[g],E=E[g])
    if is_mc:
        tid=ak.flatten(a["showerTrueTID"]).to_numpy()[g]
        pp=ak.flatten(a["showerTruePhPurity"]).to_numpy()[g]
        te=ak.flatten(a["showerTrueE"]).to_numpy()[g]
        tp=np.abs(ak.flatten(a["showerTruePID"]).to_numpy()[g])
        out.update(sig=(tid>0)&(pp>0.5)&(tp==22), trueE=te)
    return out

def curve(sig_sc, bkg_sc, ths):
    eff=np.array([(sig_sc>=t).mean() for t in ths])
    rej=np.array([1-(bkg_sc>=t).mean() for t in ths])
    return eff,rej

if sys.argv[1]=="A":
    mc=photons(NT["mc"],True); mo=photons(NT["mc_old"],True)
    # A0: calibration provenance
    for tag,d in (("cew6",mc),("old",mo)):
        m=d["sig"]&(d["trueE"]>100)&(d["E"]>0)
        r=np.median(d["E"][m]/d["trueE"][m])
        print(f"[A0] {tag}: median RecoE/TrueE (matched photons, E_true>100) = {r:.3f}")
        if tag=="cew6": mode = "baked" if r<1.05 else "needs_recal"
    open(os.path.join(HERE,"cew6_calib_mode.txt"),"w").write(mode+"\n")
    print(f"[A0] cew6 calib mode: {mode}  (old-ntuple July constants give ~1.29x; recal3-baked gives ~1.0 -> ratio vs TrueE ~0.8-0.9 given TrueE=full E)")
    ex=photons(NT["ext"],False,row_min=100000); eo=photons(NT["ext_old"],False,row_min=100000)
    ths=np.linspace(0,0.9,181)
    for tag,m,e in (("cew6",mc,ex),("old",mo,eo)):
        eff,rej=curve(m["sc"][m["sig"]],e["sc"],ths)
        k192=np.argmin(np.abs(ths-0.192))
        print(f"[A] {tag}: N sig {int(m['sig'].sum())} | N EXT {len(e['sc'])} | at thr 0.192: eff {eff[k192]:.3f} rej {rej[k192]:.3f}")
        for target in (0.985,0.97,0.95):
            j=int(np.argmin(np.abs(eff-target)))
            print(f"      eff {target}: thr {ths[j]:.3f} rej {rej[j]:.3f}")
elif sys.argv[1]=="B":
    ga,gb=float(sys.argv[2]),float(sys.argv[3])
    sys.path.insert(0,HERE)
    from datamc_diagnostics import load
    import joblib
    M=joblib.load(os.path.join(HERE,"ext_bdt_model_flashblind.joblib"))
    clf,FEATS=M["clf"],M["feats"]
    smp={}
    for s,tb in (("mc","mc_s1ep2p8cew6_table.npz"),("ext","ext_s1ep2p8cew6_table.npz")):
        d=load(NT[s],os.path.join(HERE,tb),ga,gb,50.0,1e12,1e12,
               shower_bdt_min=0.192,muon_finder="union")
        d["escore"]=clf.predict_proba(np.column_stack([d[f].astype(float) for f in FEATS]))[:,1]
        smp[s]=d
    cat=np.load(os.path.join(HERE,"mc_s1ep2p8cew6_table.npz"))["cat"][smp["mc"]["row"]]
    sig=cat<2; odd=smp["mc"]["event"]%2==1
    ws=smp["mc"]["w"][sig&odd]*2; ss=smp["mc"]["escore"][sig&odd]
    ek=(smp["ext"]["row"]>=100000)&(smp["ext"]["event"]%2==1)
    se=smp["ext"]["escore"][ek]
    for te in (0.15,0.21,0.28,0.35):
        print(f"[B] event-BDT te={te}: sig eff {ws[ss>=te].sum()/ws.sum():.3f} | EXT rej {1-(se>=te).mean():.3f} (N ext {len(se)})")
