"""isogamma_run45: does the nu-slice flash chi2 separate vertices placed on the gun
photon (<5 cm from the true conversion point) from vertices placed on a cosmic?
Run in the pointcept container: python3 isogamma_flashchi2_study.py <DATADIR>.
Results (2026-10-09) are summarized in model_and_output_file_versions.md section 3c.
"""
import uproot, numpy as np, h5py, glob, sys
D=sys.argv[1]
ev=uproot.open(f"{D}/dlgen2_larformer_ntuple_isogamma_run45.root")["EventTree"]
fb=[k for k in ev.keys() if "lash" in k]; print("flash branches:", fb)
br=["run","subrun","event","foundVertex","vtxX","vtxY","vtxZ","trueNuE","nuSliceFlashChi2","primaryVtxStream","flashObsPE","flashTimeUs"]
a=ev.arrays(br,library="np")
idx={k:i for i,k in enumerate(zip(a["run"],a["subrun"],a["event"]))}; n=len(idx)
ed=np.full((n,3),np.nan)
for fn in glob.glob(f"{D}/truth_sidecar/truth_fileno*.h5"):
    with h5py.File(fn) as h:
        for k in h:
            g=h[k]; i=idx.get((g.attrs["run"],g.attrs["subrun"],g.attrs["event"]))
            if i is None: continue
            sp=g["trueSimPart"]; ph=np.where(sp["PDG"][:]==22)[0]
            if len(ph): ed[i]=(sp["EDepX"][ph[0]],sp["EDepY"][ph[0]],sp["EDepZ"][ph[0]])
conv=np.all(np.isfinite(ed),1)&(ed[:,0]>0)&(ed[:,0]<256)&(np.abs(ed[:,1])<116)&(ed[:,2]>0)&(ed[:,2]<1036)
fv=a["foundVertex"]==1
V=np.stack([a["vtxX"],a["vtxY"],a["vtxZ"]],1); d=np.linalg.norm(V-ed,axis=1)
chi=np.asarray(a["nuSliceFlashChi2"],float); print("primaryVtxStream uniq:", np.unique(a["primaryVtxStream"],return_counts=True)); print("flashTimeUs pct 5/50/95:", np.nanpercentile(a["flashTimeUs"][a["flashTimeUs"]>0],[5,50,95]).round(2), " no-flash events:", int(np.sum(a["flashTimeUs"]<=0)))
print("recoVtxFlashChi2 sentinels/uniques<=0:", np.unique(chi[chi<=0],return_counts=True))
for k in []:
    v=a[k]; 
    try: print(f"  {k}: uniq<=0 {np.unique(v[v<=0])[:5]}  pct 50/90 {np.nanpercentile(v[v>0],[50,90]).round(2) if (v>0).any() else '-'}")
    except Exception as e: print("  ",k,"skip",e)
cats={"photon vertex (<5 cm of conversion)": fv&conv&(d<5),
      "ambiguous (5-20 cm)": fv&conv&(d>=5)&(d<20),
      "cosmic pick, photon converted (>20 cm)": fv&conv&(d>=20),
      "cosmic pick, photon never converted": fv&~conv}
print(f"\nfound vertices: {fv.sum()} of {n}")
print(f"{'category':42s} {'N':>6s} {'frac':>6s} {'noflash':>8s} | log10(chi2) pct 10/50/90")
L={}
for lbl,m in cats.items():
    c=chi[m]; ok=c>0; L[lbl]=np.log10(c[ok])
    print(f"{lbl:42s} {m.sum():6d} {m.sum()/fv.sum():6.3f} {np.mean(~ok):8.3f} | {np.percentile(L[lbl],[10,50,90]).round(2)}")
sig=L["photon vertex (<5 cm of conversion)"]; bkg=np.concatenate([L["cosmic pick, photon converted (>20 cm)"],L["cosmic pick, photon never converted"]])
print("\ncut log10(chi2) < c :  photon-vertex kept | cosmic-pick kept | purity among found vertices (photon / (photon+ambig+cosmic))")
allL=np.log10(np.where(chi>0,chi,np.nan))
for c in [1.5,2.0,2.5,3.0,3.5,4.0,99]:
    keep=lambda m: np.sum(m&(allL<c))
    ph=keep(cats["photon vertex (<5 cm of conversion)"]); am=keep(cats["ambiguous (5-20 cm)"]); co=keep(cats["cosmic pick, photon converted (>20 cm)"])+keep(cats["cosmic pick, photon never converted"])
    print(f"  c={c:4.1f}: {np.mean(sig<c):.3f} | {np.mean(bkg<c):.3f} | {ph/(ph+am+co):.3f}   (N photon {ph}, cosmic {co})")
E=a["trueNuE"]
print("\nphoton-vertex log10(chi2) median by E:", [(lo,hi,round(float(np.median(allL[cats['photon vertex (<5 cm of conversion)']&(E>=lo)&(E<hi)&(chi>0)])),2)) for lo,hi in [(0,.1),(.1,.3),(.3,1),(1,10)]])

# ---- follow-up: cosmic picks that pass the chi2 cut -- is the photon in the slice?
a2=ev.arrays(["nShowers","showerStartPosX","showerStartPosY","showerStartPosZ","showerRecoE","vtxFracHitsOnCosmic","vtxScore"],library="np")
near=np.zeros(n,bool)
for i in np.where(conv)[0]:
    if a2["nShowers"][i]==0: continue
    P=np.stack([a2["showerStartPosX"][i],a2["showerStartPosY"][i],a2["showerStartPosZ"][i]],1)
    near[i]=np.linalg.norm(P-ed[i],axis=1).min()<10
hasflash=chi>0
print("\nINCLUDING no-flash events as failing the cut (an analysis cut needs a flash):")
S=cats["photon vertex (<5 cm of conversion)"]; B=cats["cosmic pick, photon converted (>20 cm)"]|cats["cosmic pick, photon never converted"]; A=cats["ambiguous (5-20 cm)"]
for c in [2.0,2.5,3.0,3.5]:
    ks=S&hasflash&(allL<c); kb=B&hasflash&(allL<c); ka=A&hasflash&(allL<c)
    print(f"  log10 chi2 < {c}: photon-vertex eff {ks.sum()/S.sum():.3f}, cosmic-pick leak {kb.sum()/B.sum():.3f}, purity {ks.sum()/(ks.sum()+ka.sum()+kb.sum()):.3f}")
Bc=cats["cosmic pick, photon converted (>20 cm)"]
lo=Bc&hasflash&(allL<2.5); hi=Bc&hasflash&(allL>=2.5)
print(f"\ncosmic picks in CONVERTING events: with a reco shower starting <10 cm from the conversion point (photon reconstructed, vertex elsewhere):")
print(f"  passing log10chi2<2.5 : {near[lo].mean():.3f} of {lo.sum()}")
print(f"  failing               : {near[hi].mean():.3f} of {hi.sum()}")
print(f"  photon-vertex events  : {near[S].mean():.3f} (sanity)")
fc=a2["vtxFracHitsOnCosmic"]; vs=a2["vtxScore"]
for lbl,m in [("photon vertex",S),("cosmic pick, chi2<2.5",lo),("cosmic pick, chi2>=2.5",hi),("cosmic pick, never converted",cats["cosmic pick, photon never converted"])]:
    print(f"  {lbl:30s} vtxFracHitsOnCosmic pct 10/50/90 {np.nanpercentile(fc[m],[10,50,90]).round(2)}   vtxScore pct 10/50/90 {np.nanpercentile(vs[m],[10,50,90]).round(2)}")
