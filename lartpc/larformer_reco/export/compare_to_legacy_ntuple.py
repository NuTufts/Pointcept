"""Truth-quality comparison: old v0 reco vs exported LArFormer reco, on shared events."""
import numpy as np, uproot

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--new", required=True)
_ap.add_argument("--old", required=True)
_args = _ap.parse_args()
NEW = _args.new
OLD = _args.old
MASS = {11: 0.511, 22: 0.0, 13: 105.6584, 211: 139.5704, 2212: 938.2721}
BR = ["run","subrun","event","foundVertex","vtxX","vtxY","vtxZ","trueVtxX","trueVtxY","trueVtxZ",
      "nTracks","trackPID","trackTruePID","trackTrueTID","trackRecoE","trackTrueE","trackTrueComp","trackTruePurity","trackClassified",
      "nShowers","showerPID","showerTruePID","showerTrueTID","showerRecoE","showerTrueE","showerTrueComp","showerTruePurity","showerClassified",
      "nTrueSimParts","trueSimPartPDG","trueSimPartTID","trueSimPartProcess","trueSimPartE"]

new = uproot.open(NEW)["EventTree"].arrays(BR + ["trueVtxInWCFV","primaryVtxStream","nRecoVtx","recoVtxX","recoVtxY","recoVtxZ"], library="np")
old = uproot.open(OLD)["EventTree"].arrays(BR, library="np")
newk = {(r,s,e): i for i,(r,s,e) in enumerate(zip(new["run"],new["subrun"],new["event"]))}
pairs = [(newk[(r,s,e)], j) for j,(r,s,e) in enumerate(zip(old["run"],old["subrun"],old["event"])) if (r,s,e) in newk]
print(f"shared events (true vtx in WC-FV by old preselection): {len(pairs)}")
ni = np.array([p[0] for p in pairs]); oj = np.array([p[1] for p in pairs])

def vtx_metrics(tag, found, dx, dy, dz):
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    f = found.astype(bool)
    dd = d[f]
    print(f"  {tag:28s} found {f.mean():.3f} | median dist {np.median(dd):6.2f} cm | <1cm {np.mean(dd<1):.3f} | <3cm {np.mean(dd<3):.3f} | <5cm {np.mean(dd<5):.3f}")

print("\n== VERTEX (denominator: shared events; dist for found-vertex events) ==")
vtx_metrics("old v0 reco", old["foundVertex"][oj],
            old["vtxX"][oj]-old["trueVtxX"][oj], old["vtxY"][oj]-old["trueVtxY"][oj], old["vtxZ"][oj]-old["trueVtxZ"][oj])
vtx_metrics("larformer (primary vtx)", new["foundVertex"][ni],
            new["vtxX"][ni]-new["trueVtxX"][ni], new["vtxY"][ni]-new["trueVtxY"][ni], new["vtxZ"][ni]-new["trueVtxZ"][ni])
nu_primary = new["primaryVtxStream"][ni] == 0
vtx_metrics("larformer (nu-stream primary)", (new["foundVertex"][ni].astype(bool) & nu_primary),
            new["vtxX"][ni]-new["trueVtxX"][ni], new["vtxY"][ni]-new["trueVtxY"][ni], new["vtxZ"][ni]-new["trueVtxZ"][ni])
# best vertex in the full recoVtx table
best = np.full(len(ni), np.inf)
for k, i in enumerate(ni):
    if new["nRecoVtx"][i]:
        tv = np.array([new["trueVtxX"][i], new["trueVtxY"][i], new["trueVtxZ"][i]])
        vv = np.stack([new["recoVtxX"][i], new["recoVtxY"][i], new["recoVtxZ"][i]], 1)
        best[k] = np.linalg.norm(vv - tv, axis=1).min()
bb = best[np.isfinite(best)]
print(f"  {'larformer (best of table)':28s} found {np.isfinite(best).mean():.3f} | median dist {np.median(bb):6.2f} cm | <1cm {np.mean(bb<1):.3f} | <3cm {np.mean(bb<3):.3f} | <5cm {np.mean(bb<5):.3f}")

def prong_metrics(tag, a, idx, pre):
    pid, tpid, reco, true = [], [], [], []
    comp, pur = [], []
    for i in idx:
        for j in range(a["n"+ ("Tracks" if pre=="track" else "Showers")][i]):
            if not a[pre+"Classified"][i][j] or a[pre+"TruePID"][i][j] == 0: continue
            pid.append(a[pre+"PID"][i][j]); tpid.append(abs(a[pre+"TruePID"][i][j]))
            m = MASS.get(abs(a[pre+"TruePID"][i][j]), 0.0)
            ket = a[pre+"TrueE"][i][j] - (m if pre=="track" else 0.0)
            if ket > 25:
                reco.append(a[pre+"RecoE"][i][j]); true.append(ket)
            comp.append(a[pre+"TrueComp"][i][j]); pur.append(a[pre+"TruePurity"][i][j])
    pid, tpid = np.array(pid), np.array(tpid)
    fr = (np.array(reco)-np.array(true))/np.array(true) if reco else np.array([np.nan])
    print(f"  {tag:22s} {pre}s: N={len(pid):5d} | PID==true {np.mean(pid==tpid):.3f} | median |dE/E| {np.median(np.abs(fr)):.3f}"
          f" | median TrueComp {np.median(comp):.3f} | median TruePurity {np.median(pur):.3f}")

print("\n== PRONGS (classified + truth-matched; energy for trueKE>25 MeV) ==")
for pre in ("track", "shower"):
    prong_metrics("old v0 reco", old, oj, pre)
    prong_metrics("larformer", new, ni, pre)

SPECIES = {11: "e", 22: "gamma", 13: "mu", 211: "pi", 2212: "p"}

def truth_side(tag, a, idx):
    """Truth-side denominator: true PRIMARY particles (Process==0, KE>25 MeV)
    of reconstructable species; numerator = a classified prong truth-matched
    to that TID exists (found), and one of them has PID == |true PDG|."""
    n_true = {s: 0 for s in SPECIES.values()}
    n_found = {s: 0 for s in SPECIES.values()}
    n_pid = {s: 0 for s in SPECIES.values()}
    for i in idx:
        pid_by_tid = {}
        for pre, cn in (("track", "nTracks"), ("shower", "nShowers")):
            for j in range(a[cn][i]):
                if not a[pre + "Classified"][i][j]:
                    continue
                t = int(a[pre + "TrueTID"][i][j])
                if t > 0:
                    pid_by_tid.setdefault(t, set()).add(
                        int(a[pre + "PID"][i][j]))
        for j in range(a["nTrueSimParts"][i]):
            pdg = abs(int(a["trueSimPartPDG"][i][j]))
            if pdg not in SPECIES or a["trueSimPartProcess"][i][j] != 0:
                continue
            ke = float(a["trueSimPartE"][i][j]) - MASS.get(pdg, 0.0)
            if not np.isfinite(ke) or ke <= 25.0:
                continue
            sp = SPECIES[pdg]
            n_true[sp] += 1
            pids = pid_by_tid.get(int(a["trueSimPartTID"][i][j]))
            if pids:
                n_found[sp] += 1
                if pdg in pids:
                    n_pid[sp] += 1
    row = " | ".join(
        f"{s}: {n_found[s]/n_true[s]:.3f}/{n_pid[s]/n_true[s]:.3f} (N={n_true[s]})"
        if n_true[s] else f"{s}: -" for s in ("e", "gamma", "mu", "pi", "p"))
    print(f"  {tag:12s} {row}")

print("\n== TRUTH-SIDE EFFICIENCY (true primaries, KE>25 MeV; "
      "found/found-with-correct-PID fractions) ==")
truth_side("old v0 reco", old, oj)
truth_side("larformer", new, ni)
