"""Start-point / direction resolution of vertex-less photon showers vs truth,
for one or more novtx ntuples (e.g. old vs new start finder). Per true-nu
photon shower (showerTruePID==22, purity>=0.5, unlabeled<0.5, E>20):
angle(reco dir, true photon momentum) and |reco start - true conversion pt|,
split attached / vertex-less.

    python3 validate_novtx_start.py <label>=<ntuple> [<label>=<ntuple> ...]
"""
import sys
import numpy as np
import uproot
import awkward as ak

BR = ["showerLArFormerPID", "showerRecoE", "showerVtxIdx", "showerTruePID",
      "showerTrueTID", "showerTruePurity", "showerTrueUnlabeledPurity",
      "showerStartDirX", "showerStartDirY", "showerStartDirZ",
      "showerStartPosX", "showerStartPosY", "showerStartPosZ",
      "trueSimPartTID", "trueSimPartPx", "trueSimPartPy", "trueSimPartPz",
      "trueSimPartEDepX", "trueSimPartEDepY", "trueSimPartEDepZ"]
TPC_LO = np.array([0.0, -116.5, 0.0]); TPC_HI = np.array([256.35, 116.5, 1036.8])


def run(label, ntuple):
    a = uproot.open(ntuple)["EventTree"].arrays(BR)
    g = ((a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > 20)
         & (a["showerTruePID"] == 22) & (a["showerTruePurity"] >= 0.5)
         & (a["showerTrueUnlabeledPurity"] < 0.5))
    ang = {0: [], 1: []}; dst = {0: [], 1: []}; intpc = {0: [], 1: []}
    for i in np.nonzero(ak.to_numpy(ak.any(g, axis=1)))[0]:
        tid = np.asarray(a["trueSimPartTID"][i])
        P = np.stack([np.asarray(a["trueSimPart" + c][i]) for c in ("Px", "Py", "Pz")], 1).astype(float)
        ED = np.stack([np.asarray(a["trueSimPartEDep" + c][i]) for c in "XYZ"], 1).astype(float)
        for j in np.nonzero(np.asarray(g[i]))[0]:
            k = np.nonzero(tid == a["showerTrueTID"][i][j])[0]
            if not len(k):
                continue
            tp = P[k[0]]; nrm = np.linalg.norm(tp)
            if nrm < 1e-6:
                continue
            d = np.array([a["showerStartDir" + c][i][j] for c in "XYZ"], float)
            s = np.array([a["showerStartPos" + c][i][j] for c in "XYZ"], float)
            key = 1 if a["showerVtxIdx"][i][j] < 0 else 0
            ang[key].append(np.degrees(np.arccos(np.clip(d @ tp / nrm, -1, 1))))
            dst[key].append(np.linalg.norm(s - ED[k[0]]))
            intpc[key].append(bool(np.all(s >= TPC_LO) and np.all(s <= TPC_HI)))
    for key, lab in ((0, "attached"), (1, "vertex-less")):
        r = np.array(ang[key]); ds = np.array(dst[key]); it = np.array(intpc[key])
        if not len(r):
            continue
        print(f"{label:>10s} {lab:12s} N={len(r):5d} | angle median {np.median(r):5.1f} deg, "
              f"<20deg {np.mean(r < 20):.2f}, flipped(>90) {np.mean(r > 90):.2f} | "
              f"|start-conv| median {np.median(ds):5.1f} cm, 90% {np.percentile(ds, 90):5.1f} | "
              f"start in TPC {np.mean(it):.2f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        lab, path = arg.split("=", 1)
        run(lab, path)
