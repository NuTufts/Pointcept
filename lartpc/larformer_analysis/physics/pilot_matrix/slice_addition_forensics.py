"""Is the nu-slice charge S1 ADDED (vs the ep4 reference) deposit
PERIPHERY or genuine contamination? For each event, take points in the
NEW nu slice but not in the REF nu slice, and measure their 3D distance
to the nearest TRUE nu-origin (origin==1) point. Periphery => sub-cm.

  PYTHONPATH=./ python3 slice_addition_forensics.py --label X \
     --ref-dir R --new-dir N --msp-list L --sig-npy S [--n-events 40]
"""
import argparse, glob, os, re
import numpy as np, h5py
from scipy.spatial import cKDTree
from lartpc.larformer_reco.trajfit.calo import dedup_charge
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow import _cell_keys

ap = argparse.ArgumentParser()
for k in ("--label","--ref-dir","--new-dir","--msp-list","--sig-npy"): ap.add_argument(k, required=True)
ap.add_argument("--n-events", type=int, default=40)
a = ap.parse_args()
sig = [int(x) for x in np.load(a.sig_npy)]
msp = [l.strip() for l in open(a.msp_list) if l.strip()]
def index(d):
    o={}
    for p in glob.glob(os.path.join(d,"sliceid_event*.h5")):
        m=re.search(r"sliceid_event0*(\d+)",os.path.basename(p))
        if m: o[int(m.group(1))]=p
    return o
R,N = index(a.ref_dir), index(a.new_dir)
dists=[]; q_added_near=0.0; q_added_far=0.0; q_added_cos=0.0
for rank, ev in enumerate(sig[:a.n_events]):
    if rank not in R or rank not in N: continue
    def sids(p):
        with h5py.File(p,"r") as f:
            return dict(zip(_cell_keys(f["full_slice/coord_cm"][()].astype(np.float32)),
                            f["full_slice/slice_id"][()].astype(np.int64)))
    sr, sn = sids(R[rank]), sids(N[rank])
    with h5py.File(msp[ev],"r") as f:
        td=f["entry_0/triplet_data"]
        pos=np.ascontiguousarray(td["pos"][()],np.float32)
        org=td["origin"][()].astype(np.int64)
        _,q=dedup_charge(td["pixval"][()],td["tick"][()],td["uwire"][()],td["vwire"][()],td["ywire"][()])
    keys=_cell_keys(pos)
    truenu=pos[org==1]
    if len(truenu)<10: continue
    tree=cKDTree(truenu.astype(np.float64))
    added=[i for i,k in enumerate(keys) if sn.get(k)==-5 and sr.get(k)!=-5]
    if not added: continue
    d,_=tree.query(pos[added].astype(np.float64))
    dists.append(d)
    for i,di in zip(added,d):
        if org[i]==2: q_added_cos+=q[i]
        elif di<=1.0: q_added_near+=q[i]
        else: q_added_far+=q[i]
d=np.concatenate(dists); tot=q_added_near+q_added_far+q_added_cos
print(f"[{a.label}] added points={len(d)} over {len(dists)} events")
print(f"  dist to nearest TRUE-nu point: median {np.median(d):.2f} cm  p75 {np.percentile(d,75):.2f}  p95 {np.percentile(d,95):.2f}")
print(f"  frac of added points within 1 cm of true nu: {(d<=1.0).mean():.3f}   within 3 cm: {(d<=3.0).mean():.3f}")
print(f"  ADDED CHARGE split: periphery(<1cm) {q_added_near/tot:.3f} | far(>1cm) {q_added_far/tot:.3f} | true-cosmic {q_added_cos/tot:.3f}")
