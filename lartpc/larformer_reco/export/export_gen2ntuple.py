"""Export larformer_reco products to the gen2ntuple flat ROOT format.

Event universe = the merged_sp list (every event gets an EventTree row; events
with no reconstructed interaction get foundVertex=0 + defaults). Per event it
joins: the truth sidecar (GENIE + mcreco truth, POT), the xsecWeight pickle,
and BOTH slice streams' nu_reco_larpid shards + keypoint2 files.

Interaction merging (user design): interactions are ranked nu-stream first
(by seed vertex score, desc), then flashmatch-stream (by score). Rank 1 fills
the legacy single-vertex scalars (vtxX..., foundVertex; a flashmatch vertex
MAY be primary — analyzers filter on primaryVtxStream / recoVtxStream). ALL
interactions land in the recoVtx* table and every prong carries
trackVtxIdx/showerVtxIdx into it.

Documented redefinitions vs the legacy maker (user-approved):
- prong True{Purity,Comp} are DE-DOUBLE-COUNTED-CHARGE based (2026-08-31,
  user-approved; previously point counts): purity = dedup charge of pred∩GT
  over dedup charge of the predicted cluster (dedup within the predicted set,
  i.e. the currency the energy reco sums); comp = the same intersection charge
  in the GT-set dedup frame over the GT cluster's charge. TrueXxPurity =
  charge fractions of the predicted cluster by true species;
  TrueUnlabeledPurity = charge fraction with no truth owner (trackid<=0 /
  unmatched — real-cosmic + unlabeled periphery in overlay originals).
  Purity/comp are TID-based (cluster charge with merged_sp trackid ==
  matched TID; comp vs the TID's total event dedup charge, capped 1.5) —
  they live entirely off merged_sp labels, so re-export alone tracks
  label completion; kp2 gt_point_idx (frozen at inference) is not used.
- trackRecoE = part_energy - m(pred class) (our range/calo pipeline), not the
  LArPID-hypothesis switch; showerRecoE = part_energy.
- HitFrac/ChargeFrac denominators = sums over ALL exported prongs of the
  event (all interactions, both streams).
- vtxFracHitsOnCosmic = -1 (no thrumu in this chain).
- MC events are NOT dropped: out-of-WC-FV true vertices are flagged
  (trueVtxInWCFV), missing/inf xsecWeight -> -1 (counted in the log).
- VERTEX-LESS PRONGS (2026-09-06): every segmenter particle of each stream's
  kp2 slice that is NOT part of a nu_reco interaction (incl. all particles
  of events with no interaction) is exported with VtxIdx=-1 straight from
  the kp2 particle groups: start / direction from shower_start_dir()
  (2026-09-07): the shower's points are clustered with DBSCAN
  (--orphan-dbscan-eps) and only clusters carrying >= --orphan-min-cluster-mev
  of calibrated charge are kept (noise-deposit rejection; all points if none
  qualifies); the two extreme kept points along the 1st principal axis are the
  candidate ends; the end closer to the kp2 keypoint start (the model's
  origin-point prediction) is the START, the direction is the principal axis
  oriented start -> other end. Charge = de-double-counted comb
  charge of its own points, showerRecoE from the deployed calo calib,
  LArPID block at defaults, truth match from the kp2 gt_trackid attr
  (--orphan-min-points, --no-orphans). Objectness (1-P(no_object)) and
  Stream are filled for ALL prongs; nuSliceFlashChi2/fmSliceFlashChi2 carry
  the slice chi2 so vertex-less candidates can be flash-cut.
- SHOWER ENERGY (2026-09-06): showerRecoE is recomputed at export from the
  prong's comb Charge with the DEPLOYED trajfit calo_calib (gamma affine,
  e linear) so attached and vertex-less showers share one calibration
  (--no-recalib-showers keeps nu_reco's part_energy). Analyses must then NOT
  apply the old invert-and-reapply recal flags to these ntuples.

    PYTHONPATH=./ python3 lartpc/larformer_reco/export/export_gen2ntuple.py \
      --merged-sp-list ... --truth-dir ... \
      --kp2-nu-list ... --kp2-fm-list ... \
      --nu-reco-nu-dir ... --nu-reco-fm-dir ... \
      --weights-pkl gen2ntuple/event_weighting/weights_forCV_v48_Sep24_bnb_run3.pkl \
      --out dlgen2_larformer_ntuple.root [--start 0 --n -1]
"""
import os
import re
import sys
import glob
import pickle
import ctypes
import argparse

import numpy as np
import h5py
import uproot

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from lartpc.larformer_reco.export import schema  # noqa: E402
from lartpc.larformer_reco.utils import read_list  # noqa: E402
from lartpc.larformer_reco.trajfit.calo import dedup_charge  # noqa: E402
from lartpc.larformer_reco.trajfit.particle_momentum import load_shower_calib  # noqa: E402

MASS = {0: 0.511, 1: 0.0, 2: 105.6584, 3: 139.5704, 4: 938.2721, 5: 0.0}
LARFORMER_PDG = {0: 11, 1: 22, 2: 13, 3: 211, 4: 2212, 5: 0}
SPECIES_PIDS = {"El": (11, -11), "Ph": (22,), "Mu": (13, -13),
                "Pi": (211, -211), "Pr": (2212,)}
STREAM_CODE = {"nu": 0, "flashmatch": 1}


def _attr_str(attrs, k):
    v = attrs.get(k, "")
    return v.decode() if isinstance(v, bytes) else v


def load_wcfv(libpath):
    class WCFiducial(ctypes.Structure):
        pass
    lib = ctypes.cdll.LoadLibrary(libpath)
    lib.WCFiducial_new.restype = ctypes.POINTER(WCFiducial)
    lib.WCFiducial_insideFV.argtypes = (ctypes.POINTER(WCFiducial),
                                        ctypes.c_double, ctypes.c_double,
                                        ctypes.c_double)
    lib.WCFiducial_insideFV.restype = ctypes.c_bool
    obj = lib.WCFiducial_new()
    return lambda x, y, z: bool(lib.WCFiducial_insideFV(
        obj, float(x), float(y), float(z)))


def build_kp_map(kp_list):
    """src_file -> (gidx, kp_path)."""
    out = {}
    for gidx, p in enumerate(kp_list):
        try:
            with h5py.File(p, "r") as f:
                src = _attr_str(f.attrs, "src_file")
            if src:
                out[src] = (gidx, p)
        except Exception:
            continue
    return out


def build_reco_map(nu_reco_dir):
    """gidx -> (shard path, event group name)."""
    out = {}
    for shard in sorted(glob.glob(os.path.join(nu_reco_dir, "*.h5"))):
        try:
            with h5py.File(shard, "r") as f:
                for ev in f:
                    out[int(ev.split("_")[-1])] = (shard, ev)
        except Exception as e:
            print(f"  [warn] {shard}: {e}")
    return out


class TruthIndex:
    """fileno -> sidecar; (run, subrun, event) -> entry group, lazily."""

    def __init__(self, truth_dir):
        self.dir = truth_dir
        self._rse = {}          # fileno -> {(r,s,e): entry name}
        self.pot = {}           # fileno -> (totPOT, totGoodPOT)

    def entry(self, fileno, rse):
        if fileno not in self._rse:
            path = os.path.join(self.dir, f"truth_fileno{fileno:05d}.h5")
            idx = {}
            try:
                with h5py.File(path, "r") as f:
                    self.pot[fileno] = (float(f.attrs["totPOT"]),
                                        float(f.attrs["totGoodPOT"]))
                    for ev in f:
                        a = f[ev].attrs
                        idx[(int(a["run"]), int(a["subrun"]),
                             int(a["event"]))] = ev
            except Exception as e:
                print(f"  [warn] truth sidecar fileno {fileno}: {e}")
            self._rse[fileno] = idx
        name = self._rse[fileno].get(rse)
        if name is None:
            return None, None
        return os.path.join(self.dir, f"truth_fileno{fileno:05d}.h5"), name


def fill_truth(ev, ftruth, entry):
    g = ftruth[entry]
    for k in ("trueNuE", "trueNuPDG", "trueNuCCNC", "trueNuMode",
              "trueNuIntrxnType", "trueVtxX", "trueVtxY", "trueVtxZ",
              "trueVtxInWCFV", "trueLepE", "trueLepPDG"):
        ev[k] = g.attrs[k]
    pp, sp = g["truePrimPart"], g["trueSimPart"]
    for b in schema.GROUPS["truePrimPart"][1]:
        ev["truePrimPart"][b[0]] = list(pp[b[0][len("truePrimPart"):]][()])
    for b in schema.GROUPS["trueSimPart"][1]:
        key = b[0][len("trueSimPart"):]
        if key not in sp:
            continue           # exporter-computed branch (PixelSumQ)
        ev["trueSimPart"][b[0]] = list(sp[key][()])
    return {int(t): (int(p), float(e))
            for t, p, e in zip(sp["TID"][()], sp["PDG"][()], sp["E"][()])}


def msp_qvis(msp_path, tids):
    """Uncalibrated visible-charge sum per trackid: de-double-counted
    unique-pixel charge (comb: Y else mean(U,V)) of the merged_sp triplet_data
    spacepoints truth-matched to each tid -- identical to the reco eval's
    q_true (E_vis ~= shower gamma calib factor * this)."""
    out = {int(t): 0.0 for t in tids}
    if not out:
        return out
    with h5py.File(msp_path, "r") as f:
        td = f["entry_0/triplet_data"]
        tid = np.asarray(td["trackid"][()], np.int64)
        sel = np.isin(tid, np.asarray(list(out), np.int64))
        if not sel.any():
            return out
        pix = td["pixval"][()][sel]
        tick = td["tick"][()][sel]
        uw = td["uwire"][()][sel]
        vw = td["vwire"][()][sel]
        yw = td["ywire"][()][sel]
        tid = tid[sel]
    for t in np.unique(tid):
        m = tid == t
        _, q_comb = dedup_charge(pix[m], tick[m], uw[m], vw[m], yw[m])
        out[int(t)] = float(q_comb.sum())
    return out


class MspTruthPoints:
    """Per-point truth + de-double-counted charge lookup for slice coords
    (charge-based purity/completeness/species fractions)."""

    def __init__(self, msp_path):
        with h5py.File(msp_path, "r") as f:
            td = f["entry_0/triplet_data"]
            pos = td["pos"][()].astype(np.float32)
            self.pid = td["pid"][()].astype(np.int64)
            self.tid = td["trackid"][()].astype(np.int64)
            self.pix = td["pixval"][()].astype(np.float64)
            self.tick = td["tick"][()].astype(np.int64)
            self.uw = td["uwire"][()].astype(np.int64)
            self.vw = td["vwire"][()].astype(np.int64)
            self.yw = td["ywire"][()].astype(np.int64)
            self._row = {pos[i].tobytes(): i for i in range(len(pos))}

    def rows_for(self, coords):
        c = np.asarray(coords, np.float32)
        return np.asarray([self._row.get(c[i].tobytes(), -1)
                           for i in range(len(c))], np.int64)

    def pids_for(self, coords):
        rows = self.rows_for(coords)
        out = np.full(len(rows), 0, np.int64)
        ok = rows >= 0
        out[ok] = self.pid[rows[ok]]
        return out

    def dedup_comb(self, rows):
        """Per-row de-double-counted comb charge, dedup defined WITHIN the
        given row set (rows<0 -> 0)."""
        q = np.zeros(len(rows), np.float64)
        ok = rows >= 0
        if ok.any():
            r = rows[ok]
            _, qc = dedup_charge(self.pix[r], self.tick[r], self.uw[r],
                                 self.vw[r], self.yw[r])
            q[ok] = qc
        return q

    def unlabeled_mask(self, rows):
        """True where a point has no truth owner (unmatched row, trackid<=0,
        or a sentinel pid)."""
        out = np.ones(len(rows), bool)
        ok = rows >= 0
        r = rows[ok]
        out[ok] = (self.tid[r] <= 0) | (self.pid[r] == 0) | (self.pid[r] == -1)
        return out



def _model_path(env, default_name):
    """Deployed joblib under export/data/ by default (2026-09-07); the env var
    overrides the path, and env == 'none' disables the model."""
    v = os.environ.get(env, "").strip()
    if v.lower() == "none":
        return None
    if v:
        return v if os.path.exists(v) else None
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", default_name)
    return p if os.path.exists(p) else None


# ---- per-shower cosmic BDT (default export/data/shower_cosmic_bdt.joblib;
# env LARFORMER_SHOWER_BDT overrides, 'none' disables) --------------------
_SHOWER_BDT = None
def _load_shower_bdt():
    global _SHOWER_BDT
    if _SHOWER_BDT is not None:
        return _SHOWER_BDT
    path = _model_path("LARFORMER_SHOWER_BDT", "shower_cosmic_bdt.joblib")
    if not path:
        _SHOWER_BDT = False
        return False
    import joblib
    _SHOWER_BDT = joblib.load(path)
    print(f">>> shower cosmic BDT loaded: {path} feats={len(_SHOWER_BDT['feats'])}",
          flush=True)
    return _SHOWER_BDT


def _dwall(p):
    lo = np.array([0.0, -116.5, 0.0]); hi = np.array([256.35, 116.5, 1036.8])
    return float(min((p - lo).min(), (hi - p).min()))


def _pca_dir(pts, start):
    """Unit principal axis of pts, oriented from `start` toward the centroid.
    Falls back to start->centroid, then +z."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return np.array([0.0, 0.0, 1.0], np.float32)
    c = pts.mean(0)
    v = c - np.asarray(start, np.float64)
    if len(pts) >= 3:
        try:
            _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
            ax = vt[0]
            if np.dot(ax, v) < 0:
                ax = -ax
            v = ax
        except np.linalg.LinAlgError:
            pass
    n = np.linalg.norm(v)
    return (v / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])).astype(np.float32)


def shower_start_dir(pts, qpt, kp_start, slope, min_mev=5.0, eps=2.0,
                     min_samples=5):
    """Vertex-less shower start point + direction (2026-09-07 recipe).

    pts (n,3) cm, qpt (n,) per-point de-double-counted comb charge, kp_start
    the kp2 keypoint-model start (origin prediction; may be NaN), slope the
    calo MeV/charge factor of the shower's class.
      1. principal axis via SVD of the points;
      2. DBSCAN(eps, min_samples) on the points; clusters with
         slope * sum(charge) >= min_mev are 'real'; if any, the extrema are
         searched among their points only (else among all points);
      3. the extreme point (along the axis) closer to kp_start is the start
         (kp_start absent -> the lower-projection extremum);
      4. direction = principal axis, sign start -> other extremum.
    Returns (start[3], dir[3], n_good_clusters)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n == 0:
        return (np.asarray(kp_start, np.float32), np.array([0, 0, 1], np.float32), 0)
    if n < 3:
        st = pts[0]
        return (st.astype(np.float32), _pca_dir(pts, st), 0)
    c = pts.mean(0)
    try:
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        ax = vt[0]
    except np.linalg.LinAlgError:
        ax = np.array([0.0, 0.0, 1.0])
    proj = (pts - c) @ ax
    sel = np.ones(n, bool)
    ngood = 0
    if qpt is not None and len(qpt) == n:
        try:
            from sklearn.cluster import DBSCAN
            lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
            good = [l for l in np.unique(lab)
                    if l >= 0 and slope * float(np.sum(qpt[lab == l])) >= min_mev]
            ngood = len(good)
            if good:
                sel = np.isin(lab, good)
        except Exception:
            pass
    i_lo = int(np.argmin(np.where(sel, proj, np.inf)))
    i_hi = int(np.argmax(np.where(sel, proj, -np.inf)))
    lo, hi = pts[i_lo], pts[i_hi]
    ks = np.asarray(kp_start, np.float64)
    if np.all(np.isfinite(ks)) and np.linalg.norm(hi - ks) < np.linalg.norm(lo - ks):
        start, other = hi, lo
    else:
        start, other = lo, hi
    d = ax if np.dot(ax, other - start) >= 0 else -ax
    return start.astype(np.float32), d.astype(np.float32), ngood


def shower_energy(cls_id, charge, calib):
    """Deployed calo calib: E = a*Q (+ b for types with an affine offset)."""
    if calib is None or not np.isfinite(charge) or charge < 0:
        return None
    key = {0: "e", 1: "gamma"}.get(int(cls_id), "gamma")
    if key not in calib:
        return None
    return max(calib[key] * float(charge) + calib.get(key + "_b", 0.0), 0.0)


def fill_truth_match(p, pts, npts, tid, tid_lookup, msp_pts, qv,
                     dominant_fallback=False, rows_qp=None):
    """Truth-match block shared by attached and vertex-less prongs. Returns
    the prong's own de-double-counted comb charge (or None).
    dominant_fallback: when tid<=0 (no kp2 IoU match, the vertex-less case)
    take the labeled trackid owning the most comb charge in the prong."""
    p["TruePurity"] = p["TrueComp"] = -1.0
    for nm in SPECIES_PIDS:
        p["True" + nm + "Purity"] = -1.0
    p["TrueUnlabeledPurity"] = -1.0
    Qp, rows_p, qp = None, None, None
    if rows_qp is not None:
        rows_p, qp = rows_qp
        Qp = float(qp.sum())
    elif pts is not None and npts and msp_pts is not None:
        rows_p = msp_pts.rows_for(pts)
        qp = msp_pts.dedup_comb(rows_p)
        Qp = float(qp.sum())
    if dominant_fallback and tid <= 0 and Qp and Qp > 0:
        okr = rows_p >= 0
        rt = msp_pts.tid[rows_p[okr]]
        qq = qp[okr]
        lab = rt > 0
        if lab.any():
            u, inv = np.unique(rt[lab], return_inverse=True)
            tid = int(u[np.argmax(np.bincount(inv, weights=qq[lab]))])
    p["TrueTID"] = tid
    tp, te = tid_lookup.get(tid, (0, -1.0))
    p["TruePID"], p["TrueE"] = tp, te
    if Qp is None:
        return None
    if Qp <= 0:
        return Qp
    row_tid = np.full(len(rows_p), -1, np.int64)
    okr0 = rows_p >= 0
    row_tid[okr0] = msp_pts.tid[rows_p[okr0]]
    in_gt = row_tid == tid
    p["TruePurity"] = float(qp[in_gt].sum() / Qp)
    qtot = qv.get(int(tid), 0.0)
    if qtot > 0:
        p["TrueComp"] = float(min(qp[in_gt].sum() / qtot, 1.5))
    pids = np.full(len(rows_p), 0, np.int64)
    pids[okr0] = msp_pts.pid[rows_p[okr0]]
    unl = msp_pts.unlabeled_mask(rows_p)
    for nm, pdgs in SPECIES_PIDS.items():
        p["True" + nm + "Purity"] = float(qp[np.isin(pids, pdgs) & ~unl].sum() / Qp)
    p["TrueUnlabeledPurity"] = float(qp[unl].sum() / Qp)
    return Qp


# ---- vertex-FREE per-shower BDT (env LARFORMER_SHOWER_BDT_NOVTX -> joblib) --
# model dict: {"clf", "feats": [names]}; feature names resolved by NOVTX_FEATS.
_NOVTX_BDT = None
def _load_novtx_bdt():
    global _NOVTX_BDT
    if _NOVTX_BDT is not None:
        return _NOVTX_BDT
    path = _model_path("LARFORMER_SHOWER_BDT_NOVTX", "shower_novtx_bdt.joblib")
    if not path:
        _NOVTX_BDT = False
        return False
    import joblib
    _NOVTX_BDT = joblib.load(path)
    print(f">>> vertex-free shower BDT loaded: {path} feats={_NOVTX_BDT['feats']}",
          flush=True)
    return _NOVTX_BDT


def novtx_features(p, ev):
    """Per-shower feature map for the vertex-free BDT (all vertex-independent;
    the trainer picks a subset by name and stores it in the model)."""
    st = np.array([p["StartPosX"], p["StartPosY"], p["StartPosZ"]], float)
    chi2 = ev["nuSliceFlashChi2"] if p["Stream"] == 0 else ev["fmSliceFlashChi2"]
    return {"E": p["RecoE"], "cosZ": p["CosTheta"], "cosY": p["CosThetaY"],
            "sdwall": _dwall(st), "startX": st[0], "startY": st[1],
            "startZ": st[2], "objectness": p["Objectness"],
            "phS": p["LArFormerPhScore"], "elS": p["LArFormerElScore"],
            "muS": p["LArFormerMuScore"], "piS": p["LArFormerPiScore"],
            "prS": p["LArFormerPrScore"], "nhits": p["NHits"],
            "charge": p["Charge"], "chargefrac": p.get("ChargeFrac", -1.0),
            "hasVtx": int(p["VtxIdx"] >= 0), "stream": p["Stream"],
            "nSlicePart": (ev["nuSliceNParticles"] if p["Stream"] == 0
                           else ev["fmSliceNParticles"]),
            "logchi2": (float(np.log10(chi2)) if chi2 > 0 else -1.0)}


def score_showers_novtx(prongs, ev):
    for key, p in prongs:
        if key == "shower":
            p["NoVtxScore"] = -9.0
    M = _load_novtx_bdt()
    if not M:
        return
    rows, refs = [], []
    for key, p in prongs:
        if key != "shower":
            continue
        if p["LArFormerPID"] == 11:
            p["NoVtxScore"] = 1.0
            continue
        if p["LArFormerPID"] != 22 or p["RecoE"] <= 0:
            continue
        f = novtx_features(p, ev)
        rows.append([f[k] for k in M["feats"]])
        refs.append(p)
    if rows:
        sc = M["clf"].predict_proba(np.asarray(rows, float))[:, 1]
        for p, s in zip(refs, sc):
            p["NoVtxScore"] = float(s)


def score_showers(prongs, vtx_by_idx, vtx_score_by_idx):
    """Fill p['CosmicScore'] for every shower prong: electrons 1.0 (autopass),
    photons = BDT score, others -9. Features mirror
    shower_cosmic_bdt.py (recal applied from the model's stored constants)."""
    M = _load_shower_bdt()
    for key, p in prongs:
        if key == "shower":
            p["CosmicScore"] = -9.0
    if not M:
        return
    ga, gb = M["recal"]
    OA, OB = 0.020101, -15.49

    def bdt_E(p):
        """Photon energy in the BDT's recal convention. Prefer the comb
        Charge (exact, calibration-independent); fall back to inverting
        the OLD deployed calib from RecoE (pre-2026-09-06 ntuples)."""
        if p["LArFormerPID"] != 22:
            return p["RecoE"]
        q = p.get("Charge", np.nan)
        if np.isfinite(q) and q >= 0:
            return ga * q + gb
        return (p["RecoE"] - OB) / OA * ga + gb if p["RecoE"] > 0 else p["RecoE"]
    # interaction context per VtxIdx
    ctx = {}
    for key, p in prongs:
        c = ctx.setdefault(p["VtxIdx"], {"nprim": 0, "nsh": 0, "nph": 0})
        if key == "track" and p.get("IsSecondary", 0) == 0:
            c["nprim"] += 1
        if key == "shower":
            E = bdt_E(p)
            if E > 20:
                c["nsh"] += 1
                if p["LArFormerPID"] == 22:
                    c["nph"] += 1
    rows, refs = [], []
    for key, p in prongs:
        if key != "shower":
            continue
        if p["LArFormerPID"] == 11:
            p["CosmicScore"] = 1.0
            continue
        if p["LArFormerPID"] != 22 or p["RecoE"] <= 0:
            continue
        v = vtx_by_idx.get(p["VtxIdx"])
        if v is None:
            continue
        E = bdt_E(p)
        st = np.array([p["StartPosX"], p["StartPosY"], p["StartPosZ"]], float)
        c = ctx[p["VtxIdx"]]
        rows.append([E, p["CosTheta"], p["CosThetaY"], p["DistToVtx"], _dwall(st),
                     p.get("AttScore", -9.0), p.get("AttConfident", 1),
                     p["LArFormerPhScore"], p["LArFormerElScore"], p["LArFormerMuScore"],
                     p["LArFormerPiScore"], p["LArFormerPrScore"], p["NHits"],
                     p.get("ChargeFrac", -1.0), v[0], v[1], v[2], _dwall(np.asarray(v, float)),
                     vtx_score_by_idx.get(p["VtxIdx"], -1.0), c["nprim"], c["nsh"], c["nph"]])
        refs.append(p)
    if rows:
        sc = M["clf"].predict_proba(np.asarray(rows, float))[:, 1]
        for p, s in zip(refs, sc):
            p["CosmicScore"] = float(s)

def prong_rows(gr):
    """nu_reco event group -> dict of part_* arrays."""
    keys = ["part_att_score", "part_att_confident",
            "part_interaction", "part_kind", "part_pred_class", "part_energy",
            "part_charge", "part_gt_trackid", "part_vtx", "part_inst_idx",
            "part_pred_class",
            "part_start_cm", "part_direction", "part_npoly", "part_poly_cm",
            "larpid_classified", "larpid_scores", "larpid_completeness",
            "larpid_purity", "larpid_process_scores", "larpid_pid",
            "larpid_process"]
    d = {k: gr[k][()] for k in keys if k in gr}
    d["polys"] = (np.split(d["part_poly_cm"],
                           np.cumsum(d["part_npoly"])[:-1])
                  if len(d.get("part_npoly", [])) else [])
    d["vtx_depth"] = gr["vtx_depth"][()]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--kp2-nu-list", required=True)
    ap.add_argument("--kp2-fm-list", required=True)
    ap.add_argument("--nu-reco-nu-dir", required=True,
                    help="nu_reco_larpid shards for the nu stream")
    ap.add_argument("--nu-reco-fm-dir", required=True)
    ap.add_argument("--weights-pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--no-recalib-showers", action="store_true",
                    help="keep nu_reco's part_energy for attached showers "
                         "instead of recomputing E from the comb Charge with "
                         "the deployed calo_calib (default: recompute)")
    ap.add_argument("--no-orphans", action="store_true",
                    help="do not export vertex-less segmenter particles")
    ap.add_argument("--orphan-min-points", type=int, default=10,
                    help="min slice points for a vertex-less prong")
    ap.add_argument("--orphan-dbscan-eps", type=float, default=2.0,
                    help="DBSCAN eps [cm] for the vertex-less shower start finder")
    ap.add_argument("--orphan-min-cluster-mev", type=float, default=5.0,
                    help="min calibrated charge [MeV] of a DBSCAN cluster to host "
                         "the vertex-less shower start")
    ap.add_argument("--wcfv-lib",
                    default=os.path.join(os.path.dirname(os.path.abspath(
                        __file__)), "lib_wirecell_fiducial_volume.so"))
    args = ap.parse_args()

    msp = read_list(args.merged_sp_list)
    # first-occurrence index of each fileno over the FULL list: sharded runs
    # write a fileno's potTree entry ONLY in the shard holding its first
    # event, so hadd-merged shards counts POT exactly once even when a
    # fileno's events straddle a shard boundary.
    first_by_fileno = {}
    for i, pth in enumerate(msp):
        m = re.search(r"fileno(\d+)", os.path.basename(pth))
        if m and int(m.group(1)) not in first_by_fileno:
            first_by_fileno[int(m.group(1))] = i
    shard_end = len(msp) if args.n < 0 else min(args.start + args.n,
                                                 len(msp))
    msp = msp[args.start:shard_end]
    print(f">>> {len(msp)} events [{args.start}:{shard_end})", flush=True)

    in_wcfv = load_wcfv(args.wcfv_lib)
    truth = TruthIndex(args.truth_dir)
    weights = pickle.load(open(args.weights_pkl, "rb"))
    calib = load_shower_calib() or None
    if args.no_recalib_showers:
        print(">>> showerRecoE = nu_reco part_energy (no export-time recalib)")
    else:
        print(f">>> showerRecoE recomputed from comb Charge with calo_calib {calib}")
    streams = {}
    for s, klist, rdir in (("nu", args.kp2_nu_list, args.nu_reco_nu_dir),
                           ("flashmatch", args.kp2_fm_list,
                            args.nu_reco_fm_dir)):
        streams[s] = {"kp": build_kp_map(read_list(klist)),
                      "reco": build_reco_map(rdir)}
        print(f">>> stream {s}: {len(streams[s]['kp'])} kp2, "
              f"{len(streams[s]['reco'])} reco events", flush=True)

    events, n_now, n_nownr = [], 0, 0
    fout = uproot.recreate(args.out)
    tree = schema.mktree(fout)
    FLUSH = 4000                # events per extend() batch (bounds RAM)
    stats = dict(noweight=0, notruth=0, found=0, orphans=0, orphan_events=0)
    for msp_path in msp:
        base = os.path.basename(msp_path)
        ev = schema.new_event()
        m = re.search(r"fileno(\d+)", base)
        fileno = int(m.group(1)) if m else -1
        ev["fileid"] = fileno
        with h5py.File(msp_path, "r") as f:
            a = f["entry_0"].attrs
            rse = (int(a["run"]), int(a["subrun"]), int(a["event"]))
        ev["run"], ev["subrun"], ev["event"] = rse

        # ---- truth + weight -------------------------------------------------
        tpath, tentry = truth.entry(fileno, rse)
        tid_lookup = {}
        if tpath is not None:
            with h5py.File(tpath, "r") as ft:
                tid_lookup = fill_truth(ev, ft, tentry)
        else:
            stats["notruth"] += 1
        qv = {}
        if tid_lookup:
            try:
                qv = msp_qvis(msp_path, tid_lookup.keys())
            except Exception as ex:
                print(f"  [warn] qvis {base}: {ex}", flush=True)
        ev["trueSimPart"]["trueSimPartPixelSumQ"] = [
            float(qv.get(int(t), -1.0))
            for t in ev["trueSimPart"]["trueSimPartTID"]]
        try:
            w = float(weights[rse[0]][rse[1]][rse[2]])
            ev["xsecWeight"] = w if np.isfinite(w) else -1.0
            if not np.isfinite(w):
                stats["noweight"] += 1
        except Exception:
            stats["noweight"] += 1

        # ---- collect interactions from both streams -------------------------
        vtx_rows = []                      # (score, stream, x, y, z, chi2, ref)
        stream_ev = {}
        kp_paths = {}                      # stream -> kp2 path (reco or not)
        for s in ("nu", "flashmatch"):
            hit = streams[s]["kp"].get(base)
            if hit is None:
                continue
            gidx, kp_path = hit
            kp_paths[s] = kp_path
            rr = streams[s]["reco"].get(gidx)
            if rr is None:
                continue
            fr = h5py.File(rr[0], "r")
            gr = fr[rr[1]]
            if _attr_str(gr.attrs, "src_file") != base:
                fr.close()
                continue
            stream_ev[s] = (fr, gr, kp_path)
            vcm = gr["vertices_cm"][()]
            vsc = gr["vertices_score"][()]
            chi2 = float(gr.attrs.get("flash_chi2", np.nan))
            for ii in range(len(vcm)):
                vtx_rows.append((float(vsc[ii]), s, vcm[ii], chi2, ii))
        vtx_rows.sort(key=lambda r: (0 if r[1] == "nu" else 1, -r[0]))

        for sc, s, pos, chi2, ii in vtx_rows:
            ev["recoVtx"]["recoVtxX"].append(float(pos[0]))
            ev["recoVtx"]["recoVtxY"].append(float(pos[1]))
            ev["recoVtx"]["recoVtxZ"].append(float(pos[2]))
            ev["recoVtx"]["recoVtxScore"].append(sc)
            ev["recoVtx"]["recoVtxStream"].append(STREAM_CODE[s])
            ev["recoVtx"]["recoVtxFlashChi2"].append(chi2)
        if vtx_rows:
            sc, s, pos, chi2, ii = vtx_rows[0]
            ev["foundVertex"] = 1
            ev["vtxX"], ev["vtxY"], ev["vtxZ"] = (float(pos[0]),
                                                  float(pos[1]), float(pos[2]))
            ev["vtxScore"] = sc
            ev["primaryVtxStream"] = STREAM_CODE[s]
            ev["vtxIsFiducial"] = int(in_wcfv(*pos))
            if np.isfinite([ev["trueVtxX"]]).all() and ev["trueVtxX"] > -9000:
                ev["vtxDistToTrue"] = float(np.linalg.norm(
                    np.asarray(pos) - np.asarray(
                        [ev["trueVtxX"], ev["trueVtxY"], ev["trueVtxZ"]])))
            stats["found"] += 1

        # ---- prongs over all interactions, both streams ----------------------
        tot_hits, tot_charge, reco_e = 0, 0.0, 0.0
        prongs = []                    # (group key, per-branch dict)
        contained = True
        msp_pts = None
        vtx_by_idx, vtx_score_by_idx = {}, {}
        attached = {"nu": set(), "flashmatch": set()}
        for gv, (sc, s, pos, chi2, ii) in enumerate(vtx_rows):
            vtx_by_idx[gv] = np.asarray(pos, float)
            vtx_score_by_idx[gv] = float(sc)
            fr, gr, kp_path = stream_ev[s]
            d = prong_rows(gr)
            sel = np.nonzero(d["part_interaction"] == ii)[0]
            if sel.size and msp_pts is None:
                msp_pts = MspTruthPoints(msp_path)
            with h5py.File(kp_path, "r") as fkp:
                slice_coords = fkp["slice/coord_cm"][()]
                for i in sel:
                    kind = int(d["part_kind"][i])
                    key = "track" if kind == 0 else "shower"
                    p = {}
                    inst = int(d["part_inst_idx"][i])
                    if inst >= 0:
                        attached[s].add(inst)
                    p["Stream"] = STREAM_CODE[s]
                    pidx = (fkp[f"particle/{inst}/point_idx"][()]
                            if inst >= 0 else np.zeros(0, np.int64))
                    gtpidx = (fkp[f"particle/{inst}/gt_point_idx"][()]
                              if inst >= 0 else np.zeros(0, np.int64))
                    pts = slice_coords[pidx] if pidx.size else None
                    if pts is not None and contained:
                        for q in pts:
                            if not in_wcfv(*q):
                                contained = False
                                break
                    vloc = int(d["part_vtx"][i])
                    p["IsSecondary"] = int(
                        vloc >= 0 and d["vtx_depth"][vloc] > 0)
                    p["NHits"] = int(pidx.size)
                    p["Charge"] = float(d["part_charge"][i])
                    tot_hits += p["NHits"]
                    tot_charge += max(p["Charge"], 0.0) \
                        if np.isfinite(p["Charge"]) else 0.0
                    dirv = d["part_direction"][i]
                    p["CosTheta"] = float(dirv[2])
                    p["CosThetaY"] = float(-dirv[1])
                    st = d["part_start_cm"][i]
                    p["StartPosX"], p["StartPosY"], p["StartPosZ"] = map(
                        float, st)
                    p["StartDirX"], p["StartDirY"], p["StartDirZ"] = map(
                        float, dirv)
                    p["DistToVtx"] = float(np.linalg.norm(st - pos))
                    if kind == 0:
                        end = (d["polys"][i][-1] if len(d["polys"][i])
                               else np.full(3, -9.0))
                        p["EndPosX"], p["EndPosY"], p["EndPosZ"] = map(
                            float, end)
                    # LArFormer segmenter PID (its own classifier)
                    p["LArFormerPID"] = LARFORMER_PDG.get(
                        int(d["part_pred_class"][i]), 0)
                    if key == "shower":   # NOTE key, not kind (kind is int)
                        p["AttScore"] = (float(d["part_att_score"][i])
                                         if "part_att_score" in d else -9.0)
                        p["AttConfident"] = (
                            int(d["part_att_confident"][i])
                            if "part_att_confident" in d else 1)
                    lfs = (fkp[f"particle/{inst}/class_scores"][()]
                           if inst >= 0
                           and f"particle/{inst}/class_scores"
                           in fkp else None)
                    for j, nm in enumerate(("El", "Ph", "Mu", "Pi", "Pr")):
                        p["LArFormer" + nm + "Score"] = (
                            float(lfs[j]) if lfs is not None and j < len(lfs)
                            else -9.0)
                    p["Objectness"] = (float(1.0 - lfs[-1])
                                       if lfs is not None and len(lfs) >= 8
                                       else -1.0)
                    # LArPID block
                    p["Classified"] = int(d["larpid_classified"][i])
                    p["PID"] = int(d["larpid_pid"][i])
                    scs = d["larpid_scores"][i]
                    for j, nm in enumerate(("El", "Ph", "Mu", "Pi", "Pr")):
                        p[nm + "Score"] = float(scs[j])
                    p["Comp"] = float(d["larpid_completeness"][i])
                    p["Purity"] = float(d["larpid_purity"][i])
                    p["Process"] = int(d["larpid_process"][i])
                    prc = d["larpid_process_scores"][i]
                    p["PrimaryScore"] = float(prc[0])
                    p["FromNeutralScore"] = float(prc[1])
                    p["FromChargedScore"] = float(prc[2])
                    # reco energy (KE for tracks, E for showers)
                    e = float(d["part_energy"][i])
                    ke = (e - MASS.get(int(d["part_pred_class"][i]), 0.0)
                          if kind == 0 else e)
                    if (key == "shower" and not args.no_recalib_showers):
                        e2 = shower_energy(int(d["part_pred_class"][i]),
                                           p["Charge"], calib)
                        if e2 is not None:
                            ke = e2
                    p["RecoE"] = ke if np.isfinite(ke) else -1.0
                    if np.isfinite(ke):
                        reco_e += ke
                    # truth match (3D point sets; TID-based off merged_sp
                    # labels, 2026-08-31)
                    fill_truth_match(p, pts, pidx.size,
                                     int(d["part_gt_trackid"][i]),
                                     tid_lookup, msp_pts, qv)
                    p["VtxIdx"] = gv
                    prongs.append((key, p))
        # ---- slice-level scalars + VERTEX-LESS prongs (per stream) ----------
        # HitFrac/ChargeFrac denominators stay = attached prongs only, so the
        # existing per-shower BDT sees unchanged inputs; orphans use the same
        # denominator (-1 when there are no attached prongs).
        n_orph = 0
        for s, kp_path in kp_paths.items():
            with h5py.File(kp_path, "r") as fkp:
                sattr = _attr_str(fkp.attrs, "stream")
                chi2s = float(fkp.attrs.get("flash_chi2", np.nan))
                npart = int(fkp.attrs.get("n_particles", 0))
                targets = [s]
                if s == "nu" and "flashmatch" in sattr and "flashmatch" not in kp_paths:
                    targets.append("flashmatch")   # one slice serves both
                for tg in targets:
                    pre = "nu" if tg == "nu" else "fm"
                    ev[pre + "SliceFlashChi2"] = (chi2s if np.isfinite(chi2s)
                                                  else -1.0)
                    ev[pre + "SliceNParticles"] = npart
                if args.no_orphans or npart == 0:
                    continue
                slice_coords = None
                for inst in range(npart):
                    if inst in attached[s] or f"particle/{inst}" not in fkp:
                        continue
                    g = fkp[f"particle/{inst}"]
                    pidx = g["point_idx"][()]
                    if pidx.size < args.orphan_min_points:
                        continue
                    if slice_coords is None:
                        slice_coords = fkp["slice/coord_cm"][()]
                    if msp_pts is None:
                        msp_pts = MspTruthPoints(msp_path)
                    pts = slice_coords[pidx]
                    cls = int(g.attrs.get("cls", 5))
                    key = "shower" if cls in (0, 1) else "track"
                    kp_st = np.asarray(g["start_cm"][()], np.float64)
                    rows_p = msp_pts.rows_for(pts)
                    qp = msp_pts.dedup_comb(rows_p)
                    if key == "shower":
                        ckey = {0: "e", 1: "gamma"}.get(cls, "gamma")
                        slope = (calib or {}).get(ckey, 0.01553)
                        st, dirv, _ = shower_start_dir(
                            pts, qp, kp_st, slope, args.orphan_min_cluster_mev,
                            args.orphan_dbscan_eps)
                        st = np.asarray(st, np.float64)
                    else:
                        st = kp_st
                        dirv = _pca_dir(pts, st)
                    p = {"Stream": STREAM_CODE[s], "VtxIdx": -1,
                         "IsSecondary": -1, "NHits": int(pidx.size),
                         "DistToVtx": -9.0,
                         "CosTheta": float(dirv[2]), "CosThetaY": float(-dirv[1]),
                         "StartPosX": float(st[0]), "StartPosY": float(st[1]),
                         "StartPosZ": float(st[2]),
                         "StartDirX": float(dirv[0]), "StartDirY": float(dirv[1]),
                         "StartDirZ": float(dirv[2]),
                         "LArFormerPID": LARFORMER_PDG.get(cls, 0)}
                    if key == "track":
                        end = np.asarray(g["end_cm"][()], np.float64)
                        p["EndPosX"], p["EndPosY"], p["EndPosZ"] = map(float, end)
                    else:
                        p["AttScore"], p["AttConfident"] = -9.0, 0
                    lfs = (g["class_scores"][()] if "class_scores" in g
                           else None)
                    for j, nm in enumerate(("El", "Ph", "Mu", "Pi", "Pr")):
                        p["LArFormer" + nm + "Score"] = (
                            float(lfs[j]) if lfs is not None and j < len(lfs)
                            else -9.0)
                    p["Objectness"] = (float(1.0 - lfs[-1])
                                       if lfs is not None and len(lfs) >= 8
                                       else -1.0)
                    p["Classified"], p["PID"], p["Process"] = 0, 0, -1
                    for nm in ("El", "Ph", "Mu", "Pi", "Pr"):
                        p[nm + "Score"] = -9.0
                    p["Comp"] = p["Purity"] = -9.0
                    p["PrimaryScore"] = p["FromNeutralScore"] = -9.0
                    p["FromChargedScore"] = -9.0
                    Qp = fill_truth_match(p, pts, pidx.size,
                                          int(g.attrs.get("gt_trackid", -1)),
                                          tid_lookup, msp_pts, qv,
                                          dominant_fallback=True,
                                          rows_qp=(rows_p, qp))
                    p["Charge"] = float(Qp) if Qp is not None else -1.0
                    e2 = (shower_energy(cls, p["Charge"], calib)
                          if key == "shower" else None)
                    p["RecoE"] = float(e2) if e2 is not None else -1.0
                    prongs.append((key, p))
                    n_orph += 1
        if n_orph:
            stats["orphans"] += n_orph
            stats["orphan_events"] += 1
        for key, p in prongs:
            p["HitFrac"] = p["NHits"] / tot_hits if tot_hits else -1.0
            p["ChargeFrac"] = (p["Charge"] / tot_charge
                               if tot_charge > 0 and np.isfinite(p["Charge"])
                               else -1.0)
        score_showers(prongs, vtx_by_idx, vtx_score_by_idx)
        score_showers_novtx(prongs, ev)
        for key, p in prongs:
            pref = key
            for b, _ in schema.GROUPS[key][1]:
                ev[key][b].append(p.get(b[len(pref):], -9.0))
        if vtx_rows:
            ev["recoNuE"] = reco_e
            ev["vtxContainment"] = (0 if not ev["vtxIsFiducial"]
                                    else (2 if contained else 1))
        for s in stream_ev:
            stream_ev[s][0].close()
        events.append(ev)
        n_now += 1
        if n_now % 200 == 0:
            print(f"  [{n_now}/{len(msp)}]", flush=True)
        if len(events) >= FLUSH:
            tree.extend(schema.extend_payload(events))
            events.clear()

    # ---- write --------------------------------------------------------------
    if events:
        tree.extend(schema.extend_payload(events))
    pot = fout.mktree("potTree", {"totPOT": "float32",
                                  "totGoodPOT": "float32"})
    # NOTE shard_end, not a bare `end`: the prong loop reuses `end` for
    # track endpoints, which clobbered the range bound (array truth error).
    fn = sorted(k for k in truth.pot
                if args.start <= first_by_fileno.get(k, -1) < shard_end)
    pot.extend({"totPOT": np.asarray([truth.pot[k][0] for k in fn],
                                     np.float32),
                "totGoodPOT": np.asarray([truth.pot[k][1] for k in fn],
                                         np.float32)})
    fout.close()
    print(f">>> {len(events)} events ({stats['found']} with a vertex; "
          f"{stats['orphans']} vertex-less prongs in "
          f"{stats['orphan_events']} events), "
          f"{stats['noweight']} missing/inf weight, "
          f"{stats['notruth']} missing truth; {len(fn)} potTree entries "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
