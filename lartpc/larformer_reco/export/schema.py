"""Declarative gen2ntuple EventTree schema for the LArFormer exporter.

ONE table drives everything: branch declaration (uproot mktree types with
shared leaflist counters), per-event defaults, and the extend() payload —
replacing the legacy maker's four hand-maintained copies (allocate / Branch /
default / fill).

Deviations from the legacy v7 tree (all user-approved):
- OMITTED: the KPSReco-specific `kp*` keypoint block and `eventPC*` PCA block.
- NEW: `trueVtxInWCFV` (the old maker DROPPED out-of-WC-FV events; we keep
  them flagged), the multi-interaction vertex table `recoVtx*` with
  `recoVtxStream` (0=nu, 1=flashmatch) + `primaryVtxStream`, and per-prong
  `trackVtxIdx`/`showerVtxIdx` into that table.
- ADDED (post-v7 gen2ntuple HEAD parity): `trueSimPartEndPx/Py/Pz/EndE`.
"""
import numpy as np
import awkward as ak

I, F = "int32", "float32"

# ---- per-event scalars: (branch, type, default) ---------------------------------
SCALARS = [
    ("fileid", I, -1), ("run", I, -1), ("subrun", I, -1), ("event", I, -1),
    ("xsecWeight", F, -1.0),
    ("trueNuE", F, -9.0), ("trueNuPDG", I, 0), ("trueNuCCNC", I, -1),
    ("trueNuMode", I, -1), ("trueNuIntrxnType", I, -1),
    ("trueVtxX", F, -9999.0), ("trueVtxY", F, -9999.0),
    ("trueVtxZ", F, -9999.0), ("trueVtxInWCFV", I, -1),
    ("trueLepE", F, -9.0), ("trueLepPDG", I, 0),
    ("recoNuE", F, -9.0), ("foundVertex", I, 0),
    ("vtxX", F, -9999.0), ("vtxY", F, -9999.0), ("vtxZ", F, -9999.0),
    ("vtxIsFiducial", I, -1), ("vtxContainment", I, -1),
    ("vtxDistToTrue", F, -99.0), ("vtxScore", F, -1.0),
    ("vtxFracHitsOnCosmic", F, -1.0),
    ("primaryVtxStream", I, -1),
]

# ---- jagged groups: group -> (counter branch, [(branch, type), ...]) -------------
_TRUE_MATCH = [("TruePID", I), ("TrueTID", I), ("TrueE", F),
               ("TruePurity", F), ("TrueComp", F),
               ("TrueElPurity", F), ("TruePhPurity", F), ("TrueMuPurity", F),
               ("TruePiPurity", F), ("TruePrPurity", F)]
_LARFORMER = [("LArFormerPID", I),
              ("LArFormerElScore", F), ("LArFormerPhScore", F),
              ("LArFormerMuScore", F), ("LArFormerPiScore", F),
              ("LArFormerPrScore", F)]
_LARPID = [("Classified", I), ("PID", I),
           ("ElScore", F), ("PhScore", F), ("MuScore", F), ("PiScore", F),
           ("PrScore", F), ("Comp", F), ("Purity", F), ("Process", I),
           ("PrimaryScore", F), ("FromNeutralScore", F),
           ("FromChargedScore", F)]
_PRONG_COMMON = [("IsSecondary", I), ("NHits", I), ("HitFrac", F),
                 ("Charge", F), ("ChargeFrac", F),
                 ("CosTheta", F), ("CosThetaY", F), ("DistToVtx", F),
                 ("StartPosX", F), ("StartPosY", F), ("StartPosZ", F),
                 ("StartDirX", F), ("StartDirY", F), ("StartDirZ", F)]

GROUPS = {
    "recoVtx": ("nRecoVtx", [
        ("recoVtxX", F), ("recoVtxY", F), ("recoVtxZ", F),
        ("recoVtxScore", F), ("recoVtxStream", I), ("recoVtxFlashChi2", F)]),
    "truePrimPart": ("nTruePrimParts", [
        ("truePrimPartPDG", I),
        ("truePrimPartX", F), ("truePrimPartY", F), ("truePrimPartZ", F),
        ("truePrimPartPx", F), ("truePrimPartPy", F), ("truePrimPartPz", F),
        ("truePrimPartE", F), ("truePrimPartContained", I)]),
    "trueSimPart": ("nTrueSimParts", [
        ("trueSimPartPDG", I), ("trueSimPartTID", I), ("trueSimPartMID", I),
        ("trueSimPartProcess", I),
        ("trueSimPartX", F), ("trueSimPartY", F), ("trueSimPartZ", F),
        ("trueSimPartEDepX", F), ("trueSimPartEDepY", F),
        ("trueSimPartEDepZ", F),
        ("trueSimPartPx", F), ("trueSimPartPy", F), ("trueSimPartPz", F),
        ("trueSimPartE", F),
        ("trueSimPartEndX", F), ("trueSimPartEndY", F), ("trueSimPartEndZ", F),
        ("trueSimPartEndPx", F), ("trueSimPartEndPy", F),
        ("trueSimPartEndPz", F), ("trueSimPartEndE", F),
        ("trueSimPartContained", I)]),
    "track": ("nTracks",
              [("track" + n, t) for n, t in _PRONG_COMMON]
              + [("trackEndPosX", F), ("trackEndPosY", F), ("trackEndPosZ", F)]
              + [("track" + n, t) for n, t in _LARFORMER]
              + [("track" + n, t) for n, t in _LARPID]
              + [("trackRecoE", F)]
              + [("track" + n, t) for n, t in _TRUE_MATCH]
              + [("trackVtxIdx", I)]),
    "shower": ("nShowers",
               [("shower" + n, t) for n, t in _PRONG_COMMON]
               + [("shower" + n, t) for n, t in _LARFORMER]
               + [("shower" + n, t) for n, t in _LARPID]
               + [("showerRecoE", F)]
               + [("shower" + n, t) for n, t in _TRUE_MATCH]
               + [("showerVtxIdx", I)]),
}


def new_event():
    """Fresh per-event record: scalar defaults + empty group row-lists."""
    ev = {n: d for n, _, d in SCALARS}
    for gname, (_, branches) in GROUPS.items():
        ev[gname] = {b: [] for b, _ in branches}
    return ev


def mktree(fout, name="EventTree"):
    """Declare the tree with shared-counter leaflist branches (verified: uproot
    record types + field_name/counter_name -> `trackPID[nTracks]/I` etc.)."""
    branch_types = {n: t for n, t, _ in SCALARS}
    counter = {}
    for gname, (cname, branches) in GROUPS.items():
        rec = ", ".join(f"{b}: {t}" for b, t in branches)
        branch_types[gname] = f"var * {{{rec}}}"
        counter[gname] = cname
    return fout.mktree(name, branch_types,
                       field_name=lambda outer, inner: inner,
                       counter_name=lambda n: counter[n])


def extend_payload(events):
    """events: list of new_event() dicts -> dict for WritableTree.extend."""
    out = {}
    for n, t, _ in SCALARS:
        out[n] = np.asarray([ev[n] for ev in events],
                            np.int32 if t == I else np.float32)
    for gname, (_, branches) in GROUPS.items():
        out[gname] = ak.zip({
            b: ak.values_astype(
                ak.Array([ev[gname][b] for ev in events]),
                np.int32 if t == I else np.float32)
            for b, t in branches})
    return out
