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
- prong True{Purity,Comp} are 3D-spacepoint based (pred vs GT point sets from
  the keypoint2 instance), TrueXxPurity = species fractions of the predicted
  points' true pids; not wire-pixel-intensity based.
- trackRecoE = part_energy - m(pred class) (our range/calo pipeline), not the
  LArPID-hypothesis switch; showerRecoE = part_energy.
- HitFrac/ChargeFrac denominators = sums over ALL exported prongs of the
  event (all interactions, both streams).
- vtxFracHitsOnCosmic = -1 (no thrumu in this chain).
- MC events are NOT dropped: out-of-WC-FV true vertices are flagged
  (trueVtxInWCFV), missing/inf xsecWeight -> -1 (counted in the log).

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

MASS = {0: 0.511, 1: 0.0, 2: 105.6584, 3: 139.5704, 4: 938.2721, 5: 0.0}
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
        ev["trueSimPart"][b[0]] = list(sp[b[0][len("trueSimPart"):]][()])
    return {int(t): (int(p), float(e))
            for t, p, e in zip(sp["TID"][()], sp["PDG"][()], sp["E"][()])}


class MspTruthPoints:
    """Per-point true pid lookup for slice coords (species purities)."""

    def __init__(self, msp_path):
        with h5py.File(msp_path, "r") as f:
            td = f["entry_0/triplet_data"]
            pos = td["pos"][()].astype(np.float32)
            self.pid = td["pid"][()].astype(np.int64)
            self._row = {pos[i].tobytes(): i for i in range(len(pos))}

    def pids_for(self, coords):
        c = np.asarray(coords, np.float32)
        rows = [self._row.get(c[i].tobytes(), -1) for i in range(len(c))]
        rows = np.asarray(rows, np.int64)
        out = np.full(len(c), 0, np.int64)
        ok = rows >= 0
        out[ok] = self.pid[rows[ok]]
        return out


def prong_rows(gr):
    """nu_reco event group -> dict of part_* arrays."""
    keys = ["part_interaction", "part_kind", "part_pred_class", "part_energy",
            "part_charge", "part_gt_trackid", "part_vtx", "part_inst_idx",
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
    ap.add_argument("--wcfv-lib",
                    default=os.path.join(os.path.dirname(os.path.abspath(
                        __file__)), "lib_wirecell_fiducial_volume.so"))
    args = ap.parse_args()

    msp = read_list(args.merged_sp_list)
    end = len(msp) if args.n < 0 else min(args.start + args.n, len(msp))
    msp = msp[args.start:end]
    print(f">>> {len(msp)} events [{args.start}:{end})", flush=True)

    in_wcfv = load_wcfv(args.wcfv_lib)
    truth = TruthIndex(args.truth_dir)
    weights = pickle.load(open(args.weights_pkl, "rb"))
    streams = {}
    for s, klist, rdir in (("nu", args.kp2_nu_list, args.nu_reco_nu_dir),
                           ("flashmatch", args.kp2_fm_list,
                            args.nu_reco_fm_dir)):
        streams[s] = {"kp": build_kp_map(read_list(klist)),
                      "reco": build_reco_map(rdir)}
        print(f">>> stream {s}: {len(streams[s]['kp'])} kp2, "
              f"{len(streams[s]['reco'])} reco events", flush=True)

    events, n_now, n_nownr = [], 0, 0
    stats = dict(noweight=0, notruth=0, found=0)
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
        for s in ("nu", "flashmatch"):
            hit = streams[s]["kp"].get(base)
            if hit is None:
                continue
            gidx, kp_path = hit
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
        for gv, (sc, s, pos, chi2, ii) in enumerate(vtx_rows):
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
                    p["RecoE"] = ke if np.isfinite(ke) else -1.0
                    if np.isfinite(ke):
                        reco_e += ke
                    # truth match (3D point sets)
                    tid = int(d["part_gt_trackid"][i])
                    p["TrueTID"] = tid
                    tp, te = tid_lookup.get(tid, (0, -1.0))
                    p["TruePID"], p["TrueE"] = tp, te
                    inter = (np.intersect1d(pidx, gtpidx).size
                             if pidx.size and gtpidx.size else 0)
                    p["TruePurity"] = (inter / pidx.size if pidx.size else -1.)
                    p["TrueComp"] = (inter / gtpidx.size if gtpidx.size
                                     else -1.)
                    if pts is not None and pidx.size:
                        pids = msp_pts.pids_for(pts)
                        for nm, pdgs in SPECIES_PIDS.items():
                            p["True" + nm + "Purity"] = float(
                                np.isin(pids, pdgs).mean())
                    else:
                        for nm in SPECIES_PIDS:
                            p["True" + nm + "Purity"] = -1.0
                    p["VtxIdx"] = gv
                    prongs.append((key, p))
        for key, p in prongs:
            p["HitFrac"] = p["NHits"] / tot_hits if tot_hits else -1.0
            p["ChargeFrac"] = (p["Charge"] / tot_charge
                               if tot_charge > 0 and np.isfinite(p["Charge"])
                               else -1.0)
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

    # ---- write --------------------------------------------------------------
    fout = uproot.recreate(args.out)
    tree = schema.mktree(fout)
    tree.extend(schema.extend_payload(events))
    pot = fout.mktree("potTree", {"totPOT": "float32",
                                  "totGoodPOT": "float32"})
    fn = sorted(truth.pot)
    pot.extend({"totPOT": np.asarray([truth.pot[k][0] for k in fn],
                                     np.float32),
                "totGoodPOT": np.asarray([truth.pot[k][1] for k in fn],
                                         np.float32)})
    fout.close()
    print(f">>> {len(events)} events ({stats['found']} with a vertex), "
          f"{stats['noweight']} missing/inf weight, "
          f"{stats['notruth']} missing truth; {len(fn)} potTree entries "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
