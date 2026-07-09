"""Extract the gen2ntuple truth block from a merged_dlreco file into an H5
"truth sidecar", so the flat-ntuple exporter can run pure-h5py (no larlite).

Ports the MC-truth section of gen2ntuple/make_dlgen2_flat_ntuples.py
(GENIE neutrino/lepton variables, truePrimPart* from mctruth status-1
particles, trueSimPart* from mcreco MCTrack+MCShower, SCE-corrected
positions, WC-fiducial-volume containment) with ONE deliberate change:
no event is dropped — the old maker skipped events whose true vertex fell
outside the Wire-Cell fiducial volume; here every entry is written and the
`trueVtxInWCFV` flag carries that information instead (analyzers filter).

Environment: pointcept container + ubdl stack sourced (same as
lartpc/data_prep/uboone_official/convert_dlmerged_to_larformer_h5.py):
larlite, larutil, ublarcvapp, ROOT, h5py.

    python3 extract_truth_sidecar.py \
        --input-dlmerged merged_dlreco_<hash>.root \
        --out truth_<TAG>_fileno<N>.h5 \
        [--wcfv-lib lib_wirecell_fiducial_volume.so]

Output schema (one group per larlite entry):
  file attrs : src_dlmerged, totPOT, totGoodPOT, n_entries
  entry_{i}/ attrs: run, subrun, event,
      trueNuE (GeV), trueNuPDG, trueNuCCNC, trueNuMode, trueNuIntrxnType,
      trueLepE (GeV; -9 for NC), trueLepPDG (0 for NC),
      trueVtxX/Y/Z (SCE-corrected, cm), trueVtxInWCFV (0/1)
  entry_{i}/truePrimPart/{PDG, X, Y, Z, Px, Py, Pz, E, Contained}   (GeV)
  entry_{i}/trueSimPart/{PDG, TID, MID, Process, X, Y, Z,
      EDepX, EDepY, EDepZ, Px, Py, Pz, E,
      EndX, EndY, EndZ, EndPx, EndPy, EndPz, EndE, Contained}       (MeV)
Process codes: 0=primary, 1=Decay, 2=other. Positions SCE-corrected
(x - dx + 0.7, y + dy, z + dz), matching gen2ntuple conventions.
"""
import os
import ctypes
import argparse

import numpy as np
import h5py
import ROOT as rt
from larlite import larlite
from larlite import larutil
from ublarcvapp import ublarcvapp


def load_wcfv(libpath):
    class WCFiducial(ctypes.Structure):
        pass
    lib = ctypes.cdll.LoadLibrary(libpath)
    lib.WCFiducial_new.argtypes = ()
    lib.WCFiducial_new.restype = ctypes.POINTER(WCFiducial)
    lib.WCFiducial_insideFV.argtypes = (ctypes.POINTER(WCFiducial),
                                        ctypes.c_double, ctypes.c_double,
                                        ctypes.c_double)
    lib.WCFiducial_insideFV.restype = ctypes.c_bool
    obj = lib.WCFiducial_new()
    return lambda x, y, z: bool(lib.WCFiducial_insideFV(obj, x, y, z))


def sce_corrected(point, sce):
    """gen2ntuple's getSCECorrectedPos: (x - dx + 0.7, y + dy, z + dz)."""
    off = sce.GetPosOffsets(point.X(), point.Y(), point.Z())
    return (point.X() - off[0] + 0.7, point.Y() + off[1], point.Z() + off[2])


def sum_pot(path):
    """gen2ntuple event_weight_helper.SumPOT (bare ROOT)."""
    tot, good = 0.0, 0.0
    try:
        f = rt.TFile(path)
        t = f.Get("potsummary_generator_tree")
        for i in range(t.GetEntries()):
            t.GetEntry(i)
            tot += t.potsummary_generator_branch.totpot
            good += t.potsummary_generator_branch.totgoodpot
        f.Close()
    except Exception:
        return -1.0, -1.0
    return tot, good


PRIM_KEYS = ["PDG", "X", "Y", "Z", "Px", "Py", "Pz", "E", "Contained"]
SIM_KEYS = ["PDG", "TID", "MID", "Process", "X", "Y", "Z",
            "EDepX", "EDepY", "EDepZ", "Px", "Py", "Pz", "E",
            "EndX", "EndY", "EndZ", "EndPx", "EndPy", "EndPz", "EndE",
            "Contained"]
_INT_KEYS = {"PDG", "TID", "MID", "Process", "Contained"}


def _write_table(g, name, keys, rows):
    sub = g.create_group(name)
    # larlite's invalid-value sentinel (~1.7e308, e.g. DetProfile of a photon
    # that never converts) overflows float32 to inf — the SAME convention the
    # legacy ntuple uses (v0 file stores inf there), so keep it, quietly.
    with np.errstate(over="ignore"):
        for j, k in enumerate(keys):
            vals = [r[j] for r in rows]
            dt = np.int32 if k in _INT_KEYS else np.float32
            sub.create_dataset(k, data=np.asarray(vals, dt))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-dlmerged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wcfv-lib",
                    default=os.path.join(os.path.dirname(os.path.abspath(
                        __file__)), "lib_wirecell_fiducial_volume.so"))
    ap.add_argument("--max-events", type=int, default=-1)
    args = ap.parse_args()

    in_wcfv = load_wcfv(args.wcfv_lib)
    sce = larutil.SpaceChargeMicroBooNE()
    nu_vertexer = ublarcvapp.mctools.NeutrinoVertex()

    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(args.input_dlmerged)
    ioll.open()

    n = ioll.get_entries()
    if args.max_events > 0:
        n = min(n, args.max_events)
    tot_pot, tot_good = sum_pot(args.input_dlmerged)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as fout:
        fout.attrs["src_dlmerged"] = os.path.basename(args.input_dlmerged)
        fout.attrs["totPOT"] = float(tot_pot)
        fout.attrs["totGoodPOT"] = float(tot_good)
        fout.attrs["n_entries"] = n

        for i in range(n):
            ioll.go_to(i)
            g = fout.create_group(f"entry_{i}")
            g.attrs["run"] = int(ioll.run_id())
            g.attrs["subrun"] = int(ioll.subrun_id())
            g.attrs["event"] = int(ioll.event_id())

            mctruth = ioll.get_data(larlite.data.kMCTruth, "generator")
            nu_int = mctruth.at(0).GetNeutrino()
            lep = nu_int.Lepton()
            vtx = nu_vertexer.getPos3DwSCE(ioll, sce)
            g.attrs["trueVtxX"] = float(vtx[0])
            g.attrs["trueVtxY"] = float(vtx[1])
            g.attrs["trueVtxZ"] = float(vtx[2])
            g.attrs["trueVtxInWCFV"] = int(in_wcfv(vtx[0], vtx[1], vtx[2]))

            g.attrs["trueNuPDG"] = int(nu_int.Nu().PdgCode())
            g.attrs["trueNuCCNC"] = int(nu_int.CCNC())
            g.attrs["trueNuMode"] = int(nu_int.Mode())
            g.attrs["trueNuIntrxnType"] = int(nu_int.InteractionType())
            g.attrs["trueNuE"] = float(nu_int.Nu().Momentum().E())
            if nu_int.CCNC() == 0:
                g.attrs["trueLepPDG"] = int(lep.PdgCode())
                g.attrs["trueLepE"] = float(lep.Momentum().E())
            else:
                g.attrs["trueLepPDG"] = 0
                g.attrs["trueLepE"] = -9.0

            # --- GENIE final-state primaries (status code 1) ------------------
            prim = []
            for p in mctruth.at(0).GetParticles():
                if p.StatusCode() != 1:
                    continue
                s = sce_corrected(p.Position(0), sce)
                e = sce_corrected(p.Position(p.Trajectory().size() - 1), sce)
                prim.append((p.PdgCode(), s[0], s[1], s[2],
                             p.Momentum(0).Px(), p.Momentum(0).Py(),
                             p.Momentum(0).Pz(), p.Momentum(0).E(),
                             int(in_wcfv(*e))))
            _write_table(g, "truePrimPart", PRIM_KEYS, prim)

            # --- Geant4/detsim particles (mcreco MCTrack + MCShower) ----------
            sim = []
            for coll in (ioll.get_data(larlite.data.kMCTrack, "mcreco"),
                         ioll.get_data(larlite.data.kMCShower, "mcreco")):
                for p in coll:
                    proc = (0 if p.Process() == "primary"
                            else 1 if p.Process() == "Decay" else 2)
                    s = sce_corrected(p.Start(), sce)
                    e = sce_corrected(p.End(), sce)
                    if p.PdgCode() == 22:          # photon: first-conversion pt
                        dp = p.DetProfile()
                        edep = (dp.X(), dp.Y(), dp.Z())
                    else:
                        edep = s
                    sim.append((p.PdgCode(), p.TrackID(), p.MotherTrackID(),
                                proc, s[0], s[1], s[2],
                                edep[0], edep[1], edep[2],
                                p.Start().Px(), p.Start().Py(), p.Start().Pz(),
                                p.Start().E(),
                                e[0], e[1], e[2],
                                p.End().Px(), p.End().Py(), p.End().Pz(),
                                p.End().E(), int(in_wcfv(*e))))
            _write_table(g, "trueSimPart", SIM_KEYS, sim)
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  [{i + 1}/{n}]", flush=True)

    ioll.close()
    print(f">>> {n} entries, POT={tot_pot:.4g} -> {args.out}")


if __name__ == "__main__":
    main()
