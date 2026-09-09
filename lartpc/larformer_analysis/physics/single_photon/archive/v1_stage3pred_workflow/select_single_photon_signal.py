"""
Step 1 of the single-photon study: ntuple-level signal selection.

Signal definition (BNB nu overlay flat ntuple, MC):
  (1) true nu vertex inside the TPC box
        0 < trueVtxX < 255, -116.5 < trueVtxY < 116.5, 0 < trueVtxZ < 1036  (cm, SCE-corrected)
  (2) at least one neutrino-origin photon that is plausibly detectable:
        - trueSimPart with PDG == 22, AND
        - trueSimPartE > 20 MeV (initial photon energy; a photon below this can
          never make a >=20 MeV ionization cluster, so it can't contribute), AND
        - first-deposit point (trueSimPartEDep{X,Y,Z}) inside the TPC box.
      In an overlay sample every trueSimPart (mcreco MCTrack/MCShower) is
      neutrino-induced, so any PDG-22 entry is nu-origin.

This 20 MeV initial-energy cut is a necessary-but-not-sufficient proxy for the
real "single ionization cluster >=20 MeV" detectability cut, which is applied
downstream on the official sim files (it needs per-cluster ionization, absent
from the ntuple). For context the summary also reports the LOOSE count (>=1
photon of any energy, EDep anywhere).

NOTE: the ntuple is already pre-filtered to the Wire-Cell fiducial volume
(>3 cm from the SCE-corrected detector edge), which is a SUBSET of the TPC box.
So this count is a lower bound for the full TPC; the box cut here is mostly a
sanity/explicitness cut. See docs/reference/Gen2_Flat_Ntuple_Spec.md.

Outputs (into --outdir):
  signal_events.csv   one row per signal event (run,subrun,event,fileid,...)
  signal_fileids.txt  unique fileids that contain signal, with signal-event counts
  summary printed to stdout (raw + POT-scaled counts)

Run inside the pointcept container (after sourcing setenv_pointcept_container.sh):
  python3 select_single_photon_signal.py --outdir workdir [-n NEVENTS]
"""
import argparse
import os
import ROOT as rt

DEFAULT_NTUPLE = ("/cluster/tufts/wongjiradlabnu/nutufts/data/ntuples/"
                  "dlgen2_reco_v2me05_gen2ntuple_v7_run3b_bnb_nu_overlay_nocrtremerge.root")

# TPC active volume (cm)
TPC_X = (0.0, 255.0)
TPC_Y = (-116.5, 116.5)
TPC_Z = (0.0, 1036.0)

PHOTON_PDG = 22


def in_tpc(x, y, z):
    return (TPC_X[0] < x < TPC_X[1] and
            TPC_Y[0] < y < TPC_Y[1] and
            TPC_Z[0] < z < TPC_Z[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=DEFAULT_NTUPLE)
    ap.add_argument("--outdir", default="workdir")
    ap.add_argument("-n", "--nevents", type=int, default=-1,
                    help="limit number of EventTree entries (-1 = all)")
    ap.add_argument("--target-pot", type=float, default=6.67e20,
                    help="POT to scale the weighted count to")
    ap.add_argument("--photon-emin", type=float, default=20.0,
                    help="min initial photon energy in MeV for the signal cut")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    f = rt.TFile(args.infile)
    et = f.Get("EventTree")
    pt = f.Get("potTree")

    # ---- total POT in the ntuple ----
    ntuple_pot = 0.0
    for i in range(pt.GetEntries()):
        pt.GetEntry(i)
        ntuple_pot += pt.totGoodPOT
    pot_scale = args.target_pot / ntuple_pot if ntuple_pot > 0 else 0.0

    nentries = et.GetEntries()
    if args.nevents >= 0:
        nentries = min(nentries, args.nevents)

    # counters
    n_total = 0
    n_fid = 0                # passes vertex-in-TPC
    n_loose = 0              # vertex + >=1 photon of any energy (context only)
    n_signal = 0            # vertex + >=1 photon E>emin & EDep in TPC
    w_signal = 0.0           # xsecWeight sum of signal
    n_signal_cc = 0
    n_signal_nc = 0
    fileid_counts = {}       # fileid -> n signal events

    csv_path = os.path.join(args.outdir, "signal_events.csv")
    csv = open(csv_path, "w")
    csv.write("run,subrun,event,fileid,trueNuPDG,trueNuCCNC,"
              "trueVtxX,trueVtxY,trueVtxZ,nPhotonsLoose,nPhotonsSig,"
              "maxPhotonE,xsecWeight\n")

    for i in range(nentries):
        et.GetEntry(i)
        n_total += 1

        if not in_tpc(et.trueVtxX, et.trueVtxY, et.trueVtxZ):
            continue
        n_fid += 1

        nphot_loose = 0      # any photon
        nphot_sig = 0        # photon with E>emin and EDep in TPC
        max_phot_e = 0.0
        for j in range(et.nTrueSimParts):
            if et.trueSimPartPDG[j] != PHOTON_PDG:
                continue
            nphot_loose += 1
            e = et.trueSimPartE[j]
            if e > max_phot_e:
                max_phot_e = e
            if e > args.photon_emin and in_tpc(et.trueSimPartEDepX[j],
                                               et.trueSimPartEDepY[j],
                                               et.trueSimPartEDepZ[j]):
                nphot_sig += 1

        if nphot_loose > 0:
            n_loose += 1
        if nphot_sig == 0:
            continue

        # ---- signal ----
        n_signal += 1
        w_signal += et.xsecWeight
        if et.trueNuCCNC == 0:
            n_signal_cc += 1
        else:
            n_signal_nc += 1
        fid = et.fileid
        fileid_counts[fid] = fileid_counts.get(fid, 0) + 1

        csv.write("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%d,%.3f,%.6g\n" % (
            et.run, et.subrun, et.event, et.fileid,
            et.trueNuPDG, et.trueNuCCNC,
            et.trueVtxX, et.trueVtxY, et.trueVtxZ,
            nphot_loose, nphot_sig, max_phot_e, et.xsecWeight))

    csv.close()

    # ---- signal fileid list ----
    fid_path = os.path.join(args.outdir, "signal_fileids.txt")
    with open(fid_path, "w") as ff:
        ff.write("# fileid  n_signal_events\n")
        for fid in sorted(fileid_counts):
            ff.write("%d %d\n" % (fid, fileid_counts[fid]))

    # ---- summary ----
    print("=" * 70)
    print("SINGLE-PHOTON NTUPLE SELECTION")
    print("infile        :", args.infile)
    print("entries scanned:", n_total, "/", et.GetEntries())
    print("ntuple POT    : %.4e   (scaling to %.3e POT, x%.4f)"
          % (ntuple_pot, args.target_pot, pot_scale))
    print("photon cut    : E > %.1f MeV  AND  first-EDep inside TPC" % args.photon_emin)
    print("-" * 70)
    print("vertex-in-TPC              : %d  (%.1f%% of scanned)"
          % (n_fid, 100.0 * n_fid / max(n_total, 1)))
    print("LOOSE (vtx + any photon)   : %d  (%.1f%% of in-TPC)  [context only]"
          % (n_loose, 100.0 * n_loose / max(n_fid, 1)))
    print("SIGNAL (vtx + phot cut)    : %d  (%.1f%% of scanned, %.1f%% of in-TPC)"
          % (n_signal, 100.0 * n_signal / max(n_total, 1),
             100.0 * n_signal / max(n_fid, 1)))
    print("   CC / NC                 : %d / %d" % (n_signal_cc, n_signal_nc))
    print("-" * 70)
    print("RAW signal events      : %d" % n_signal)
    print("POT-scaled signal      : %.1f  events @ %.3e POT"
          % (w_signal * pot_scale, args.target_pot))
    print("   (sum xsecWeight     : %.4f)" % w_signal)
    print("-" * 70)
    print("unique signal files    : %d" % len(fileid_counts))
    print("wrote: %s" % csv_path)
    print("wrote: %s" % fid_path)
    print("=" * 70)

    f.Close()


if __name__ == "__main__":
    main()
