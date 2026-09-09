"""
Step 0 verification for the single-photon study.

Opens the BNB nu overlay flat ntuple and confirms the assumptions the signal
selection relies on:
  - EventTree / potTree present, branch list as documented
  - trueSimPart* arrays carry photons (PDG 22)
  - trueSimPart particles look neutrino-induced (start positions cluster near
    the true nu vertex, NOT scattered detector-wide like cosmics would be)
  - fileid / run / subrun / event present for file isolation

Run inside the pointcept container:
  apptainer exec --bind /cluster:/cluster <pointcept_cuml.sif> \
      python3 verify_ntuple.py [--infile PATH] [-n NEVENTS]
"""
import argparse
import math
import ROOT as rt

DEFAULT_NTUPLE = ("/cluster/tufts/wongjiradlabnu/nutufts/data/ntuples/"
                  "dlgen2_reco_v2me05_gen2ntuple_v7_run3b_bnb_nu_overlay_nocrtremerge.root")

ap = argparse.ArgumentParser()
ap.add_argument("--infile", default=DEFAULT_NTUPLE)
ap.add_argument("-n", "--nevents", type=int, default=2000,
                help="number of EventTree entries to scan for the sanity checks")
args = ap.parse_args()

f = rt.TFile(args.infile)
et = f.Get("EventTree")
pt = f.Get("potTree")

print("=" * 70)
print("FILE:", args.infile)
print("EventTree entries:", et.GetEntries() if et else "MISSING")
print("potTree   entries:", pt.GetEntries() if pt else "MISSING")

# ---- POT sum ----
pot = 0.0
if pt:
    for i in range(pt.GetEntries()):
        pt.GetEntry(i)
        pot += pt.totGoodPOT
print("sum(totGoodPOT) = %.4e POT" % pot)

# ---- branch presence ----
needed = ["run", "subrun", "event", "fileid", "xsecWeight",
          "trueNuPDG", "trueNuCCNC", "trueVtxX", "trueVtxY", "trueVtxZ",
          "nTrueSimParts", "trueSimPartPDG", "trueSimPartProcess",
          "trueSimPartE", "trueSimPartX", "trueSimPartY", "trueSimPartZ",
          "trueSimPartEDepX", "trueSimPartEDepY", "trueSimPartEDepZ"]
present = set(b.GetName() for b in et.GetListOfBranches())
print("-" * 70)
for b in needed:
    print("  %-22s %s" % (b, "OK" if b in present else "*** MISSING ***"))

# ---- scan: photons + nu-origin sanity ----
n = min(args.nevents, et.GetEntries())
nwith_photon = 0
nphot_total = 0
dist_vals = []       # |simpart_start - nu_vtx| for photons, to test nu-origin
pdg_counts = {}
print("-" * 70)
print("scanning %d entries ..." % n)
for i in range(n):
    et.GetEntry(i)
    vx, vy, vz = et.trueVtxX, et.trueVtxY, et.trueVtxZ
    has_phot = False
    for j in range(et.nTrueSimParts):
        pdg = et.trueSimPartPDG[j]
        pdg_counts[pdg] = pdg_counts.get(pdg, 0) + 1
        if pdg == 22:
            has_phot = True
            nphot_total += 1
            dx = et.trueSimPartX[j] - vx
            dy = et.trueSimPartY[j] - vy
            dz = et.trueSimPartZ[j] - vz
            dist_vals.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    if has_phot:
        nwith_photon += 1

print("events with >=1 trueSimPart photon: %d / %d  (%.1f%%)"
      % (nwith_photon, n, 100.0 * nwith_photon / max(n, 1)))
print("total photons found:", nphot_total)
if dist_vals:
    dist_vals.sort()
    med = dist_vals[len(dist_vals) // 2]
    print("photon start dist to nu vtx (cm): min=%.1f  median=%.1f  max=%.1f"
          % (dist_vals[0], med, dist_vals[-1]))
    print("  -> if median is small/moderate (not detector-scale ~1000cm),")
    print("     photons are nu-induced, NOT cosmic. (cosmics would be scattered)")

# top PDG codes seen in trueSimPart
print("-" * 70)
print("top trueSimPart PDG codes (count):")
for pdg, c in sorted(pdg_counts.items(), key=lambda kv: -kv[1])[:12]:
    print("  %8d : %d" % (pdg, c))

f.Close()
print("=" * 70)
