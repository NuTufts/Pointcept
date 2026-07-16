"""Emit a (run, subrun, event) list of MC events showing the saturated-PMT
signature, for cross-checking the run3 optical simulation stream against the
official MicroBooNE production files.

Each row carries everything needed to look the event up elsewhere:
  - run/subrun/event  (the lookup key in any other production copy)
  - the offending PMT as BOTH opdet and opchannel. Use OPCHANNEL against raw
    optical data -- opdet is the Geant4/GDML sort order. NOTE the two differ on
    all 32 entries, and viz/pmtpos.py's table is WRONG; the map here is
    larlite larutil::Geometry::OpDetFromOpChannel, which is what the stage-A
    converter used.
  - observed PE (what the opflash says: ~0) vs predicted PE (what the charge
    says it should be: often >1000) at that tube
  - the ophit-level evidence at the beam time: amplitude and reconstructed PE.
    A healthy tube runs PE/amplitude ~ 0.11-0.21; these run <0.02, and the
    neighbouring tube often exceeds the 12-bit ADC ceiling of 4096.
  - the source dlmerged file + entry it came from.

Context: the hole rate rises to 34.7% at >8000 observed PE in run3b overlay MC
but stays at ~0-1% at ALL brightnesses in both run1 beam data and run3 EXT data,
so the effect looks like a simulation artifact rather than a detector one.

    python3 make_saturation_event_list.py --cache <rse_*.npz> \
        --inputlist <scale1500_bnb_nu_overlay.txt> --out saturation_events.csv
"""
import argparse
import os
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.saturation import find_saturated          # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.ophit_saturation_probe import (
    OPCH2OPDET, OPHIT_PROD, BEAM_LO, BEAM_HI)                    # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "physics", "pi0mass_peak"))
from flash_correction import neyman_masked                       # noqa: E402

OD2CH = {v: k for k, v in OPCH2OPDET.items()}


def collect(cache, dead, min_pe, want_n):
    """Events with >=1 hole candidate, brightest-prediction first."""
    z = np.load(cache, allow_pickle=True)
    rows = []
    for key, p in zip(z["keys"], z["paths"]):
        p = str(p)
        try:
            with h5py.File(p, "r") as f:
                if "flash" not in f or "observed_pe" not in f["flash"]:
                    continue
                obs = f["flash/observed_pe"][()]
                if obs.sum() < min_pe:
                    continue
                hole = find_saturated(obs, dead=dead, max_masked=None)
                if not hole:
                    continue
                labs = [l.decode() if isinstance(l, bytes) else str(l)
                        for l in f["slices/label"][()]]
                if "nu" not in labs:
                    continue
                pred = f["slices/pred_pe"][()][labs.index("nu")]
                s = f.attrs.get("src_file")
                s = s.decode() if isinstance(s, bytes) else str(s)
        except Exception:
            continue
        m = re.search(r"fileno(\d+)_entry(\d+)", s)
        if not m:
            continue
        rows.append(dict(
            run=int(key[0]), subrun=int(key[1]), event=int(key[2]),
            fileno=int(m.group(1)), entry=int(m.group(2)),
            hole=hole, obs=obs, pred=pred,
            worst=max(float(pred[t]) for t in hole)))
    rows.sort(key=lambda r: -r["worst"])
    return rows[:want_n]


def add_ophit(rows, lines, old, new):
    """Attach beam-window ophit amplitude/PE at each flagged tube."""
    import uproot
    by = {}
    for r in rows:
        by.setdefault(r["fileno"], []).append(r)
    for fno, rs in sorted(by.items()):
        if fno < 1 or fno > len(lines):
            continue
        rp = lines[fno - 1].replace(old, new)
        if not os.path.exists(rp):
            continue
        for r in rs:
            r["dlmerged"] = rp
        try:
            br = ("ophit_%s_branch/vector<larlite::ophit>/"
                  "vector<larlite::ophit>." % OPHIT_PROD)
            t = uproot.open(rp)["ophit_%s_tree" % OPHIT_PROD]
            ents = sorted(r["entry"] for r in rs)
            lo, hi = ents[0], ents[-1] + 1
            ch = t[br + "fOpChannel"].array(entry_start=lo, entry_stop=hi)
            pk = t[br + "fPeakTime"].array(entry_start=lo, entry_stop=hi)
            am = t[br + "fAmplitude"].array(entry_start=lo, entry_stop=hi)
            pe = t[br + "fPE"].array(entry_start=lo, entry_stop=hi)
        except Exception as ex:
            print("  [warn] fileno %d ophit: %s" % (fno, ex))
            continue
        for r in rs:
            k = r["entry"] - lo
            c = np.asarray(ch[k]); p = np.asarray(pk[k])
            a = np.asarray(am[k]); q = np.asarray(pe[k])
            bw = (p > BEAM_LO) & (p < BEAM_HI)
            d = {}
            for od in r["hole"]:
                cc = OD2CH[od]
                m = bw & (c == cc)
                d[od] = ((float(a[m].max()), float(q[m].sum()))
                         if m.any() else (0.0, 0.0))
            r["ophit"] = d
            r["maxamp_ev"] = float(a[bw].max()) if bw.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--inputlist", required=True)
    ap.add_argument("--out", default="saturation_events.csv")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-pe", type=float, default=3000.0,
                    help="only bright flashes -- saturation needs one")
    ap.add_argument("--dead", default="15")
    ap.add_argument("--no-ophit", action="store_true")
    ap.add_argument("--path-sub", default="/data/mcc9/:/data/mcc9_scratch/")
    args = ap.parse_args()
    dead = tuple(int(x) for x in args.dead.split(",") if x != "")
    old, new = args.path_sub.split(":")
    lines = [l.strip() for l in open(args.inputlist) if l.strip()]

    print(">>> scanning for hole events with total obs PE > %.0f ..." % args.min_pe)
    rows = collect(args.cache, dead, args.min_pe, args.n)
    print(">>> %d events" % len(rows))
    if not args.no_ophit:
        print(">>> attaching ophit evidence ...")
        add_ophit(rows, lines, old, new)

    with open(args.out, "w") as f:
        f.write("run,subrun,event,opdet,opchannel,obs_pe,pred_pe,total_obs_pe,"
                "ophit_amp,ophit_pe,ophit_pe_over_amp,event_max_amp,"
                "fileno,entry,dlmerged\n")
        for r in rows:
            for od in r["hole"]:
                amp, ope = r.get("ophit", {}).get(od, (float("nan"),) * 2)
                f.write("%d,%d,%d,%d,%d,%.2f,%.1f,%.0f,%.1f,%.2f,%.4f,%.0f,"
                        "%d,%d,%s\n"
                        % (r["run"], r["subrun"], r["event"], od, OD2CH[od],
                           float(r["obs"][od]), float(r["pred"][od]),
                           r["obs"].sum(), amp, ope,
                           (ope / amp) if amp else float("nan"),
                           r.get("maxamp_ev", float("nan")),
                           r["fileno"], r["entry"], r.get("dlmerged", "")))
    print(">>> wrote %s" % args.out)

    print("\n%-6s %-7s %-7s | %-5s %-5s | %8s %9s %9s | %8s %7s %6s"
          % ("run", "subrun", "event", "opdet", "opch", "obs_PE", "pred_PE",
             "totObsPE", "oph_amp", "oph_PE", "PE/amp"))
    for r in rows[:25]:
        for od in r["hole"]:
            amp, ope = r.get("ophit", {}).get(od, (float("nan"),) * 2)
            print("%-6d %-7d %-7d | %-5d %-5d | %8.2f %9.1f %9.0f | %8.1f %7.2f %6.4f"
                  % (r["run"], r["subrun"], r["event"], od, OD2CH[od],
                     float(r["obs"][od]), float(r["pred"][od]), r["obs"].sum(),
                     amp, ope, (ope / amp) if amp else float("nan")))


if __name__ == "__main__":
    main()
