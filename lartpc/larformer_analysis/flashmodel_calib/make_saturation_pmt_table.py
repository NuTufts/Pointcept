"""Per-PMT table for MC events that contain a GENUINE saturated tube, for
comparing the run3 optical sim against official MicroBooNE production files.

Selects events where at least one PMT shows the true saturation signature --
beam-window ophit amplitude > --amp-min with PE/amplitude < --pe-amp-max (a
healthy tube runs ~0.11-0.21) -- and emits ALL 32 PMTs for each, so the whole
light pattern can be compared, not just the offending tube.

Deliberately EXCLUDES the "dark" events (tube reads ~0 with NO pulse at all
despite a large prediction): those are a separate, unexplained population and
mixing them in would muddy the comparison. Use --want dark to get those instead.

The true (generator) neutrino vertex is included so the event can be confirmed
to be the same neutrino in the other file. NOTE this is the raw generator vertex
from merged_sp mc_particle_tree/nu_vertices, NOT the SCE-shifted one the cascade
stores as gt_nu_vertex_cm -- the two differ by a few cm.

Long format: one row per (event, opdet). Use OPCHANNEL against raw optical data;
opdet is the Geant4/GDML sort order and the two differ on all 32 entries.

    python3 make_saturation_pmt_table.py --cache <rse_*.npz> \
        --inputlist <scale1500_...txt> --merged-sp-list <merged_sp_*_tree.txt> \
        --out saturation_pmt_table.csv
"""
import argparse
import os
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.saturation import find_saturated              # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.ophit_saturation_probe import (
    OPCH2OPDET, OPHIT_PROD, BEAM_LO, BEAM_HI)                        # noqa: E402

OD2CH = {v: k for k, v in OPCH2OPDET.items()}
N_PMT = 32


def collect(cache, dead, min_pe, max_events):
    """Bright events with >=1 hole candidate + their nu-slice prediction."""
    z = np.load(cache, allow_pickle=True)
    out = []
    for key, p in zip(z["keys"], z["paths"]):
        if len(out) >= max_events:
            break
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
                src = f.attrs.get("src_file")
                src = src.decode() if isinstance(src, bytes) else str(src)
                t0 = np.nan
                if "flash/all/time_us" in f:
                    tid = f["flash/all/producer_id"][()]
                    tpe = f["flash/all/total_pe"][()]
                    tus = f["flash/all/time_us"][()]
                    b = np.nonzero(tid == 0)[0]
                    if b.size:
                        t0 = float(tus[b[int(np.argmax(tpe[b]))]])
        except Exception:
            continue
        m = re.search(r"fileno(\d+)_entry(\d+)", src)
        if not m:
            continue
        out.append(dict(run=int(key[0]), subrun=int(key[1]), event=int(key[2]),
                        fileno=int(m.group(1)), entry=int(m.group(2)),
                        src=src, obs=obs, pred=pred, hole=hole, t0=t0))
    return out


def attach_ophit(evs, lines, old, new):
    """Beam-window sum PE + max amplitude for ALL 32 opdets, per event."""
    import uproot
    by = {}
    for e in evs:
        by.setdefault(e["fileno"], []).append(e)
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
            sp = np.zeros(N_PMT); mx = np.zeros(N_PMT); nh = np.zeros(N_PMT, int)
            for od in range(N_PMT):
                m = bw & (c == OD2CH[od])
                if m.any():
                    sp[od] = float(q[m].sum()); mx[od] = float(a[m].max())
                    nh[od] = int(m.sum())
            r["oph_pe"] = sp; r["oph_amp"] = mx; r["oph_nhit"] = nh


def attach_truth(evs, msp_list):
    """Raw generator nu vertex from merged_sp mc_particle_tree/nu_vertices."""
    idx = {}
    for l in open(msp_list):
        l = l.strip()
        if l:
            idx[os.path.basename(l)] = l
    for e in evs:
        e["nu_vtx"] = (np.nan,) * 3
        e["n_nu"] = 0
        p = idx.get(e["src"])
        if p is None or not os.path.exists(p):
            continue
        try:
            with h5py.File(p, "r") as f:
                g = f["entry_0/mc_particle_tree/nu_vertices"]
                v = g[()]
                e["n_nu"] = int(v.shape[0])
                if v.shape[0]:
                    e["nu_vtx"] = tuple(float(x) for x in v[0][:3])
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--inputlist", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--out", default="saturation_pmt_table.csv")
    ap.add_argument("--max-events", type=int, default=40,
                    help="cap on candidate events scanned into the table")
    ap.add_argument("--min-pe", type=float, default=3000.0)
    ap.add_argument("--amp-min", type=float, default=100.0)
    ap.add_argument("--pe-amp-max", type=float, default=0.02)
    ap.add_argument("--dead", default="15")
    ap.add_argument("--want", default="saturated",
                    choices=["saturated", "dark"])
    ap.add_argument("--path-sub", default="/data/mcc9/:/data/mcc9_scratch/")
    args = ap.parse_args()
    dead = tuple(int(x) for x in args.dead.split(",") if x != "")
    old, new = args.path_sub.split(":")
    lines = [l.strip() for l in open(args.inputlist) if l.strip()]

    print(">>> scanning for bright hole events ...")
    evs = collect(args.cache, dead, args.min_pe, args.max_events)
    print(">>> %d candidate events; reading ophits ..." % len(evs))
    attach_ophit(evs, lines, old, new)
    evs = [e for e in evs if "oph_amp" in e]

    # keep only events with a GENUINE saturation among the flagged tubes
    def sat_tubes(e):
        return tuple(od for od in e["hole"]
                     if e["oph_amp"][od] > args.amp_min
                     and e["oph_pe"][od] < args.pe_amp_max * e["oph_amp"][od])
    for e in evs:
        e["sat"] = sat_tubes(e)
    if args.want == "saturated":
        evs = [e for e in evs if e["sat"]]
    else:
        evs = [e for e in evs if not e["sat"]]
    print(">>> %d events with a genuine saturated PMT" % len(evs))
    if not evs:
        return
    print(">>> reading truth vertices ...")
    attach_truth(evs, args.merged_sp_list)
    evs.sort(key=lambda e: -max(e["pred"][od] for od in e["sat"]) if e["sat"]
             else 0.0)

    with open(args.out, "w") as f:
        f.write("run,subrun,event,nu_vtx_x,nu_vtx_y,nu_vtx_z,n_true_nu,"
                "flash_t0_us,total_obs_pe,opdet,opchannel,obs_pe,pred_pe,"
                "ophit_pe,ophit_maxamp,ophit_nhit,ophit_pe_over_amp,"
                "is_saturated,is_dead,is_hole_flagged,fileno,entry,dlmerged\n")
        for e in evs:
            vx, vy, vz = e["nu_vtx"]
            for od in range(N_PMT):
                amp = e["oph_amp"][od]
                f.write("%d,%d,%d,%.3f,%.3f,%.3f,%d,%.3f,%.0f,"
                        "%d,%d,%.2f,%.2f,%.2f,%.1f,%d,%s,%d,%d,%d,%d,%d,%s\n"
                        % (e["run"], e["subrun"], e["event"], vx, vy, vz,
                           e["n_nu"], e["t0"], e["obs"].sum(), od, OD2CH[od],
                           float(e["obs"][od]), float(e["pred"][od]),
                           e["oph_pe"][od], amp, e["oph_nhit"][od],
                           ("%.4f" % (e["oph_pe"][od] / amp)) if amp else "",
                           int(od in e["sat"]), int(od in dead),
                           int(od in e["hole"]), e["fileno"], e["entry"],
                           e.get("dlmerged", "")))
    print(">>> wrote %s  (%d events x 32 PMTs = %d rows)"
          % (args.out, len(evs), 32 * len(evs)))

    print("\n== events (saturated tubes listed as opdet/opchannel) ==")
    print("%-6s %-6s %-7s | %-24s | %8s | %s"
          % ("run", "subrun", "event", "true nu vertex (x,y,z) cm", "totObsPE",
             "saturated od/ch: pred_PE, amp"))
    for e in evs:
        s = "  ".join("od%d/ch%d: pred=%.0f amp=%.0f"
                      % (od, OD2CH[od], e["pred"][od], e["oph_amp"][od])
                      for od in e["sat"])
        print("%-6d %-6d %-7d | (%7.1f,%7.1f,%7.1f) | %8.0f | %s"
              % (e["run"], e["subrun"], e["event"], e["nu_vtx"][0],
                 e["nu_vtx"][1], e["nu_vtx"][2], e["obs"].sum(), s))


if __name__ == "__main__":
    main()
