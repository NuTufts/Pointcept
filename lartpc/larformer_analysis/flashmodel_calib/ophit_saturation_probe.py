"""Does the ophit signature (negative integrated area at the beam time, with a
large positive amplitude) tag the same PMTs as the observed-PE "hole" finder?

Motivation: the hole finder (lartpc/flashmatch/saturation.py) infers saturation
from the observed light pattern alone -- a tube reading ~0 surrounded by bright
ones. That is cheap (works on merged_sp/cascade as they stand) but indirect. The
ophit route is the mechanism itself: a saturated PMT rails, the baseline
restoration undershoots, and the hit's integrated AREA goes negative while its
AMPLITUDE stays large. If the two agree, the hole finder is validated and can be
used as-is; where they disagree tells us what a sidecar would buy.

This probes the question WITHOUT building any sidecar: it walks a sample of
cascade events, traces each back to its source dlmerged via the cascade's
src_file attr (-> fileno/entry -> the stage-A INPUTLIST line), and reads the
ophit tree directly.

    python3 ophit_saturation_probe.py --cascade-dir <keypoint2_streams> \
        --inputlist <scale1500_bnb_nu_overlay.txt> --max-files 40
"""
import argparse
import os
import re
import sys

import numpy as np
import h5py
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.saturation import find_saturated        # noqa: E402

# larlite larutil::Geometry::OpDetFromOpChannel, opch -> opdet. This is the
# authoritative map (it is what the stage-A converter used to fill
# merged_sp flashes/pe). NOTE: viz/pmtpos.py's table disagrees on 32/32
# channels and must NOT be used here.
OPCH2OPDET = {0: 3, 1: 5, 2: 1, 3: 6, 4: 0, 5: 2, 6: 4, 7: 9, 8: 11, 9: 7,
              10: 12, 11: 8, 12: 10, 13: 14, 14: 17, 15: 13, 16: 18, 17: 15,
              18: 16, 19: 21, 20: 22, 21: 19, 22: 24, 23: 20, 24: 23, 25: 26,
              26: 29, 27: 30, 28: 25, 29: 31, 30: 27, 31: 28}

OPHIT_PROD = "ophitBeam::OverlayStage1OpticalDLrerun"
BEAM_LO, BEAM_HI = 2.0, 6.0     # in-time window [us]


def src_fileno_entry(name):
    m = re.search(r"fileno(\d+)_entry(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def scan_cascade(cascade_dir, dead, limit_files, want):
    """{fileno: [(entry, cascade_path, obs, hole_mask)]} for events that have a
    nu slice + an in-time flash. `want`: 'hole' (>=1 candidate), 'nohole', 'any'.
    """
    out = {}
    done = False
    for root, _d, files in os.walk(cascade_dir):
        if done:
            break
        for n in sorted(files):
            # break out as soon as we have enough SOURCE files -- the MC
            # cascade dir is flat with ~77k entries, so waiting for os.walk to
            # finish the directory would read all of them
            if len(out) >= limit_files:
                done = True
                break
            if not n.startswith("keypoint2_event") or not n.endswith("_0.h5") \
                    or n.endswith("_fm_0.h5"):
                continue
            p = os.path.join(root, n)
            try:
                with h5py.File(p, "r") as f:
                    s = f.attrs.get("src_file")
                    if s is None or "flash" not in f or \
                            "observed_pe" not in f["flash"]:
                        continue
                    s = s.decode() if isinstance(s, bytes) else str(s)
                    obs = f["flash/observed_pe"][()]
            except Exception:
                continue
            if not np.isfinite(obs).any() or obs.sum() <= 0:
                continue
            fno, ent = src_fileno_entry(s)
            if fno is None:
                continue
            hole = find_saturated(obs, dead=dead, max_masked=None)
            if want == "hole" and not hole:
                continue
            if want == "nohole" and hole:
                continue
            out.setdefault(fno, []).append((ent, p, obs, hole))
        nfile = len(out)
        if nfile >= limit_files:
            break
    return out


def ophit_flags(root_path, entries, amp_min, prod=OPHIT_PROD,
                pe_amp_max=0.02):
    """{entry: {opdet: (sumPE, maxAmp)}} for tubes whose beam-window pulse is
    LARGE in amplitude but reconstructs to almost no PE.

    The naive criterion "integrated area < 0" turns out to be only the extreme
    tail: measured on the run3b overlay, a healthy tube runs PE/amplitude ~ 0.21
    (opch 13: amp 5200 -> 1095 PE), while a saturated one runs ~0.005 (amp 518
    -> 2.8 PE) with area still slightly POSITIVE. So flag on the ratio, which
    catches the negative-area cases as a subset. Requiring amplitude > amp_min
    is essential: without it, ~150 tiny baseline wiggles per file (amp ~6) have
    negative area and would swamp the tag.
    """
    br = ("ophit_%s_branch/vector<larlite::ophit>/vector<larlite::ophit>." % prod)
    f = uproot.open(root_path)
    tn = "ophit_%s_tree" % prod
    if tn not in [k.split(";")[0] for k in f.keys(recursive=False)]:
        return {}
    t = f[tn]
    lo, hi = min(entries), max(entries) + 1
    ch = t[br + "fOpChannel"].array(entry_start=lo, entry_stop=hi)
    pk = t[br + "fPeakTime"].array(entry_start=lo, entry_stop=hi)
    am = t[br + "fAmplitude"].array(entry_start=lo, entry_stop=hi)
    pe = t[br + "fPE"].array(entry_start=lo, entry_stop=hi)
    out = {}
    for e in entries:
        k = e - lo
        c = np.asarray(ch[k]); p = np.asarray(pk[k])
        a = np.asarray(am[k]); q = np.asarray(pe[k])
        bw = (p > BEAM_LO) & (p < BEAM_HI)
        d = {}
        for ci in np.unique(c[bw]):
            od = OPCH2OPDET.get(int(ci))
            if od is None:
                continue
            m = bw & (c == ci)
            mx = float(a[m].max())
            sp = float(q[m].sum())
            if mx > amp_min and sp < pe_amp_max * mx:
                d[od] = (sp, mx)
        out[e] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--inputlist", required=True)
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--amp-min", type=float, default=100.0)
    ap.add_argument("--dead", default="15")
    ap.add_argument("--want", default="any", choices=["any", "hole", "nohole"])
    ap.add_argument("--path-sub", default="/data/mcc9/:/data/mcc9_scratch/",
                    help="OLD:NEW prefix rewrite for stale INPUTLIST paths")
    args = ap.parse_args()
    dead = tuple(int(x) for x in args.dead.split(",") if x != "")
    old, new = args.path_sub.split(":")

    lines = [l.strip() for l in open(args.inputlist) if l.strip()]
    print(">>> scanning cascade for events (want=%s) ..." % args.want)
    by_file = scan_cascade(args.cascade_dir, dead, args.max_files, args.want)
    print(">>> %d source files, %d events"
          % (len(by_file), sum(len(v) for v in by_file.values())))

    n_ev = n_agree = n_hole_only = n_oph_only = 0
    rows = []
    for fno, evs in sorted(by_file.items()):
        if fno < 1 or fno > len(lines):
            continue
        rp = lines[fno - 1].replace(old, new)
        if not os.path.exists(rp):
            continue
        try:
            fl = ophit_flags(rp, sorted(e for e, _, _, _ in evs), args.amp_min)
        except Exception as ex:
            print("  [warn] fileno %d: %s" % (fno, ex))
            continue
        for ent, path, obs, hole in evs:
            oph = set(fl.get(ent, {}).keys()) - set(dead)
            hs = set(hole)
            n_ev += 1
            n_agree += len(hs & oph)
            n_hole_only += len(hs - oph)
            n_oph_only += len(oph - hs)
            if hs or oph:
                rows.append((fno, ent, sorted(hs), sorted(oph),
                             {k: v for k, v in fl.get(ent, {}).items()},
                             obs))
    print("\n== per-TUBE agreement over %d events ==" % n_ev)
    print("  both hole-finder AND negative-area ophit : %d" % n_agree)
    print("  hole-finder only (no neg-area ophit)     : %d" % n_hole_only)
    print("  neg-area ophit only (not a hole)         : %d" % n_oph_only)
    if n_agree + n_hole_only:
        print("  -> hole-finder tubes confirmed by ophit   : %.0f%%"
              % (100.0 * n_agree / (n_agree + n_hole_only)))

    print("\n== examples (fileno, entry, hole, ophit_neg_area) ==")
    for fno, ent, hs, oph, det, obs in rows[:18]:
        d = "  ".join("od%d:PE=%.1f,amp=%.0f" % (k, v[0], v[1])
                       for k, v in sorted(det.items()))
        print("  f%05d e%06d hole=%-10s ophit=%-10s obsPE=%7.0f | %s"
              % (fno, ent, hs, oph, obs.sum(), d))


if __name__ == "__main__":
    main()
