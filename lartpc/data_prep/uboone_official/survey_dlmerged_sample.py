"""Pre-flight survey of a dlmerged input list (Sample_Processing_Procedure.md step 1).

Reports what the processing decisions depend on: file presence + size, event
count, run range -> run period(s), MC vs data (generator/corsika truth trees),
POT, generator particle content + start times, and the brightest in-time
beam-flash time histogram (to choose / verify FLASH_WINDOW).

    python3 survey_dlmerged_sample.py <inputlist.txt> [--sample-every 12] [--events-per-file 200]

Run inside the pointcept container with the ubdl env sourced (needs ROOT +
larlite dictionaries). Reads only a few small trees; ~20 s for 250 files.
"""
import argparse
import os
import sys

import numpy as np
import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lartpc.flashmatch.dead_channels import run_period  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputlist")
    ap.add_argument("--sample-every", type=int, default=12,
                    help="read truth/flash content from every Nth file")
    ap.add_argument("--events-per-file", type=int, default=200)
    args = ap.parse_args()

    files = [l.strip() for l in open(args.inputlist) if l.strip()]
    missing = [f for f in files if not os.path.isfile(f)]
    present = [f for f in files if os.path.isfile(f)]
    size_gb = sum(os.path.getsize(f) for f in present) / 1e9
    on_tier2 = sum(f.startswith("/cluster/tier2/") for f in files)
    print(f"files: {len(files)} listed, {len(present)} present, {len(missing)} missing, "
          f"{size_gb:.1f} GB; on tier2 (login-only, must be copied): {on_tier2}")
    if len(set(files)) != len(files):
        print(f"WARNING: {len(files) - len(set(files))} duplicate lines")

    runs, nevs, pot = [], [], 0.0
    trees = None
    gen_pdg, gen_t, gen_e, flash_t = [], [], [], []
    for k, fn in enumerate(present):
        f = ROOT.TFile.Open(fn)
        if trees is None:
            trees = sorted(x.GetName() for x in f.GetListOfKeys())
        t = f.Get("larlite_id_tree")
        n = t.GetEntries()
        nevs.append(n)
        t.GetEntry(0)
        r0 = t._run_id
        t.GetEntry(n - 1)
        runs.append((r0, t._run_id))
        p = f.Get("potsummary_generator_tree")
        if p:
            for i in range(p.GetEntries()):
                p.GetEntry(i)
                pot += p.potsummary_generator_branch.totpot
        if k % args.sample_every == 0:
            g = f.Get("mctruth_generator_tree")
            b = f.Get("opflash_simpleFlashBeam_tree")
            for i in range(min(n, args.events_per_file)):
                if g:
                    g.GetEntry(i)
                    for m in g.mctruth_generator_branch:
                        for q in m.GetParticles():
                            if q.StatusCode() == 1:
                                gen_pdg.append(q.PdgCode())
                                gen_t.append(q.Trajectory()[0].T() / 1000.0)
                                gen_e.append(q.Trajectory()[0].E())
                b.GetEntry(i)
                fl = [(x.TotalPE(), x.Time()) for x in b.opflash_simpleFlashBeam_branch
                      if 0 < x.Time() < 20]
                if fl:
                    flash_t.append(max(fl)[1])
        f.Close()

    r = np.array(runs)
    periods = sorted({run_period(x) for x in r.ravel()})
    print(f"events: {sum(nevs)} (per file min/median/max {min(nevs)}/{int(np.median(nevs))}/{max(nevs)})")
    print(f"runs: {r.min()}-{r.max()} -> run period(s) {periods}")
    print(f"trees ({len(trees)}): {' '.join(trees)}")
    is_mc = "mctruth_generator_tree" in trees
    has_corsika = "mctruth_corsika_tree" in trees
    print(f"MC truth: {is_mc}; simulated CORSIKA cosmics: {has_corsika} "
          f"({'pure MC' if has_corsika else 'overlay or data'})")
    print(f"sum totPOT: {pot:.4g}")
    if gen_pdg:
        u, c = np.unique(gen_pdg, return_counts=True)
        print("generator status-1 PDG counts:", dict(zip(u.tolist(), c.tolist())))
        print("generator start time [us] pct 0/1/50/99/100:",
              np.percentile(gen_t, [0, 1, 50, 99, 100]).round(3))
        print("generator E [GeV] pct 0/50/100:", np.percentile(gen_e, [0, 50, 100]).round(3))
    if flash_t:
        ft = np.array(flash_t)
        h, e = np.histogram(ft, bins=np.arange(2.0, 7.01, 0.2))
        print(f"brightest beam flash (0-20 us) N={len(ft)}, pct 5/50/95:",
              np.percentile(ft, [5, 50, 95]).round(3))
        print("flash-time histogram, 0.2 us bins from 2.0 us:")
        for lo, cnt in zip(e[:-1], h):
            print(f"  {lo:4.1f} {'#' * int(60 * cnt / max(h.max(), 1))} {cnt}")


if __name__ == "__main__":
    main()
