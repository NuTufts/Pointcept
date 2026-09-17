"""Step-by-step signal / eff / purity / EXT / data tables for the pi0 selection
at a given working point, from the pre-built tables of one table prefix.

Generalisation of cew6_step_tables.py (which stays as the frozen cew6 talk
version): the ntuples, table prefix, working point, EXT normalisation and
classifier-hygiene convention are all arguments.

  --hygiene cew6 : signal weights = odd-event x2 (event-BDT hygiene), EXT =
                   rows>=100k AND odd, x2 -- the run-3 cew6 samples.
  --hygiene none : samples no classifier trained on (run-1 table-gamma set):
                   all rows, table weights as they are.

Combined >=2-photon table (signal = true CC+NC pi0, per-stream chi2) and
per-stream exactly-2 tables (CC: signal = true CC pi0; NC: true NC pi0).
Event-BDT pass masks come from the *_nochi2bdt tables (chi2 gates open at
scoring time). Optional --data-ntuple adds data counts and data/pred.

    python3 pi0_step_tables.py --prefix run1tg_ts0164 --mc-ntuple .. \
        --ext-ntuple .. --data-ntuple .. --ext-scale 16.355 --hygiene none
"""
import argparse
import os

import awkward as ak
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))


def steps(nt, ts, mu_ke):
    t = uproot.open(nt)["EventTree"]
    a = t.arrays(["run", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "showerLArFormerPID", "showerRecoE",
                  "showerCosmicScore", "trackLArFormerPID", "trackIsSecondary",
                  "trackRecoE", "trackClassified", "trackMuScore",
                  "trackElScore", "trackPhScore", "trackPiScore",
                  "trackPrScore"])
    E = a["showerRecoE"]                       # recal3 baked: no transform
    vtx = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    g_nos = (a["showerLArFormerPID"] == 22) & (E > 20.0)
    g_sc = g_nos & (a["showerCosmicScore"] >= ts)
    seg = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
           & (a["trackRecoE"] > mu_ke))
    lp = ((a["trackClassified"] == 1) & (a["trackMuScore"] > a["trackElScore"])
          & (a["trackMuScore"] > a["trackPhScore"])
          & (a["trackMuScore"] > a["trackPiScore"])
          & (a["trackMuScore"] > a["trackPrScore"])
          & (a["trackIsSecondary"] == 0) & (a["trackRecoE"] > mu_ke))
    cc = ak.to_numpy(ak.any(seg | lp, axis=1))
    n_nos = ak.to_numpy(ak.sum(g_nos, axis=1))
    n_sc = ak.to_numpy(ak.sum(g_sc, axis=1))
    return dict(vtx=vtx, ge2_nos=vtx & (n_nos >= 2), ge2=vtx & (n_sc >= 2),
                eq2_nos=vtx & (n_nos == 2), eq2=vtx & (n_sc == 2),
                cc=cc, event=np.asarray(a["event"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefix", required=True, help="e.g. run1tg_ts0164")
    ap.add_argument("--table-dir", default=HERE)
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--ext-ntuple", required=True)
    ap.add_argument("--data-ntuple", default=None)
    ap.add_argument("--ts", type=float, default=0.164, help="shower score cut")
    ap.add_argument("--te", type=float, default=0.21, help="event BDT cut (label)")
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--ext-scale", type=float, required=True)
    ap.add_argument("--hygiene", default="none", choices=["none", "cew6"])
    ap.add_argument("--chi2-cc", type=float, default=1e4)
    ap.add_argument("--chi2-nc", type=float, default=1778.0)
    ap.add_argument("--out", default=None, help="also write the text here")
    args = ap.parse_args()

    def tb(sample, sfx=""):
        return np.load(os.path.join(args.table_dir,
                                    f"{sample}_{args.prefix}{sfx}_table.npz"))

    mc = steps(args.mc_ntuple, args.ts, args.mu_ke_min)
    ex = steps(args.ext_ntuple, args.ts, args.mu_ke_min)
    da = steps(args.data_ntuple, args.ts, args.mu_ke_min) if args.data_ntuple else None
    mt, mn = tb("mc"), tb("mc", "_nochi2bdt")
    et, en = tb("ext"), tb("ext", "_nochi2bdt")
    dt, dn = (tb("data"), tb("data", "_nochi2bdt")) if da else (None, None)
    cat = mt["cat"]; wmc = np.asarray(mt["w"], float)
    if args.hygiene == "cew6":
        odd = mc["event"] % 2 == 1
        sig_w = np.where(odd, 2.0, 0.0)
        e_keep = (np.arange(len(ex["vtx"])) >= 100000) & (ex["event"] % 2 == 1)
        we = np.where(e_keep, args.ext_scale * 2.0, 0.0)
        hyg = "odd x2 signal; EXT rows>=100k & odd x2"
    else:
        sig_w = np.ones(len(wmc))
        we = args.ext_scale * np.asarray(et["w"], float)
        hyg = "none (all rows)"
    chi = np.asarray(mn["flash_chi2"], float)
    echi = np.asarray(en["flash_chi2"], float)
    m_chi = np.where(mc["cc"], chi < args.chi2_cc, chi < args.chi2_nc)
    e_chi = np.where(ex["cc"], echi < args.chi2_cc, echi < args.chi2_nc)
    if da:
        dchi = np.asarray(dn["flash_chi2"], float)
        d_chi = np.where(da["cc"], dchi < args.chi2_cc, dchi < args.chi2_nc)

    lines = [f"# {args.prefix}: shower score >= {args.ts}, event BDT >= "
             f"{args.te}, union muon finder (KE>{args.mu_ke_min:.0f}), chi2 "
             f"CC<{args.chi2_cc:g} / NC<{args.chi2_nc:g}; EXT scale "
             f"{args.ext_scale:.4f}; hygiene: {hyg}"]

    def table(title, sel_key, strm=None, truecat=None, cut=None):
        sigm = (cat < 2) & (cat >= 0) if truecat is None else (cat == truecat)
        ws = np.where(sigm, wmc * sig_w, 0.0)
        wb = np.where(~sigm & (cat >= 0), wmc, 0.0)
        sm = np.ones(len(wmc), bool) if strm is None else (mc["cc"] == strm)
        se = np.ones(len(we), bool) if strm is None else (ex["cc"] == strm)
        DEN = ws.sum()
        key_nos = "ge2_nos" if sel_key == "ge2" else "eq2_nos"
        bdt_m = np.asarray(mn["sel_" + sel_key], bool)
        bdt_e = np.asarray(en["sel_" + sel_key], bool)
        base_m = np.asarray(mt["sel_" + sel_key], bool)
        base_e = np.asarray(et["sel_" + sel_key], bool)
        chi_m = m_chi if cut is None else (chi < cut)
        chi_e = e_chi if cut is None else (echi < cut)
        nph = ">=2" if sel_key == "ge2" else "exactly 2"
        rows = [("all events (stream-tagged)", sm, se),
                ("reco vtx (nu-stream, FV)", sm & mc["vtx"], se & ex["vtx"]),
                (f"{nph} photons >20 MeV", sm & mc[key_nos], se & ex[key_nos]),
                (f"+ shower score >={args.ts}", sm & mc[sel_key], se & ex[sel_key]),
                ("+ mass ok (pair found)", sm & base_m, se & base_e),
                (f"+ event BDT >={args.te}", sm & bdt_m, se & bdt_e),
                ("+ flash chi2", sm & bdt_m & chi_m, se & bdt_e & chi_e)]
        if da:
            sd = np.ones(len(da["vtx"]), bool) if strm is None else (da["cc"] == strm)
            bdt_d = np.asarray(dn["sel_" + sel_key], bool)
            base_d = np.asarray(dt["sel_" + sel_key], bool)
            chi_d = d_chi if cut is None else (dchi < cut)
            drows = [sd, sd & da["vtx"], sd & da[key_nos], sd & da[sel_key],
                     sd & base_d, sd & bdt_d, sd & bdt_d & chi_d]
        lines.append(f"\n== {title} | signal denominator (POT-wtd): {DEN:.1f} ==")
        hdr = (f"{'cut':<30}{'signal':>8}{'eff':>7}{'purity':>8}"
               f"{'MC other':>10}{'EXT':>9}")
        if da:
            hdr += f"{'data':>8}{'d/p':>6}"
        lines.append(hdr)
        for k, (nm, mm, em) in enumerate(rows):
            s = ws[mm].sum(); b = wb[mm].sum(); e = we[em].sum()
            line = (f"{nm:<30}{s:8.1f}{s/DEN:7.3f}{s/max(s+b+e,1e-9):8.3f}"
                    f"{b:10.1f}{e:9.1f}")
            if da:
                d = int(drows[k].sum())
                line += f"{d:8d}{d/max(s+b+e,1e-9):6.2f}"
            lines.append(line)

    table("COMBINED CC+NC, >=2 photons (signal = true CC+NC pi0; chi2 "
          "per-stream)", "ge2")
    table("reco-CC stream, exactly 2 (signal = true CC pi0)", "eq2",
          strm=True, truecat=0, cut=args.chi2_cc)
    table("reco-NC stream, exactly 2 (signal = true NC pi0)", "eq2",
          strm=False, truecat=1, cut=args.chi2_nc)
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(txt + "\n")
        print(f">>> wrote {args.out}")


if __name__ == "__main__":
    main()
