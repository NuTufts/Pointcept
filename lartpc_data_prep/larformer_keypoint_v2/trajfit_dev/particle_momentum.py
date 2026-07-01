"""Assign 4-momenta to reco particles (spec §6): tracks via range, showers via
calorimetry. Also derives the primary-level shower calo calibration and evaluates
both estimators against truth.

    python particle_momentum.py            # fit calib + eval vs truth (pi0 set)
"""
import os
import glob
import argparse
import statistics as st

import numpy as np

import trajfit_io as tio
from calo import event_particle_charges
from range_momentum import RangeMomentum

TRACK_CLASSES = {"mu", "pi", "p"}


def _entry(fp, msp_dir):
    """Open the parent merged_sp entry for a keypoint2 file (or None, None)."""
    if not msp_dir:
        return None, None
    import h5py
    with h5py.File(fp, "r") as f:
        src = f.attrs.get("src_file", "")
    if isinstance(src, bytes):
        src = src.decode()
    return tio._open_merged_sp(msp_dir, src)


# ---------------------------------------------------------------------------
# shower calo calibration: KE = a * Q_comb  (per type, through origin)
# ---------------------------------------------------------------------------
def fit_shower_calib(files, msp_dir, min_points=20):
    data = {"e": [], "gamma": []}
    for fp in files:
        entry, fh = _entry(fp, msp_dir)
        if entry is None:
            continue
        try:
            recs = tio.load_instances(fp, msp_dir, tracks_only=False,
                                      min_points=min_points)
            sh = [r for r in recs if r.pred_cls_name in ("e", "gamma")]
            charges = event_particle_charges([r.points for r in sh], entry)
            for r, c in zip(sh, charges):
                if c and c["comb"] > 0 and np.isfinite(r.energy_mev) \
                        and r.energy_mev > 0:
                    data[r.pred_cls_name].append((c["comb"], r.energy_mev))
        finally:
            fh.close()
    calib = {}
    for typ, v in data.items():
        if len(v) < 3:
            continue
        Q = np.array([a for a, _ in v]); KE = np.array([b for _, b in v])
        calib[typ] = float((Q * KE).sum() / (Q * Q).sum())   # KE = a*Q
    return calib, data


def _calib_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "calo_calib.npz")


def save_shower_calib(calib, path=None):
    np.savez(path or _calib_path(),
             **{f"{k}_a": np.float32(v) for k, v in calib.items()})


def load_shower_calib(path=None):
    """{type: a}. Empty if no file (reco then can't calorimeter showers)."""
    p = path or _calib_path()
    if not os.path.exists(p):
        return {}
    d = np.load(p)
    return {k[:-2]: float(d[k]) for k in d.files if k.endswith("_a")}


# ---------------------------------------------------------------------------
# assignment (for integration into nu_interaction)
# ---------------------------------------------------------------------------
def assign_momenta(interactions, entry, range_mom, shower_calib):
    """Return one 4-momentum dict per reco particle across all interactions."""
    items, pts = [], []
    for I in interactions:
        for T in I["tracks"]:
            ei = T["attach"]["end"] if T.get("attach") else 0
            d = T["ends"][ei]["u_in"]
            items.append(("track", T, d, T["length"]))
            pts.append(T["points"])
        for s in I["showers"]:
            if not s["attached"]:
                continue
            items.append(("shower", s, s["trunk"].direction, None))
            pts.append(s["points"])
    charges = (event_particle_charges(pts, entry) if entry is not None
               else [None] * len(items))
    out = []
    for (kind, obj, d, L), c in zip(items, charges):
        cls = obj["cls_name"]
        rec = dict(kind=kind, pred_class=cls,
                   direction=np.asarray(d, np.float32),
                   charge_comb=(c["comb"] if c else float("nan")),
                   ke_range=float("nan"), ke_calo=float("nan"))
        if kind == "track" and cls in TRACK_CLASSES:
            fm = range_mom.fourmom(L, cls, d)
            rec.update(ke_range=fm["ke"], energy=fm["energy"],
                       momentum=fm["momentum"], fourvec=fm["fourvec"],
                       length_cm=L, momentum_method="range")
        else:                                          # shower (or non-track)
            a = shower_calib.get(cls, shower_calib.get("gamma", 0.0))
            E = a * c["comb"] if c else float("nan")
            dd = np.asarray(d, np.float32)
            rec.update(ke_calo=E, energy=E, momentum=(E * dd).astype(np.float32),
                       fourvec=np.array([E, *(E * dd)], np.float32),
                       momentum_method="calo")
        obj["mom"] = rec                               # attach onto the particle
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# eval vs truth
# ---------------------------------------------------------------------------
def _res(pairs):
    """(reco, true) pairs -> median fractional bias and resolution."""
    fr = [(r - t) / t for r, t in pairs if t > 0]
    if not fr:
        return None
    return (st.median(fr),
            (np.percentile(fr, 84) - np.percentile(fr, 16)) / 2.0, len(fr))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data", "bnb_pi0_valdata")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    args = ap.parse_args()
    import nu_interaction as ni                # lazy: avoid import cycle
    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    rmom = RangeMomentum()

    calib, data = fit_shower_calib(files, args.merged_sp_dir)
    save_shower_calib(calib)
    print(">>> shower calo calibration  KE[MeV] = a * Q_comb (saved to npz):")
    for typ in ("e", "gamma"):
        if typ in calib:
            Q = np.array([a for a, _ in data[typ]])
            KE = np.array([b for _, b in data[typ]])
            corr = np.corrcoef(Q, KE)[0, 1]
            print(f"    {typ:5s} a={calib[typ]:.3f}  (n={len(data[typ])} "
                  f"corr={corr:.2f})")

    # eval: showers (calo) and tracks (range) vs true KE
    calo_pairs = {"e": [], "gamma": []}
    range_pairs = {"mu": [], "pi": [], "p": []}
    for fp in files:
        entry, fh = _entry(fp, args.merged_sp_dir)
        if entry is None:
            continue
        try:
            recs = tio.load_instances(fp, args.merged_sp_dir, tracks_only=False,
                                      min_points=20)
            sh = [r for r in recs if r.pred_cls_name in ("e", "gamma")]
            charges = event_particle_charges([r.points for r in sh], entry)
            for r, c in zip(sh, charges):
                if c and c["comb"] > 0 and r.energy_mev > 0:
                    calo_pairs[r.pred_cls_name].append(
                        (calib.get(r.pred_cls_name, 0) * c["comb"], r.energy_mev))
            trk = [r for r in recs if r.pred_cls_name in TRACK_CLASSES]
            tracks = ni.build_tracks(trk)
            e_by = {r.gt_trackid: r.energy_mev for r in trk}
            for T in tracks:
                ke_true = e_by.get(T["gt_trackid"], float("nan"))
                if np.isfinite(ke_true) and ke_true > 0:
                    ke_r = rmom.ke(T["length"], {"mu": "muon", "pi": "pion",
                                                 "p": "proton"}[T["cls_name"]])
                    range_pairs[T["cls_name"]].append((ke_r, ke_true))
        finally:
            fh.close()

    print("\n>>> energy resolution vs true KE  (median frac bias +/- resolution):")
    print("  CALO (showers):")
    for typ in ("gamma", "e"):
        r = _res(calo_pairs[typ])
        if r:
            print(f"    {typ:5s} bias={r[0]:+.2f} res={r[1]:.2f} (n={r[2]})")
    print("  RANGE (tracks; all incl. non-stopping/cosmics):")
    for typ in ("p", "pi", "mu"):
        r = _res(range_pairs[typ])
        if r:
            print(f"    {typ:5s} bias={r[0]:+.2f} res={r[1]:.2f} (n={r[2]})")


if __name__ == "__main__":
    main()
