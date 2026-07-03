"""Assign 4-momenta to reco particles (spec §6): tracks via range, showers via
calorimetry. Also derives the shower calo calibration (sums de-double-counted
pixel charge per predicted shower cluster vs true KE -- no interaction reco
needed) and evaluates both estimators against truth.

    # dev (dir mode):
    python particle_momentum.py
    # production (list mode -- valdata spread over subdirs, like run_nu_reco.py):
    python particle_momentum.py --keypoint2-list KP.txt --merged-sp-list MSP.txt \
        [--start 0 --n 2000]      # optional: calibrate on a subsample
The fitted calibration is saved to data/calo_calib.npz (loaded by run_nu_reco.py).
"""
import os
import argparse
import statistics as st

import numpy as np

from . import trajfit_io as tio
from .calo import event_particle_charges
from .range_momentum import RangeMomentum

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
def _read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_msp_map(path):
    """basename -> full path, for resolving a keypoint2 file's src_file."""
    return {os.path.basename(p): p for p in _read_list(path)}


def make_msp_resolver(kp_dir=None, kp_list=None, msp_dir=None, msp_list=None):
    """Returns (kp_files, resolve) where resolve(kp_path) -> merged_sp DIR for that
    file. Dir mode: constant `msp_dir`. List mode: per-file via the src_file attr
    matched by basename against `msp_list`."""
    import glob as _glob
    if kp_list:
        kp_files = _read_list(kp_list)
        mmap = _build_msp_map(msp_list) if msp_list else {}

        def resolve(kp):
            import h5py
            with h5py.File(kp, "r") as f:
                src = f.attrs.get("src_file", "")
            src = src.decode() if isinstance(src, bytes) else src
            p = mmap.get(os.path.basename(src))
            return os.path.dirname(p) if p else None
        return kp_files, resolve
    kp_files = sorted(_glob.glob(os.path.join(kp_dir, "*.h5")))
    return kp_files, (lambda kp: msp_dir)


def fit_shower_calib(files, resolve_msp, min_points=20):
    data = {"e": [], "gamma": []}
    for fp in files:
        msp_dir = resolve_msp(fp)
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
    return _fit_pairs(data), data


def _calib_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "calo_calib.npz")


def save_shower_calib(calib, path=None):
    np.savez(path or _calib_path(),
             **{f"{k}_a": np.float32(v) for k, v in calib.items()})


def _fit_pairs(data):
    """data {type:[(Q,KE)]} -> {type: a} with KE = a*Q (through origin)."""
    calib = {}
    for typ, v in data.items():
        if len(v) < 3:
            continue
        Q = np.array([a for a, _ in v]); KE = np.array([b for _, b in v])
        calib[typ] = float((Q * KE).sum() / (Q * Q).sum())
    return calib


def save_calib_data(data, path):
    """Per-shard collected (Q,KE) pairs -> npz (for later merge+fit)."""
    d = {}
    for typ, v in data.items():
        if v:
            d[f"{typ}_Q"] = np.array([a for a, _ in v], np.float32)
            d[f"{typ}_KE"] = np.array([b for _, b in v], np.float32)
    np.savez(path, **d)


def merge_calib_data(paths):
    """Concatenate per-shard (Q,KE) npz -> data {type:[(Q,KE)]}."""
    acc = {"e": [[], []], "gamma": [[], []]}
    for p in paths:
        d = np.load(p)
        for typ in acc:
            if f"{typ}_Q" in d.files:
                acc[typ][0].append(d[f"{typ}_Q"])
                acc[typ][1].append(d[f"{typ}_KE"])
    out = {}
    for typ, (Q, KE) in acc.items():
        if Q:
            out[typ] = list(zip(np.concatenate(Q), np.concatenate(KE)))
    return out


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
    # dir mode (dev) OR list mode (production valdata, spread over subdirs)
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--keypoint2-list", default=None,
                    help="list of keypoint2 files (overrides --keypoint2-dir)")
    ap.add_argument("--merged-sp-list", default=None,
                    help="list of merged_sp files (src_file resolved by basename)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1,
                    help="subsample the file list for calibration (-1 = all)")
    # grid chunking: collect (Q,KE) pairs per shard, then merge+fit
    ap.add_argument("--out-data", default=None,
                    help="collect mode: write this shard's (Q,KE) pairs to this "
                         "npz and STOP (no fit); merge later with --merge-data")
    ap.add_argument("--merge-data", default=None,
                    help="merge mode: glob of per-shard *_data.npz -> fit + save "
                         "calo_calib.npz (no file processing)")
    args = ap.parse_args()

    if args.merge_data:                            # merge shards -> fit -> save
        import glob as _glob
        paths = sorted(_glob.glob(args.merge_data))
        data = merge_calib_data(paths)
        calib = _fit_pairs(data)
        save_shower_calib(calib)
        print(f">>> merged {len(paths)} shards -> calo_calib.npz")
        for typ in ("e", "gamma"):
            if typ in data:
                Q = np.array([a for a, _ in data[typ]])
                KE = np.array([b for _, b in data[typ]])
                E = calib[typ] * Q
                r = _res(list(zip(E, KE)))
                print(f"    {typ:5s} a={calib.get(typ, float('nan')):.4f} "
                      f"n={len(data[typ])} corr={np.corrcoef(Q, KE)[0,1]:.2f}"
                      + (f" | E_calo vs KE: bias={r[0]:+.2f} res={r[1]:.2f}"
                         if r else ""))
        return

    from . import nu_interaction as ni                # lazy: avoid import cycle
    files, resolve_msp = make_msp_resolver(
        kp_dir=args.keypoint2_dir, kp_list=args.keypoint2_list,
        msp_dir=args.merged_sp_dir, msp_list=args.merged_sp_list)
    end = len(files) if args.n < 0 else min(args.start + args.n, len(files))
    files = files[args.start:end]
    print(f">>> {len(files)} keypoint2 files "
          f"({'list' if args.keypoint2_list else 'dir'} mode)")
    rmom = RangeMomentum()

    calib, data = fit_shower_calib(files, resolve_msp)
    if args.out_data:                              # collect mode: dump pairs, stop
        save_calib_data(data, args.out_data)
        print(f">>> collected {sum(len(v) for v in data.values())} (Q,KE) pairs "
              f"({ {k: len(v) for k, v in data.items()} }) -> {args.out_data}")
        return
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
        msp_dir = resolve_msp(fp)
        entry, fh = _entry(fp, msp_dir)
        if entry is None:
            continue
        try:
            recs = tio.load_instances(fp, msp_dir, tracks_only=False,
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
