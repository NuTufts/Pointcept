#!/usr/bin/env python3
"""
SBND flash-model closure on MC larcv2 files: predict the beam flash from TRUE
energy deposits and compare with the simulated opflash.

For each entry:
  - deposits: sparse3d_pcluster (true MeV on reco voxels; overlap-resolved owner in
    sparse3d_pcluster_index -> particle_corrected) of particles whose start time is in
    the beam window [t_lo, t_hi] ns (in-spill neutrino and any in-time cosmic)
  - photons = E[MeV] x scint_yield (SBND LArProperties ScintYield/ScintPreScale = 24000/MeV
    by default; recombination/quenching ignored -> the fitted ratio absorbs it)
  - prediction: SBNDFlashModel.predict(..., charge_is_photons=True), PMT channels
  - observation: the brightest opflash_tpc0 + opflash_tpc1 flashes inside [f_lo, f_hi] us,
    PEPerOpDet padded to 312 channels, summed
  - metrics over unmasked PMTs: cosine(pred, obs), obs/pred total ratio, and the cosine
    for a "mirrored" prediction (TPCs swapped) as a sanity baseline

Run inside the pointcept container with larcv2 set up:
    source lartpc/data_prep/sbnd_official/test/setenv_larcv2.sh
    source lartpc/data_prep/sbnd_official/test/larcv2/configure.sh -q
    python3 lartpc/flashmatch/model/tests/sbnd_larcv2_flash_closure.py FILE.root [--detectors all]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))


def sparse_ids_vals(io, name):
    s = io.get_data("sparse3d", name)
    v = s.as_vector()
    n = v.size()
    ids = np.fromiter((v[i].id() for i in range(n)), np.int64, n)
    vals = np.fromiter((v[i].value() for i in range(n)), np.float64, n)
    return ids, vals, s.meta()


def voxel_xyz(meta, ids):
    nx, ny = meta.num_voxel_x(), meta.num_voxel_y()
    ix = ids % nx
    iy = (ids // nx) % ny
    iz = ids // (nx * ny)
    return np.stack([meta.min_x() + (ix + 0.5) * meta.size_voxel_x(),
                     meta.min_y() + (iy + 0.5) * meta.size_voxel_y(),
                     meta.min_z() + (iz + 0.5) * meta.size_voxel_z()], 1)


def best_flash(io, producer, f_lo, f_hi, n_ch):
    fv = io.get_data("opflash", producer).as_vector()
    best, best_pe = None, -1.0
    for i in range(fv.size()):
        f = fv[i]
        if f_lo <= f.time() <= f_hi and f.TotalPE() > best_pe:
            best, best_pe = f, f.TotalPE()
    if best is None:
        return None, None
    pe = np.zeros(n_ch)
    v = np.asarray(list(best.PEPerOpDet()), dtype=np.float64)
    pe[: min(n_ch, v.size)] = v[:n_ch]
    return pe, best.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--detectors", default="pmt", choices=["pmt", "all"])
    ap.add_argument("--scint-yield", type=float, default=None, help="photons/MeV (default: detinfo LArProperties)")
    ap.add_argument("--particle-window-ns", type=float, nargs=2, default=(0.0, 2500.0))
    ap.add_argument("--flash-window-us", type=float, nargs=2, default=(-0.5, 3.0))
    ap.add_argument("--max-entries", type=int, default=-1)
    ap.add_argument("--deposits", default="pcluster", choices=["pcluster", "sed"],
                    help="pcluster: true MeV on reconstructed voxels (active volume only); "
                         "sed: all true energy deposits (cluster3d_sed, incl. outside the AV)")
    a = ap.parse_args()

    # build the model (h5py) BEFORE importing larcv: larcv's OpenCV/GDAL loads the system
    # HDF5 1.10, after which conda's h5py (built for 1.14) can no longer initialize
    from lartpc.flashmatch.model import make_flash_model
    fm = make_flash_model("sbnd", detectors="all" if a.detectors == "all" else
                          ("pmt_coated", "pmt_uncoated"))
    fm.photonlib                                  # load the outside-AV library now, same reason
    from larcv import larcv
    lp = fm.di.larprop
    sy = a.scint_yield or float(lp.ScintYield()) / float(lp.ScintPreScale())
    nch = fm.n_channels
    print(f"model: {len(fm.selected)} channels ({fm.detector_types}); scint yield {sy:.0f} photons/MeV")

    io = larcv.IOManager(larcv.IOManager.kREAD)
    for f in a.files:
        io.add_in_file(f)
    io.set_verbosity(2)
    io.initialize()
    n = io.get_n_entries() if a.max_entries < 0 else min(a.max_entries, io.get_n_entries())
    rows = []
    print(f"{'entry':>5} {'E_in[MeV]':>9} {'npts':>6} {'obsPE':>8} {'predPE':>9} {'obs/pred':>8} "
          f"{'cos':>6} {'cos_mirror':>10} {'tflash':>7}")
    for e in range(n):
        io.read_entry(e)
        pv = io.get_data("particle", "corrected").as_vector()
        tpart = np.array([pv[int(k)].t() for k in range(pv.size())])
        intime = (tpart >= a.particle_window_ns[0]) & (tpart <= a.particle_window_ns[1])
        if a.deposits == "pcluster":
            ids, mev, meta = sparse_ids_vals(io, "pcluster")
            iid, owner, _ = sparse_ids_vals(io, "pcluster_index")
            if not np.array_equal(ids, iid):
                print(f"{e:5d} pcluster / pcluster_index voxel sets differ; skipped")
                continue
            sel = intime[owner.astype(int)]
        else:
            c3 = io.get_data("cluster3d", "sed")
            meta = c3.meta()
            cv = c3.as_vector()
            ids_l, mev_l, own_l = [], [], []
            for k in range(min(cv.size(), pv.size())):        # last cluster = unassigned leftovers
                vs = cv[k].as_vector()
                m = vs.size()
                if m == 0:
                    continue
                ids_l.append(np.fromiter((vs[j].id() for j in range(m)), np.int64, m))
                mev_l.append(np.fromiter((vs[j].value() for j in range(m)), np.float64, m))
                own_l.append(np.full(m, k))
            if not ids_l:
                continue
            ids, mev, owner = np.concatenate(ids_l), np.concatenate(mev_l), np.concatenate(own_l)
            sel = intime[owner]
        if sel.sum() < 10:
            continue
        pts = voxel_xyz(meta, ids[sel])
        photons = mev[sel] * sy
        obs = np.zeros(nch)
        tfl = []
        for prod in ("tpc0", "tpc1"):
            pe, t = best_flash(io, prod, a.flash_window_us[0], a.flash_window_us[1], nch)
            if pe is not None:
                obs += pe
                tfl.append(t)
        if not tfl:
            continue
        t0 = float(np.mean(tfl))
        pred = fm.predict(pts, photons, flash_t0_us=t0, light_yield=1.0, charge_is_photons=True)[0]
        mirror = pts.copy()
        mirror[:, 0] *= -1
        pred_m = fm.predict(mirror, photons, flash_t0_us=t0, light_yield=1.0, charge_is_photons=True)[0]
        mask = np.zeros(nch, bool)
        mask[list(fm.channel_mask(None, mask_saturated=False))] = True
        use = ~mask
        o, p, pm = obs[use], pred[use], pred_m[use]
        cos = float(o @ p / (np.linalg.norm(o) * np.linalg.norm(p) + 1e-30))
        cosm = float(o @ pm / (np.linalg.norm(o) * np.linalg.norm(pm) + 1e-30))
        ratio = float(o.sum() / max(p.sum(), 1e-30))
        rows.append((e, mev[sel].sum(), sel.sum(), o.sum(), p.sum(), ratio, cos, cosm))
        print(f"{e:5d} {mev[sel].sum():9.1f} {int(sel.sum()):6d} {o.sum():8.0f} {p.sum():9.0f} {ratio:8.3f} "
              f"{cos:6.3f} {cosm:10.3f} {t0:7.3f}")
    if rows:
        r = np.array(rows)
        good = r[:, 3] > 500
        print(f"\n{len(r)} entries; with obs PE > 500: {good.sum()}")
        if good.any():
            print(f"  cosine     median {np.median(r[good, 6]):.3f}  (mirrored-TPC baseline median {np.median(r[good, 7]):.3f})")
            print(f"  obs/pred   median {np.median(r[good, 5]):.3f}  (spread 16-84%: "
                  f"{np.percentile(r[good, 5], 16):.3f}-{np.percentile(r[good, 5], 84):.3f})")


if __name__ == "__main__":
    main()
