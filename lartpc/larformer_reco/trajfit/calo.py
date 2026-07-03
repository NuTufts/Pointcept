"""Calorimetric charge sums with de-double-counting (spec §4).

Multiple 3D spacepoints project to the same wire-plane pixel and each carries that
pixel's full ADC in `pixval`, so summing `pixval` double-counts shared pixels. We
identify a spacepoint's pixel in plane p by `(tick, wire_p)`, count how many
spacepoints share it, and give each a charge share `pixval/count`. Per-plane sums
(U,V,Y) plus a combined "Y, else mean(U,V)".

CLI study: per true particle, de-double-counted combined charge vs true KE (to see
the charge->energy relation before fitting the calibration).
    python calo.py <merged_sp dir-or-file>
"""
import os
import glob
import argparse

import numpy as np
import h5py

_WBASE = 100000        # tick*_WBASE + wire, per-plane unique pixel key (wires < 1e5)


def dedup_charge(pixval, tick, uwire, vwire, ywire):
    """De-double-counted per-spacepoint charge. Inputs are parallel arrays over a
    spacepoint SET (the set defines the sharing). Returns:
      q (N,3)  per-plane de-double-counted charge (U,V,Y)
      q_comb (N,)  Y if pixval_Y>0 else mean(U,V), on the de-double-counted values
    """
    pixval = np.asarray(pixval, np.float64)
    tick = np.asarray(tick, np.int64)
    wires = [np.asarray(uwire, np.int64), np.asarray(vwire, np.int64),
             np.asarray(ywire, np.int64)]
    q = np.zeros_like(pixval)
    for p in range(3):
        key = tick * _WBASE + wires[p]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        q[:, p] = pixval[:, p] / cnt[inv]          # split pixel ADC among sharers
    q_comb = np.where(pixval[:, 2] > 0, q[:, 2], 0.5 * (q[:, 0] + q[:, 1]))
    return q, q_comb


def particle_charge(q, q_comb, idx):
    """Sum de-double-counted charge over a particle's spacepoint indices (into the
    same set dedup_charge was computed on). Returns dict U/V/Y/comb."""
    idx = np.asarray(idx, np.int64)
    return dict(U=float(q[idx, 0].sum()), V=float(q[idx, 1].sum()),
                Y=float(q[idx, 2].sum()), comb=float(q_comb[idx].sum()))


def event_particle_charges(points_list, entry, max_match_cm=0.5):
    """De-double-counted charge per reco particle. `points_list` = list of (n_i,3)
    reco-particle point clouds (slice cm); `entry` = an open merged_sp `entry_0`
    group. KD-matches the UNION of particle points into triplet_data (they're an
    exact subset), de-double-counts over that union, and returns one charge dict
    per particle. Particles that don't match (no merged_sp) -> None."""
    from scipy.spatial import cKDTree
    if not points_list:
        return []
    counts = [len(P) for P in points_list]
    allP = np.concatenate([np.asarray(P, np.float64) for P in points_list], 0)
    tpos = entry["triplet_data/pos"][()].astype(np.float64)
    td = entry["triplet_data"]
    tree = cKDTree(tpos)
    d, idx = tree.query(allP, k=1)
    if np.median(d) > max_match_cm:
        return [None] * len(points_list)
    q, q_comb = dedup_charge(td["pixval"][()][idx], td["tick"][()][idx],
                             td["uwire"][()][idx], td["vwire"][()][idx],
                             td["ywire"][()][idx])
    out, s = [], 0
    for n in counts:
        sl = slice(s, s + n)
        out.append(dict(U=float(q[sl, 0].sum()), V=float(q[sl, 1].sum()),
                        Y=float(q[sl, 2].sum()), comb=float(q_comb[sl].sum())))
        s += n
    return out


# ---------------------------------------------------------------------------
# calibration study: charge vs true KE per particle type
# ---------------------------------------------------------------------------
_PDG_NAME = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
             211: "pi", -211: "pi", 2212: "p"}


def _study(files):
    import statistics as st
    by_type = {}
    for fp in files:
        with h5py.File(fp, "r") as f:
            e = f["entry_0"]
            td = e["triplet_data"]
            trk = td["trackid"][()].astype(np.int64)
            real = trk > 0                          # drop ghosts (proxy for slice)
            if real.sum() == 0:
                continue
            pixval = td["pixval"][()][real]
            tick = td["tick"][()][real]
            uw, vw, yw = (td["uwire"][()][real], td["vwire"][()][real],
                          td["ywire"][()][real])
            trk = trk[real]
            mt = e["mc_particle_tree"]
            ke_by = {int(t): float(k) for t, k in
                     zip(mt["trackid"][()], mt["energy_mev"][()])}
            pid_by = {int(t): int(p) for t, p in
                      zip(mt["trackid"][()], mt["pid"][()])}
        q, q_comb = dedup_charge(pixval, tick, uw, vw, yw)
        for t in np.unique(trk):
            typ = _PDG_NAME.get(pid_by.get(int(t), 0))
            ke = ke_by.get(int(t))
            if typ is None or ke is None or ke <= 0:
                continue
            Q = q_comb[trk == t].sum()
            if Q <= 0:
                continue
            by_type.setdefault(typ, []).append((Q, ke))
    print(f"{'type':>6} {'n':>4} {'medQ/KE':>8} {'spread%':>7}  "
          f"(Q=de-double-counted combined charge [ADC], KE=true [MeV])")
    for typ in ("e", "gamma", "mu", "pi", "p"):
        v = by_type.get(typ, [])
        if len(v) < 3:
            continue
        ratios = [Q / ke for Q, ke in v]
        r = st.median(ratios)
        spread = (np.percentile(ratios, 84) - np.percentile(ratios, 16)) / (2 * r)
        # linearity: correlation of Q vs KE
        Q = np.array([a for a, _ in v]); KE = np.array([b for _, b in v])
        corr = np.corrcoef(Q, KE)[0, 1]
        print(f"{typ:>6} {len(v):>4} {r:>8.1f} {100*spread:>6.0f}%  "
              f"corr(Q,KE)={corr:.2f}  KE range {KE.min():.0f}-{KE.max():.0f}MeV")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged_sp", help="merged_sp dir or single .h5")
    args = ap.parse_args()
    files = ([args.merged_sp] if args.merged_sp.endswith(".h5")
             else sorted(glob.glob(os.path.join(args.merged_sp, "*.h5"))))
    print(f">>> {len(files)} merged_sp files")
    _study(files)


if __name__ == "__main__":
    main()
