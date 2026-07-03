"""Aggregate slice-id analysis for the single-visible-photon sample.

For each event's visible photon (nu-origin gamma with the most charge), partition
its ionization CHARGE (de-double-counted, dedup once over the photon so shares sum
to 100%) across the cascade slice categories -- pre-filtered / ghost (deghosted) /
kept-unclustered / nu-slice / cosmic-slice -- using the full-event slice_id
sidecars. Then quantify how often the photon is CLEANLY IN ONE cosmic slice:

  f_cosmic_max  = charge in the single LARGEST cosmic slice / photon charge
  f_cosmic_all  = charge in ANY cosmic slice / photon charge
  cosmic_purity = f_cosmic_max / f_cosmic_all  (of the cosmic part, how much is
                  in the top slice; ~1 => the photon owns one coherent cosmic slice)

Reads merged_sp truth + the sliceid_event*.h5 sidecars via the visualizer's
loader.  Pure h5py/numpy/scipy.

    python analyze_photon_slices.py \
        --browse-list inputlists/merged_sp_valdata_singlephoton.txt \
        --slice-id-dir output/slice_ids_singlephoton
"""
import os
import sys
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import visualize_cascade_output as V   # noqa: E402


def photon_slice_partition(ev):
    """Return (qtot, cat_frac dict, {cosmic_id: frac}) for the visible photon,
    or None. cat_frac keys: prefiltered/ghost/unclustered/nu."""
    gam = [r for r in ev["particle_rows"]
           if r["origin"] == V.ORIGIN_NU and r["name"] == "gamma"]
    if not gam:
        return None
    ph = max(gam, key=lambda r: r["q_true"])
    idx = np.where(ev["td_tid"] == ph["trackid"])[0]
    if idx.size == 0:
        return None
    _, qc = V.dedup_charge(ev["td_pixval"][idx], ev["td_tick"][idx],
                           ev["td_uw"][idx], ev["td_vw"][idx], ev["td_yw"][idx])
    qtot = float(qc.sum())
    if qtot <= 0:
        return None
    sidp = ev["td_sid"][idx]
    cat = {"prefiltered": float(qc[sidp == -3].sum()) / qtot,
           "ghost": float(qc[sidp == -2].sum()) / qtot,
           "unclustered": float(qc[sidp == -1].sum()) / qtot,
           "nu": float(qc[sidp == V.NU_SID].sum()) / qtot}
    cosmic = {int(ci): float(qc[sidp == ci].sum()) / qtot
              for ci in sorted(set(int(x) for x in sidp[sidp >= 0]))}
    return qtot, cat, cosmic, ph["ke"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--browse-list", required=True,
                    help="single-photon merged_sp list")
    ap.add_argument("--slice-id-dir", required=True)
    ap.add_argument("--clean-thresh", type=float, default=0.5,
                    help="cosmic_purity >= this => 'cleanly one cosmic slice'")
    args = ap.parse_args()

    ctx = V.Context(V.DEF_CASCADE_DIR, V.DEF_MSP_LIST, V.DEF_KP_LIST,
                    V.DEF_NURECO_DIR, slice_id_dir=args.slice_id_dir)
    paths = [ln.strip() for ln in open(args.browse_list) if ln.strip()]
    fmax, fall, purity, cats, n_nosid = [], [], [], [], 0
    for p in paths:
        ev = V.load_event(ctx, p)
        if not ev["has_sliceid"]:
            n_nosid += 1
            continue
        res = photon_slice_partition(ev)
        if res is None:
            continue
        _, cat, cosmic, _ke = res
        fa = sum(cosmic.values())
        fm = max(cosmic.values()) if cosmic else 0.0
        fmax.append(fm); fall.append(fa)
        purity.append(fm / fa if fa > 0 else 0.0)
        cats.append(cat)
    n = len(fmax)
    fmax, fall, purity = np.array(fmax), np.array(fall), np.array(purity)
    print(f"\n=== single-photon slice analysis: {n} photons "
          f"({n_nosid} had no sliceid) ===")
    mc = {k: np.mean([c[k] for c in cats]) for k in cats[0]} if cats else {}
    print("mean photon charge by category: " +
          "  ".join(f"{k}={100*v:.0f}%" for k, v in mc.items()) +
          f"  cosmic_any={100*fall.mean():.0f}%")
    print(f"\nlargest single cosmic slice (f_cosmic_max): "
          f"mean {100*fmax.mean():.0f}%  median {100*np.median(fmax):.0f}%")
    for thr in (0.3, 0.5, 0.7):
        print(f"  photons with >= {thr:.0%} of charge in ONE cosmic slice: "
              f"{int((fmax >= thr).sum()):3d}/{n}  ({100*(fmax>=thr).mean():.0f}%)")
    heavy = fall >= 0.3
    print(f"\namong the {int(heavy.sum())} photons with >=30% total cosmic charge:")
    if heavy.any():
        print(f"  cosmic_purity (top slice / all cosmic): mean "
              f"{100*purity[heavy].mean():.0f}%  median "
              f"{100*np.median(purity[heavy]):.0f}%")
        clean = heavy & (purity >= args.clean_thresh)
        print(f"  'cleanly in one cosmic slice' (purity >= "
              f"{args.clean_thresh:.0%}): {int(clean.sum())}/{int(heavy.sum())}")
    # coarse histogram of f_cosmic_max
    edges = [0, .1, .2, .3, .5, .7, 1.01]
    h, _ = np.histogram(fmax, bins=edges)
    print("\nf_cosmic_max histogram:")
    for lo, hi, c in zip(edges[:-1], edges[1:], h):
        print(f"  {lo:.0%}-{min(hi,1):.0%}: {'#'*int(c)} {c}")


if __name__ == "__main__":
    main()
