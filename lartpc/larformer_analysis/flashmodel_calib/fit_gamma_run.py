"""Per-run flash light-yield (gamma) calibration from in-time MIP muons.

The flash prediction is pred_pe = gamma * sum_sp q_sp * Vis(sp). gamma is the
q->PE light-yield scale, which drifts run-to-run (MICROBOONE-NOTE-1120: Run 1
~0%, dropping to ~-25% by Run 3). We fit it PER RUN from a clean minimally-
ionizing muon sample -- MIPs keep q->L in the linear, recombination-stable
regime the model assumes (protons/high-dE/dx have suppressed collected charge
but enhanced scintillation -> biased q/L, so they are vetoed).

Because pred_pe is LINEAR in gamma and is already stored per slice in the
cascade flash tables (at gamma_beam=5.25) next to the observed in-time flash,
the fit needs no re-run:

    gamma(run) = gamma_beam * median_events[ sum_live obs_pe / sum_live pred_pe ]

summed over LIVE PMTs only (dead opdets excluded, run-aware). This MEASURES the
per-run gamma from data already on disk; bake it into the prediction for the
cascade re-run.

Calibration muon selection (reco-CC in-time muon, data-selectable, no truth):
  - nu-stream FV vertex; a primary muon track (LArFormerPID==13) that is the
    longest/dominant track; length > MIN_LEN cm;
  - STOPPING + ENTERING: exactly one endpoint on a TPC boundary face (entry),
    the other inside (Bragg stop);
  - entry endpoint drift-x > X_ENTRY_MIN cm (cathode side) so the out-of-TPC
    segment's light is attenuated before reaching the anode PMTs;
  - proton-poor: veto events with an energetic reco proton (LArFormerPID==2212)
    to avoid high-dE/dx recombination bias.

Caveats: this first pass relies on the median's robustness to residual
contamination (cosmic/hadronic, out-of-time). A pred-vs-obs y/z-centroid + t0
in-time consistency cut and a per-PMT weighted fit are follow-ups.
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak
import h5py

# reuse the RSE->cascade-path cache built for the pi0 study
_PI0 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "physics", "pi0mass_peak")
sys.path.insert(0, _PI0)
from flash_correction import rse_map              # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.dead_channels import dead_opdets_for_run  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "larformer_reco", "viz"))
from pmtpos import getPMTPosByOpDet                # noqa: E402

# PMT (opdet) y,z positions in TPC coords, opdet-indexed (matches the PE arrays)
_PMT_YZ = np.array([getPMTPosByOpDet(i)[1:] for i in range(32)])

TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])


def on_boundary(pt, margin=10.0):
    """True if pt is within `margin` cm of any active-TPC face."""
    return bool(np.any(np.abs(pt - TPC_LO) < margin)
                or np.any(np.abs(pt - TPC_HI) < margin))


def _cache_path(cascade_dir):
    base = ("rse_" + os.path.basename(os.path.dirname(cascade_dir.rstrip("/")))
            + "_" + os.path.basename(cascade_dir.rstrip("/")) + ".npz")
    p = os.path.join(_PI0, base)
    return p if os.path.exists(p) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), base)


def _centroid(pe, live):
    """PE-weighted (y,z) centroid over live PMTs, or (nan,nan)."""
    w = pe * live
    s = w.sum()
    return (_PMT_YZ[:, 0] @ w / s, _PMT_YZ[:, 1] @ w / s) if s > 0 \
        else (np.nan, np.nan)


def nu_slice_pred_obs(path, dead):
    """For the cascade file's nu slice, return dict with sum_live pred/obs PE
    and the pred vs obs (y,z) light centroids (for the in-time spatial match),
    or None. Live = all opdets except `dead`."""
    live = np.ones(32, bool)
    if dead:
        live[list(dead)] = False
    try:
        with h5py.File(path, "r") as f:
            if "slices" not in f or "flash" not in f \
                    or "observed_pe" not in f["flash"]:
                return None
            labs = [l.decode() if isinstance(l, bytes) else str(l)
                    for l in f["slices/label"][()]]
            if "nu" not in labs:
                return None
            j = labs.index("nu")
            pred = np.nan_to_num(np.asarray(f["slices/pred_pe"][()][j], float))
            obs = np.nan_to_num(np.asarray(f["flash/observed_pe"][()], float))
            pcy, pcz = _centroid(pred, live)
            ocy, ocz = _centroid(obs, live)
            return dict(pred=float(pred[live].sum()), obs=float(obs[live].sum()),
                        dy=abs(pcy - ocy), dz=abs(pcz - ocz))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--sample-tag", required=True, help="label for plots/out")
    ap.add_argument("--gamma-beam", type=float, default=5.25)
    ap.add_argument("--min-muon-len", type=float, default=50.0)
    ap.add_argument("--x-entry-min", type=float, default=125.0)
    ap.add_argument("--proton-veto-e", type=float, default=50.0,
                    help="veto events with a primary proton track recoE above")
    ap.add_argument("--min-pred-pe", type=float, default=50.0,
                    help="require sum_live pred_pe above (drop empty slices)")
    ap.add_argument("--match-dz", type=float, default=100.0,
                    help="in-time spatial match: |pred-obs| z-centroid < this "
                         "cm (confirm the muon IS the observed-flash source; "
                         "spatial, so unbiased for the PE-scale fit)")
    ap.add_argument("--match-dy", type=float, default=60.0)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "vtxX",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE",
                  "trackStartPosX", "trackStartPosY", "trackStartPosZ",
                  "trackEndPosX", "trackEndPosY", "trackEndPosZ"])
    n = len(a["run"])
    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])
    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))

    # ---- select calibration muons -----------------------------------------
    sel_idx, sel_xentry = [], []
    for i in np.nonzero(vok)[0]:
        pid = ak.to_numpy(a["trackLArFormerPID"][i])
        sec = ak.to_numpy(a["trackIsSecondary"][i])
        e = ak.to_numpy(a["trackRecoE"][i])
        if not len(pid):
            continue
        # proton veto: any primary energetic proton -> high-dE/dx, skip event
        prot = (pid == 2212) & (sec == 0) & (e > args.proton_veto_e)
        if prot.any():
            continue
        mu = (pid == 13) & (sec == 0)
        if not mu.any():
            continue
        sp = np.stack([ak.to_numpy(a[f"trackStartPos{c}"][i]) for c in "XYZ"], 1)
        ep = np.stack([ak.to_numpy(a[f"trackEndPos{c}"][i]) for c in "XYZ"], 1)
        length = np.linalg.norm(ep - sp, axis=1)
        cand = np.nonzero(mu & (length > args.min_muon_len))[0]
        if not len(cand):
            continue
        k = cand[np.argmax(length[cand])]        # dominant (longest) muon
        s_bd, e_bd = on_boundary(sp[k]), on_boundary(ep[k])
        if s_bd == e_bd:                          # need exactly one end on wall
            continue
        entry = sp[k] if s_bd else ep[k]          # boundary end = entry
        if entry[0] <= args.x_entry_min:          # cathode-side entry only
            continue
        sel_idx.append(int(i)); sel_xentry.append(float(entry[0]))
    sel_idx = np.array(sel_idx, int)
    print(f">>> {args.sample_tag}: {len(sel_idx)} calibration muons of "
          f"{int(vok.sum())} nu-stream vertices")

    # ---- pred/obs per selected muon (nu-slice, live PMTs, run-aware dead) --
    rmap = rse_map(args.cascade_dir, _cache_path(args.cascade_dir))
    pred, obs, xent, runs = [], [], [], []
    n_nomatch = 0
    for i, xe in zip(sel_idx, sel_xentry):
        p = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p is None:
            continue
        dead = dead_opdets_for_run(int(run[i]))
        r = nu_slice_pred_obs(p, dead)
        if r is None or r["pred"] < args.min_pred_pe or r["obs"] <= 0:
            continue
        # in-time spatial match: the muon's predicted light must land where the
        # observed flash is (else it's not the source -> uncorrelated ratio)
        if not (r["dz"] < args.match_dz and r["dy"] < args.match_dy):
            n_nomatch += 1
            continue
        pred.append(r["pred"]); obs.append(r["obs"])
        xent.append(xe); runs.append(int(run[i]))
    print(f">>> spatial in-time match: kept {len(pred)}, dropped {n_nomatch} "
          "(pred/obs centroid mismatch)")
    pred = np.array(pred); obs = np.array(obs); xent = np.array(xent)
    ratio = obs / pred
    if not len(ratio):
        raise SystemExit("no calibration muons survived pred/obs extraction")
    gamma_fit = args.gamma_beam * float(np.median(ratio))
    lo, hi = np.percentile(ratio, [16, 84])
    print(f">>> {args.sample_tag}: N={len(ratio)} | median obs/pred="
          f"{np.median(ratio):.3f} (16-84%: {lo:.2f}-{hi:.2f}) | "
          f"gamma_beam {args.gamma_beam:.3f} -> gamma_fit {gamma_fit:.3f}")

    if args.out:
        np.savez(args.out, pred=pred, obs=obs, ratio=ratio, x_entry=xent,
                 run=np.array(runs), gamma_beam=args.gamma_beam,
                 gamma_fit=gamma_fit)

    # ---- plots -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(np.clip(ratio, 0, 4), bins=40, color="#1f77b4", alpha=0.85)
    ax[0].axvline(np.median(ratio), color="r", ls="--",
                  label=f"median {np.median(ratio):.2f}")
    ax[0].axvline(1.0, color="0.4", ls=":")
    ax[0].set(xlabel="obs / pred PE (live PMTs)", ylabel="muons",
              title=f"{args.sample_tag}: gamma_fit={gamma_fit:.2f}\n"
                    f"(gamma_beam={args.gamma_beam:.2f}, N={len(ratio)})")
    ax[0].legend(fontsize=8)
    hi_pe = np.percentile(np.r_[pred, obs], 99)
    ax[1].plot([0, hi_pe], [0, hi_pe], "0.4", ls=":")
    ax[1].plot([0, hi_pe], [0, hi_pe / np.median(ratio)], "r--", lw=1,
               label="fit")
    ax[1].scatter(obs, pred, s=6, alpha=0.4)
    ax[1].set(xlabel="observed PE", ylabel="predicted PE",
              xlim=(0, hi_pe), ylim=(0, hi_pe),
              title="pred vs obs (per muon)")
    ax[1].legend(fontsize=8)
    # ratio vs drift-x of the entry point (attenuation residual check)
    xb = np.linspace(args.x_entry_min, TPC_HI[0], 8)
    xc = 0.5 * (xb[:-1] + xb[1:])
    med = [np.median(ratio[(xent >= lo) & (xent < hi_)]) if
           ((xent >= lo) & (xent < hi_)).any() else np.nan
           for lo, hi_ in zip(xb[:-1], xb[1:])]
    ax[2].scatter(xent, np.clip(ratio, 0, 4), s=6, alpha=0.3)
    ax[2].plot(xc, med, "ro-", label="median")
    ax[2].axhline(np.median(ratio), color="0.4", ls=":")
    ax[2].set(xlabel="entry drift-x [cm]", ylabel="obs/pred",
              ylim=(0, 4), title="attenuation residual (ratio vs x)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/gamma_fit_{args.sample_tag}.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}/gamma_fit_{args.sample_tag}.png")


if __name__ == "__main__":
    main()
