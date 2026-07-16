"""Analysis-level flash-chi2 correction: recompute the nu-slice Neyman chi2
with dead PMTs masked, from the per-PMT arrays already stored in the cascade
files (slices/pred_pe, flash/observed_pe). Approximates the effect of the
proper flash-prediction fix WITHOUT re-running the GPU cascade.

Motivation: in run3 (MC overlay + EXT), opdet 15 is dead (observed PE == 0),
but the PhotonLib prediction is not masked there, so the Neyman term
(0 - pred)^2 / eps ~ pred^2 dominates the chi2 and inflates genuine-neutrino
chi2 into the cosmic range. Masking opdet 15 (in ALL samples, incl. run1 beam
data, for a fair comparison) collapses neutrino chi2 (~9000 -> ~130) while
cosmic chi2 stays high (~22000) -> the cosmic background then separates.

CAVEAT: this only corrects the nu-slice chi2 of the already-reconstructed
nu-stream vertex. The proper fix also changes which slices the cascade
flashmatch stream selects (chi2 ranking), which needs a cascade re-run.

Cascade files are matched to ntuple events by (run, subrun, event) via a
scanned map (cached per cascade dir) -- entry-index == cascade-index does NOT
hold for the beam sample (the veto5M list surgery shifted the indexing).
"""
import os

import numpy as np
import h5py

DEAD_DEFAULT = (15,)          # opdet 15: dead in run3 (MC + EXT)


def neyman_masked(pred, obs, dead=DEAD_DEFAULT, f_sys=0.10, eps=1.0):
    """Dead-PMT-masked Neyman chi2 (same var floor as flash_chi2.neyman_chi2)."""
    pred = np.nan_to_num(np.asarray(pred, np.float64))
    obs = np.nan_to_num(np.asarray(obs, np.float64))
    var = obs + (f_sys * obs) ** 2 + eps
    per_pmt = (obs - pred) ** 2 / var
    if dead:
        m = np.ones(per_pmt.shape[0], bool)
        m[list(dead)] = False
        per_pmt = per_pmt[m]
    return float(per_pmt.sum())


def _scan_rse(cascade_dir):
    """{(run,subrun,event): path} over the nu-slice cascade files. Recurses
    (os.walk) so it handles both a flat dir and the index-tree layout."""
    m = {}
    for root, _dirs, files in os.walk(cascade_dir):
        for n in files:
            if (not n.startswith("keypoint2_event") or not n.endswith("_0.h5")
                    or n.endswith("_fm_0.h5")):
                continue
            path = os.path.join(root, n)
            try:
                with h5py.File(path, "r") as f:
                    key = (int(f.attrs["run"]), int(f.attrs["subrun"]),
                           int(f.attrs["event"]))
                m[key] = path
            except Exception:
                continue
    return m


def rse_map(cascade_dir, cache_npz=None):
    """Load or build the (run,subrun,event) -> cascade-path map, cached."""
    if cache_npz and os.path.exists(cache_npz):
        z = np.load(cache_npz, allow_pickle=True)
        return {tuple(int(x) for x in k): str(p)
                for k, p in zip(z["keys"], z["paths"])}
    m = _scan_rse(cascade_dir)
    if cache_npz and m:
        try:
            np.savez(cache_npz, keys=np.array(list(m.keys()), np.int64),
                     paths=np.array(list(m.values())))
        except Exception:
            pass
    return m


def corrected_chi2_by_rse(cascade_dir, run, subrun, event, entries,
                          dead=DEAD_DEFAULT, cache_npz=None, saturation=False,
                          max_saturated=4):
    """{entry_idx: masked nu-slice chi2} matched by (run,subrun,event); NaN if
    the event has no cascade file, no 'nu' slice, or no observed flash.

    saturation: also mask saturated PMTs (lartpc/flashmatch/saturation.py) --
        tubes reading far below their brightest neighbour, which the run3
        optical sim produces and which otherwise contribute ~pred^2/eps to the
        chi2. Derived from the OBSERVED flash only, so it is applied identically
        to every sample and is not circular. Use this to give MC and data the
        SAME treatment: the MC cascade may already bake the mask into its stored
        chi2 while data/EXT cascades predate it.
    """
    m = rse_map(cascade_dir, cache_npz)
    if saturation:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
        from lartpc.flashmatch.saturation import find_saturated
    out = {}
    for i in entries:
        i = int(i)
        p = m.get((int(run[i]), int(subrun[i]), int(event[i])))
        val = np.nan
        if p is not None:
            try:
                with h5py.File(p, "r") as f:
                    if ("slices" in f and "flash" in f
                            and "observed_pe" in f["flash"]):
                        labs = [l.decode() if isinstance(l, bytes) else str(l)
                                for l in f["slices/label"][()]]
                        if "nu" in labs:
                            j = labs.index("nu")
                            obs = f["flash/observed_pe"][()]
                            msk = tuple(dead)
                            if saturation:
                                msk = tuple(sorted(set(msk) | set(
                                    find_saturated(obs, dead=dead,
                                                   max_masked=max_saturated))))
                            val = neyman_masked(f["slices/pred_pe"][()][j],
                                                obs, msk)
            except Exception:
                val = np.nan
        out[i] = val
    return out
