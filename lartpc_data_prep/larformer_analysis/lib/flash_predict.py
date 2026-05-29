"""PhotonLib wrapper for slice → predicted PE.

Thin wrapper around `pointcept.models.event_slicer.photonlib.PhotonLibLookup`
that:

  - lazily loads the photon library cache once per process
  - takes numpy inputs (positions in cm, per-SP charges) and returns a
     numpy (N_PMTs,) PE vector
  - applies per-producer γ + readout-factor scaling identical to
     `tools/visualize_slice_flash_match.py`

The analysis script computes a flash prediction per slice (per
predicted-slice query plus the GT-baseline slice) and feeds the result
into `flash_chi2.chi2_with_oob(...)` along with the observed in-time
flash PE vector.

Constants:
  PHOTONLIB_DEFAULT_CACHE: production photon-library file (70 kV).

Producer IDs (matched to `prepare_flashinfo_h5.PRODUCER_NAMES`):
    0 = simpleFlashBeam   — long readout window, readout_factor ≈ 1.0
    1 = simpleFlashCosmic — 312.5 ns window, readout_factor < 1
                            (computed via the fast/slow split model)
"""

import math
import os
import sys
from typing import Optional

import numpy as np


# Default photon-library cache (70 kV production). Overridable per-call.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PHOTONLIB_DEFAULT_CACHE = os.path.join(
    _REPO_ROOT, "lartpc_data_prep", "dat", "photonlib_v6_70kV.npz",
)

# Scintillation-readout defaults (match `visualize_slice_flash_match.py`).
DEFAULT_FAST_FRACTION       = 0.3
DEFAULT_TAU_SLOW_US         = 1.6
DEFAULT_WINDOW_COSMIC_US    = 0.3125
DEFAULT_READOUT_FACTOR_BEAM = 1.0
DEFAULT_V_DRIFT_CM_PER_US   = 0.1098
DEFAULT_GAMMA               = 1.0   # placeholder — pass calibrated value


def compute_readout_factor(fast_fraction: float,
                           tau_slow_us: float,
                           window_us: float) -> float:
    """Fraction of scintillation PE collected within a fixed readout window.

    Same model as `visualize_slice_flash_match.compute_readout_factor`:
    fast component fully collected; slow component truncated to the
    window via exp CDF. Beam window is much longer than the slow decay,
    so the beam producer's factor is ~1.0 by default.
    """
    slow_collected = 1.0 - math.exp(-window_us / tau_slow_us)
    return fast_fraction + (1.0 - fast_fraction) * slow_collected


def default_readout_factors() -> tuple:
    """Return (beam, cosmic) readout factors from the defaults above."""
    return (
        DEFAULT_READOUT_FACTOR_BEAM,
        compute_readout_factor(
            DEFAULT_FAST_FRACTION, DEFAULT_TAU_SLOW_US,
            DEFAULT_WINDOW_COSMIC_US,
        ),
    )


# -----------------------------------------------------------------------
# PhotonLib loader (lazy singleton — load is ~seconds, ~hundreds of MB)
# -----------------------------------------------------------------------

_PL_CACHE_KEY: Optional[str] = None
_PL_LOOKUP = None


def get_photonlib(cache_path: str = None,
                  device: str = None,
                  fp16: bool = False):
    """Return a process-singleton `PhotonLibLookup`, loading on first call.

    Re-loads only if `cache_path` differs from the last call (rare).
    The torch dependency is imported lazily so this module is importable
    on hosts without torch (e.g., for `flash_chi2.py` standalone use).
    """
    global _PL_CACHE_KEY, _PL_LOOKUP
    cache_path = cache_path or PHOTONLIB_DEFAULT_CACHE
    if _PL_LOOKUP is not None and _PL_CACHE_KEY == cache_path:
        return _PL_LOOKUP
    # Import here so callers that only need readout helpers don't pay.
    sys.path.insert(0, _REPO_ROOT)
    import torch
    from pointcept.models.event_slicer.photonlib import PhotonLibLookup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # PhotonLibLookup's ctor doesn't take a device kwarg — place via
    # `.to(device)` (it's an nn.Module with registered buffers).
    pl = PhotonLibLookup(cache_path, fp16=fp16).to(device)
    _PL_LOOKUP = pl
    _PL_CACHE_KEY = cache_path
    return _PL_LOOKUP


def select_charge_y_with_uv_fallback_np(pixval: np.ndarray) -> np.ndarray:
    """Numpy port of `photonlib.select_charge_y_with_uv_fallback`.

    Pixval columns are (U, V, Y) charge in the same order the merged-H5
    triplet_data writes them. Y is the collection plane (most reliable);
    when Y is non-positive, fall back to the mean of (U, V).

    Returns (N,) float charge in the same units as the input columns.
    """
    if pixval.ndim != 2 or pixval.shape[1] != 3:
        raise ValueError(f"pixval must be (N, 3); got {pixval.shape!r}")
    u = pixval[:, 0].astype(np.float32)
    v = pixval[:, 1].astype(np.float32)
    y = pixval[:, 2].astype(np.float32)
    out = y.copy()
    bad = y <= 0
    out[bad] = 0.5 * (u[bad] + v[bad])
    return np.clip(out, a_min=0.0, a_max=None)


# -----------------------------------------------------------------------
# Per-slice flash prediction
# -----------------------------------------------------------------------

def predict_slice_pe(
    pos_cm: np.ndarray,
    charge: np.ndarray,
    flash_t0_us: float,
    producer_id: int,
    gamma_by_producer: tuple = (DEFAULT_GAMMA, DEFAULT_GAMMA),
    readout_factor_by_producer: tuple = None,
    photonlib_cache: str = None,
    device: str = None,
    v_drift_cm_per_us: float = DEFAULT_V_DRIFT_CM_PER_US,
) -> np.ndarray:
    """Predicted PE for ONE slice, scaled by per-producer γ + readout factor.

    Drift correction is applied INTERNALLY (`x_true = x_apparent -
    v_drift * t_flash`), matching the convention `flash_chi2.chi2_with_oob`
    uses for OOB rejection. Callers should pass RAW positions (no
    pre-correction).

    Args:
        pos_cm:                (N_sp, 3) detector cm.
        charge:                (N_sp,)  per-SP charge (from
                               `select_charge_y_with_uv_fallback_np`).
        flash_t0_us:           in-time flash t0 (μs vs trigger).
        producer_id:           0=beam, 1=cosmic (selects γ + readout factor).
        gamma_by_producer:     (γ_beam, γ_cosmic). Photons per charge.
        readout_factor_by_producer:
                               (beam, cosmic). When None, uses
                               `default_readout_factors()`.
        photonlib_cache, device: forwarded to `get_photonlib`.
        v_drift_cm_per_us:     drift velocity for the t0 correction.

    Returns:
        pe: (N_PMTs,) float32. All-zero when N_sp == 0.
    """
    import torch
    if pos_cm.shape[0] == 0:
        from .flash_chi2 import N_PMTS
        return np.zeros(N_PMTS, dtype=np.float32)
    if charge.shape[0] != pos_cm.shape[0]:
        raise ValueError(
            f"pos_cm and charge length mismatch: "
            f"{pos_cm.shape[0]} vs {charge.shape[0]}"
        )
    if readout_factor_by_producer is None:
        readout_factor_by_producer = default_readout_factors()
    pid = int(producer_id)
    gamma = float(gamma_by_producer[pid])
    rf = float(readout_factor_by_producer[pid])
    eff_scale = gamma * rf

    pl = get_photonlib(cache_path=photonlib_cache, device=device)
    dev = pl.vis_table.device
    pos = torch.from_numpy(pos_cm.astype(np.float32, copy=False)).to(dev)
    q = torch.from_numpy(charge.astype(np.float32, copy=False)).to(dev)
    # Internal drift correction.
    pos = pos.clone()
    pos[:, 0] = pos[:, 0] - float(v_drift_cm_per_us) * float(flash_t0_us)
    cid = torch.zeros(pos.shape[0], dtype=torch.int64, device=dev)
    pe = pl.predict_flash(pos, q, cid, n_clusters=1)[0]
    pe = (pe * eff_scale).detach().cpu().numpy().astype(np.float32)
    return pe


def predict_many_slices_pe(
    pos_cm: np.ndarray,
    charge: np.ndarray,
    cluster_id: np.ndarray,
    n_clusters: int,
    flash_t0_us: float,
    producer_id: int,
    gamma_by_producer: tuple = (DEFAULT_GAMMA, DEFAULT_GAMMA),
    readout_factor_by_producer: tuple = None,
    photonlib_cache: str = None,
    device: str = None,
    v_drift_cm_per_us: float = DEFAULT_V_DRIFT_CM_PER_US,
) -> np.ndarray:
    """Batched version: one flash-prediction per cluster.

    All clusters share the same flash_t0_us (the in-time flash, applied
    via a single drift correction). For per-cluster t0s use
    `predict_slice_pe` in a loop or extend with `predict_flash_pairs`.

    Returns:
        pe: (n_clusters, N_PMTs) float32.
    """
    import torch
    if pos_cm.shape[0] == 0:
        from .flash_chi2 import N_PMTS
        return np.zeros((n_clusters, N_PMTS), dtype=np.float32)
    if readout_factor_by_producer is None:
        readout_factor_by_producer = default_readout_factors()
    pid = int(producer_id)
    eff_scale = float(gamma_by_producer[pid]) * float(readout_factor_by_producer[pid])

    pl = get_photonlib(cache_path=photonlib_cache, device=device)
    dev = pl.vis_table.device
    pos = torch.from_numpy(pos_cm.astype(np.float32, copy=False)).to(dev).clone()
    pos[:, 0] = pos[:, 0] - float(v_drift_cm_per_us) * float(flash_t0_us)
    q = torch.from_numpy(charge.astype(np.float32, copy=False)).to(dev)
    cid = torch.from_numpy(cluster_id.astype(np.int64, copy=False)).to(dev)
    pe = pl.predict_flash(pos, q, cid, n_clusters=int(n_clusters))
    return (pe * eff_scale).detach().cpu().numpy().astype(np.float32)
