"""Flash-vs-PE chi-2 with systematic floor + OOB rejection.

The chi-2 model is Neyman-style (denominator = PE_obs²-derived variance,
i.e. uncertainty estimated from observation, not from prediction) with
two regularizers:

  - `eps`:     small absolute floor on the per-PMT variance so dark
                PMTs (PE_obs == 0) don't blow up the chi-2.
  - `f_sys`:   fractional systematic uncertainty on the PE_obs scale.
                Added in quadrature with sqrt(PE_obs). 0.10–0.15 absorbs
                photon-library + detector + readout systematics that
                aren't in the per-PMT Poisson statistic.

Per-PMT term:

    χ²_i = (PE_obs_i − PE_pred_i)² / (PE_obs_i + (f_sys · PE_obs_i)² + ε)

Total χ² is Σ_i. Reduced χ² = χ² / dof where dof = N_PMTs - 1.

`oob_frac` is computed by the CALLER (the analysis script does the
drift correction, then this module's helper checks fraction outside
the TPC bounds). When `oob_frac > threshold`, the slice is REJECTED:
the chi-2 returns NaN so downstream ranking skips it cleanly.

Conventions used elsewhere in the repo (kept identical here for
consistency with `visualize_slice_flash_match.py`):
  - N_PMTs = 32 (active opdets only — opdet IDs 0..31)
  - drift velocity ≈ 0.1098 cm/μs at 70 kV
  - PE arrays are 1-D (N_PMTs,) float32 in OpDet order
"""

import numpy as np


N_PMTS = 32
DRIFT_VELOCITY_CM_PER_US = 0.1098

# Active-TPC bounds (the standard MicroBooNE active volume).
# Used to compute oob_frac per slice after drift correction.
TPC_X_MIN, TPC_X_MAX = 0.0, 256.35
TPC_Y_MIN, TPC_Y_MAX = -116.5, 116.5
TPC_Z_MIN, TPC_Z_MAX = 0.0, 1036.8


def oob_mask(pos_cm: np.ndarray,
             margin_cm: float = 0.0) -> np.ndarray:
    """Bool mask, True for points OUTSIDE the active TPC volume.

    Args:
        pos_cm:    (N, 3) float positions in cm, post-drift-correction.
        margin_cm: shrink the TPC bounds by this much on each side.
                   Default 0 = active-volume edges; set ~10 for a
                   fiducial cut.
    """
    if pos_cm.size == 0:
        return np.zeros(0, dtype=bool)
    return ~(
        (pos_cm[:, 0] >= TPC_X_MIN + margin_cm) &
        (pos_cm[:, 0] <= TPC_X_MAX - margin_cm) &
        (pos_cm[:, 1] >= TPC_Y_MIN + margin_cm) &
        (pos_cm[:, 1] <= TPC_Y_MAX - margin_cm) &
        (pos_cm[:, 2] >= TPC_Z_MIN + margin_cm) &
        (pos_cm[:, 2] <= TPC_Z_MAX - margin_cm)
    )


def drift_correct(pos_cm: np.ndarray, flash_t0_us: float,
                  v_drift: float = DRIFT_VELOCITY_CM_PER_US) -> np.ndarray:
    """Apply x_true = x_apparent - v_drift * t_flash to a copy of pos.

    Same convention as `PhotonLibLookup.predict_flash_pairs`. For a flash
    at t = 0 (in-time, beam-aligned) this is a no-op.
    """
    if pos_cm.size == 0:
        return pos_cm.copy()
    out = pos_cm.copy().astype(np.float32, copy=False)
    out[:, 0] = out[:, 0] - float(v_drift) * float(flash_t0_us)
    return out


def neyman_chi2(
    pe_pred: np.ndarray,
    pe_obs: np.ndarray,
    f_sys: float = 0.10,
    eps: float = 1.0,
    dead_opdets=None,
    return_per_pmt: bool = False,
):
    """Neyman χ² with systematic floor.

    χ²_i = (obs - pred)² / (obs + (f_sys · obs)² + ε)
    χ²   = Σ_{i ∉ dead} χ²_i

    Args:
        pe_pred:        (N_PMTs,) predicted PE.
        pe_obs:         (N_PMTs,) observed PE.
        f_sys:          fractional systematic. Default 0.10.
        eps:            absolute variance floor. Default 1.0 PE² —
                        prevents dark-PMT division blowups while not
                        masking real signal differences.
        dead_opdets:    iterable of opdet indices to EXCLUDE from the sum
                        (dead/disabled PMTs read obs≈0 but the PhotonLib
                        still predicts light there, so the Neyman term
                        pred²/ε would spuriously dominate — see
                        lartpc/flashmatch/dead_channels.py). Their per-PMT
                        contribution is zeroed. None = score all 32.
        return_per_pmt: if True, also return the (N_PMTs,) per-PMT
                        contribution vector for diagnostics (dead channels
                        are 0 there too).

    Returns:
        chi2: float scalar.
        (optional) per_pmt: (N_PMTs,) array.
    """
    pe_pred = np.asarray(pe_pred, dtype=np.float64)
    pe_obs = np.asarray(pe_obs, dtype=np.float64)
    if pe_pred.shape != pe_obs.shape:
        raise ValueError(f"pe_pred/pe_obs shape mismatch: "
                         f"{pe_pred.shape} vs {pe_obs.shape}")
    var = pe_obs + (float(f_sys) * pe_obs) ** 2 + float(eps)
    per_pmt = (pe_obs - pe_pred) ** 2 / var
    if dead_opdets is not None:
        dead = [d for d in dead_opdets if 0 <= int(d) < per_pmt.shape[0]]
        if dead:
            per_pmt = per_pmt.copy()
            per_pmt[dead] = 0.0
    chi2 = float(per_pmt.sum())
    if return_per_pmt:
        return chi2, per_pmt.astype(np.float32)
    return chi2


def chi2_with_oob(
    pos_cm: np.ndarray,
    pe_pred: np.ndarray,
    pe_obs: np.ndarray,
    flash_t0_us: float,
    oob_thresholds: np.ndarray,
    margin_cm: float = 0.0,
    f_sys: float = 0.10,
    eps: float = 1.0,
    dead_opdets=None,
):
    """Per-slice chi-2 across an OOB-threshold sweep.

    Computes `oob_frac` (fraction of post-drift-correction spacepoints
    OUTSIDE the active TPC), then for each threshold T returns NaN if
    `oob_frac > T` else the Neyman chi-2.

    Args:
        pos_cm:         (N_sp, 3) PRE-drift-correction positions. The
                        function applies drift correction internally
                        using flash_t0_us, so callers can stay agnostic.
        pe_pred:        (N_PMTs,) predicted PE for THIS slice (already
                        includes γ and readout factor — produced by
                        `flash_predict.py`).
        pe_obs:         (N_PMTs,) observed in-time flash PE vector.
        flash_t0_us:    in-time flash time (μs vs trigger).
        oob_thresholds: 1-D array of OOB thresholds (e.g. [0.0, 0.05,
                        0.10, 0.20, 0.50]). Output chi-2 array is
                        aligned to this.
        margin_cm:      fiducial margin for the OOB test.
        f_sys, eps:     passed to neyman_chi2.

    Returns dict:
        oob_frac:    float — fraction of pos_cm OUTSIDE TPC after drift.
        chi2:        (len(oob_thresholds),) — chi-2 at each threshold;
                     NaN where oob_frac > threshold.
        n_sp:        int — len(pos_cm).
        n_oob:       int — number of OOB SPs.
    """
    pos_drift = drift_correct(pos_cm, flash_t0_us)
    oob = oob_mask(pos_drift, margin_cm=margin_cm)
    n_sp = int(pos_drift.shape[0])
    n_oob = int(oob.sum())
    oob_frac = (n_oob / n_sp) if n_sp > 0 else 1.0

    chi2_base = neyman_chi2(pe_pred, pe_obs, f_sys=f_sys, eps=eps,
                            dead_opdets=dead_opdets)
    chi2_arr = np.where(
        np.asarray(oob_thresholds, dtype=np.float64) >= oob_frac,
        chi2_base,
        np.nan,
    ).astype(np.float32)
    return {
        "oob_frac": float(oob_frac),
        "chi2":     chi2_arr,
        "n_sp":     n_sp,
        "n_oob":    n_oob,
    }
