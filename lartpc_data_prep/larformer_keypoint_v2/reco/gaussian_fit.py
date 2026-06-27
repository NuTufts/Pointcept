"""Gaussian-peak fitters for keypoint reconstruction (v1, score-field only).

All fits localize the MEAN of an isotropic 3D Gaussian of FIXED width ``sigma``
(= the 3 cm width the GT keypoint scores were generated with,
``lartpc_data_prep/keypoint_labels.py``) from a set of points carrying a score
field ``y_j ~ A * exp(-||x_j - mu||^2 / (2 sigma^2))``.

The fitters are written to be robust to the truncated / one-sided support that
the slicer + particle masks routinely produce (track/shower *start* points see
only a half-Gaussian; a masked peak leaves only the descending tail, with the
true peak OUTSIDE the surviving points). See
``../keypoint_reco_spec.md`` section 4.1 for the rationale. In order of
robustness:

* ``fit_gaussian_nls``       — Method 1 (default): direct nonlinear, score-
                               weighted least squares on the Gaussian SHAPE with
                               sigma fixed. Fits shape not centroid (so one-
                               sidedness does not bias it), down-weights the
                               noisy tail, and ``mu`` may leave the point cloud.
* ``fit_gaussian_loglinear`` — Method 2: fixed-sigma, y^2-weighted, JOINT-3D
                               log-linear fit. Closed form; a good seed for
                               Method 1 or a fast standalone fit.
* ``fit_gaussian_centroid``  — score^p-weighted centroid. BIASED under one-sided
                               truncation; degenerate-case fallback only.
* ``caruana_full``           — variable-sigma per-axis Caruana (the C++
                               ``_fit_cluster_CARUANA`` math), kept for parity
                               tests / a broad-peak fallback.

Convention: callers should CENTER ``coords`` near the peak (e.g. subtract the
running-max point) before fitting — it conditions the linear systems and makes
the returned ``mu`` a small offset. The fitters themselves are frame-agnostic.

All functions take/return ``torch`` tensors (float64 internally for
conditioning) and run on whatever device the inputs live on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_EPS = 1e-12


@dataclass
class FitResult:
    """One Gaussian-peak fit. ``mu`` is in the SAME frame as the input coords."""
    mu: torch.Tensor          # (3,) fitted mean
    amplitude: float          # fitted (or seeded) peak amplitude A
    rmse: float               # sqrt(mean (y - f)^2) over the fit points
    rsqr: float               # 1 - SS_res/SS_tot (<=1; can be negative)
    method: str               # which fitter actually produced mu
    n: int                    # number of points used


def _prep(coords, scores):
    coords = torch.as_tensor(coords, dtype=torch.float64)
    scores = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    return coords, scores


def _goodness(coords, scores, mu, amp, sigma):
    """rmse + R^2 of the fitted fixed-sigma Gaussian against the scores."""
    d2 = ((coords - mu.reshape(1, 3)) ** 2).sum(1)
    f = amp * torch.exp(-0.5 * d2 / (sigma * sigma))
    res = scores - f
    ss_res = float((res * res).sum())
    n = scores.numel()
    rmse = float((ss_res / max(n, 1)) ** 0.5)
    ss_tot = float(((scores - scores.mean()) ** 2).sum())
    rsqr = 1.0 - ss_res / ss_tot if ss_tot > _EPS else float("nan")
    return rmse, rsqr


# ---------------------------------------------------------------------------
# Method 2 — fixed-sigma, y^2-weighted, joint-3D log-linear fit
# ---------------------------------------------------------------------------

def fit_gaussian_loglinear(coords, scores, sigma, weight_power=2.0):
    """Closed-form fixed-sigma Gaussian mean.

    With sigma KNOWN the log-parabola curvature is fixed, so moving it to the
    LHS linearizes the problem::

        z_j = ln(y_j) + ||x_j||^2 / (2 sigma^2)
            = (ln A - ||mu||^2 / 2 sigma^2)  +  (mu / sigma^2) . x_j
            =        c0                       +        g . x_j

    A single weighted (w_j = y_j^power) linear least squares in 4 unknowns
    (c0, g in R^3) — JOINTLY in 3D, not per-axis — gives ``mu = g * sigma^2``.
    Points with y_j <= 0 are dropped (ln undefined).
    """
    coords, scores = _prep(coords, scores)
    sigma = float(sigma)
    pos = scores > _EPS
    c = coords[pos]
    y = scores[pos]
    if c.shape[0] < 4:
        raise ValueError("loglinear fit needs >= 4 positive-score points")
    w = y ** weight_power
    z = torch.log(y) + (c * c).sum(1) / (2.0 * sigma * sigma)
    # design D = [1, x, y, z]  (M, 4)
    D = torch.cat([torch.ones(c.shape[0], 1, dtype=c.dtype), c], dim=1)
    Dw = D * w.reshape(-1, 1)
    A = D.transpose(0, 1) @ Dw            # (4,4) normal matrix
    b = Dw.transpose(0, 1) @ z            # (4,)
    # tiny ridge for conditioning (coords pre-centered by the caller)
    A = A + 1e-9 * torch.eye(4, dtype=A.dtype)
    sol = torch.linalg.solve(A, b)
    g = sol[1:4]
    mu = g * (sigma * sigma)
    # amplitude from the intercept: ln A = c0 + ||mu||^2 / 2 sigma^2
    lnA = sol[0] + (mu * mu).sum() / (2.0 * sigma * sigma)
    amp = float(torch.exp(lnA))
    rmse, rsqr = _goodness(coords, scores, mu, amp, sigma)
    return FitResult(mu=mu, amplitude=amp, rmse=rmse, rsqr=rsqr,
                     method="loglinear", n=int(c.shape[0]))


# ---------------------------------------------------------------------------
# Method 1 — direct nonlinear score-weighted fit (sigma fixed)
# ---------------------------------------------------------------------------

def fit_gaussian_nls(coords, scores, sigma, mu0=None, amp0=None,
                     weight_power=2.0, iters=8, lm_damping=1e-3):
    """Gauss-Newton (LM-damped) fit of ``y ~ A exp(-||x-mu||^2/2 sigma^2)``.

    sigma is FIXED; free parameters are ``A`` (scalar) and ``mu`` (3,). Weights
    ``w_j = y_j^power`` emphasise the high-SNR peak and down-weight the noisy
    tail, so a one-sided / truncated sample does not bias ``mu`` the way a
    centroid would. Seeded at ``mu0`` (default: weighted centroid) / ``amp0``
    (default: max score). A handful of iterations suffice.

    Jacobian columns:
        df/dA      = g_j                       with g_j = exp(-r_j^2/2 sigma^2)
        df/dmu_d   = A g_j (x_{j,d} - mu_d) / sigma^2
    """
    coords, scores = _prep(coords, scores)
    sigma = float(sigma)
    s2 = sigma * sigma
    n = scores.numel()
    if n < 4:
        raise ValueError("nls fit needs >= 4 points")
    w = (scores.clamp_min(0.0)) ** weight_power
    if mu0 is None:
        sw = float(w.sum())
        mu = (coords * w.reshape(-1, 1)).sum(0) / (sw + _EPS)
    else:
        mu = torch.as_tensor(mu0, dtype=torch.float64).reshape(3).clone()
    amp = float(scores.max()) if amp0 is None else float(amp0)

    for _ in range(int(iters)):
        diff = coords - mu.reshape(1, 3)              # (M,3)
        r2 = (diff * diff).sum(1)                     # (M,)
        g = torch.exp(-0.5 * r2 / s2)                 # (M,)
        f = amp * g
        res = scores - f                              # (M,)
        # Jacobian (M,4): [dA, dmu_x, dmu_y, dmu_z]
        J = torch.empty(n, 4, dtype=coords.dtype)
        J[:, 0] = g
        J[:, 1:4] = (amp * g).reshape(-1, 1) * diff / s2
        Jw = J * w.reshape(-1, 1)
        JtJ = J.transpose(0, 1) @ Jw                  # (4,4)
        Jtr = Jw.transpose(0, 1) @ res                # (4,)
        JtJ = JtJ + lm_damping * torch.diag(torch.diagonal(JtJ).clamp_min(_EPS))
        JtJ = JtJ + 1e-12 * torch.eye(4, dtype=JtJ.dtype)
        try:
            delta = torch.linalg.solve(JtJ, Jtr)
        except Exception:
            break
        amp = amp + float(delta[0])
        mu = mu + delta[1:4]
        if float(delta.abs().max()) < 1e-7:
            break

    rmse, rsqr = _goodness(coords, scores, mu, amp, sigma)
    return FitResult(mu=mu, amplitude=float(amp), rmse=rmse, rsqr=rsqr,
                     method="nls", n=int(n))


# ---------------------------------------------------------------------------
# Fallback — score^p-weighted centroid (BIASED under truncation)
# ---------------------------------------------------------------------------

def fit_gaussian_centroid(coords, scores, sigma, weight_power=2.0):
    """Score^power-weighted centroid. Mirrors the C++ ``center_avg_pt_v`` path.

    Biased toward the visible mass for one-sided / truncated support — use only
    when the shape fits are not applicable (too few points / singular system).
    """
    coords, scores = _prep(coords, scores)
    w = (scores.clamp_min(0.0)) ** weight_power
    sw = float(w.sum())
    if sw <= _EPS:
        mu = coords.mean(0)
    else:
        mu = (coords * w.reshape(-1, 1)).sum(0) / sw
    amp = float(scores.max())
    rmse, rsqr = _goodness(coords, scores, mu, amp, float(sigma))
    return FitResult(mu=mu, amplitude=amp, rmse=rmse, rsqr=rsqr,
                     method="centroid", n=int(scores.numel()))


# ---------------------------------------------------------------------------
# Parity — variable-sigma per-axis Caruana (the C++ algorithm)
# ---------------------------------------------------------------------------

def caruana_full(coords, scores, sigma_for_goodness=3.0):
    """Per-axis variable-sigma Caruana fit (parity with C++ _fit_cluster_CARUANA).

    Solves, per axis, the 3x3 moment system for the parabola coefficients of
    ``ln(y)`` and returns ``mu_d = -sol1/(2 sol2)``. Kept for unit-test parity
    and as a broad-/unknown-width fallback; the production path fixes sigma.
    """
    coords, scores = _prep(coords, scores)
    pos = scores > _EPS
    c = coords[pos]
    y = scores[pos]
    if c.shape[0] < 3:
        raise ValueError("caruana fit needs >= 3 positive-score points")
    lny = torch.log(y)
    N = float(c.shape[0])
    mu = torch.zeros(3, dtype=torch.float64)
    for d in range(3):
        x = c[:, d]
        x_sum = [float((x ** k).sum()) for k in range(1, 5)]
        A = torch.tensor(
            [[N, x_sum[0], x_sum[1]],
             [x_sum[0], x_sum[1], x_sum[2]],
             [x_sum[1], x_sum[2], x_sum[3]]], dtype=torch.float64)
        b = torch.tensor(
            [float(lny.sum()), float((x * lny).sum()),
             float((x * x * lny).sum())], dtype=torch.float64)
        sol = torch.linalg.solve(A + 1e-12 * torch.eye(3, dtype=A.dtype), b)
        denom = 2.0 * float(sol[2])
        mu[d] = -float(sol[1]) / denom if abs(denom) > _EPS else float(x.mean())
    amp = float(y.max())
    rmse, rsqr = _goodness(coords, scores, mu, amp, float(sigma_for_goodness))
    return FitResult(mu=mu, amplitude=amp, rmse=rmse, rsqr=rsqr,
                     method="caruana_full", n=int(c.shape[0]))
