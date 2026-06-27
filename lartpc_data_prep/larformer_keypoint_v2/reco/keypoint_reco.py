"""Greedy iterative keypoint reconstruction from a dense score field (v1).

PyTorch port of larflow ``larflow::reco::KeypointReco`` (KeypointReco.cxx),
re-targeted from the larmatch ``larflow3dhit`` score columns to the attempt-2
LArFormer cascade's dense per-level score maps. See ``../keypoint_reco_spec.md``.

Algorithm (per score map, all in detector cm):

    residual = scores.clone()
    loop:
        i* = argmax(residual);  smax = residual[i*]
        if smax < score_thresh: break
        nbr = points within radius_cm of coords[i*]
        if |nbr| < min_neighbors: residual[i*] = 0; continue   # lone hot point
        mu, A = fit_fixed_sigma_gaussian(coords[nbr], residual[nbr])   # 4.1
        clamp mu to coords[i*] +/- clamp_extrap_sigma * sigma
        emit keypoint(mu, smax, ...)
        residual -= A_sub * exp(-||coords - mu||^2 / (2 sigma^2))      # peel
        residual.clamp_(min=0)

The subtraction is what enforces non-maximum suppression: a found peak's
Gaussian is removed before the next argmax, so the next iteration finds the next
distinct keypoint rather than a neighbor of the same one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from . import gaussian_fit as gf

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:                                   # pragma: no cover
    _HAVE_KDTREE = False


@dataclass
class KeypointRecoParams:
    radius_cm: float = 10.0          # neighborhood isolation radius
    sigma_cm: float = 3.0            # fixed Gaussian width (= GT label sigma)
    score_thresh: float = 0.67       # stop when running max < this
    max_keypoints: int = 64          # safety cap per map
    min_neighbors: int = 4           # below this: suppress the point, no fit
    fit_radius_cm: Optional[float] = None  # FIT neighborhood (None -> 2*sigma);
    #   kept tighter than radius_cm so a neighbor peak inside the isolation
    #   radius does not bias the local Gaussian fit (spec 4.2).
    amplitude: str = "peak"          # subtraction amplitude: "peak" | "fit"
    fit_method: str = "nls"          # "nls" | "loglinear" | "centroid"
    fit_weight_power: float = 2.0    # score^p weighting in the fits
    nls_iters: int = 8
    clamp_extrap_sigma: float = 3.0  # cap ||mu - argmax|| at this * sigma
    onesided_sigma: float = 1.0      # nbr-centroid offset > this*sigma -> flag
    subtract_radius_sigma: float = 4.0  # restrict the subtraction for speed
    device: str = "cpu"

    def __post_init__(self):
        if self.amplitude not in ("peak", "fit"):
            raise ValueError(f"amplitude must be peak|fit, got {self.amplitude}")
        if self.fit_method not in ("nls", "loglinear", "centroid"):
            raise ValueError(f"bad fit_method {self.fit_method}")


@dataclass
class Keypoint:
    """One reconstructed keypoint (the v1 subset of the C++ ``KPCluster``)."""
    pos_cm: np.ndarray               # (3,) fitted mean, detector cm
    peak_score: float                # residual score at detection (~ max_score)
    fit_rmse: float                  # fit quality (residual vs fitted Gaussian)
    fit_rsqr: float
    n_support: int                   # neighborhood size used for the fit
    extrapolated: bool               # peak off-support / one-sided (low trust)
    method: str                      # fitter that produced pos_cm

    def as_dict(self):
        return dict(pos_cm=self.pos_cm, peak_score=self.peak_score,
                    fit_rmse=self.fit_rmse, fit_rsqr=self.fit_rsqr,
                    n_support=self.n_support, extrapolated=self.extrapolated,
                    method=self.method)


class KeypointRecoTorch:
    """Greedy peak-finder on dense keypoint score maps (one map at a time)."""

    def __init__(self, params: Optional[KeypointRecoParams] = None):
        self.p = params or KeypointRecoParams()

    # -- the fit dispatch + robustness guards (spec 4.1) -------------------
    def _fit_neighborhood(self, coords_c, scores):
        """Fit one peak in a CENTERED neighborhood frame (origin = argmax).

        Returns (mu_centered (3,) tensor, amplitude, rmse, rsqr, method).
        Falls back loglinear/centroid -> centroid on any numerical failure.
        """
        p = self.p
        order = [p.fit_method]
        for m in ("loglinear", "centroid"):           # graceful degradation
            if m not in order:
                order.append(m)
        last_exc = None
        for method in order:
            try:
                if method == "nls":
                    # seed from the cheap closed-form when possible
                    seed = None
                    try:
                        seed = gf.fit_gaussian_loglinear(
                            coords_c, scores, p.sigma_cm,
                            weight_power=p.fit_weight_power).mu
                    except Exception:
                        seed = None
                    r = gf.fit_gaussian_nls(
                        coords_c, scores, p.sigma_cm, mu0=seed,
                        weight_power=p.fit_weight_power, iters=p.nls_iters)
                elif method == "loglinear":
                    r = gf.fit_gaussian_loglinear(
                        coords_c, scores, p.sigma_cm,
                        weight_power=p.fit_weight_power)
                else:
                    r = gf.fit_gaussian_centroid(
                        coords_c, scores, p.sigma_cm,
                        weight_power=p.fit_weight_power)
                if torch.all(torch.isfinite(r.mu)):
                    return r
            except Exception as exc:                   # pragma: no cover
                last_exc = exc
                continue
        # absolute last resort: the argmax itself (origin in centered frame)
        rmse, rsqr = gf._goodness(
            torch.as_tensor(coords_c, dtype=torch.float64),
            torch.as_tensor(scores, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), float(scores.max()),
            self.p.sigma_cm)
        return gf.FitResult(mu=torch.zeros(3, dtype=torch.float64),
                            amplitude=float(scores.max()), rmse=rmse,
                            rsqr=rsqr, method="argmax", n=int(len(scores)))

    def reconstruct(self, coords_cm, scores,
                    max_keypoints: Optional[int] = None) -> List[Keypoint]:
        """Greedy peeling on ONE score map. coords_cm (M,3), scores (M,)."""
        p = self.p
        coords_np = np.ascontiguousarray(
            np.asarray(coords_cm, dtype=np.float64).reshape(-1, 3))
        m = coords_np.shape[0]
        residual = torch.as_tensor(
            np.asarray(scores, dtype=np.float64).reshape(-1),
            dtype=torch.float64, device=p.device)
        if m == 0 or residual.numel() == 0:
            return []
        cap = p.max_keypoints if max_keypoints is None else int(max_keypoints)
        s2 = p.sigma_cm * p.sigma_cm
        fit_radius = (p.fit_radius_cm if p.fit_radius_cm is not None
                      else 2.0 * p.sigma_cm)

        tree = cKDTree(coords_np) if _HAVE_KDTREE else None
        coords_t = torch.as_tensor(coords_np, dtype=torch.float64,
                                   device=p.device)

        def ball(center, radius):
            if tree is not None:
                return np.asarray(
                    tree.query_ball_point(np.asarray(center), radius),
                    dtype=np.int64)
            d2 = ((coords_t - torch.as_tensor(center, dtype=torch.float64,
                                              device=p.device)) ** 2).sum(1)
            return torch.nonzero(d2 <= radius * radius).reshape(-1).cpu().numpy()

        out: List[Keypoint] = []
        # hard backstop on iterations (suppress + emit both make progress)
        for _ in range(8 * cap + m + 16):
            if len(out) >= cap:
                break
            imax = int(torch.argmax(residual).item())
            smax = float(residual[imax].item())
            if smax < p.score_thresh:
                break
            x0 = coords_np[imax]
            # FIT on a tight neighborhood (localizes the Gaussian; spec 4.2),
            # isolation/subtraction use the wider radii below.
            nbr = ball(x0, fit_radius)
            # only positive-residual points inform the fit
            nbr = nbr[np.asarray(residual[nbr].cpu()) > 1e-9] if nbr.size else nbr
            if nbr.size < p.min_neighbors:
                residual[imax] = 0.0                   # lone hot point
                continue

            cc = coords_t[nbr] - torch.as_tensor(x0, dtype=torch.float64,
                                                 device=p.device)
            yy = residual[nbr]
            r = self._fit_neighborhood(cc, yy)
            mu_c = r.mu.to(p.device)

            # --- off-support / one-sided guards (spec 4.1) ---------------
            limit = p.clamp_extrap_sigma * p.sigma_cm
            clamped = bool(torch.any(mu_c.abs() > limit))
            mu_c = torch.clamp(mu_c, -limit, limit)
            wnb = yy.clamp_min(0.0) ** p.fit_weight_power
            nb_centroid = (cc * wnb.reshape(-1, 1)).sum(0) / (
                float(wnb.sum()) + 1e-12)
            onesided = bool(float(torch.linalg.vector_norm(nb_centroid))
                            > p.onesided_sigma * p.sigma_cm)
            mu = mu_c + torch.as_tensor(x0, dtype=torch.float64,
                                        device=p.device)

            # subtraction amplitude: peak (C++) is robust; "fit" clamped
            a_sub = smax if p.amplitude == "peak" else min(
                float(r.amplitude), 1.5 * smax)
            a_sub = max(a_sub, 1e-6)

            out.append(Keypoint(
                pos_cm=mu.cpu().numpy().astype(np.float32),
                peak_score=smax, fit_rmse=float(r.rmse),
                fit_rsqr=float(r.rsqr), n_support=int(nbr.size),
                extrapolated=bool(clamped or onesided), method=r.method))

            # --- peel: subtract the fitted Gaussian (restricted support) --
            sub_idx = ball(mu.cpu().numpy(),
                           p.subtract_radius_sigma * p.sigma_cm)
            if sub_idx.size:
                d2 = ((coords_t[sub_idx] - mu.reshape(1, 3)) ** 2).sum(1)
                residual[sub_idx] = torch.clamp(
                    residual[sub_idx] - a_sub * torch.exp(-0.5 * d2 / s2),
                    min=0.0)
            # guarantee progress at the argmax (avoid infinite loop if the
            # fit pushed mu away and the peak barely moved)
            if float(residual[imax].item()) >= smax:
                residual[imax] = 0.0

        return out

    # -- event-level convenience -----------------------------------------
    def reconstruct_event(self, score_maps: Dict[str, dict]) -> Dict[str, dict]:
        """Run reco on every head in a score-map dict.

        ``score_maps`` = ``{name: {"coords_cm": (M,3), "score": (M,),
        "level": str, "kp_types": [int]}}`` (the ``io.load_score_maps`` output).

        Returns ``{name: {"keypoints": [Keypoint...], "level", "kp_types",
        "is_nu": bool}}``. The nu-vertex head (kp_types containing 0, or named
        ``nu_vertex``) is capped at a few peaks; other heads use the param cap.
        """
        result = {}
        for name, sm in score_maps.items():
            kp_types = list(sm.get("kp_types", []) or [])
            is_nu = (name == "nu_vertex") or (0 in kp_types)
            cap = 3 if is_nu else self.p.max_keypoints
            kps = self.reconstruct(sm["coords_cm"], sm["score"],
                                   max_keypoints=cap)
            result[name] = dict(keypoints=kps, level=sm.get("level", ""),
                                kp_types=kp_types, is_nu=is_nu)
        return result


def best_nu_vertex(event_result: Dict[str, dict]) -> Optional[Keypoint]:
    """Highest-score peak among the nu-vertex head(s), or None."""
    best = None
    for r in event_result.values():
        if not r.get("is_nu"):
            continue
        for kp in r["keypoints"]:
            if best is None or kp.peak_score > best.peak_score:
                best = kp
    return best
