"""Method B: variable-tolerance n-D RDP with an MCS-tied tolerance (brief §5).

Turns an *ordered* centerline (the stitched sliding-PCA output of
cluster_fit_stitch) into a minimal polyline whose break points sit at physically
real kinks, while multiple-Coulomb-scattering (MCS) wiggle is absorbed into
straight segments. The novelty vs off-the-shelf RDP: the tolerance is not a
constant -- it is `eps(L)`, a function of the span chord length `L` derived from
the Highland MCS model, so the simplifier knows how much transverse wander is
"just MCS" at each scale.

Removing the MCS/noise wiggle is what makes the path length usable as a *range*
estimate for momentum (raw sliding-PCA length over-counts because it traces the
wiggle).

`eps(L) = kappa * sqrt( sigma_MCS(L)^2 + sigma_reso^2 )`, with
`sigma_MCS(L) ~ L * theta0(L) / sqrt(3)` and Highland `theta0` below.
"""
import numpy as np

from run_elpigraph import point_to_polyline_distance

X0_LAr = 14.0      # radiation length of liquid argon [cm]
SIGMA_RESO = 0.3   # per-point reco resolution floor [cm] (~3 mm spacing, brief §2.0)


def beta_from_p(p_mev, mass_mev):
    """beta = p / E = p / sqrt(p^2 + m^2)."""
    if p_mev is None or not np.isfinite(p_mev) or p_mev <= 0:
        return float("nan")
    return float(p_mev / np.sqrt(p_mev ** 2 + mass_mev ** 2))


def highland_theta0(p_mev, beta, x_cm, X0=X0_LAr, z=1):
    """Highland multiple-scattering RMS angle [rad] for path length x in LAr."""
    if p_mev <= 0 or beta <= 0 or x_cm <= 0:
        return 0.0
    ln_arg = x_cm * z * z / (X0 * beta * beta)
    corr = 1.0 + 0.038 * np.log(max(ln_arg, 1e-6))
    corr = max(corr, 0.1)                       # guard: bracket -> negative for tiny x
    return (13.6 / (beta * p_mev)) * z * np.sqrt(x_cm / X0) * corr


def make_mcs_tolerance(p_mev, beta, kappa=3.0, sigma_reso=SIGMA_RESO,
                       X0=X0_LAr, z=1):
    """Return `tol(L)` giving the RDP tolerance [cm] for a span of chord length L.

    p_mev/beta set the MCS scale (momentum_source = 'truth' supplies them; a fixed
    p is the fallback). kappa is the significance multiplier (higher -> keep fewer,
    sharper kinks). If p/beta are unusable, tolerance collapses to kappa*sigma_reso
    (pure detector-resolution floor)."""
    def tol(L):
        L = max(float(L), 1e-3)
        if not (np.isfinite(p_mev) and p_mev > 0 and np.isfinite(beta) and beta > 0):
            return kappa * sigma_reso
        theta0 = highland_theta0(p_mev, beta, L, X0=X0, z=z)
        sigma_mcs = L * theta0 / np.sqrt(3.0)
        return kappa * float(np.sqrt(sigma_mcs ** 2 + sigma_reso ** 2))
    return tol


def rdp_variable(P, tol):
    """n-D Ramer-Douglas-Peucker with a callable, span-dependent tolerance.

    P: (M,d) ORDERED polyline vertices. tol: callable L->eps. Returns the kept
    subset (M'<=M, d) including both endpoints. Iterative (explicit stack) to
    avoid recursion limits on long centerlines. Uses 3D point-to-segment distance.
    """
    P = np.asarray(P, np.float64)
    n = len(P)
    if n < 3:
        return P.copy()
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = np.array([P[i], P[j]])
        d = point_to_polyline_distance(P[i + 1:j], seg)
        kmax = int(d.argmax())
        dmax = float(d[kmax])
        k = i + 1 + kmax
        L = float(np.linalg.norm(P[j] - P[i]))
        if dmax > tol(L):
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return P[keep]


def path_length(P):
    P = np.asarray(P)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
