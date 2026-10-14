"""
Semi-analytic photon visibility: a vectorized torch port of larsim's
phot::SemiAnalyticalModel (larsim v10_09_01, used by sbndcode v10_14_02).

    from lartpc.detinfo import load
    sam = SemiAnalyticVisibility.from_detinfo(load("sbnd"))           # all 312 opdets
    sam = SemiAnalyticVisibility.from_detinfo(load("sbnd"), opdets=pmt_ids)
    direct, reflected = sam.visibility(points)                        # (N, D) each

Every number comes from the detinfo cache (schema >= 1.3, /optical_model): the
VUV Gaisser-Hillas and VIS correction tables, the per-opdet type/orientation/size
as the model builds them, cathode centre, active volume, absorption length.

Faithful to the C++ including its numerical quirks (checked against LArSoft
reference values in /validation_optical by tests/test_semianalytic.py):
  - fast_acos (cubic approximation) for every angle -> GH / VIS angle bins
  - interpolate(..., extrapolate=false) returns the LEFT value beyond the right end
  - interpolate2 brackets x once, then interpolates each radial row
  - Rectangle_SolidAngle compares |v.z| against the width for every orientation,
    and uses machine-epsilon "definitely greater" tests (gaps -> 0)
  - GH correction set to 0 unless finite and in [0, 10]
  - optical path: SBND -> a detector sees only its own side of the cathode
Disk PMTs (type 2, elliptic integrals) are not ported; construction raises if present.

Precision: use the default float64. In float32 the 4-term rectangle solid angle
cancels badly: X-ARAPUCA direct visibilities are off by up to ~17% and reflected
visibilities (cathode-plane solid angle) by up to ~15% per channel; only PMT
direct light stays at ~2e-4. (Measured against the LArSoft reference.)
"""
import math

import numpy as np
import torch

_EPS = float(np.finfo(np.float64).eps)
_PI = math.pi

# Omega_Dome_Model apparent-radius corrections (9 angular bins of 10 deg)
_DOME_PAR0 = [0., 0., 0., 0., 0., 0.597542, 1.00872, 1.46993, 2.04221]
_DOME_PAR1 = [0., 0., 0.19569, 0.300449, 0.555598, 0.854939, 1.39166, 2.19141, 2.57732]

OPTICAL_PATHS = ("standard", "sbnd")


# ----------------------------------------------------------------------------
# larsim PhotonPropagationUtils
# ----------------------------------------------------------------------------
def fast_acos(x):
    """phot::fast_acos (cubic approximation, |error| ~ 1e-4 rad)."""
    a3, a2, a1, a0 = (-2.08730442907856008e-02, 7.68769404161671888e-02,
                      -2.12871094165645952e-01, 1.57075835365209659e+00)
    ax = x.abs()
    ret = ((a3 * ax + a2) * ax + a1) * ax + a0
    ret = ret * torch.sqrt(1.0 - ax)
    return torch.where(x >= 0, ret, _PI - ret)


def _bracket(xdata, x):
    """Left index i used by phot::interpolate for x (size >= 2 table)."""
    n = xdata.shape[0]
    i = torch.searchsorted(xdata[1:].contiguous(), x.contiguous(), right=False)
    i = torch.where(x >= xdata[n - 2], torch.full_like(i, n - 2), i)
    return i.clamp(0, n - 2)


def interpolate(xdata, ydata, x, extrapolate, i=None):
    """phot::interpolate, vectorized over x. ydata: (n,) or (..., n) broadcast
    against x (last axis = table). i: optional precomputed bracket."""
    if i is None:
        i = _bracket(xdata, x)
    xl, xr = xdata[i], xdata[i + 1]
    if ydata.dim() == 1:
        yl, yr = ydata[i], ydata[i + 1]
    else:
        yl = torch.gather(ydata, -1, i.unsqueeze(-1)).squeeze(-1)
        yr = torch.gather(ydata, -1, (i + 1).unsqueeze(-1)).squeeze(-1)
    val = yl + (yr - yl) / (xr - xl) * (x - xl)
    if not extrapolate:
        val = torch.where((x < xl) | (x > xr), yl, val)    # sic: yL on both ends
    return val


def interpolate2(xdist, rdist, params, x, r, k):
    """phot::interpolate2. params: (K, R, X) table; x, r, k broadcast to one shape."""
    i = _bracket(xdist, x)                                 # bracket x once
    rows = params[k]                                       # (..., R, X)
    ii = i.unsqueeze(-1).expand(*i.shape, rows.shape[-2])
    xx = x.unsqueeze(-1).expand_as(ii)
    vals = interpolate(xdist, rows, xx, False, ii)         # (..., R)
    return interpolate(rdist, vals, r, False)


def _definitely_greater(a, b):
    d = a - b
    return (d > _EPS) | (d > torch.maximum(a.abs(), b.abs()) * _EPS)


def _approximately_zero(a):
    return a.abs() <= _EPS


# ----------------------------------------------------------------------------
# solid angles
# ----------------------------------------------------------------------------
def _rect_sa(a, b, d):
    """Rectangle_SolidAngle(a, b, d): on-axis rectangle a x b at distance d."""
    aa = a / (2.0 * d)
    bb = b / (2.0 * d)
    aux = (1.0 + aa * aa + bb * bb) / ((1.0 + aa * aa) * (1.0 + bb * bb))
    return 4.0 * fast_acos(torch.sqrt(aux))


def rectangle_solid_angle(h, w, v, orientation):
    """Rectangle_SolidAngle(Dims{h,w}, v, orientation); v = |relative| (..., 3)."""
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    d1 = torch.where(orientation == 0, vy.abs(), vx.abs())
    d2 = torch.where(orientation == 2, vz.abs(), torch.where(orientation == 1, vy.abs(), vx.abs()))
    az = vz.abs()
    h2, w2 = 0.5 * h, 0.5 * w

    c0 = _approximately_zero(d1) & _approximately_zero(vz)
    g1, gz = _definitely_greater(d1, h2), _definitely_greater(az, w2)
    l1, lz = d1 <= h2, az <= w2

    out = torch.zeros_like(d1)
    # case: both outside
    A, B = d1 - h2, az - w2
    v_oo = (_rect_sa(2 * (A + h), 2 * (B + w), d2) - _rect_sa(2 * A, 2 * (B + w), d2)
            - _rect_sa(2 * (A + h), 2 * B, d2) + _rect_sa(2 * A, 2 * B, d2)) * 0.25
    # case: both inside
    A, B = -d1 + h2, -az + w2
    v_ii = (_rect_sa(2 * (h - A), 2 * (w - B), d2) + _rect_sa(2 * A, 2 * (w - B), d2)
            + _rect_sa(2 * (h - A), 2 * B, d2) + _rect_sa(2 * A, 2 * B, d2)) * 0.25
    # case: d1 outside, z inside
    A, B = d1 - h2, -az + w2
    v_oi = (_rect_sa(2 * (A + h), 2 * (w - B), d2) - _rect_sa(2 * A, 2 * (w - B), d2)
            + _rect_sa(2 * (A + h), 2 * B, d2) - _rect_sa(2 * A, 2 * B, d2)) * 0.25
    # case: d1 inside, z outside
    A, B = -d1 + h2, az - w2
    v_io = (_rect_sa(2 * (h - A), 2 * (B + w), d2) - _rect_sa(2 * (h - A), 2 * B, d2)
            + _rect_sa(2 * A, 2 * (B + w), d2) - _rect_sa(2 * A, 2 * B, d2)) * 0.25
    # the C++ if-chain, last branch first so earlier ones win
    out = torch.where(l1 & gz, v_io, out)
    out = torch.where(g1 & lz, v_oi, out)
    out = torch.where(l1 & lz, v_ii, out)
    out = torch.where(g1 & gz, v_oo, out)
    out = torch.where(c0, _rect_sa(h, w, d2), out)
    return out


def dome_solid_angle(distance, theta_deg, radius):
    """Omega_Dome_Model(distance, theta)."""
    j = (theta_deg / 10.0).to(torch.long).clamp(0, 8)
    par0 = torch.tensor(_DOME_PAR0, dtype=distance.dtype, device=distance.device)[j]
    par1 = torch.tensor(_DOME_PAR1, dtype=distance.dtype, device=distance.device)[j]
    r_app = torch.where(distance >= 5 * radius, radius - par1, radius - par0)
    ratio = r_app * r_app / (distance * distance)
    return 2 * _PI * (1 - torch.sqrt(1 - ratio))


def gaisser_hillas(x, p0, p1, p2, p3):
    diff = p1 - p3
    term = torch.pow((x - p3) / diff, diff / p2)
    return p0 * term * torch.exp((p1 - x) / p2)


def _sane_gh(gh):
    return torch.where(torch.isfinite(gh) & (gh >= 0) & (gh <= 10), gh, torch.zeros_like(gh))


# ----------------------------------------------------------------------------
class SemiAnalyticVisibility:
    """Direct (VUV) and reflected (visible) visibilities of a set of optical detectors."""

    def __init__(self, *, centers, types, orientations, h, w, vuv, vis, cathode_centre,
                 cathode_plane_hw, plane_depth, absorption_length, do_reflected=True,
                 include_anode_reflections=False, optical_path="sbnd", opdet_ids=None,
                 dtype=torch.float64, device="cpu", chunk_size=8192):
        if include_anode_reflections:
            raise NotImplementedError("anode reflections are not ported (SBND does not use them)")
        if optical_path not in OPTICAL_PATHS:
            raise ValueError(f"optical_path must be one of {OPTICAL_PATHS}")
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        self.dtype, self.device, self.chunk_size = dtype, device, int(chunk_size)
        self.opdet_ids = (np.arange(len(types)) if opdet_ids is None else np.asarray(opdet_ids))
        self.O = t(centers)
        self.type = ti(types)
        self.orient = ti(orientations)
        if (self.type == 2).any():
            raise NotImplementedError("disk PMTs (type 2) are not ported")
        self._rect_cols = torch.nonzero(self.type == 0).flatten()
        self._dome_cols = torch.nonzero(self.type == 1).flatten()
        self.h, self.w = t(h), t(w)
        self.optical_path = optical_path
        self.do_reflected = bool(do_reflected)
        self.lam = float(absorption_length)
        self.cc = t(cathode_centre)
        self.cath_h, self.cath_w = float(cathode_plane_hw[0]), float(cathode_plane_hw[1])
        self.plane_depth = float(plane_depth)
        # VUV parameterization
        self.radius = float(vuv.get("PMT_radius", 10.16))
        self.d_ang_vuv = float(vuv.get("delta_angulo_vuv", 10))
        self.max_dist = float(vuv.get("MaxPDDistance", 1000.0))
        self.vertical_border = bool(vuv.get("VerticalBorderCorrectionMode", False))
        self.flat = bool(vuv.get("FlatPDCorr", False))
        self.flat_lat = bool(vuv.get("FlatPDCorrLat", False))
        self.dome = bool(vuv.get("DomePDCorr", False))
        if not (self.flat or self.flat_lat or self.dome):
            raise ValueError("no GH parameterization enabled (FlatPDCorr/FlatPDCorrLat/DomePDCorr)")
        self.gh = {}
        for kind, flag in (("flat", self.flat), ("flat_lateral", self.flat_lat), ("dome", self.dome)):
            if flag:
                self.gh[kind] = dict(pars=t(vuv[f"GH_PARS_{kind}"]),
                                     ang=t(vuv[f"GH_border_angulo_{kind}"]),
                                     border=t(vuv[f"GH_border_{kind}"]))
        # VIS parameterization
        self.vis = {}
        if self.do_reflected:
            self.d_ang_vis = float(vis["delta_angulo_vis"])
            for kind, flag in (("flat", self.flat), ("flat_lateral", self.flat_lat), ("dome", self.dome)):
                if flag and f"VIS_correction_{kind}" in vis:
                    self.vis[kind] = dict(x=t(vis[f"VIS_distances_x_{kind}"]),
                                          r=t(vis[f"VIS_distances_r_{kind}"]),
                                          corr=t(vis[f"VIS_correction_{kind}"]))

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_detinfo(cls, di, opdets=None, **kw):
        """opdets: optional list of opdet indices to keep (e.g. PMTs only)."""
        om = di.optical_model
        sa = om["semianalytic"]
        od = sa["opdets"]
        centers = di.geometry.OpDetPositions()
        ids = np.arange(len(od["type"])) if opdets is None else np.asarray(opdets)
        av = np.asarray(sa["active_volumes"])[0]
        cath = np.asarray(sa["cathode_centre"], dtype=np.float64)
        vis = om["vis_hits"]["params"] if "vis_hits" in om else {}
        return cls(centers=centers[ids], types=od["type"][ids], orientations=od["orientation"][ids],
                   h=od["h"][ids], w=od["w"][ids], vuv=om["vuv_hits"]["params"], vis=vis,
                   cathode_centre=cath, cathode_plane_hw=(av[4] - av[1], av[5] - av[2]),
                   plane_depth=abs(cath[0]), absorption_length=sa["vuv_absorption_length"],
                   do_reflected=bool(sa["DoReflectedLight"]),
                   include_anode_reflections=bool(sa["IncludeAnodeReflections"]),
                   optical_path="sbnd" if "SBND" in str(sa["optical_path_class"]) else "standard",
                   opdet_ids=ids, **kw)

    # ------------------------------------------------------------------ helpers
    def _visible(self, P):
        if self.optical_path == "sbnd":
            return (P[:, None, 0] < 0) == (self.O[None, :, 0] < 0)
        return torch.ones(P.shape[0], self.O.shape[0], dtype=torch.bool, device=P.device)

    def _cos_theta(self, rel, dist):
        o = self.orient[None, :]
        num = torch.where(o == 2, rel[..., 2].abs(), torch.where(o == 1, rel[..., 1].abs(), rel[..., 0].abs()))
        cosine = num / dist
        return cosine, fast_acos(cosine) * 180.0 / _PI

    def _detector_solid_angle(self, rel, dist, theta):
        # evaluate each shape's formula only on its own detector columns
        sa = torch.zeros_like(dist)
        if self._rect_cols.numel():
            c = self._rect_cols
            n = dist.shape[0]
            sa[:, c] = rectangle_solid_angle(self.h[c][None, :].expand(n, -1),
                                             self.w[c][None, :].expand(n, -1),
                                             rel[:, c].abs(), self.orient[c][None, :].expand(n, -1))
        if self._dome_cols.numel():
            c = self._dome_cols
            sa[:, c] = dome_solid_angle(dist[:, c], theta[:, c], self.radius)
        return sa

    def _border_r(self, P, lateral_like):
        """r for the border correction; lateral_like: (D,) bool (orientation 1/2 vs 0)."""
        c = self.cc
        dx, dy, dz = P[:, 0] - c[0], P[:, 1] - c[1], P[:, 2] - c[2]
        r_yz = torch.hypot(dy, dz)[:, None]
        if self.vertical_border:
            return dy.abs()[:, None].expand(-1, self.O.shape[0])
        o = self.orient[None, :]
        r = torch.where(o == 2, torch.hypot(dx, dy)[:, None],
                        torch.where(o == 1, torch.hypot(dx, dz)[:, None], r_yz))
        return r

    # ------------------------------------------------------------------ direct
    def _direct_chunk(self, P):
        rel = P[:, None, :] - self.O[None, :, :]                 # (N, D, 3)
        dist = torch.linalg.norm(rel, dim=-1)
        cosine, theta = self._cos_theta(rel, dist)
        sa = self._detector_solid_angle(rel, dist, theta)
        vis_geo = torch.exp(-dist / self.lam) * (sa / (4 * _PI))
        j = (theta / self.d_ang_vuv).to(torch.long)
        r = self._border_r(P, None)

        o, ty = self.orient[None, :].expand_as(dist), self.type[None, :].expand_as(dist)
        pars = torch.zeros(4, *dist.shape, dtype=dist.dtype, device=dist.device)
        s = torch.zeros(3, *dist.shape, dtype=dist.dtype, device=dist.device)

        def fill(kind, m):
            if not m.any():
                return
            g = self.gh[kind]
            jj = j.clamp(0, g["pars"].shape[1] - 1)
            for q in range(4):
                pars[q] = torch.where(m, g["pars"][q][jj], pars[q])
            for q in range(3):
                s[q] = torch.where(m, interpolate(g["ang"], g["border"][q], theta, True), s[q])

        flat_like = (ty == 0) | (ty == 2)
        if self.flat or self.flat_lat:
            lat = (o == 1) | (o == 2)
            if self.flat_lat:
                fill("flat_lateral", flat_like & lat)
            if self.flat:
                fill("flat", flat_like & (o == 0))
            bad = flat_like & ((lat & (not self.flat_lat)) | ((o == 0) & (not self.flat)))
            if bad.any():
                raise ValueError("flat optical detectors found but their GH parameters are missing")
        if self.dome:
            fill("dome", ty == 1)
        elif (ty == 1).any():
            raise ValueError("dome PMTs found but DomePDCorr is off")

        p0 = pars[0] + s[0] * r
        p1 = pars[1] + s[1] * r
        p2 = pars[2] + s[2] * r
        gh = _sane_gh(gaisser_hillas(dist, p0, p1, p2, pars[3]))
        val = gh * vis_geo / cosine
        keep = self._visible(P) & (dist <= self.max_dist)
        return torch.where(keep, val, torch.zeros_like(val))

    # ------------------------------------------------------------------ reflected
    def _reflected_chunk(self, P):
        N, D = P.shape[0], self.O.shape[0]
        c = self.cc
        plane_depth = torch.where(P[:, 0] < 0, torch.full_like(P[:, 0], -self.plane_depth),
                                  torch.full_like(P[:, 0], self.plane_depth))
        rel_c = torch.stack([(P[:, 0] - plane_depth).abs(), (P[:, 1] - c[1]).abs(),
                             (P[:, 2] - c[2]).abs()], dim=-1)
        sa_cath = rectangle_solid_angle(torch.full_like(rel_c[:, 0], self.cath_h),
                                        torch.full_like(rel_c[:, 0], self.cath_w), rel_c,
                                        torch.zeros(N, dtype=torch.long, device=P.device))
        d_cath = (plane_depth - P[:, 0]).abs()
        cath_geo = torch.exp(-d_cath / self.lam) * (sa_cath / (4 * _PI))
        r_yz = torch.hypot(P[:, 1] - c[1], P[:, 2] - c[2])
        if not self.flat:
            raise ValueError("flat VUV correction required for reflected light")
        g = self.gh["flat"]
        zero = torch.zeros(1, dtype=P.dtype, device=P.device)
        s = [interpolate(g["ang"], g["border"][q], zero, True)[0] for q in range(3)]
        gh = _sane_gh(gaisser_hillas(d_cath, g["pars"][0][0] + s[0] * r_yz,
                                     g["pars"][1][0] + s[1] * r_yz,
                                     g["pars"][2][0] + s[2] * r_yz, g["pars"][3][0]))
        cath_vis = gh * cath_geo                                   # (N,)

        hotspot = torch.stack([plane_depth, P[:, 1], P[:, 2]], dim=-1)
        rel = hotspot[:, None, :] - self.O[None, :, :]
        dist = torch.linalg.norm(rel, dim=-1)
        cosine, theta = self._cos_theta(rel, dist)
        sa = self._detector_solid_angle(rel, dist, theta)
        vis_geo = sa / (2 * _PI) * cath_vis[:, None]
        k = (theta / self.d_ang_vis).to(torch.long)
        o, ty = self.orient[None, :].expand(N, D), self.type[None, :].expand(N, D)
        lat = ((o == 1) | (o == 2)) & self.flat_lat
        r = torch.where(lat, torch.hypot(P[:, 0] - c[0], P[:, 2] - c[2])[:, None], r_yz[:, None])
        d_c = (P[:, 0] - plane_depth).abs()[:, None].expand(N, D)
        border = torch.zeros_like(dist)

        def corr(kind, m):
            if not m.any():
                return
            if kind not in self.vis:
                raise ValueError(f"VIS correction '{kind}' missing")
            t = self.vis[kind]
            kk = k.clamp(0, t["corr"].shape[0] - 1)
            val = interpolate2(t["x"], t["r"], t["corr"], d_c, r, kk)
            border.copy_(torch.where(m, val, border))

        flat_like = (ty == 0) | (ty == 2)
        corr("flat", flat_like & (o == 0) & self.flat)
        corr("flat_lateral", flat_like & lat)
        corr("dome", (ty == 1) & self.dome)
        val = border * vis_geo / cosine
        return torch.where(self._visible(P), val, torch.zeros_like(val))

    # ------------------------------------------------------------------ public
    def _run(self, fn, points):
        P = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points,
                            dtype=self.dtype, device=self.device).reshape(-1, 3)
        out = [fn(P[i:i + self.chunk_size]) for i in range(0, P.shape[0], self.chunk_size)]
        return torch.cat(out, 0) if out else torch.zeros(0, self.O.shape[0], dtype=self.dtype)

    @torch.no_grad()
    def direct(self, points):
        """(N, D) direct VUV visibilities."""
        return self._run(self._direct_chunk, points)

    @torch.no_grad()
    def reflected(self, points):
        """(N, D) reflected (visible) visibilities; zeros if the model has no reflected light."""
        if not self.do_reflected:
            P = torch.as_tensor(points, dtype=self.dtype).reshape(-1, 3)
            return torch.zeros(P.shape[0], self.O.shape[0], dtype=self.dtype, device=self.device)
        return self._run(self._reflected_chunk, points)

    def visibility(self, points):
        return self.direct(points), self.reflected(points)
