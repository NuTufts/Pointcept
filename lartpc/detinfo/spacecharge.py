"""
Space-charge offsets from the maps cached in /spacecharge/histograms.

Interface follows LArSoft spacecharge::SpaceCharge:
    GetPosOffsets(points)              forward  (true -> reco) spatial offsets [cm]
    GetCalPosOffsets(points, tpcid)    backward (reco -> true) spatial offsets [cm]
    GetEfieldOffsets(points)           E-field offsets (fraction of nominal field)
All accept (3,) or (N,3) arrays.

Each experiment's provider has its own map conventions; the per-detector
classes reproduce the LArSoft implementation of the extracted code version:
    SpaceChargeSBND        <- sbndcode/SpaceCharge/SpaceChargeSBND.cxx (v10_14_02, Voxelized_TH3)
    SpaceChargeMicroBooNE  <- ubevt/SpaceCharge/SpaceChargeMicroBooNE.cxx (uboonecode v10_04_07_13,
                              Voxelized_TH3; forward map = InputFilename, backward = CalibrationInputFilename)
The python classes always evaluate the maps; the Enable* flags of the extracted
configuration are reported by Enable*() but do not switch the methods off.
"""
import numpy as np

from .io import read_attrs


class Hist3D:
    """A TH3 stored as values[ix,iy,iz] + bin edges; Interpolate() reproduces
    ROOT TH3::Interpolate (trilinear between bin centres; points outside the
    centre-to-centre domain return 0, as ROOT does after its error message)."""

    def __init__(self, dset):
        self.values = dset[()].astype(np.float64)
        self.edges = [np.asarray(dset.attrs[f"edges_{a}"], dtype=np.float64) for a in "xyz"]
        self.centers = [0.5 * (e[1:] + e[:-1]) for e in self.edges]
        self.name = dset.name.split("/")[-1]

    def _lower_bin(self, axis, x):
        """0-based index of the bin centre at or below x (ROOT: ubx-1)."""
        e = self.edges[axis]
        b = np.searchsorted(e, x, side="right") - 1          # 0-based bin containing x
        b = np.clip(b, -1, len(e) - 1)
        c = self.centers[axis]
        bc = np.where((b >= 0) & (b < len(c)), c[np.clip(b, 0, len(c) - 1)], np.inf)
        return np.where(x < bc, b - 1, b)

    def Interpolate(self, x, y, z):
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        lo = [self._lower_bin(a, v) for a, v in enumerate((x, y, z))]
        n = self.values.shape
        ok = np.ones(x.shape, dtype=bool)
        for a in range(3):
            ok &= (lo[a] >= 0) & (lo[a] + 1 <= n[a] - 1)
        li = [np.clip(l, 0, n[a] - 2) for a, l in enumerate(lo)]
        d = []
        for a, v in enumerate((x, y, z)):
            c = self.centers[a]
            d.append((v - c[li[a]]) / (c[li[a] + 1] - c[li[a]]))
        V = self.values
        ix, iy, iz = li
        xd, yd, zd = d
        i1 = V[ix, iy, iz] * (1 - zd) + V[ix, iy, iz + 1] * zd
        i2 = V[ix, iy + 1, iz] * (1 - zd) + V[ix, iy + 1, iz + 1] * zd
        j1 = V[ix + 1, iy, iz] * (1 - zd) + V[ix + 1, iy, iz + 1] * zd
        j2 = V[ix + 1, iy + 1, iz] * (1 - zd) + V[ix + 1, iy + 1, iz + 1] * zd
        w1 = i1 * (1 - yd) + i2 * yd
        w2 = j1 * (1 - yd) + j2 * yd
        return np.where(ok, w1 * (1 - xd) + w2 * xd, 0.0)


def _pts(points):
    p = np.asarray(points, dtype=np.float64)
    return p.reshape(-1, 3), p.ndim == 1


def _ret(arr, single):
    return arr[0] if single else arr


class SpaceCharge:
    """Base: no distortion."""

    def __init__(self, h5file=None):
        self.config = read_attrs(h5file["spacecharge"]) if h5file is not None else {}

    def EnableSimSpatialSCE(self):
        return bool(self.config.get("EnableSimSpatialSCE", False))

    def EnableSimEfieldSCE(self):
        return bool(self.config.get("EnableSimEfieldSCE", False))

    def EnableCalSpatialSCE(self):
        return bool(self.config.get("EnableCalSpatialSCE", False))

    def EnableCalEfieldSCE(self):
        return bool(self.config.get("EnableCalEfieldSCE", False))

    def GetPosOffsets(self, points):
        p, s = _pts(points)
        return _ret(np.zeros_like(p), s)

    def GetCalPosOffsets(self, points, tpcid=1):
        p, s = _pts(points)
        return _ret(np.zeros_like(p), s)

    def GetEfieldOffsets(self, points):
        p, s = _pts(points)
        return _ret(np.zeros_like(p), s)


class SpaceChargeSBND(SpaceCharge):
    """Port of SpaceChargeSBND (Voxelized_TH3) from sbndcode v10_14_02.

    - coordinates clamped to |x|,|y| <= 199.999, 0.001 <= z <= 499.999
    - east map (`_E`) for x < 0, west map (`_W`) for x >= 0
    - forward x offset multiplied by corr = -1 for x < 0, or everywhere if
      is2DdriftSimHack (as configured in the extracted services)
    - GetCalPosOffsets clamps x to <= -2.5 (TPC 0) / >= +2.5 (TPC 1)
    """
    XY_CLAMP = 199.999
    Z_LO, Z_HI = 0.001, 499.999
    CATHODE_CLAMP = 2.5

    def __init__(self, h5file):
        super().__init__(h5file)
        if self.config.get("RepresentationType") != "Voxelized_TH3":
            raise NotImplementedError(f"SBND SCE representation {self.config.get('RepresentationType')!r}")
        hg = h5file["spacecharge/histograms"]
        self.hack_2d = bool(self.config.get("is2DdriftSimHack", False))
        self.maps = {}
        for side in "EW":
            self.maps[side] = dict(
                fwd=[Hist3D(hg[f"TrueFwd_Displacement_{a}_{side}"]) for a in "XYZ"],
                bkwd=[Hist3D(hg[f"TrueBkwd_Displacement_{a}_{side}"]) for a in "XYZ"],
                efield=[Hist3D(hg[f"True_ElecField_{a}_{side}"]) for a in "XYZ"])

    def _clamp(self, p):
        x = np.clip(p[:, 0], -self.XY_CLAMP, self.XY_CLAMP)
        y = np.clip(p[:, 1], -self.XY_CLAMP, self.XY_CLAMP)
        z = np.clip(p[:, 2], self.Z_LO, self.Z_HI)
        return x, y, z

    def _lookup(self, kind, x, y, z):
        out = np.zeros((len(x), 3))
        east = x < 0
        for side, m in (("E", east), ("W", ~east)):
            if m.any():
                for a in range(3):
                    out[m, a] = self.maps[side][kind][a].Interpolate(x[m], y[m], z[m])
        return out

    def GetPosOffsets(self, points):
        p, s = _pts(points)
        x, y, z = self._clamp(p)
        out = self._lookup("fwd", x, y, z)
        corr = np.where(x < 0, -1.0, 1.0)
        if self.hack_2d:
            corr[:] = -1.0
        out[:, 0] *= corr
        return _ret(out, s)

    def GetCalPosOffsets(self, points, tpcid=1):
        p, s = _pts(points)
        x, y, z = self._clamp(p)
        tpcid = np.broadcast_to(np.asarray(tpcid), x.shape)
        x = np.where((tpcid == 0) & (x > -self.CATHODE_CLAMP), -self.CATHODE_CLAMP, x)
        x = np.where((tpcid == 1) & (x < self.CATHODE_CLAMP), self.CATHODE_CLAMP, x)
        return _ret(self._lookup("bkwd", x, y, z), s)

    def GetEfieldOffsets(self, points):
        p, s = _pts(points)
        x, y, z = self._clamp(p)
        return _ret(self._lookup("efield", x, y, z), s)


class SpaceChargeMicroBooNE(SpaceCharge):
    """Port of SpaceChargeMicroBooNE, Voxelized_TH3 representation.

    - points beyond the "too far" box (x<0 | x>266.6, |y|>126.5, z<-10 | z>1047) -> 0
    - points between the active box (0.001..255.999, +-116.499, 0.001..1036.999) and the
      "too far" box are moved onto the active boundary (PretendAtBoundary)
    - map frame: x' = 2.5 - (2.5/2.56)(x/100), y' = (2.5/2.33)(y/100 + 1.165),
      z' = (10/10.37)(z/100); ROOT TH3::Interpolate on hDx/hDy/hDz (positions) and
      hEx/hEy/hEz (E field)
    - E-field offsets = -map x EfieldOffsetScale (pset default 1.0; 3.65 in v10_04_07_13)
    - GetCalPosOffsets ignores the TPC id (single TPC)
    """

    def __init__(self, h5file):
        super().__init__(h5file)
        if self.config.get("RepresentationType") != "Voxelized_TH3":
            raise NotImplementedError(f"MicroBooNE SCE representation {self.config.get('RepresentationType')!r}")
        hg = h5file["spacecharge/histograms"]
        self.fwd = [Hist3D(hg[n]) for n in ("hDx", "hDy", "hDz")]
        self.fwd_e = [Hist3D(hg[n]) for n in ("hEx", "hEy", "hEz")]
        cal = hg["CalibrationInputFilename"] if "CalibrationInputFilename" in hg else None
        self.bkwd = [Hist3D(cal[n]) for n in ("hDx", "hDy", "hDz")] if cal is not None else None
        self.bkwd_e = [Hist3D(cal[n]) for n in ("hEx", "hEy", "hEz")] if cal is not None else None
        self.efield_scale = float(self.config.get("EfieldOffsetScale", 1.0))

    @staticmethod
    def _prepare(p):
        x, y, z = p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy()
        too_far = (x < 0.0) | (x > 266.6) | (y < -126.5) | (y > 126.5) | (z < -10) | (z > 1047.0)
        x = np.where(x < 0.001, 0.001, np.where(x > 255.999, 255.999, x))
        y = np.where(y < -116.499, -116.499, np.where(y > 116.499, 116.499, y))
        z = np.where(z < 0.001, 0.001, np.where(z > 1036.999, 1036.999, z))
        xt = 2.50 - (2.50 / 2.56) * (x / 100.0)
        yt = (2.50 / 2.33) * ((y / 100.0) + 1.165)
        zt = (10.0 / 10.37) * (z / 100.0)
        return xt, yt, zt, too_far

    def _eval(self, hists, points, sign=1.0):
        p, s = _pts(points)
        xt, yt, zt, far = self._prepare(p)
        out = np.stack([h.Interpolate(xt, yt, zt) for h in hists], axis=1) * sign
        out[far] = 0.0
        return _ret(out, s)

    def GetPosOffsets(self, points):
        return self._eval(self.fwd, points)

    def GetCalPosOffsets(self, points, tpcid=1):
        if self.bkwd is None:
            raise ValueError("no backward (CalibrationInputFilename) map in this detinfo file")
        return self._eval(self.bkwd, points)

    def GetEfieldOffsets(self, points):
        return self._eval(self.fwd_e, points, -self.efield_scale)

    def GetCalEfieldOffsets(self, points, tpcid=1):
        return self._eval(self.bkwd_e, points, -self.efield_scale)


SPACECHARGE_CLASSES = {"sbnd": SpaceChargeSBND, "uboone": SpaceChargeMicroBooNE}


def make_spacecharge(detector, h5file):
    cls = SPACECHARGE_CLASSES.get(detector)
    if cls is None or "spacecharge/histograms" not in h5file:
        return SpaceCharge(h5file if "spacecharge" in h5file else None)
    return cls(h5file)
