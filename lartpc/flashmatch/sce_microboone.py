"""MicroBooNE space-charge correction (MCC9 backward AND forward maps) in
pure numpy.

Replicates larutil::SpaceChargeMicroBooNE(kMCC9_Backward).GetPosOffsets():
reads the TH3F offset maps (hDx/hDy/hDz) from
SCEoffsets_dataDriven_combined_bkwd_Jan18.root with uproot and does the same
trilinear (bin-center) interpolation after the MCC9 coordinate transforms
(SpaceChargeMicroBooNEMCC9.cxx):

    xNew = 2.50 - (2.50/2.56)*(x/100)     # [0,256]   -> [2.5,0]  (flipped)
    yNew = (2.50/2.33)*((y/100)+1.165)    # [-116.5,116.5] -> [0,2.5]
    zNew = (10.0/10.37)*(z/100)           # [0,1037]  -> [0,10]

Backward convention (SpaceChargeMicroBooNE.cxx kMCC9_Backward branch):
    corrected = reco + offset   (all three axes; outside map bounds -> 0)

The pyROOT binding of the C++ class is unavailable in the current larlite
build (stale dictionary: only the copy/default ctors are exposed), which is
why this standalone implementation exists.

Fully self-contained: the map ships with the repo as a compressed npz
(data/sce_offsets_mcc9_bkwd.npz, ~536 KB, converted from the larlite
TH3Fs) — no ROOT, uproot, or ubdl checkout needed. Passing a .root path
still works (read via uproot) for alternate maps.

    sce = SCEBackward()                     # bundled npz map
    xyz_true = sce.correct(xyz_reco)        # (N,3) or (3,)

Forward mode (true ionization-deposit position -> expected RECO position,
SpaceChargeMicroBooNE.cxx kMCC9_Forward branch:
x_reco = x_true - off_x + 0.7 [the anode tick-offset hack]; y,z: + off):

    fwd = SCEForward()
    xyz_reco_expected = fwd.apply(xyz_true)
"""
import os
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_MAP = os.path.join(_DATA_DIR, "sce_offsets_mcc9_bkwd.npz")
DEFAULT_MAP_FWD = os.path.join(_DATA_DIR, "sce_offsets_mcc9_fwd.npz")


class SCEBackward:
    def __init__(self, path=DEFAULT_MAP):
        from scipy.interpolate import RegularGridInterpolator
        self.interp = []
        if path.endswith(".npz"):
            z = np.load(path)
            for name in ("hDx", "hDy", "hDz"):
                centers = [z[f"{name}_c{ax}"] for ax in "xyz"]
                self.centers = centers
                self.interp.append(RegularGridInterpolator(
                    centers, z[name].astype(np.float64), method="linear",
                    bounds_error=False, fill_value=None))
        else:                                   # .root via uproot
            import uproot
            f = uproot.open(path)
            for name in ("hDx", "hDy", "hDz"):
                h = f[name]
                centers = [0.5 * (ax.edges()[:-1] + ax.edges()[1:])
                           for ax in (h.axis(0), h.axis(1), h.axis(2))]
                self.centers = centers
                self.interp.append(RegularGridInterpolator(
                    centers, h.values(), method="linear",
                    bounds_error=False, fill_value=None))
        self.lo = np.array([c[0] for c in self.centers])
        self.hi = np.array([c[-1] for c in self.centers])

    @staticmethod
    def _transform(xyz):
        xyz = np.atleast_2d(np.asarray(xyz, np.float64))
        t = np.empty_like(xyz)
        t[:, 0] = 2.50 - (2.50 / 2.56) * (xyz[:, 0] / 100.0)
        t[:, 1] = (2.50 / 2.33) * (xyz[:, 1] / 100.0 + 1.165)
        t[:, 2] = (10.0 / 10.37) * (xyz[:, 2] / 100.0)
        return t

    def offsets(self, xyz):
        """(N,3) reco cm -> (N,3) backward offsets (0 outside map bounds)."""
        xyz = np.atleast_2d(np.asarray(xyz, np.float64))
        t = self._transform(xyz)
        inside = np.all((t > self.lo) & (t < self.hi), axis=1)
        out = np.zeros_like(xyz)
        if inside.any():
            tc = np.clip(t[inside], self.lo, self.hi)
            for i in range(3):
                out[inside, i] = self.interp[i](tc)
        return out

    def correct(self, xyz):
        """reco position(s) -> space-charge-corrected position(s)."""
        xyz = np.asarray(xyz, np.float64)
        one = xyz.ndim == 1
        res = np.atleast_2d(xyz) + self.offsets(xyz)
        return res[0] if one else res


class SCEForward(SCEBackward):
    """Forward map: true deposit position -> expected reconstructed position.
    Replicates larutil::SpaceChargeMicroBooNE(kMCC9_Forward)
    ApplySpaceChargeEffect: x_reco = x_true - off_x + 0.7 (anode tick-offset
    hack from the C++); y_reco = y_true + off_y; z_reco = z_true + off_z.
    Offsets zero outside the map bounds (then the +0.7 x-shift is also
    suppressed, matching the C++ 'applied=false' path)."""

    def __init__(self, path=DEFAULT_MAP_FWD):
        super().__init__(path)

    def apply(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        one = xyz.ndim == 1
        pts = np.atleast_2d(xyz)
        off = self.offsets(pts)
        applied = np.any(off != 0, axis=1)
        out = pts.copy()
        out[:, 0] -= off[:, 0]
        out[:, 1] += off[:, 1]
        out[:, 2] += off[:, 2]
        out[applied, 0] += 0.7
        return out[0] if one else out

    def correct(self, xyz):  # avoid silently inheriting the backward semantic
        raise AttributeError("SCEForward has no 'correct'; use apply() "
                             "(true -> expected reco)")
