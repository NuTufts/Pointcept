"""
Geometry: cryostats, TPCs, wire planes, wires, optical detectors and aux
detectors (CRT), read from the detinfo HDF5 cache.

Method names follow larlite's (icdl) `larutil::Geometry` so code ports easily:
    Ncryostats, NTPCs, Nplanes, Nwires, GetCryostat, GetTPC, GetPlane,
    TPCBoundaries, TPCDriftDir, WireCoordinate, NearestWire,
    PlaneWireToChannel, ChannelToWireID, ChannelWireStart/End,
    GetContainingCryoAndTPCIDs, GetSimplePlaneIndexFromCTP, GetCTPfromSimplePlaneIndex,
    NOpDets, OpDetPosition, OpDetFromOpChannel, Nchannels
Argument order follows larlite: (plane, tpc=0, cryo=0).

Every position argument accepts a single (3,) point or an (N,3) array; the
vectorized form returns arrays.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .ids import PlaneID, SigType, TPCID, View, WireID
from .io import read_table


def _as_points(pos):
    p = np.asarray(pos, dtype=np.float64)
    return p.reshape(-1, 3), p.ndim == 1


@dataclass
class BBox:
    min: np.ndarray
    max: np.ndarray

    @classmethod
    def from_row(cls, row):
        row = np.asarray(row, dtype=np.float64)
        return cls(row[:3].copy(), row[3:].copy())

    @property
    def center(self):
        return 0.5 * (self.min + self.max)

    @property
    def size(self):
        return self.max - self.min

    def contains(self, pos, wiggle=1.0):
        """LArSoft BoxBoundedGeo::ContainsPosition semantics (relative wiggle)."""
        p, single = _as_points(pos)
        lo = np.where(self.min > 0, self.min / wiggle, self.min * wiggle)
        hi = np.where(self.max < 0, self.max / wiggle, self.max * wiggle)
        r = np.all((p >= lo) & (p <= hi), axis=1)
        return bool(r[0]) if single else r

    def corners(self):
        lo, hi = self.min, self.max
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                         for z in (lo[2], hi[2])])


@dataclass
class WireGeo:
    id: WireID
    channel: int
    start: np.ndarray
    end: np.ndarray

    @property
    def center(self):
        return 0.5 * (self.start + self.end)

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))


@dataclass
class PlaneGeo:
    id: PlaneID
    view: View
    signal_type: SigType
    nwires: int
    wire_pitch: float
    theta_z: float
    center: np.ndarray
    normal_dir: np.ndarray
    wire_dir: np.ndarray
    increasing_wire_dir: np.ndarray
    first_wire_center: np.ndarray
    first_channel: int
    width: float
    depth: float
    _wires: "Geometry" = field(repr=False, default=None)

    def WireCoordinate(self, pos):
        """Continuous wire number of the projection of pos (LArSoft PlaneGeo::WireCoordinate)."""
        p, single = _as_points(pos)
        w = (p - self.first_wire_center) @ self.increasing_wire_dir / self.wire_pitch
        return float(w[0]) if single else w

    def NearestWire(self, pos, clamp=False):
        """Nearest wire index (LArSoft rounds with lround). Off-plane -> -1, or clamped."""
        w = np.floor(np.asarray(self.WireCoordinate(pos)) + 0.5).astype(np.int64)
        if clamp:
            w = np.clip(w, 0, self.nwires - 1)
        else:
            w = np.where((w < 0) | (w >= self.nwires), -1, w)
        return int(w) if np.ndim(w) == 0 else w

    def Wire(self, iwire) -> WireGeo:
        return self._wires._wire(self.id, iwire)

    def WireEndPoints(self):
        """(nwires,3) start and end arrays for this plane."""
        return self._wires._plane_wire_arrays(self.id)

    @property
    def ViewName(self):
        return View(self.view).name[1:]


@dataclass
class TPCGeo:
    id: TPCID
    bbox: BBox
    active_bbox: BBox
    drift_dir: np.ndarray
    drift_distance: float
    cathode_center: np.ndarray
    planes: List[PlaneGeo]

    def Nplanes(self):
        return len(self.planes)

    def ActiveBoundingBox(self):
        return self.active_bbox

    def ContainsPosition(self, pos, wiggle=1.0):
        return self.bbox.contains(pos, wiggle)


@dataclass
class CryoGeo:
    id: int
    bbox: BBox
    tpcs: List[TPCGeo]


@dataclass
class OpDetGeo:
    opdet: int
    center: np.ndarray
    shape: str
    half_size: np.ndarray       # dx, dy, dz (bounding half-widths)
    rmax: float
    volume_name: str


class Geometry:
    # LArSoft GeometryCore::FindTPCAtPosition uses a relative wiggle of 1+1e-4
    TPC_WIGGLE = 1.0 + 1.0e-4

    def __init__(self, h5file):
        g = h5file["geometry"]
        self.detector_name = str(g["detector"].attrs["name"])
        self.attrs = dict(g["detector"].attrs)
        cr = read_table(g["cryostats"])
        tp = read_table(g["tpcs"])
        pl = read_table(g["planes"])
        self._wire_tab = read_table(g["wires"])
        self._opdet_tab = read_table(g["opdets"])
        self._opch_tab = read_table(g["opchannels"])
        self._aux_tab = read_table(g["auxdets"]) if "auxdets" in g else None
        self._auxs_tab = read_table(g["auxdet_sensitive"]) if "auxdet_sensitive" in g else None

        # planes
        planes = {}
        for i in range(len(pl["plane"])):
            pid = PlaneID(int(pl["cryo"][i]), int(pl["tpc"][i]), int(pl["plane"][i]))
            planes[pid] = PlaneGeo(
                id=pid, view=View(int(pl["view"][i])), signal_type=SigType(int(pl["signal_type"][i])),
                nwires=int(pl["n_wires"][i]), wire_pitch=float(pl["wire_pitch"][i]),
                theta_z=float(pl["theta_z"][i]), center=pl["center"][i],
                normal_dir=pl["normal_dir"][i], wire_dir=pl["wire_dir"][i],
                increasing_wire_dir=pl["increasing_wire_dir"][i],
                first_wire_center=pl["first_wire_center"][i],
                first_channel=int(pl["first_channel"][i]),
                width=float(pl["width"][i]), depth=float(pl["depth"][i]), _wires=self)
        # tpcs
        tpcs = {}
        for i in range(len(tp["tpc"])):
            tid = TPCID(int(tp["cryo"][i]), int(tp["tpc"][i]))
            tpcs[tid] = TPCGeo(
                id=tid, bbox=BBox.from_row(tp["bbox"][i]),
                active_bbox=BBox.from_row(tp["active_bbox"][i]),
                drift_dir=tp["drift_dir"][i], drift_distance=float(tp["drift_distance"][i]),
                cathode_center=tp["cathode_center"][i],
                planes=[planes[PlaneID(tid.cryo, tid.tpc, p)] for p in range(int(tp["n_planes"][i]))])
        self._cryos = []
        for i in range(len(cr["cryo"])):
            c = int(cr["cryo"][i])
            self._cryos.append(CryoGeo(c, BBox.from_row(cr["bbox"][i]),
                                       [tpcs[TPCID(c, t)] for t in range(int(cr["n_tpcs"][i]))]))
        self._tpc_list = [t for c in self._cryos for t in c.tpcs]
        self._plane_list = [p for t in self._tpc_list for p in t.planes]

        # wire lookup: row offset of each plane's wires (rows are ordered by c,t,p,w)
        wt = self._wire_tab
        key = (wt["cryo"].astype(np.int64) * 1000 + wt["tpc"]) * 100 + wt["plane"]
        self._plane_row0 = {}
        for p in self._plane_list:
            k = (p.id.cryo * 1000 + p.id.tpc) * 100 + p.id.plane
            rows = np.nonzero(key == k)[0]
            self._plane_row0[p.id] = int(rows[0])
        # channel -> wire rows (a channel may map to several wires)
        order = np.argsort(wt["channel"], kind="stable")
        self._ch_sorted = wt["channel"][order]
        self._ch_order = order

    # ------------------------------------------------------------- counts
    def Ncryostats(self):
        return len(self._cryos)

    def NTPCs(self, cryo=0):
        return len(self._cryos[cryo].tpcs)

    def Nplanes(self, tpc=0, cryo=0):
        return len(self._cryos[cryo].tpcs[tpc].planes)

    def Nwires(self, plane, tpc=0, cryo=0):
        return self.GetPlane(plane, tpc, cryo).nwires

    def Nchannels(self):
        return int(self.attrs.get("n_channels", int(self._wire_tab["channel"].max()) + 1))

    def NOpDets(self):
        return len(self._opdet_tab["opdet"])

    def NOpChannels(self):
        return len(self._opch_tab["opchannel"])

    # ------------------------------------------------------------- objects
    def GetCryostat(self, cryo=0) -> CryoGeo:
        return self._cryos[cryo]

    def GetTPC(self, tpc=0, cryo=0) -> TPCGeo:
        return self._cryos[cryo].tpcs[tpc]

    def GetPlane(self, plane, tpc=0, cryo=0) -> PlaneGeo:
        return self._cryos[cryo].tpcs[tpc].planes[plane]

    def IterateTPCs(self):
        return iter(self._tpc_list)

    def IteratePlanes(self):
        return iter(self._plane_list)

    def CryostatBoundaries(self, cryo=0):
        b = self._cryos[cryo].bbox
        return np.array([b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]])

    def TPCBoundaries(self, tpc=0, cryo=0, active=True):
        """(min, max) xyz arrays of the TPC (active volume by default)."""
        t = self.GetTPC(tpc, cryo)
        b = t.active_bbox if active else t.bbox
        return b.min.copy(), b.max.copy()

    def TPCDriftDir(self, tpc=0, cryo=0):
        return self.GetTPC(tpc, cryo).drift_dir.copy()

    def DetectorBoundingBox(self, active=True):
        """Union of all TPC (active) volumes."""
        bs = [t.active_bbox if active else t.bbox for t in self._tpc_list]
        return BBox(np.min([b.min for b in bs], 0), np.max([b.max for b in bs], 0))

    # ------------------------------------------------------------- positions
    def FindTPC(self, pos, wiggle=None):
        """Vectorized GeometryCore::FindTPCAtPosition -> (cryo, tpc) int arrays, -1 if none."""
        p, single = _as_points(pos)
        wiggle = self.TPC_WIGGLE if wiggle is None else wiggle
        cryo = np.full(len(p), -1, dtype=np.int64)
        tpc = np.full(len(p), -1, dtype=np.int64)
        for t in self._tpc_list:
            m = (tpc < 0) & t.bbox.contains(p, wiggle)
            cryo[m], tpc[m] = t.id.cryo, t.id.tpc
        return (int(cryo[0]), int(tpc[0])) if single else (cryo, tpc)

    def GetContainingCryoAndTPCIDs(self, pos):
        """larlite name; returns [cryo, tpc] ([-1,-1] outside)."""
        c, t = self.FindTPC(pos)
        return [c, t]

    def WireCoordinate(self, pos, plane, tpc=0, cryo=0):
        return self.GetPlane(plane, tpc, cryo).WireCoordinate(pos)

    def NearestWire(self, pos, plane, tpc=0, cryo=0, clamp=False):
        return self.GetPlane(plane, tpc, cryo).NearestWire(pos, clamp=clamp)

    # ------------------------------------------------------------- channels
    def PlaneWireToChannel(self, plane, wire, tpc=0, cryo=0):
        row = self._plane_row0[PlaneID(cryo, tpc, plane)] + np.asarray(wire)
        ch = self._wire_tab["channel"][row]
        return int(ch) if np.ndim(ch) == 0 else ch

    def ChannelToWireID(self, channel) -> List[WireID]:
        """All wires read out by a channel (several for split/wrapped wires)."""
        lo = np.searchsorted(self._ch_sorted, channel, "left")
        hi = np.searchsorted(self._ch_sorted, channel, "right")
        wt = self._wire_tab
        return [WireID(int(wt["cryo"][r]), int(wt["tpc"][r]), int(wt["plane"][r]), int(wt["wire"][r]))
                for r in self._ch_order[lo:hi]]

    def ChannelToPlane(self, channel):
        w = self.ChannelToWireID(channel)
        return PlaneID(*w[0][:3]) if w else None

    def ChannelToWire(self, channel):
        w = self.ChannelToWireID(channel)
        return w[0].wire if w else -1

    def ChannelWireStart(self, channel):
        w = self.ChannelToWireID(channel)[0]
        return self._wire(PlaneID(*w[:3]), w.wire).start

    def ChannelWireEnd(self, channel):
        w = self.ChannelToWireID(channel)[0]
        return self._wire(PlaneID(*w[:3]), w.wire).end

    # ------------------------------------------------------------- simple plane index
    def GetSimplePlaneIndexFromCTP(self, cryo, tpc, plane):
        """Flat index over all planes, ordered (cryo, tpc, plane)."""
        return self._plane_list.index(self.GetPlane(plane, tpc, cryo))

    def GetCTPfromSimplePlaneIndex(self, index):
        p = self._plane_list[index].id
        return [p.cryo, p.tpc, p.plane]

    # ------------------------------------------------------------- wires (internal)
    def _wire(self, pid, iwire) -> WireGeo:
        r = self._plane_row0[pid] + int(iwire)
        wt = self._wire_tab
        return WireGeo(WireID(*pid, int(iwire)), int(wt["channel"][r]),
                       wt["start"][r].astype(np.float64), wt["end"][r].astype(np.float64))

    def _plane_wire_arrays(self, pid):
        r0 = self._plane_row0[pid]
        n = self.GetPlane(pid.plane, pid.tpc, pid.cryo).nwires
        wt = self._wire_tab
        return wt["start"][r0:r0 + n], wt["end"][r0:r0 + n]

    # ------------------------------------------------------------- optical
    def OpDet(self, i) -> OpDetGeo:
        t = self._opdet_tab
        return OpDetGeo(int(t["opdet"][i]), t["center"][i], str(t["shape"][i]),
                        np.array([t["dx"][i], t["dy"][i], t["dz"][i]]), float(t["rmax"][i]),
                        str(t["volume_name"][i]))

    def OpDetPositions(self):
        return self._opdet_tab["center"].copy()

    def OpDetPosition(self, i):
        return self._opdet_tab["center"][i].copy()

    def OpDetFromOpChannel(self, ch):
        return int(self._opch_tab["opdet"][ch])

    def OpDetTable(self):
        return self._opdet_tab

    # ------------------------------------------------------------- aux dets (CRT)
    def NAuxDets(self):
        return 0 if self._aux_tab is None else len(self._aux_tab["auxdet"])

    def AuxDetTable(self):
        return self._aux_tab

    def AuxDetSensitiveTable(self):
        return self._auxs_tab
