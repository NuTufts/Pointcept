"""
detinfo -- detector-agnostic detector constants for SBN experiments.

    from lartpc.detinfo import load
    di = load("sbnd")                    # or load("sbnd", "v10_14_02") / load(path="...h5")
    geo = di.geometry
    geo.NTPCs(); geo.GetPlane(2, tpc=1).WireCoordinate(points)
    di.detprop.ConvertXToTicks(x, plane=2, tpc=1)
    di.clocks.TPCClock().TickPeriod
    di.sce.GetCalPosOffsets(points, tpcid=0)
    di.photonlib.GetVisibility(points)   # lazy, loads photonlib_<det>_<ver>.h5
    di.project(points)                   # vectorized: TPC id, wire coordinate + tick per plane

Data: one HDF5 file per experiment code version, made by extract/run_extract.sh;
layout in SCHEMA.md. numpy + h5py only.
"""
import json
import os
from functools import cached_property

import numpy as np

from .conventions import make_conventions
from .geometry import Geometry
from .io import find_detinfo_file, read_attrs
from .properties import DetectorClocks, DetectorProperties, LArProperties
from .spacecharge import make_spacecharge

__all__ = ["load", "DetInfo"]

_CACHE = {}


class DetInfo:
    def __init__(self, path):
        import h5py
        self.path = os.path.abspath(path)
        self.cache_dir = os.path.dirname(self.path)
        with h5py.File(self.path, "r") as f:
            self.schema_version = str(f.attrs["schema_version"])
            self.detector = str(f.attrs["detector"])
            self.provenance = read_attrs(f["provenance"])
            self.geometry = Geometry(f)
            self.clocks = DetectorClocks(f)
            self.detprop = DetectorProperties(f)
            self.larprop = LArProperties(f)
            self.sce = make_spacecharge(self.detector, f)
            self.photon_visibility_config = (read_attrs(f["photon_visibility"])
                                             if "photon_visibility" in f else {})
        self.conventions = make_conventions(self.detector, self)

    @property
    def version(self):
        return self.provenance.get("experiment_version")

    @property
    def product_versions(self):
        return json.loads(self.provenance.get("product_versions_json", "{}"))

    @property
    def failed_queries(self):
        return json.loads(self.provenance.get("failed_queries_json", "{}"))

    @cached_property
    def photonlib(self):
        from .photonlib import PhotonLibrary
        name = self.photon_visibility_config.get("converted_h5")
        if not name:
            raise FileNotFoundError(f"no photon library extracted for {self.detector} "
                                    f"({self.photon_visibility_config.get('note', '')})")
        return PhotonLibrary(os.path.join(self.cache_dir, name))

    def project(self, points):
        """Vectorized projection of (N,3) points into the readout.

        Returns dict:
          cryo, tpc         (N,) containing TPC (-1 outside)
          wire_coordinate   (N, max_planes) continuous wire number per plane of that TPC
          nearest_wire      (N, max_planes) int, -1 off-plane / outside
          tick              (N, max_planes) TPC tick of the drifted charge (no SCE)
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        g = self.geometry
        cryo, tpc = g.FindTPC(p)
        nmax = max(t.Nplanes() for t in g.IterateTPCs())
        wc = np.full((len(p), nmax), np.nan)
        nw = np.full((len(p), nmax), -1, dtype=np.int64)
        tk = np.full((len(p), nmax), np.nan)
        for t in g.IterateTPCs():
            m = (cryo == t.id.cryo) & (tpc == t.id.tpc)
            if not m.any():
                continue
            for pl in t.planes:
                k = pl.id.plane
                wc[m, k] = pl.WireCoordinate(p[m])
                nw[m, k] = pl.NearestWire(p[m])
                tk[m, k] = self.detprop.ConvertXToTicks(p[m, 0], k, t.id.tpc, t.id.cryo)
        return dict(cryo=cryo, tpc=tpc, wire_coordinate=wc, nearest_wire=nw, tick=tk)

    def summary(self):
        g = self.geometry
        lines = [f"{self.detector} ({g.detector_name}) {self.provenance.get('experiment_product')} "
                 f"{self.version}  schema {self.schema_version}",
                 f"  cryostats {g.Ncryostats()}  TPCs {sum(g.NTPCs(c) for c in range(g.Ncryostats()))}  "
                 f"channels {g.Nchannels()}  opdets {g.NOpDets()}  auxdets {g.NAuxDets()}"]
        for t in g.IterateTPCs():
            a = t.active_bbox
            lines.append(f"  TPC {tuple(t.id)} active x[{a.min[0]:.2f},{a.max[0]:.2f}] "
                         f"y[{a.min[1]:.2f},{a.max[1]:.2f}] z[{a.min[2]:.2f},{a.max[2]:.2f}] "
                         f"drift {t.drift_dir.round(3).tolist()}")
            for pl in t.planes:
                lines.append(f"     plane {pl.id.plane} {pl.ViewName} {pl.signal_type.name} "
                             f"nwires {pl.nwires} pitch {pl.wire_pitch:.3f} "
                             f"theta_z {np.degrees(pl.theta_z):.1f} deg  ch0 {pl.first_channel}")
        lines.append(f"  tick {self.clocks.TPCClock().TickPeriod} us  samples "
                     f"{self.detprop.NumberTimeSamples()}  vdrift {self.detprop.DriftVelocity():.5f} "
                     f"cm/us  E {self.detprop.Efield():.3f} kV/cm  T {self.detprop.Temperature()} K")
        return "\n".join(lines)


def load(detector=None, version=None, path=None):
    """Load (and cache) the detinfo file for a detector/version, or an explicit path."""
    if path is None:
        if detector is None:
            raise ValueError("give detector= or path=")
        path = find_detinfo_file(detector, version)
    path = os.path.abspath(path)
    if path not in _CACHE:
        _CACHE[path] = DetInfo(path)
    return _CACHE[path]
