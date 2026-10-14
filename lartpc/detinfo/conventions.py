"""
Per-experiment facts that the LArSoft geometry does not expose, derived from
files copied into the cache (aux/) or from the geometry itself.
"""
import json
import os

import numpy as np


class Conventions:
    plane_names = ("U", "V", "Y")

    def __init__(self, detinfo):
        self.di = detinfo

    def opdet_types(self):
        """Per-opdet type string (default: derived from the TGeo shape)."""
        shapes = self.di.geometry.OpDetTable()["shape"]
        return np.array(["pmt" if s == "TGeoSphere" else "box" if s == "TGeoBBox" else "unknown"
                         for s in shapes], dtype=object)

    def crt_walls(self):
        """Per-auxdet wall label: axis of the module normal, side from its centre
        relative to the TPC volume centre."""
        g = self.di.geometry
        aux = g.AuxDetTable()
        if aux is None:
            return np.array([], dtype=object)
        det = g.DetectorBoundingBox(active=False)
        c = aux["center"]
        rel = c - det.center
        axis = np.argmax(np.abs(aux["normal"]), axis=1)
        names = {(0, -1): "x-", (0, 1): "x+", (1, -1): "bottom", (1, 1): "top",
                 (2, -1): "upstream", (2, 1): "downstream"}
        sign = np.sign(rel[np.arange(len(c)), axis]).astype(int)
        return np.array([names[(a, s)] for a, s in zip(axis, sign)], dtype=object)


class ConventionsSBND(Conventions):
    plane_names = ("U", "V", "Y")

    def pds_mapping(self):
        path = os.path.join(self.di.cache_dir, "aux", "sbnd_pds_mapping.json")
        with open(path) as f:
            return json.load(f)

    def opdet_types(self):
        """'pmt_coated' | 'pmt_uncoated' | 'xarapuca_vuv' | 'xarapuca_vis' per opdet,
        from sbnd_pds_mapping.json (keyed by optical channel)."""
        g = self.di.geometry
        types = np.full(g.NOpDets(), "unknown", dtype=object)
        for row in self.pds_mapping():
            ch = int(row["channel"])
            if ch < g.NOpChannels():
                types[g.OpDetFromOpChannel(ch)] = row["pd_type"]
        return types

    def crt_walls(self):
        """SBND names the x<0 side east (TPC 0, `_E` SCE map) and x>0 west."""
        w = super().crt_walls()
        return np.array([{"x-": "east", "x+": "west"}.get(s, s) for s in w], dtype=object)


CONVENTIONS = {"sbnd": ConventionsSBND}


def make_conventions(detector, detinfo):
    return CONVENTIONS.get(detector, Conventions)(detinfo)
