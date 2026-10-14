#!/usr/bin/env python
"""
extract_larsoft_detinfo.py

Query the official LArSoft service providers of an SBN experiment (geometry,
wire readout, aux-det/CRT geometry, detector clocks, detector properties, LAr
properties, space-charge configuration) and cache everything into one HDF5
file. That file is read later by the python-only `lartpc.detinfo` utility (no
ROOT / LArSoft / cvmfs at use time).

MUST run inside the experiment's LArSoft environment (see run_extract.sh, which
starts the SL7 cvmfs container and runs `setup <product> <version>`). Needs
h5py importable (run_extract.sh puts a vendored py3.9 h5py on PYTHONPATH).

Every provider query goes through `Q.get(key, fn)`. A query that raises is
recorded under /provenance/failed_queries instead of aborting, so one missing
API in a new LArSoft version does not lose the rest of the extraction.

Schema: ../SCHEMA.md      Table dumper: ../dump_tables.py

Usage (inside the environment):
    python extract_larsoft_detinfo.py --detector sbnd --outdir <dir>
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import traceback

import numpy as np
import h5py

SCHEMA_VERSION = "1.3"

# ---------------------------------------------------------------------------
# Per-experiment configuration. Only `sbnd` has been run so far; the `icarus`
# and `uboone` entries are best guesses to be verified when first extracted.
# ---------------------------------------------------------------------------
DETECTORS = {
    "sbnd": dict(
        product="SBNDCODE",                      # env prefix: SBNDCODE_VERSION
        services_module="SBNDservices",
        services_fcl="simulationservices_sbnd.fcl",
        service_table="sbnd_simulation_services",
        # extra files copied verbatim (searched on FW_SEARCH_PATH)
        aux_file_globs=["*pds*map*.json", "*PDS*map*.json"],
        # LArSoft SCE provider, instantiated only to compute validation values
        sce_provider=dict(header="sbndcode/SpaceCharge/SpaceChargeSBND.h",
                          library="sbndcode_SpaceCharge",
                          cls="spacecharge.SpaceChargeSBND"),
        # light model tables outside the services config: name -> (fcl, table)
        optical_fcl_tables={
            "vuv_hits": ("opticalsimparameterisations_sbnd.fcl", "sbnd_vuv_RS100cm_hits_parameterization"),
            "vis_hits": ("opticalsimparameterisations_sbnd.fcl", "sbnd_vis_RS100cm_hits_parameterization"),
            "pdfastsim_par": ("PDFastSim_sbnd.fcl", "sbnd_pdfastsim_par"),
            "pdfastsim_pvs": ("PDFastSim_sbnd.fcl", "sbnd_pdfastsim_pvs"),
            "flashmatch": ("opt0finder_sbnd.fcl", "sbnd_opt0_finder_one_to_many"),
        },
        # semi-analytic model reference build: optical-path tool (header, class)
        semianalytic=dict(optical_path=("sbndcode/OpPathTool/SBNDOpticalPath.h",
                                        "phot::SBNDOpticalPath")),
    ),
    "icarus": dict(
        product="ICARUSCODE",
        services_module="ICARUSservices",
        services_fcl="services_common_icarus.fcl",
        service_table="icarus_common_services",
        aux_file_globs=[],
        sce_provider=None,
        optical_fcl_tables={},
        semianalytic=None,
    ),
}

# product versions recorded into /provenance (env var <NAME>_VERSION)
VERSION_PRODUCTS = [
    "SBNDCODE", "ICARUSCODE", "UBOONECODE", "SBNCODE", "SBNALG", "SBNOBJ",
    "SBND_DATA", "ICARUS_DATA", "LARSOFT", "LARCORE", "LARCOREALG",
    "LARDATA", "LARDATAALG", "LARDATAOBJ", "LAREVT", "LARSIM", "LARG4",
    "ROOT", "GALLERY", "FHICLCPP", "CANVAS", "ART", "NUSIMDATA",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Queries:
    """Safe-call wrapper: records failures instead of raising."""

    def __init__(self):
        self.failed = {}

    def get(self, key, fn, default=None):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad
            msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
            # keep only the first failure per key pattern
            self.failed.setdefault(key, msg)
            return default


Q = Queries()


def vec(v):
    return [float(v.X()), float(v.Y()), float(v.Z())]


def bbox(b):
    return [float(b.MinX()), float(b.MinY()), float(b.MinZ()),
            float(b.MaxX()), float(b.MaxY()), float(b.MaxZ())]


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_on_search_path(relpath, envvar="FW_SEARCH_PATH"):
    for d in os.environ.get(envvar, "").split(":"):
        if d and os.path.isfile(os.path.join(d, relpath)):
            return os.path.join(d, relpath)
    return None


def write_table(group, name, rows, attrs=None):
    """rows: dict column -> list/array (all the same length)."""
    g = group.create_group(name)
    n = None
    for col, vals in rows.items():
        arr = np.asarray(vals)
        if arr.dtype.kind == "U" or arr.dtype == object:
            arr = np.asarray([str(x) for x in vals], dtype=h5py.string_dtype())
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float64)
        big = arr.size > 1000
        g.create_dataset(col, data=arr, compression="gzip" if big else None,
                         compression_opts=9 if big else None,
                         shuffle=big and arr.dtype.kind in "iuf")
        n = len(arr) if n is None else n
    g.attrs["n_rows"] = 0 if n is None else n
    for k, v in (attrs or {}).items():
        g.attrs[k] = v
    return g


def write_scalars(group, name, d):
    """dict of scalar / small-array values -> group attrs (None skipped)."""
    g = group.create_group(name)
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], str):
            g.attrs[k] = np.asarray(v, dtype=h5py.string_dtype())
        else:
            g.attrs[k] = v
    return g


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def extract_geometry(ROOT, geom, wr):
    out = {}
    ncryo = int(geom.Ncryostats())
    out["detector"] = dict(
        name=str(geom.DetectorName()),
        gdml_file=Q.get("geom.GDMLFile", lambda: str(geom.GDMLFile()), ""),
        n_cryostats=ncryo,
        n_tpcs_total=int(sum(geom.NTPC(ROOT.geo.CryostatID(c)) for c in range(ncryo))),
        n_opdets=Q.get("geom.NOpDets", lambda: int(geom.NOpDets()), -1),
        n_channels=Q.get("wr.Nchannels", lambda: int(wr.Nchannels()), -1),
        n_opchannels=Q.get("wr.NOpChannels", lambda: int(wr.NOpChannels()), -1),
        world_bbox=Q.get("geom.WorldBox", lambda: bbox(geom.WorldBox()), []),
        enclosure_bbox=Q.get("geom.DetectorEnclosureBox",
                             lambda: bbox(geom.DetectorEnclosureBox()), []),
        units_length="cm",
    )

    # --- cryostats
    cr = dict(cryo=[], bbox=[], center=[], n_tpcs=[], n_opdets=[])
    for c in range(ncryo):
        cid = ROOT.geo.CryostatID(c)
        cryo = geom.Cryostat(cid)
        cr["cryo"].append(c)
        cr["bbox"].append(Q.get("cryo.bbox", lambda: bbox(cryo.Boundaries()),
                                Q.get("cryo.bbox2", lambda: bbox(cryo), [np.nan] * 6)))
        cr["center"].append(Q.get("cryo.center", lambda: vec(cryo.GetCenter()), [np.nan] * 3))
        cr["n_tpcs"].append(int(geom.NTPC(cid)))
        cr["n_opdets"].append(Q.get("cryo.NOpDet", lambda: int(cryo.NOpDet()), -1))
    out["cryostats"] = cr

    # --- TPCs, planes, wires
    tp = dict(cryo=[], tpc=[], bbox=[], active_bbox=[], center=[],
              active_center=[], drift_dir=[], drift_distance=[],
              cathode_center=[], n_planes=[])
    pl = dict(cryo=[], tpc=[], plane=[], view=[], signal_type=[], n_wires=[],
              wire_pitch=[], theta_z=[], center=[], normal_dir=[], wire_dir=[],
              increasing_wire_dir=[], first_wire_center=[], last_wire_center=[],
              first_channel=[], width=[], depth=[], plane_bbox=[])
    wi = dict(cryo=[], tpc=[], plane=[], wire=[], channel=[], start=[], end=[],
              center=[])
    for c in range(ncryo):
        for t in range(int(geom.NTPC(ROOT.geo.CryostatID(c)))):
            tid = ROOT.geo.TPCID(c, t)
            tpc = geom.TPC(tid)
            npl = int(wr.Nplanes(tid))
            tp["cryo"].append(c); tp["tpc"].append(t)
            tp["bbox"].append(Q.get("tpc.bbox", lambda: bbox(tpc), [np.nan] * 6))
            tp["active_bbox"].append(Q.get("tpc.active_bbox",
                                           lambda: bbox(tpc.ActiveBoundingBox()), [np.nan] * 6))
            tp["center"].append(Q.get("tpc.center", lambda: vec(tpc.GetCenter()), [np.nan] * 3))
            tp["active_center"].append(Q.get("tpc.active_center",
                                             lambda: vec(tpc.GetActiveVolumeCenter()), [np.nan] * 3))
            tp["drift_dir"].append(Q.get("tpc.DriftDir", lambda: vec(tpc.DriftDir()), [np.nan] * 3))
            tp["drift_distance"].append(Q.get("tpc.DriftDistance",
                                              lambda: float(tpc.DriftDistance()), np.nan))
            tp["cathode_center"].append(Q.get("tpc.cathode",
                                              lambda: vec(tpc.GetCathodeCenter()), [np.nan] * 3))
            tp["n_planes"].append(npl)
            for p in range(npl):
                pid = ROOT.geo.PlaneID(c, t, p)
                plane = wr.Plane(pid)
                nw = int(plane.Nwires())
                pl["cryo"].append(c); pl["tpc"].append(t); pl["plane"].append(p)
                pl["view"].append(int(plane.View()))
                pl["signal_type"].append(Q.get("wr.SignalType",
                                               lambda: int(wr.SignalType(pid)), -1))
                pl["n_wires"].append(nw)
                pl["wire_pitch"].append(float(plane.WirePitch()))
                pl["theta_z"].append(float(plane.ThetaZ()))
                pl["center"].append(vec(plane.GetCenter()))
                pl["normal_dir"].append(Q.get("plane.normal", lambda: vec(plane.GetNormalDirection()), [np.nan] * 3))
                pl["wire_dir"].append(Q.get("plane.wiredir", lambda: vec(plane.GetWireDirection()), [np.nan] * 3))
                pl["increasing_wire_dir"].append(Q.get("plane.incdir", lambda: vec(plane.GetIncreasingWireDirection()), [np.nan] * 3))
                pl["first_wire_center"].append(vec(plane.Wire(0).GetCenter()))
                pl["last_wire_center"].append(vec(plane.Wire(nw - 1).GetCenter()))
                pl["first_channel"].append(int(wr.PlaneWireToChannel(ROOT.geo.WireID(c, t, p, 0))))
                pl["width"].append(Q.get("plane.Width", lambda: float(plane.Width()), np.nan))
                pl["depth"].append(Q.get("plane.Depth", lambda: float(plane.Depth()), np.nan))
                pl["plane_bbox"].append(Q.get("plane.bbox", lambda: bbox(plane.BoundingBox()), [np.nan] * 6))
                for w in range(nw):
                    wg = plane.Wire(w)
                    wi["cryo"].append(c); wi["tpc"].append(t); wi["plane"].append(p)
                    wi["wire"].append(w)
                    wi["channel"].append(int(wr.PlaneWireToChannel(ROOT.geo.WireID(c, t, p, w))))
                    wi["start"].append(vec(wg.GetStart()))
                    wi["end"].append(vec(wg.GetEnd()))
                    wi["center"].append(vec(wg.GetCenter()))
    out["tpcs"] = tp
    out["planes"] = pl
    out["wires"] = {k: (np.asarray(v, dtype=np.float32) if k in ("start", "end", "center")
                        else np.asarray(v, dtype=np.int32)) for k, v in wi.items()}

    # --- optical detectors
    od = dict(opdet=[], cryo=[], center=[], shape=[], dx=[], dy=[], dz=[], rmax=[],
              volume_name=[])
    nop = out["detector"]["n_opdets"]
    for i in range(max(nop, 0)):
        g = Q.get("geom.OpDetGeoFromOpDet", lambda: geom.OpDetGeoFromOpDet(i))
        if g is None:
            break
        shp = Q.get("opdet.Shape", lambda: g.Shape())
        sname = shp.IsA().GetName() if shp is not None else ""
        od["opdet"].append(i)
        od["cryo"].append(Q.get("opdet.cryo", lambda: int(geom.PositionToCryostatID(g.GetCenter()).Cryostat), -1))
        od["center"].append(vec(g.GetCenter()))
        od["shape"].append(sname)
        dx = dy = dz = rmax = np.nan
        if shp is not None and shp.InheritsFrom("TGeoBBox"):
            dx, dy, dz = shp.GetDX(), shp.GetDY(), shp.GetDZ()
        if sname in ("TGeoSphere", "TGeoTube", "TGeoTubeSeg"):
            rmax = shp.GetRmax()
        od["dx"].append(dx); od["dy"].append(dy); od["dz"].append(dz); od["rmax"].append(rmax)
        od["volume_name"].append(Q.get("opdet.node", lambda: str(g.Node().GetVolume().GetName()), ""))
    out["opdets"] = od
    oc = dict(opchannel=[], opdet=[])
    for ch in range(max(out["detector"]["n_opchannels"], 0)):
        oc["opchannel"].append(ch)
        oc["opdet"].append(Q.get("wr.OpDetFromOpChannel", lambda: int(wr.OpDetFromOpChannel(ch)), -1))
    out["opchannels"] = oc
    return out


def extract_auxdets(ROOT, aux):
    ad = dict(auxdet=[], name=[], center=[], half_width1=[], half_width2=[],
              half_height=[], length=[], normal=[], n_sensitive=[],
              axis_x=[], axis_y=[], axis_z=[])
    sv = dict(auxdet=[], sensitive=[], center=[], half_width1=[], half_width2=[],
              half_height=[], length=[], normal=[], axis_x=[], axis_y=[], axis_z=[])

    def axes(g):
        # local unit axes expressed in world coordinates
        LP = type(g).LocalPoint_t
        o = g.toWorldCoords(LP(0, 0, 0))
        res = []
        for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            p = g.toWorldCoords(LP(*e))
            res.append([p.X() - o.X(), p.Y() - o.Y(), p.Z() - o.Z()])
        return res

    n = Q.get("aux.NAuxDets", lambda: int(aux.NAuxDets()), 0)
    for i in range(n):
        g = aux.AuxDet(i)
        ad["auxdet"].append(i)
        ad["name"].append(Q.get("aux.Name", lambda: str(g.Name()), ""))
        ad["center"].append(vec(g.GetCenter()))
        ad["half_width1"].append(Q.get("aux.hw1", lambda: float(g.HalfWidth1()), np.nan))
        ad["half_width2"].append(Q.get("aux.hw2", lambda: float(g.HalfWidth2()), np.nan))
        ad["half_height"].append(Q.get("aux.hh", lambda: float(g.HalfHeight()), np.nan))
        ad["length"].append(Q.get("aux.len", lambda: float(g.Length()), np.nan))
        ad["normal"].append(Q.get("aux.normal", lambda: vec(g.GetNormalVector()), [np.nan] * 3))
        ax = Q.get("aux.axes", lambda: axes(g), [[np.nan] * 3] * 3)
        ad["axis_x"].append(ax[0]); ad["axis_y"].append(ax[1]); ad["axis_z"].append(ax[2])
        ns = Q.get("aux.NSensitiveVolume", lambda: int(g.NSensitiveVolume()), 0)
        ad["n_sensitive"].append(ns)
        for j in range(ns):
            s = g.SensitiveVolume(j)
            sv["auxdet"].append(i); sv["sensitive"].append(j)
            sv["center"].append(vec(s.GetCenter()))
            sv["half_width1"].append(Q.get("auxs.hw1", lambda: float(s.HalfWidth1()), np.nan))
            sv["half_width2"].append(Q.get("auxs.hw2", lambda: float(s.HalfWidth2()), np.nan))
            sv["half_height"].append(Q.get("auxs.hh", lambda: float(s.HalfHeight()), np.nan))
            sv["length"].append(Q.get("auxs.len", lambda: float(s.Length()), np.nan))
            sv["normal"].append(Q.get("auxs.normal", lambda: vec(s.GetNormalVector()), [np.nan] * 3))
            ax = Q.get("auxs.axes", lambda: axes(s), [[np.nan] * 3] * 3)
            sv["axis_x"].append(ax[0]); sv["axis_y"].append(ax[1]); sv["axis_z"].append(ax[2])
    return ad, sv


def extract_clocks(ROOT, clkdata):
    d = {}
    for cname in ("TPCClock", "OpticalClock", "TriggerClock", "ExternalClock"):
        clk = Q.get(f"clk.{cname}", lambda: getattr(clkdata, cname)())
        if clk is None:
            continue
        for m, unit in (("TickPeriod", "us"), ("Frequency", "MHz"), ("FramePeriod", "us"),
                        ("FrameTicks", "ticks")):
            d[f"{cname}_{m}_{unit}"] = Q.get(f"clk.{cname}.{m}", lambda: float(getattr(clk, m)()))
    for m in ("TriggerTime", "BeamGateTime", "TriggerOffsetTPC"):
        d[f"{m}_us"] = Q.get(f"clk.{m}", lambda: float(getattr(clkdata, m)()))
    d["G4ToElecTime_at_g4t0_us"] = Q.get("clk.G4ToElecTime", lambda: float(clkdata.G4ToElecTime(0.)))
    d["TPCG4Time2Tick_at_g4t0"] = Q.get("clk.TPCG4Time2Tick", lambda: float(clkdata.TPCG4Time2Tick(0.)))
    d["TPCTick2TrigTime_at_tick0_us"] = Q.get("clk.TPCTick2TrigTime", lambda: float(clkdata.TPCTick2TrigTime(0.)))
    d["TPCTick2Time_at_tick0_us"] = Q.get("clk.TPCTick2Time", lambda: float(clkdata.TPCTick2Time(0.)))
    return d


def extract_detprop(ROOT, geom, wr, detdata):
    d = {}
    for m in ("Temperature", "ElectronLifetime", "Density", "NumberTimeSamples",
              "ReadOutWindowSize", "ElectronsToADC", "DriftVelocity"):
        d[m] = Q.get(f"detp.{m}", lambda: float(getattr(detdata, m)()))
    d["Efield_kV_cm"] = [Q.get("detp.Efield", lambda: float(detdata.Efield(g)), np.nan) for g in range(3)]
    d["DriftVelocity_cm_us"] = d.pop("DriftVelocity")
    d["Temperature_K"] = d.pop("Temperature")
    d["ElectronLifetime_us"] = d.pop("ElectronLifetime")
    d["Density_g_cm3"] = d.pop("Density")
    # x <-> tick conversion per plane, as a linear map  tick = a*x + b
    xt = dict(cryo=[], tpc=[], plane=[], ticks_per_cm=[], tick_at_x0=[],
              x_at_tick0=[], tick_at_plane=[])
    for c in range(int(geom.Ncryostats())):
        for t in range(int(geom.NTPC(ROOT.geo.CryostatID(c)))):
            for p in range(int(wr.Nplanes(ROOT.geo.TPCID(c, t)))):
                pid = ROOT.geo.PlaneID(c, t, p)
                t0 = Q.get("detp.ConvertXToTicks", lambda: float(detdata.ConvertXToTicks(0., pid)), np.nan)
                t1 = Q.get("detp.ConvertXToTicks", lambda: float(detdata.ConvertXToTicks(100., pid)), np.nan)
                xp = float(wr.Plane(pid).GetCenter().X())
                xt["cryo"].append(c); xt["tpc"].append(t); xt["plane"].append(p)
                xt["ticks_per_cm"].append((t1 - t0) / 100.)
                xt["tick_at_x0"].append(t0)
                xt["x_at_tick0"].append(Q.get("detp.ConvertTicksToX", lambda: float(detdata.ConvertTicksToX(0., pid)), np.nan))
                xt["tick_at_plane"].append(Q.get("detp.ConvertXToTicks", lambda: float(detdata.ConvertXToTicks(xp, pid)), np.nan))
    return d, xt


def extract_larprop(larp):
    d = {}
    for m in ("RadiationLength", "Argon39DecayRate", "AtomicNumber", "AtomicMass",
              "ExcitationEnergy", "ScintResolutionScale", "ScintFastTimeConst",
              "ScintSlowTimeConst", "ScintBirksConstant", "ScintByParticleType",
              "ProtonScintYield", "ProtonScintYieldRatio", "MuonScintYield",
              "MuonScintYieldRatio", "PionScintYield", "PionScintYieldRatio",
              "KaonScintYield", "KaonScintYieldRatio", "ElectronScintYield",
              "ElectronScintYieldRatio", "AlphaScintYield", "AlphaScintYieldRatio",
              "CerenkovLightEnabled", "ExtraMatProperties", "TpbTimeConstant"):
        v = Q.get(f"larp.{m}", lambda: getattr(larp, m)())
        if v is not None:
            d[m] = float(v) if not isinstance(v, bool) else bool(v)
    d["ScintYield"] = Q.get("larp.ScintYield", lambda: float(larp.ScintYield(True)))
    d["ScintPreScale"] = Q.get("larp.ScintPreScale", lambda: float(larp.ScintPreScale(True)))
    d["ScintYieldRatio"] = Q.get("larp.ScintYieldRatio", lambda: float(larp.ScintYieldRatio()))
    spectra = {}
    for m in ("FastScintSpectrum", "SlowScintSpectrum", "RIndexSpectrum",
              "AbsLengthSpectrum", "RayleighSpectrum", "TpbAbs", "TpbEm"):
        mp = Q.get(f"larp.{m}", lambda: getattr(larp, m)())
        if mp is None:
            continue
        e = [float(kv.first) for kv in mp]
        v = [float(kv.second) for kv in mp]
        spectra[m] = (np.asarray(e), np.asarray(v))
    return d, spectra


def fhicl_block(text, key):
    """Return the text of the top-level `services.<key>: { ... }` block."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip() == f"{key}: {{" and l.startswith("   ") and not l.startswith("    "):
            depth, out = 0, []
            for m in lines[i:]:
                depth += m.count("{") - m.count("}")
                out.append(m)
                if depth == 0:
                    return "\n".join(out)
    return None


def fhicl_flat_values(block):
    """Parse `key: value` lines at the first nesting level of a block."""
    vals = {}
    for l in block.splitlines()[1:]:
        if l.startswith("      ") and not l.startswith("       ") and ":" in l and not l.rstrip().endswith("{"):
            k, v = l.strip().split(":", 1)
            v = v.strip()
            if v in ("true", "false"):
                v = (v == "true")
            elif v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            vals[k] = v
    return vals


def extract_spacecharge(ROOT, services_fhicl, outdir):
    """Copy the SCE map file and also store its histograms as arrays.
    The configuration is read from the dumped services FHiCL (the table is
    called `SpaceCharge` in SBND, `SpaceChargeService` elsewhere)."""
    info = {}
    block = None
    for key in ("SpaceChargeService", "SpaceCharge"):
        block = fhicl_block(services_fhicl, key)
        if block:
            info["fhicl_key"] = key
            break
    if block is None:
        Q.failed["sce.config"] = "no SpaceCharge/SpaceChargeService table in services"
        return info, {}
    info["fhicl"] = block
    info.update(fhicl_flat_values(block))
    hists = {}
    fn = info.get("InputFilename", "")
    src = find_on_search_path(fn) if fn else None
    if src is None:
        Q.failed["sce.file"] = f"could not find {fn!r} on FW_SEARCH_PATH"
        return info, hists
    dst_dir = os.path.join(outdir, "spacecharge")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(fn))
    shutil.copy2(src, dst)
    info["source_path"] = src
    info["copied_to"] = os.path.relpath(dst, outdir)
    info["md5"] = md5sum(dst)
    f = ROOT.TFile.Open(dst)
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TH1"):
            Q.failed.setdefault("sce.nonhist_objects", "")
            Q.failed["sce.nonhist_objects"] += f"{key.GetName()}({obj.ClassName()}) "
            continue
        dim = obj.GetDimension()
        axes = [obj.GetXaxis(), obj.GetYaxis(), obj.GetZaxis()][:dim]
        nb = [a.GetNbins() for a in axes]
        edges = [np.array([a.GetBinLowEdge(i) for i in range(1, a.GetNbins() + 2)]) for a in axes]
        # ROOT global bin = ix + (nx+2)*(iy + (ny+2)*iz), incl. under/overflow
        full = np.array([obj.GetBinContent(i) for i in range(obj.GetNcells())],
                        dtype=np.float64)
        full = full.reshape([n + 2 for n in nb][::-1])            # (z,y,x)
        inner = full[tuple(slice(1, -1) for _ in nb)]
        vals = np.ascontiguousarray(inner.transpose()).astype(np.float32)  # (x,y,z)
        hists[obj.GetName()] = dict(values=vals, edges=edges, title=str(obj.GetTitle()),
                                    cls=obj.ClassName(),
                                    axis_titles=[str(a.GetTitle()) for a in axes])
    f.Close()
    return info, hists


def make_sce_provider(ROOT, cfg, sce_block):
    """Instantiate the experiment's SCE provider from the services FHiCL block
    (used only to compute validation reference values)."""
    spec = cfg.get("sce_provider")
    if not spec or not sce_block:
        return None
    import LArSoftUtils
    LArSoftUtils.SourceCode.loadHeaderFromUPS(spec["header"])
    LArSoftUtils.SourceCode.loadLibrary(spec["library"])
    body = "\n".join(sce_block.splitlines()[1:-1])   # strip "Key: {" ... "}"
    pset = ROOT.fhicl.ParameterSet.make(body)
    cls = ROOT
    for part in spec["cls"].split("."):
        cls = getattr(cls, part)
    return cls(pset)


def make_voxel_def(ROOT, pvs):
    """LArSoft's own photon-library voxelization, for validation voxel ids."""
    if not pvs or "NX" not in pvs:
        return None
    import LArSoftUtils
    LArSoftUtils.SourceCode.loadHeaderFromUPS("larsim/Simulation/PhotonVoxels.h")
    return ROOT.sim.PhotonVoxelDef(float(pvs["XMin"]), float(pvs["XMax"]), int(pvs["NX"]),
                                   float(pvs["YMin"]), float(pvs["YMax"]), int(pvs["NY"]),
                                   float(pvs["ZMin"]), float(pvs["ZMax"]), int(pvs["NZ"]))


def extract_validation(ROOT, geom, wr, detdata, sce=None, voxdef=None, n_inside=100,
                       n_outside=20, n_edge=80, seed=20261014):
    """Reference answers from LArSoft for random points, used by the python
    utility's unit tests: containing TPC, and per plane of that TPC the
    continuous wire coordinate, nearest wire and x->tick conversion; plus the
    SCE provider's offsets (n_edge extra points straddle the active-volume
    faces and the cathode to exercise the map clamping)."""
    rng = np.random.default_rng(seed)
    pts = []
    for c in range(int(geom.Ncryostats())):
        for t in range(int(geom.NTPC(ROOT.geo.CryostatID(c)))):
            a = geom.TPC(ROOT.geo.TPCID(c, t)).ActiveBoundingBox()
            lo = np.array([a.MinX(), a.MinY(), a.MinZ()]) + 0.5
            hi = np.array([a.MaxX(), a.MaxY(), a.MaxZ()]) - 0.5
            pts.append(rng.uniform(lo, hi, size=(n_inside, 3)))
    e = geom.DetectorEnclosureBox()
    pts.append(rng.uniform([e.MinX(), e.MinY(), e.MinZ()], [e.MaxX(), e.MaxY(), e.MaxZ()],
                           size=(n_outside, 3)))
    # edge points: inflate the union of active volumes by 6 cm, plus a cathode slab
    act = np.array([bbox(geom.TPC(ROOT.geo.TPCID(c, t)).ActiveBoundingBox())
                    for c in range(int(geom.Ncryostats()))
                    for t in range(int(geom.NTPC(ROOT.geo.CryostatID(c))))])
    lo, hi = act[:, :3].min(0) - 6.0, act[:, 3:].max(0) + 6.0
    pts.append(rng.uniform(lo, hi, size=(n_edge // 2, 3)))
    cat = rng.uniform(lo, hi, size=(n_edge - n_edge // 2, 3))
    cat[:, 0] = rng.uniform(-4.0, 4.0, size=len(cat))
    pts.append(cat)
    pts = np.concatenate(pts)
    maxpl = 3
    out = dict(point=pts, cryo=np.full(len(pts), -1), tpc=np.full(len(pts), -1),
               wire_coordinate=np.full((len(pts), maxpl), np.nan),
               nearest_wire=np.full((len(pts), maxpl), -1),
               tick=np.full((len(pts), maxpl), np.nan),
               sce_pos_offsets=np.full((len(pts), 3), np.nan),
               sce_cal_pos_offsets=np.full((len(pts), 3), np.nan),
               sce_cal_tpcid=np.full(len(pts), -1),
               sce_efield_offsets=np.full((len(pts), 3), np.nan),
               photonlib_voxel_id=np.full(len(pts), -2))
    for i, p in enumerate(pts):
        P = ROOT.geo.Point_t(*p)
        if sce is not None:
            out["sce_pos_offsets"][i] = Q.get("val.sce.GetPosOffsets", lambda: vec(sce.GetPosOffsets(P)), [np.nan] * 3)
            out["sce_efield_offsets"][i] = Q.get("val.sce.GetEfieldOffsets", lambda: vec(sce.GetEfieldOffsets(P)), [np.nan] * 3)
            # TPC id passed to GetCalPosOffsets: alternate between the two
            # conventions-relevant choices so both cathode clamps are exercised
            ctid = int(i % 2)
            out["sce_cal_tpcid"][i] = ctid
            out["sce_cal_pos_offsets"][i] = Q.get("val.sce.GetCalPosOffsets", lambda: vec(sce.GetCalPosOffsets(P, ctid)), [np.nan] * 3)
        if voxdef is not None:
            out["photonlib_voxel_id"][i] = Q.get("val.GetVoxelID", lambda: int(voxdef.GetVoxelID(P)), -2)
        tid = Q.get("val.FindTPCAtPosition", lambda: geom.FindTPCAtPosition(P))
        if tid is None or not tid.isValid:
            continue
        out["cryo"][i] = tid.Cryostat; out["tpc"][i] = tid.TPC
        for pl in range(min(maxpl, int(wr.Nplanes(tid)))):
            pid = ROOT.geo.PlaneID(tid.Cryostat, tid.TPC, pl)
            plane = wr.Plane(pid)
            out["wire_coordinate"][i, pl] = Q.get("val.WireCoordinate", lambda: float(plane.WireCoordinate(P)), np.nan)
            out["nearest_wire"][i, pl] = Q.get("val.NearestWireID", lambda: int(plane.NearestWireID(P).Wire), -1)
            out["tick"][i, pl] = Q.get("val.ConvertXToTicks", lambda: float(detdata.ConvertXToTicks(p[0], pid)), np.nan)
    return out


def extract_photonlib(ROOT, services_fhicl, outdir, tag, n_opdets):
    """Copy the photon visibility library and convert it to a dense HDF5 file.

    LArSoft's PhotonLibrary stores a sparse TTree `PhotonLibraryData`
    (Voxel, OpChannel, Visibility[, ReflVisibility]) over a regular voxel grid
    (PhotonVisibilityService NX/NY/NZ, X/Y/ZMin/Max). We densify it to
    float32 arrays [ix, iy, iz, opchannel] (same layout as the MicroBooNE
    lartpc/flashmatch photonlib cache), voxel id = ix + iy*NX + iz*NX*NY.
    Returns (info dict for /photon_visibility attrs, path of photonlib h5 or None).
    """
    info = {}
    block = fhicl_block(services_fhicl, "PhotonVisibilityService")
    if block is None:
        Q.failed["photonlib.config"] = "no PhotonVisibilityService table in services"
        return info, None
    info.update(fhicl_flat_values(block))   # scalars only; arrays stay in services_fhicl
    if info.get("DoNotLoadLibrary") or not info.get("LibraryFile"):
        info["note"] = "no library configured (semi-analytic model only; parameters in services_fhicl)"
        return info, None
    fn = info["LibraryFile"]
    src = find_on_search_path(fn)
    if src is None:
        Q.failed["photonlib.file"] = f"could not find {fn!r} on FW_SEARCH_PATH"
        return info, None
    dst_dir = os.path.join(outdir, "photonlib")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(fn))
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
    info["source_path"] = src
    info["copied_to"] = os.path.relpath(dst, outdir)
    info["md5"] = md5sum(dst)

    nx, ny, nz = int(info["NX"]), int(info["NY"]), int(info["NZ"])
    f = ROOT.TFile.Open(dst)
    key = f.GetKey("PhotonLibraryData")          # highest cycle, as LArSoft's Get()
    info["tree_cycle"] = int(key.GetCycle())
    tree = f.Get("PhotonLibraryData")
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    cols = [c for c in ("Voxel", "OpChannel", "Visibility", "ReflVisibility") if c in branches]
    info["tree_entries"] = int(tree.GetEntries())
    info["tree_branches"] = ",".join(branches)
    arr = ROOT.RDataFrame(tree).AsNumpy(cols)
    vox = np.asarray(arr["Voxel"], dtype=np.int64)
    och = np.asarray(arr["OpChannel"], dtype=np.int64)
    nch = max(int(och.max()) + 1, n_opdets)
    info["n_opchannels_in_library"] = int(och.max()) + 1
    nvox = nx * ny * nz
    bad = (vox < 0) | (vox >= nvox) | (och < 0)
    if bad.any():
        Q.failed["photonlib.out_of_range_rows"] = str(int(bad.sum()))
    vox, och = vox[~bad], och[~bad]
    h5path = os.path.join(outdir, f"photonlib_{tag}.h5")
    with h5py.File(h5path, "w") as fo:
        fo.attrs["schema_version"] = SCHEMA_VERSION
        fo.attrs["layout"] = "[ix, iy, iz, opchannel]; voxel id = ix + iy*NX + iz*NX*NY"
        for k, v in info.items():
            if isinstance(v, (str, int, float, bool, np.number)):
                fo.attrs[k] = v
        fo.attrs["grid_min"] = [float(info["XMin"]), float(info["YMin"]), float(info["ZMin"])]
        fo.attrs["grid_max"] = [float(info["XMax"]), float(info["YMax"]), float(info["ZMax"])]
        fo.attrs["grid_n"] = [nx, ny, nz]
        for col, name in (("Visibility", "visibility"), ("ReflVisibility", "refl_visibility")):
            if col not in arr:
                continue
            dense = np.zeros((nvox, nch), dtype=np.float32)
            dense[vox, och] = np.asarray(arr[col], dtype=np.float32)[~bad]
            dense = dense.reshape(nz, ny, nx, nch).transpose(2, 1, 0, 3)
            fo.create_dataset(name, data=np.ascontiguousarray(dense), compression="gzip",
                              compression_opts=6, shuffle=True,
                              chunks=(min(nx, 16), min(ny, 16), min(nz, 16), nch))
            info[f"{name}_nonzero"] = int(np.count_nonzero(dense))
            del dense
    f.Close()
    info["converted_h5"] = os.path.basename(h5path)
    return info, h5path


# ---------------------------------------------------------------------------
# light model (semi-analytic) tables + LArSoft reference visibilities
# ---------------------------------------------------------------------------
def fhicl_table(fcl, table, workdir):
    """(resolved text, python dict) of FHiCL table `table` defined in `fcl`."""
    path = os.path.join(workdir, "_detinfo_table_dump.fcl")
    with open(path, "w") as f:
        f.write(f'#include "{fcl}"\ntab: @local::{table}\n')
    try:
        text = subprocess.run(["fhicl-dump", path], capture_output=True, text=True,
                              check=True).stdout
        import fhicl                                  # fhiclpy (UPS)
        d = fhicl.make_pset(path)["tab"]
        return text, d
    finally:
        os.remove(path)


def write_fhicl_tree(group, d):
    """Nested FHiCL dict -> HDF5: tables -> subgroups, numeric (rectangular)
    sequences -> datasets, scalars/strings -> attrs, other sequences -> JSON attrs."""
    for k, v in d.items():
        if isinstance(v, dict):
            write_fhicl_tree(group.create_group(k), v)
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v, dtype=np.float64)
                if arr.dtype == object or arr.size == 0:
                    raise ValueError
                group.create_dataset(k, data=arr)
            except (ValueError, TypeError):
                group.attrs[k + "__json"] = json.dumps(v)
        elif v is None:
            continue
        else:
            group.attrs[k] = v


def extract_optical_model(ROOT, cfg, geom, wr, larp, outdir):
    """Dump the light-model FHiCL tables and, if configured, build the LArSoft
    semi-analytic model (patched copy compiled with cling, see
    larsoft_ref/semianalytic_ref.py). Returns (tables, sam_info, model|None)."""
    tables = {}
    for name, (fcl, table) in cfg.get("optical_fcl_tables", {}).items():
        res = Q.get(f"optical.table.{name}", lambda: fhicl_table(fcl, table, outdir))
        if res is not None:
            tables[name] = dict(fcl=fcl, table=table, text=res[0], tree=res[1])
    sam_cfg = cfg.get("semianalytic")
    if not sam_cfg or "vuv_hits" not in tables:
        return tables, {}, None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "larsoft_ref"))
    import semianalytic_ref as R
    cls = R.build(ROOT, os.environ["LARSIM_DIR"], sam_cfg["optical_path"][0])
    opath = R.make_optical_path(ROOT, sam_cfg["optical_path"][1])
    vuv = R.pset_from_text(ROOT, tables["vuv_hits"]["text"], "tab")
    vis = R.pset_from_text(ROOT, tables["vis_hits"]["text"], "tab") if "vis_hits" in tables \
        else ROOT.fhicl.ParameterSet()
    par = tables.get("pdfastsim_par", {}).get("tree", {})
    flags = dict(DoReflectedLight=bool(par.get("DoReflectedLight", False)),
                 IncludeAnodeReflections=bool(par.get("IncludeAnodeReflections", False)),
                 UseXeAbsorption=bool(par.get("UseXeAbsorption", False)))
    model = R.make_model(ROOT, cls, vuv, vis, opath, flags["DoReflectedLight"],
                         flags["IncludeAnodeReflections"], flags["UseXeAbsorption"],
                         geom, wr, larp)
    sam = dict(flags,
               optical_path_class=sam_cfg["optical_path"][1],
               reference_source="larsim " + os.environ.get("LARSIM_VERSION", "?") +
                                " SemiAnalyticalModel.cxx (patched constructor, cling)",
               vuv_absorption_length=float(model.absorptionLength()),
               cathode_centre=list(model.cathodeCentre()),
               anode_centre=list(model.anodeCentre()),
               drift_distance=float(model.driftDistance()),
               active_volumes=[list(model.activeVolume(i)) for i in range(model.nActiveVolumes())],
               opdet_type=list(model.opDetType()),
               opdet_orientation=list(model.opDetOrientation()),
               opdet_h=list(model.opDetH()),
               opdet_w=list(model.opDetW()))
    return tables, sam, model


def optical_reference_points(ROOT, geom, rng):
    """Points for the semi-analytic reference: random in each TPC active volume,
    line scans through the detector, and points close in front of detectors."""
    pts = []
    act = []
    for c in range(int(geom.Ncryostats())):
        for t in range(int(geom.NTPC(ROOT.geo.CryostatID(c)))):
            a = geom.TPC(ROOT.geo.TPCID(c, t)).ActiveBoundingBox()
            lo = np.array([a.MinX(), a.MinY(), a.MinZ()]); hi = np.array([a.MaxX(), a.MaxY(), a.MaxZ()])
            act.append((lo, hi))
            pts.append(rng.uniform(lo + 0.5, hi - 0.5, size=(150, 3)))
    lo = np.min([a[0] for a in act], 0); hi = np.max([a[1] for a in act], 0)
    c = 0.5 * (lo + hi)
    pts.append(np.stack([np.linspace(lo[0] + 0.5, hi[0] - 0.5, 161), np.full(161, c[1]), np.full(161, c[2])], 1))
    for x in (lo[0] * 0.5, hi[0] * 0.5):
        pts.append(np.stack([np.full(101, x), np.full(101, c[1]), np.linspace(lo[2] + 0.5, hi[2] - 0.5, 101)], 1))
        pts.append(np.stack([np.full(81, x), np.linspace(lo[1] + 0.5, hi[1] - 0.5, 81), np.full(81, c[2])], 1))
    # close to optical detectors: 5-60 cm inside the active volume, at the
    # detector's (y, z) (+-5 cm), on the detector's side of the cathode
    nop = int(geom.NOpDets())
    for i in rng.choice(nop, size=min(40, nop), replace=False):
        o = geom.OpDetGeoFromOpDet(int(i)).GetCenter()
        face = hi[0] if o.X() > c[0] else lo[0]
        x = face - np.sign(o.X() - c[0]) * rng.uniform(5, 60)
        pts.append(np.array([[x, o.Y() + rng.uniform(-5, 5), o.Z() + rng.uniform(-5, 5)]]))
    return np.concatenate(pts)


def extract_optical_validation(ROOT, geom, model, seed=20261015):
    rng = np.random.default_rng(seed)
    pts = optical_reference_points(ROOT, geom, rng)
    nop = int(geom.NOpDets())
    direct = np.zeros((len(pts), nop)); refl = np.full((len(pts), nop), np.nan)
    v = ROOT.std.vector("double")()
    do_refl = True
    for i, p in enumerate(pts):
        P = ROOT.geo.Point_t(*p)
        model.detectedDirectVisibilities(v, P)
        direct[i] = np.fromiter(v, dtype=np.float64, count=nop)
        if do_refl:
            r = Q.get("val.sam.reflected", lambda: (model.detectedReflectedVisibilities(v, P, False),
                                                    np.fromiter(v, dtype=np.float64, count=nop))[1])
            if r is None:
                do_refl = False
            else:
                refl[i] = r
    return dict(point=pts, direct=direct, reflected=refl)


def copy_aux_files(globs, outdir):
    copied = []
    for pat in globs:
        for d in os.environ.get("FW_SEARCH_PATH", "").split(":"):
            for src in glob.glob(os.path.join(d, pat)) + glob.glob(os.path.join(d, "*", pat)):
                if not os.path.isfile(src):
                    continue
                dst_dir = os.path.join(outdir, "aux")
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied.append((src, os.path.relpath(dst, outdir), md5sum(dst)))
    return copied


def fhicl_dump(services_fcl, service_table, workdir):
    fcl = os.path.join(workdir, "_detinfo_services_dump.fcl")
    with open(fcl, "w") as f:
        f.write(f'#include "{services_fcl}"\nservices: @local::{service_table}\n')
    try:
        return subprocess.run(["fhicl-dump", fcl], capture_output=True, text=True,
                              check=True).stdout
    except Exception as e:  # noqa: BLE001
        Q.failed["fhicl-dump"] = str(e)
        return ""
    finally:
        os.remove(fcl)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--services-fcl", default=None, help="override services fcl")
    ap.add_argument("--service-table", default=None, help="override services table")
    ap.add_argument("--skip-photonlib", action="store_true",
                    help="do not copy/convert the photon visibility library")
    args = ap.parse_args()
    cfg = dict(DETECTORS[args.detector])
    if args.services_fcl:
        cfg["services_fcl"] = args.services_fcl
    if args.service_table:
        cfg["service_table"] = args.service_table

    os.makedirs(args.outdir, exist_ok=True)
    version = os.environ.get(f"{cfg['product']}_VERSION", "unknown")
    t_start = datetime.datetime.now(datetime.timezone.utc)

    S = __import__(cfg["services_module"])
    from ROOTutils import ROOT
    S.ServiceManager.setConfiguration(cfg["services_fcl"], cfg["service_table"])
    geom = S.ServiceManager("Geometry")
    wr = S.ServiceManager("WireReadout")
    clk = S.ServiceManager("DetectorClocks")
    detp = S.ServiceManager("DetectorProperties")
    larp = S.ServiceManager("LArProperties")
    aux = Q.get("svc.AuxDetGeometry", lambda: S.ServiceManager("AuxDetGeometry"))

    clkdata = clk.DataForJob()
    detdata = detp.DataFor(clkdata)

    print("extracting geometry ...", flush=True)
    G = extract_geometry(ROOT, geom, wr)
    print("extracting aux dets ...", flush=True)
    AD, SV = extract_auxdets(ROOT, aux) if aux is not None else ({}, {})
    print("extracting clocks / properties ...", flush=True)
    CLK = extract_clocks(ROOT, clkdata)
    DETP, XT = extract_detprop(ROOT, geom, wr, detdata)
    LARP, SPECTRA = extract_larprop(larp)
    services_fhicl = fhicl_dump(cfg["services_fcl"], cfg["service_table"], args.outdir)
    print("extracting space charge ...", flush=True)
    SCE, HISTS = extract_spacecharge(ROOT, services_fhicl, args.outdir)
    aux_files = copy_aux_files(cfg["aux_file_globs"], args.outdir)
    if args.skip_photonlib:
        PVS, plib_h5 = {"note": "skipped (--skip-photonlib)"}, None
    else:
        print("extracting photon visibility library ...", flush=True)
        PVS, plib_h5 = Q.get("photonlib", lambda: extract_photonlib(
            ROOT, services_fhicl, args.outdir, f"{args.detector}_{version}",
            int(G["detector"]["n_opdets"])), ({"note": "extraction failed"}, None))
    print("extracting light model tables / semi-analytic reference ...", flush=True)
    OPT_TABLES, SAM, sam_model = Q.get("optical_model", lambda: extract_optical_model(
        ROOT, cfg, geom, wr, larp, args.outdir), ({}, {}, None))
    OPT_VAL = (Q.get("optical_validation", lambda: extract_optical_validation(ROOT, geom, sam_model))
               if sam_model is not None else None)
    print("computing validation reference points ...", flush=True)
    sce = Q.get("val.sce_provider", lambda: make_sce_provider(ROOT, cfg, SCE.get("fhicl")))
    voxdef = Q.get("val.voxel_def", lambda: make_voxel_def(ROOT, PVS))
    VAL = extract_validation(ROOT, geom, wr, detdata, sce=sce, voxdef=voxdef)

    det = G["detector"]["name"]
    h5path = os.path.join(args.outdir, f"detinfo_{args.detector}_{version}.h5")
    print(f"writing {h5path}", flush=True)
    with h5py.File(h5path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["detector"] = args.detector
        f.attrs["geometry_name"] = det

        # ---- provenance
        pv = f.create_group("provenance")
        pv.attrs["experiment_product"] = cfg["product"].lower()
        pv.attrs["experiment_version"] = version
        pv.attrs["qualifiers"] = os.environ.get("DETINFO_QUALS", "")
        pv.attrs["services_fcl"] = cfg["services_fcl"]
        pv.attrs["service_table"] = cfg["service_table"]
        pv.attrs["gdml_file"] = G["detector"]["gdml_file"]
        pv.attrs["extracted_utc"] = t_start.isoformat()
        pv.attrs["host"] = socket.gethostname()
        pv.attrs["extractor"] = os.path.abspath(__file__)
        pv.attrs["extractor_md5"] = md5sum(__file__)
        pv.attrs["extractor_git_commit"] = os.environ.get("DETINFO_GIT_COMMIT", "")
        pv.attrs["container"] = os.environ.get("DETINFO_CONTAINER", "")
        pv.attrs["python"] = sys.version.split()[0]
        vers = {p.lower(): os.environ[f"{p}_VERSION"] for p in VERSION_PRODUCTS
                if f"{p}_VERSION" in os.environ}
        pv.attrs["product_versions_json"] = json.dumps(vers, sort_keys=True)
        pv.create_dataset("services_fhicl", data=services_fhicl, dtype=h5py.string_dtype())
        pv.attrs["aux_files_json"] = json.dumps(
            [dict(source=s, copied_to=d, md5=m) for s, d, m in aux_files])
        pv.attrs["failed_queries_json"] = json.dumps(Q.failed, sort_keys=True, indent=1)

        # ---- geometry
        g = f.create_group("geometry")
        write_scalars(g, "detector", G["detector"])
        write_table(g, "cryostats", G["cryostats"])
        write_table(g, "tpcs", G["tpcs"])
        write_table(g, "planes", G["planes"],
                    attrs=dict(view_enum="geo::View_t: 0=kU 1=kV 2=kZ(=kW) 3=kY 4=kX 5=k3D 6=kUnknown",
                               signal_type_enum="geo::SigType_t: 0=kInduction 1=kCollection 2=kMysteryType",
                               theta_z_units="rad"))
        write_table(g, "wires", G["wires"])
        write_table(g, "opdets", G["opdets"])
        write_table(g, "opchannels", G["opchannels"])
        if AD:
            write_table(g, "auxdets", AD)
            write_table(g, "auxdet_sensitive", SV)

        # ---- clocks / detector / LAr properties
        write_scalars(f, "clocks", CLK)
        dp = write_scalars(f, "detector_properties", DETP)
        write_table(dp, "x_to_ticks", XT,
                    attrs=dict(note="tick = ticks_per_cm * x + tick_at_x0 (TPC ticks, "
                                    "from DetectorPropertiesData::ConvertXToTicks)"))
        lp = write_scalars(f, "lar_properties", LARP)
        sg = lp.create_group("spectra")
        for name, (e, v) in SPECTRA.items():
            s = sg.create_group(name)
            s.create_dataset("energy_eV", data=e)
            s.create_dataset("value", data=v)

        # ---- light model: FHiCL tables + semi-analytic geometry descriptors
        if OPT_TABLES or SAM:
            om = f.create_group("optical_model")
            for name, t in OPT_TABLES.items():
                tg = om.create_group(name)
                tg.attrs["fcl"] = t["fcl"]
                tg.attrs["table"] = t["table"]
                tg.create_dataset("fhicl", data=t["text"], dtype=h5py.string_dtype())
                write_fhicl_tree(tg.create_group("params"), t["tree"])
            if SAM:
                sg = om.create_group("semianalytic")
                for k in ("DoReflectedLight", "IncludeAnodeReflections", "UseXeAbsorption",
                          "optical_path_class", "reference_source", "vuv_absorption_length",
                          "cathode_centre", "anode_centre", "drift_distance"):
                    sg.attrs[k] = SAM[k]
                sg.create_dataset("active_volumes", data=np.asarray(SAM["active_volumes"]))
                write_table(sg, "opdets", dict(
                    opdet=np.arange(len(SAM["opdet_type"])), type=SAM["opdet_type"],
                    orientation=SAM["opdet_orientation"], h=SAM["opdet_h"], w=SAM["opdet_w"]),
                    attrs=dict(type_enum="0=rectangle (X-ARAPUCA/bar) 1=dome PMT 2=disk PMT",
                               orientation_enum="0=faces x (anode/cathode) 1=faces y 2=faces z",
                               note="as built by SemiAnalyticalModel::opticalDetectors() from OpDetGeo"))
        if OPT_VAL is not None:
            write_table(f, "validation_optical", OPT_VAL,
                        attrs=dict(note="LArSoft SemiAnalyticalModel reference: direct = "
                                        "detectedDirectVisibilities, reflected = "
                                        "detectedReflectedVisibilities(AnodeMode=false); "
                                        "columns indexed by opdet"))

        # ---- photon visibility (library itself in photonlib_<det>_<ver>.h5)
        pg = f.create_group("photon_visibility")
        for k, v in PVS.items():
            pg.attrs[k] = v

        # ---- space charge
        sc = f.create_group("spacecharge")
        for k, v in SCE.items():
            sc.attrs[k] = v
        hg = sc.create_group("histograms")
        for name, h in HISTS.items():
            d = hg.create_dataset(name, data=h["values"], compression="gzip",
                                  compression_opts=9, shuffle=True)
            d.attrs["class"] = h["cls"]; d.attrs["title"] = h["title"]
            d.attrs["axis_titles"] = np.asarray(h["axis_titles"], dtype=h5py.string_dtype())
            for ax, e in zip("xyz", h["edges"]):
                d.attrs[f"edges_{ax}"] = e

        vg = write_table(f, "validation", VAL,
                         attrs=dict(note="LArSoft reference answers for random points; "
                                         "wire_coordinate/nearest_wire/tick are per plane of the "
                                         "containing TPC (nan/-1 when outside any TPC); sce_* from "
                                         "the experiment SCE provider (nan if unavailable); "
                                         "sce_cal_pos_offsets used TPCid=sce_cal_tpcid"))

    dt = (datetime.datetime.now(datetime.timezone.utc) - t_start).total_seconds()
    print(f"done in {dt:.0f}s -> {h5path}")
    print(f"failed queries ({len(Q.failed)}):")
    for k, v in sorted(Q.failed.items()):
        print(f"   {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
