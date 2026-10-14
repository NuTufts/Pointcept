"""
Build LArSoft service providers WITHOUT art and without an experiment python
loader (SBND has SBNDservices.py; MicroBooNE/ICARUS releases may not): compile
small factory functions with cling around larcorealg's standalone geometry
setup and lardataalg's provider test helpers -- the same calls those loaders make.

    provs = build(ROOT, services_pset, readout=dict(header=..., library=..., cls=...))
    provs["geom"], provs["wr"], provs["aux"], provs["clk"], provs["detp"], provs["larp"]

`services_pset` is the fhicl::ParameterSet of the `services` table (resolved).
`readout`: the experiment's WireReadoutGeom class if it is not the standard one
(MicroBooNE v10: geo::WireReadoutUBooNEGeom(GeometryCore const*, ParameterSet)).
"""

_DECLARED = set()

_COMMON = r'''
#include "larcorealg/Geometry/StandaloneGeometrySetup.h"
#include "larcorealg/Geometry/GeoObjectSorterStandard.h"
#include "larcorealg/Geometry/WireReadoutSorterStandard.h"
#include "larcorealg/Geometry/WireReadoutStandardGeom.h"
#include "larcorealg/Geometry/AuxDetGeoObjectSorterStandard.h"
#include "lardataalg/DetectorInfo/LArPropertiesStandardTestHelpers.h"
#include "lardataalg/DetectorInfo/DetectorClocksStandardTestHelpers.h"
#include "lardataalg/DetectorInfo/DetectorPropertiesStandardTestHelpers.h"
namespace detinfo_ref {
  geo::GeometryCore* makeGeometry(fhicl::ParameterSet const& p) {
    return lar::standalone::SetupGeometry<geo::GeoObjectSorterStandard>(p).release(); }
  geo::WireReadoutGeom* makeStandardReadout(fhicl::ParameterSet const& p, geo::GeometryCore const* g) {
    return lar::standalone::SetupReadout<geo::WireReadoutSorterStandard, geo::WireReadoutStandardGeom>(p, g).release(); }
  geo::AuxDetGeometryCore* makeAuxDet(fhicl::ParameterSet const& p) {
    return lar::standalone::SetupAuxDetGeometry<geo::AuxDetGeoObjectSorterStandard>(p).release(); }
  detinfo::LArPropertiesStandard* makeLArProp(fhicl::ParameterSet const& p) {
    return testing::setupProvider<detinfo::LArPropertiesStandard>(p).release(); }
  detinfo::DetectorClocksStandard* makeClocks(fhicl::ParameterSet const& p) {
    return testing::setupProvider<detinfo::DetectorClocksStandard>(p).release(); }
  detinfo::DetectorPropertiesStandard* makeDetProp(fhicl::ParameterSet const& p,
      geo::GeometryCore const* g, geo::WireReadoutGeom const* w, detinfo::LArProperties const* l) {
    return testing::setupProvider<detinfo::DetectorPropertiesStandard>(
      p, detinfo::DetectorPropertiesStandard::providers_type{g, w, l}).release(); }
}
'''

_LIBS = ("larcorealg_Geometry", "lardataalg_DetectorInfo")


def _load_lib(ROOT, name):
    for cand in (f"lib{name}.so", name):
        if ROOT.gSystem.Load(cand) >= 0:
            return
    raise RuntimeError(f"could not load library {name}")


def _declare(ROOT, key, code):
    if key in _DECLARED:
        return
    if not ROOT.gInterpreter.Declare(code):
        raise RuntimeError(f"cling failed to compile {key}")
    _DECLARED.add(key)


def prepare(ROOT):
    """Make fhiclcpp + the provider factories known to cling (idempotent)."""
    for lib in ("fhiclcpp", "cetlib", "cetlib_except") + _LIBS:
        try:
            _load_lib(ROOT, lib)
        except RuntimeError:
            if lib in _LIBS or lib == "fhiclcpp":
                raise
    # range-v3 (pulled in by larcorealg) includes <span> whenever __has_include(<span>) is
    # true -- which it is under cling's C++17 std modulemap, where <span> needs C++20 and
    # the interpreter then crashes. Its documented opt-out macro skips the optional
    # std::span integration (no effect on the provider code).
    _declare(ROOT, "range_span", "#define RANGES_WORKAROUND_MSVC_UNUSABLE_SPAN 1")
    _declare(ROOT, "fhicl", '#include "fhiclcpp/ParameterSet.h"')
    _declare(ROOT, "common", _COMMON)


def build(ROOT, services, readout=None, want_aux=True):
    prepare(ROOT)
    ns = ROOT.detinfo_ref
    get = lambda key: services.get[ROOT.fhicl.ParameterSet](key)
    has = lambda key: services.has_key(key)

    geom = ns.makeGeometry(get("Geometry"))
    wr_pset = get("WireReadout") if has("WireReadout") else ROOT.fhicl.ParameterSet()
    if readout:
        for lib in readout.get("libraries", ()):
            _load_lib(ROOT, lib)
        fn = "makeReadout_" + readout["cls"].replace(":", "_")
        _declare(ROOT, fn, f'#include "{readout["header"]}"\n'
                           f'namespace detinfo_ref {{ geo::WireReadoutGeom* {fn}('
                           f'geo::GeometryCore const* g, fhicl::ParameterSet const& p) '
                           f'{{ return new {readout["cls"]}({readout.get("ctor_args", "g, p")}); }} }}')
        wr = getattr(ns, fn)(geom, wr_pset)
    else:
        wr = ns.makeStandardReadout(wr_pset, geom)
    aux = None
    if want_aux and has("AuxDetGeometry"):
        try:
            aux = ns.makeAuxDet(get("AuxDetGeometry"))
        except Exception:                                        # noqa: BLE001
            aux = None
    larp = ns.makeLArProp(get("LArPropertiesService"))
    clk = ns.makeClocks(get("DetectorClocksService"))
    detp = ns.makeDetProp(get("DetectorPropertiesService"), geom, wr, larp)
    return dict(geom=geom, wr=wr, aux=aux, larp=larp, clk=clk, detp=detp)


def services_pset(ROOT, services_text):
    """fhicl::ParameterSet of the `services` table from fhicl-dump text of
    '#include "...fcl" / services: @local::<table>'."""
    prepare(ROOT)
    top = ROOT.fhicl.ParameterSet.make(services_text)
    return top.get[ROOT.fhicl.ParameterSet]("services")
