"""
Build a LArSoft-reference instance of larsim's phot::SemiAnalyticalModel inside a
python/gallery session (no art), by compiling a *patched copy* of the original
source with cling. Only the constructor's service lookups are replaced:

  fGeom        <- geo::GeometryCore const&     (from the experiment ServiceManager)
  fChannelMap  <- geo::WireReadoutGeom const&  (idem)
  AbsLengthSpectrum <- std::map<double,double> (LArPropertiesStandard::AbsLengthSpectrum())
  mf::Log*     <- a null stream

All physics code (visibilities, solid angles, Gaisser-Hillas, interpolation via
larsim's PhotonPropagationUtils) is the unmodified upstream text. The patched
class is renamed phot::SemiAnalyticalModelRef so it cannot collide with the
library symbol.
"""
import os
import re

CLASS = "SemiAnalyticalModelRef"


def _read(path):
    with open(path) as f:
        return f.read()


def patched_source(larsim_dir):
    src_dir = os.path.join(larsim_dir, "source", "larsim", "PhotonPropagation")
    h = _read(os.path.join(src_dir, "SemiAnalyticalModel.h"))
    c = _read(os.path.join(src_dir, "SemiAnalyticalModel.cxx"))

    # ---- header: rename class, new constructor signature, store abs spectrum
    h = h.replace("SEMIANALYTICALMODEL_H", "SEMIANALYTICALMODELREF_H")
    h = re.sub(r"\bSemiAnalyticalModel\b", CLASS, h)
    # our extra parameters follow the defaulted flags, so drop the flag defaults
    h = h.replace("const bool doReflectedLight = false,", "const bool doReflectedLight,")
    h = h.replace("const bool includeAnodeReflections = false,", "const bool includeAnodeReflections,")
    h = h.replace("const bool useXeAbsorption = false);",
                  "const bool useXeAbsorption,\n"
                  "                        geo::GeometryCore const& geom,\n"
                  "                        geo::WireReadoutGeom const& wireReadout,\n"
                  "                        std::map<double, double> absLengthSpectrum);\n"
                  "    // reference-build accessors\n"
                  "    double absorptionLength() const { return fvuv_absorption_length; }\n"
                  "    std::vector<double> cathodeCentre() const { return {fcathode_centre[0], fcathode_centre[1], fcathode_centre[2]}; }\n"
                  "    std::vector<double> anodeCentre() const { return {fanode_centre[0], fanode_centre[1], fanode_centre[2]}; }\n"
                  "    double driftDistance() const { return fDriftDistance; }\n"
                  "    std::vector<int> opDetType() const { std::vector<int> v; for (auto const& o : fOpDetector) v.push_back(o.type); return v; }\n"
                  "    std::vector<int> opDetOrientation() const { std::vector<int> v; for (auto const& o : fOpDetector) v.push_back(o.orientation); return v; }\n"
                  "    std::vector<double> opDetH() const { std::vector<double> v; for (auto const& o : fOpDetector) v.push_back(o.h); return v; }\n"
                  "    std::vector<double> opDetW() const { std::vector<double> v; for (auto const& o : fOpDetector) v.push_back(o.w); return v; }\n"
                  "    std::vector<double> activeVolume(size_t i) const { auto const& b = fActiveVolumes.at(i); return {b.MinX(), b.MinY(), b.MinZ(), b.MaxX(), b.MaxY(), b.MaxZ()}; }\n"
                  "    size_t nActiveVolumes() const { return fActiveVolumes.size(); }")
    h = h.replace("    std::shared_ptr<OpticalPath> fOpticalPath;",
                  "    std::shared_ptr<OpticalPath> fOpticalPath;\n"
                  "    std::map<double, double> fAbsLengthSpectrum;")
    # members are const-initialized in declaration order: fAbsLengthSpectrum is
    # only read in the constructor body (VUVAbsorptionLength), after init.

    # ---- source
    c = c.replace('#include "SemiAnalyticalModel.h"', "")
    for inc in ('#include "larcore/CoreUtils/ServiceUtil.h"',
                '#include "larcore/Geometry/Geometry.h"',
                '#include "larcore/Geometry/WireReadout.h"',
                '#include "lardata/DetectorInfoServices/LArPropertiesService.h"',
                '#include "messagefacility/MessageLogger/MessageLogger.h"'):
        c = c.replace(inc, "")
    c = re.sub(r"\bSemiAnalyticalModel\b", CLASS, c)
    c = c.replace("const bool useXeAbsorption)",
                  "const bool useXeAbsorption,\n"
                  "                                           geo::GeometryCore const& geom,\n"
                  "                                           geo::WireReadoutGeom const& wireReadout,\n"
                  "                                           std::map<double, double> absLengthSpectrum)", 1)
    c = c.replace("fGeom{*(lar::providerFrom<geo::Geometry>())}", "fGeom{geom}")
    c = c.replace("fChannelMap(art::ServiceHandle<geo::WireReadout>()->Get())", "fChannelMap(wireReadout)")
    c = c.replace("    , fOpticalPath(OpticalPath)\n  {",
                  "    , fOpticalPath(OpticalPath)\n    , fAbsLengthSpectrum(absLengthSpectrum)\n  {")
    c = c.replace("lar::providerFrom<detinfo::LArPropertiesService>()->AbsLengthSpectrum()",
                  "fAbsLengthSpectrum")
    c = re.sub(r"mf::Log(Info|Warning|Error|Debug)\(\"[A-Za-z]+\"\)", "phot_ref_null_log()", c)

    prelude = (
        '#include "larcorealg/Geometry/GeometryCore.h"\n'
        '#include "larcorealg/Geometry/WireReadoutGeom.h"\n'
        '#include "larcorealg/Geometry/OpDetGeo.h"\n'
        '#include "larcorealg/Geometry/CryostatGeo.h"\n'
        '#include "larcorealg/Geometry/TPCGeo.h"\n'
        '#include <ostream>\n#include <streambuf>\n'
        'namespace { struct PhotRefNullBuf : std::streambuf { int overflow(int c) override { return c; } };\n'
        '  inline std::ostream& phot_ref_null_log() { static PhotRefNullBuf b; static std::ostream s(&b); return s; } }\n')
    return prelude + h, c


def check_patch(h, c):
    """The patch must have replaced every service lookup; fail loudly otherwise."""
    bad = [s for s in ("providerFrom", "ServiceHandle", "mf::Log") if s in c or s in h]
    if bad:
        raise RuntimeError(f"semi-analytic reference patch incomplete: {bad} still present")


def _has_ns(ROOT, name):
    try:
        getattr(ROOT, name)
        return True
    except AttributeError:
        return False


def build(ROOT, larsim_dir, optical_path_header, extra_libraries=()):
    """Compile the patched class (once per process). Returns the class."""
    import LArSoftUtils
    existing = getattr(getattr(ROOT, "phot", None), CLASS, None) if _has_ns(ROOT, "phot") else None
    if existing is not None:
        return existing
    for lib in ("larsim_PhotonPropagation", "larsim_IonizationScintillation") + tuple(extra_libraries):
        LArSoftUtils.SourceCode.loadLibrary(lib)
    h, c = patched_source(larsim_dir)
    check_patch(h, c)
    if optical_path_header:
        ok = ROOT.gInterpreter.Declare(f'#include "{optical_path_header}"')
        if not ok:
            raise RuntimeError(f"could not include {optical_path_header}")
    if not ROOT.gInterpreter.Declare(h):
        raise RuntimeError("cling failed on patched SemiAnalyticalModel header")
    if not ROOT.gInterpreter.Declare(c):
        raise RuntimeError("cling failed on patched SemiAnalyticalModel source")
    return getattr(ROOT.phot, CLASS)


def make_model(ROOT, cls, vuv_pset, vis_pset, optical_path, do_reflected, anode_reflections,
               use_xe, geom, wire_readout, larprop):
    spec = ROOT.std.map("double", "double")()
    for kv in larprop.AbsLengthSpectrum():
        spec[kv.first] = kv.second
    return cls(vuv_pset, vis_pset, optical_path, bool(do_reflected), bool(anode_reflections),
               bool(use_xe), geom, wire_readout, spec)


def make_optical_path(ROOT, cls_name):
    """shared_ptr<phot::OpticalPath> of the experiment's optical-path tool class
    (constructed with an empty ParameterSet, as the tools ignore it)."""
    fn = "detinfo_make_optical_path_" + cls_name.replace(":", "_")
    if not hasattr(ROOT, fn):
        ROOT.gInterpreter.Declare(
            f"std::shared_ptr<phot::OpticalPath> {fn}() {{ fhicl::ParameterSet ps; "
            f"return std::make_shared<{cls_name}>(ps); }}")
    return getattr(ROOT, fn)()


def pset_from_text(ROOT, text, key):
    """fhicl::ParameterSet of table `key` from fully resolved FHiCL text (fhicl-dump output)."""
    top = ROOT.fhicl.ParameterSet.make(text)
    return top.get[ROOT.fhicl.ParameterSet](key)
