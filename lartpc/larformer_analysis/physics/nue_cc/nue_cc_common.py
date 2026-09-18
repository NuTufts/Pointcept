"""Shared constants, sample registry and selection logic for the nue-CC
inclusive analysis.

Every plotter in this folder used to re-declare `FULL_EXT_SCALE`, `COMPONENTS`,
`COLORS` and its own copy of the `base()` selection mask -- and they had drifted
(`nue_cc_larpid_scores.py` applied a two-sided flash band where
`nue_cc_overlay.py` applied a one-sided cut, on the same tables). This module is
the single source of truth for all of it.

Chain version: **v2_s1ep2p8cew6** (see
`lartpc/larformer_analysis/model_and_output_file_versions.md`). recal3 is baked
into `showerRecoE` in these ntuples, so NO analysis-side `--recal-gamma-*` is
applied anywhere in this folder.
"""
import numpy as np

# --------------------------------------------------------------------------
# Samples (v2_s1ep2p8cew6)
# --------------------------------------------------------------------------
_D = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts"
_NUE_DIR = ("/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/"
            "mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge")

#: role -> (ntuple path, note). The nue signal ntuple is produced by the cew6
#: reprocessing campaign (Phase 1); TAG = nue_overlay_s1ep2p8cew6_run3.
SAMPLES = {
    "nue": (f"{_NUE_DIR}/dlgen2_larformer_ntuple_nue_overlay_s1ep2p8cew6_run3.root",
            "intrinsic-nue signal; POT 4.709e22 over 2231 good filenos"),
    "bnb": (f"{_D}/larformer_mcoverlay67k_s1ep2p8cew6/"
            "dlgen2_larformer_ntuple_mc_overlay_s1ep2p8cew6_run3.root",
            "BNB-nu overlay background; 67,211 evts; POT 8.394e19"),
    "ext": (f"{_D}/larformer_extbnb200k_s1ep2p8cew6/"
            "dlgen2_larformer_ntuple_extbnb200k_s1ep2p8cew6.root",
            "EXT-BNB beam-off cosmic; 200,000 evts (a subset of the full sample)"),
    "data": (f"{_D}/larformer_bnb5e19_s1ep2p8cew6/"
             "dlgen2_larformer_ntuple_bnb5e19_s1ep2p8cew6.root",
             "bnb5e19 beam data; 176,302 evts; POT 4.4e19"),
}

#: The July old-chain nue ntuple, kept only for A/B regression against the
#: recorded July cutflow (79,356 -> 69,901 -> 58,552). NOT for physics.
NUE_NTUPLE_JULY = f"{_NUE_DIR}/dlgen2_larformer_ntuple_mcc9_v29e_nue_overlay.root"

DEFAULT_POT = 4.4e19          # Tufts bnb5e19 beam livetime

# --- RUN 1, table-gamma (versions doc section 3c) --------------------------
# Data, beam-on MC and beam-off EXT are ALL run 1 and share one calibrated
# flash light-yield table, so run-1-vs-run-3 detector differences can no
# longer explain a data/MC disagreement in these plots.
_L = "/cluster/tufts/wongjiradlab/larbys/data/larformer"
SAMPLES_RUN1 = {
    "nue": (f"{_L}/run1_nueintrinsics_half/dlgen2_larformer_ntuple_nue_run1_half.root",
            "run-1 intrinsic-nue RESERVED half; 82,624 evts; POT 4.9065e22"),
    "bnb": (f"{_L}/run1_bnboverlay_half/dlgen2_larformer_ntuple_bnbovl_run1_half.root",
            "run-1 BNB-nu overlay TRAINPOOL half; 182,069 evts; POT 2.2903e20 "
            "(2.1292e20 after the segmenter-training exclusion)"),
    "ext": (f"{_L}/run1_C1_extbnb_half/dlgen2_larformer_ntuple_extbnb_run1_half.root",
            "run-1 C1 EXT-BNB, 25% of the full sample; 104,516 evts"),
    "data": (f"{_L}/bnb5e19_run1_table/dlgen2_larformer_ntuple_bnb5e19_run1_table.root",
             "bnb5e19 run-1 open data, table-gamma reprocess; 176,302 evts; POT 4.4e19"),
}
#: The run-1 overlay half's first 350 filenos fed the cew6 segmenter
#: training; they are dropped whole and the POT re-summed (kept_pot) --
#: nue_cc_analysis.py --exclude-rows.
RUN1_BNB_EXCLUSION = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/"
    "lartpc/larformer_analysis/physics/pi0mass_peak/run1ovl_trainpool_exclusion.npz")

# Run-1 EXT normalization (lartpc/data_prep/README.md): EXT is scaled by
# SPILLS. Full run-1 C1 EXT-BNB = 94,414,115 spills, of which 25% was
# processed; bnb5e19 beam = 23,090,946 spills.
#
# NOTE: the same README line quotes "25% -> 5,772,737 spills", which is 25% of
# the BEAM count (23,090,946), not of the EXT count -- an arithmetic slip. The
# correct processed EXT spill count is 0.25 x 94,414,115 = 23,603,529.
RUN1_BEAM_SPILLS = 23_090_946
RUN1_EXT_FULL_SPILLS = 94_414_115
RUN1_EXT_FRACTION = 0.25
RUN1_EXT_SCALE = RUN1_BEAM_SPILLS / (RUN1_EXT_FULL_SPILLS * RUN1_EXT_FRACTION)

# --------------------------------------------------------------------------
# EXT normalization
# --------------------------------------------------------------------------
# Per-EXT-event weight = SPILL_RATIO / f, where f is the fraction of the FULL
# EXT sample actually processed (pi0mass_peak/datamc_ext_overlay.py:10-11).
#
# The July analysis used the FULL 668,388-event EXT ntuple, so its weight was
# the bare spill ratio. The cew6 EXT file is a 200,000-event SUBSET -- carrying
# the old default forward undercounts EXT by 3.3x.
SPILL_RATIO = 0.17682554549

#: fraction of the full EXT sample represented by the cew6 200k file
EXT200K_FRACTION = 0.29923

EXT_SCALE_ALL = 0.5909        # cew6 200k, all rows
EXT_SCALE_ANALYSIS_HALF = 1.1818   # rows >= 100k (shower-BDT analysis half)
EXT_SCALE_ANALYSIS_ODD = 2.3636    # rows >= 100k AND odd event (event-BDT too)

#: EXT rows below this index are the per-shower-BDT TRAINING half and must be
#: excluded from any plot that uses a shower-BDT score.
EXT_TRAIN_ROW_MAX = 100000


#: Everything that differs between the analysis sample sets. `hygiene`: the
#: run-3 EXT rows < 100k (and even events) trained the shower/event BDTs; no
#: classifier ever saw a run-1 event, so run 1 needs no hygiene at all.
SAMPLE_SETS = {
    "run3_cew6": dict(samples=SAMPLES, ext_scale_all=EXT_SCALE_ALL, hygiene=True,
                      label="run 3 cew6 (legacy flash gamma)"),
    "run1_tg": dict(samples=SAMPLES_RUN1, ext_scale_all=RUN1_EXT_SCALE,
                    hygiene=False, label="run 1 table-gamma"),
}
DEFAULT_SAMPLE_SET = "run3_cew6"      # what tables built before tagging were

_ACTIVE = {"set": DEFAULT_SAMPLE_SET, "ext_override": None, "ext_rescale": 1.0}


def set_sample_set(name, ext_scale_override=None, ext_rescale=1.0):
    """Select the sample set whose EXT weight and hygiene rules apply.

    `ext_scale_override` REPLACES the per-event EXT weight; `ext_rescale`
    MULTIPLIES whatever weight is in effect (e.g. 1.3 to test a data-driven
    correction on top of the spill-based normalization).
    """
    if name not in SAMPLE_SETS:
        raise KeyError(f"unknown sample set {name!r}; known: {list(SAMPLE_SETS)}")
    _ACTIVE["set"] = name
    _ACTIVE["ext_override"] = ext_scale_override
    _ACTIVE["ext_rescale"] = 1.0 if ext_rescale is None else float(ext_rescale)


def active_set():
    return _ACTIVE["set"]


def active_label():
    lab = SAMPLE_SETS[_ACTIVE["set"]]["label"]
    if _ACTIVE["ext_override"] is not None:
        lab += f" [EXT weight overridden to {_ACTIVE['ext_override']:.4g}]"
    if _ACTIVE["ext_rescale"] != 1.0:
        lab += f" [EXT x{_ACTIVE['ext_rescale']:g}]"
    return lab


def ext_scale(shower_bdt_used=False, event_bdt_used=False):
    """Per-event EXT weight for the ACTIVE sample set under its hygiene rules,
    times the optional --ext-rescale factor."""
    return _ACTIVE["ext_rescale"] * _ext_scale_nominal(shower_bdt_used,
                                                       event_bdt_used)


def _ext_scale_nominal(shower_bdt_used=False, event_bdt_used=False):
    if _ACTIVE["ext_override"] is not None:
        return _ACTIVE["ext_override"]
    ss = SAMPLE_SETS[_ACTIVE["set"]]
    if not ss["hygiene"]:
        return ss["ext_scale_all"]
    if event_bdt_used:
        return EXT_SCALE_ANALYSIS_ODD
    if shower_bdt_used:
        return EXT_SCALE_ANALYSIS_HALF
    return EXT_SCALE_ALL


def ext_hygiene_mask(tab, shower_bdt_used=False, event_bdt_used=False):
    """Rows of an EXT table that may be plotted under the hygiene rules.

    `row` is the ntuple entry index, stored by the table builder. Tables built
    before `row` existed fall back to "keep everything" (correct only when no
    BDT score is used).
    """
    n = len(tab["sel"])
    if not SAMPLE_SETS[_ACTIVE["set"]]["hygiene"]:
        return np.ones(n, bool)
    if "row" not in tab:
        if shower_bdt_used or event_bdt_used:
            raise KeyError(
                "EXT table has no 'row' column but a BDT score is in use -- "
                "rebuild the table with nue_cc_analysis.py so EXT hygiene "
                "(rows >= 100k) can be applied.")
        return np.ones(n, bool)
    m = np.ones(n, bool)
    if shower_bdt_used or event_bdt_used:
        m &= np.asarray(tab["row"]) >= EXT_TRAIN_ROW_MAX
    if event_bdt_used:
        m &= np.asarray(tab["event"]) % 2 == 1
    return m


# --------------------------------------------------------------------------
# Stack presentation
# --------------------------------------------------------------------------
COMPONENTS = ["nu_e CC (signal)", "nu_mu CC", "NC", "EXT cosmic"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#e5e5e5"]

#: cew6 working point for the per-shower cosmic BDT (eff 0.97); the ep8-era
#: value was 0.192. See model_and_output_file_versions.md section 5.
SHOWER_BDT_WP = 0.164
#: vertex-free per-shower BDT working point (single_photon/README).
NOVTX_BDT_WP = 0.5


# --------------------------------------------------------------------------
# Derived variables
# --------------------------------------------------------------------------
def _col(tab, key):
    """Column as float, or all-NaN when the table predates that column."""
    if key in tab:
        return np.asarray(tab[key], float)
    return np.full(len(tab["sel"]), np.nan)


def _count(tab, key):
    """Non-negative count column; -1 sentinel and missing column -> 0."""
    if key not in tab:
        return np.zeros(len(tab["sel"]), np.int64)
    v = np.asarray(tab[key])
    return np.where(v < 0, 0, v).astype(np.int64)


def derive(tab):
    """All derived discriminants for the leading electron candidate.

    LArPID scores are LOG-softmax; the LArFormer scores are stored as
    log(probability) by the table builder so the same formulas apply to both.
    """
    d = {
        # --- LArPID ---
        "el": _col(tab, "el_score"),
        "elconf": _col(tab, "el_score")
                  - 0.5 * (_col(tab, "pi_score") + _col(tab, "ph_score")),
        "egamma": _col(tab, "el_score") - _col(tab, "ph_score"),
        "primariness": _col(tab, "prim_score")
                       - np.maximum(_col(tab, "fromneut_score"),
                                    _col(tab, "fromchg_score")),
        "mu": _col(tab, "mu_score"),
        "vtx_mu": _col(tab, "vtx_mu_score"),
        # --- LArFormer (segmentation model) ---
        "elconf_lf": _col(tab, "lf_el_score")
                     - 0.5 * (_col(tab, "lf_pi_score") + _col(tab, "lf_ph_score")),
        "egamma_lf": _col(tab, "lf_el_score") - _col(tab, "lf_ph_score"),
        "mu_lf": _col(tab, "lf_mu_score"),
        "vtx_mu_lf": _col(tab, "vtx_lf_mu_score"),
        # --- pi0 / photon tags ---
        "n_photons": _count(tab, "n_photons"),
        "n_good_photons": _count(tab, "n_good_photons"),
        "n_novtx_photons": _count(tab, "n_novtx_photons"),
        # --- new cew6 handles ---
        "objectness": _col(tab, "objectness"),
        "vtx_score": _col(tab, "vtx_score"),
        "vtx_frac_cosmic": _col(tab, "vtx_frac_cosmic"),
        "reco_ele_E": _col(tab, "reco_ele_E"),
        # --- diagnostics (MC only) ---
        "vtx_dist_true": _col(tab, "vtx_dist_true"),
    }
    fc = _col(tab, "flash_chi2")
    with np.errstate(invalid="ignore", divide="ignore"):
        d["logchi2"] = np.where(np.isfinite(fc) & (fc > 0), np.log10(fc), np.nan)
    for key, br in (("nu_slice_logchi2", "nu_slice_chi2"),
                    ("fm_slice_logchi2", "fm_slice_chi2")):
        v = _col(tab, br)
        with np.errstate(invalid="ignore", divide="ignore"):
            d[key] = np.where(np.isfinite(v) & (v > 0), np.log10(v), np.nan)
    return d


#: (key, axis label, direction, default cut) for the scan tool. "above" keeps
#: values greater than the cut, "below" keeps values less than it, "max" keeps
#: values <= the cut (integer counts).
VARS = [
    ("logchi2", r"$\log_{10}(\mathrm{flash}\ \chi^2)$", "below", 3.0),
    ("elconf", r"LArPID e-confidence  $\log p_e-0.5(\log p_\pi+\log p_\gamma)$", "above", None),
    ("egamma", r"LArPID e/$\gamma$  $\log p_e-\log p_\gamma$", "above", None),
    ("el", r"LArPID electron score  $\log p_e$", "above", None),
    ("primariness", r"primariness  $\log p_{\rm prim}-\max(\log p_{fN},\log p_{fC})$", "above", None),
    ("mu", r"e-shower muon score  $\log p_\mu$", "below", None),
    ("vtx_mu", r"vertex muon score (other particles)  $\log p_\mu$", "below", None),
    ("elconf_lf", r"LArFormer e-confidence", "above", None),
    ("egamma_lf", r"LArFormer e/$\gamma$  $\log p_e-\log p_\gamma$", "above", None),
    ("mu_lf", r"LArFormer e-shower muon score", "below", None),
    ("vtx_mu_lf", r"LArFormer vertex muon score", "below", None),
    ("n_photons", r"# reco photons (>20 MeV) at the nu vertex", "max", None),
    ("n_good_photons", rf"# BDT-passing photons at the nu vertex (score>={SHOWER_BDT_WP})", "max", None),
    ("n_novtx_photons", rf"# vertex-less photons (NoVtx score>={NOVTX_BDT_WP})", "max", None),
    ("objectness", "electron-candidate objectness", "above", None),
    ("vtx_score", "reco vertex score", "above", None),
    ("vtx_frac_cosmic", "vertex fraction of hits on cosmic", "below", None),
    ("nu_slice_logchi2", r"$\log_{10}$(nu-slice flash $\chi^2$)", "below", None),
    ("vtx_dist_true", "reco-true vtx distance [cm] (MC only)", "below", None),
]

#: variables that only exist for MC (no data/EXT overlay possible)
MC_ONLY_VARS = {"vtx_dist_true"}


# --------------------------------------------------------------------------
# Cuts
# --------------------------------------------------------------------------
#: cut name -> (derived-variable key, direction). Keeping this table separate
#: from VARS lets the scan tool plot a variable that has no cut flag yet.
CUT_SPECS = {
    "flashchi2": ("logchi2", "below"),
    "elconf": ("elconf", "above"),
    "el": ("el", "above"),
    "egamma": ("egamma", "above"),
    "primariness": ("primariness", "above"),
    "mu": ("mu", "below"),
    "vtxmu": ("vtx_mu", "below"),
    "elconf_lf": ("elconf_lf", "above"),
    "egamma_lf": ("egamma_lf", "above"),
    "mu_lf": ("mu_lf", "below"),
    "vtxmu_lf": ("vtx_mu_lf", "below"),
    "nphoton_max": ("n_photons", "max"),
    "ngoodphoton_max": ("n_good_photons", "max"),
    "nnovtxphoton_max": ("n_novtx_photons", "max"),
    "objectness": ("objectness", "above"),
    "vtxscore": ("vtx_score", "above"),
    "vtxfraccosmic": ("vtx_frac_cosmic", "below"),
}

#: cuts whose variable comes from a shower-BDT score -> triggers EXT hygiene
SHOWER_BDT_CUTS = {"ngoodphoton_max", "nnovtxphoton_max"}

#: "vertex-veto" cuts: a NaN means "no other particle at the vertex", which is
#: signal-like and must be KEPT rather than dropped by the finiteness guard.
VETO_CUTS = {"vtxmu", "vtxmu_lf"}


def add_cut_args(ap):
    """Register every cut flag on an argparse parser (one definition site)."""
    ap.add_argument("--flashchi2-cut", type=float, default=3.0,
                    help="keep log10(flash_chi2) < this; -1 disables")
    ap.add_argument("--flash-lo", type=float, default=None,
                    help="optional LOWER log10(flash_chi2) band edge "
                         "(for sideband/excess studies)")
    ap.add_argument("--elconf-cut", type=float, default=None)
    ap.add_argument("--el-cut", type=float, default=None,
                    help="LArPID electron score: keep log p_e > this "
                         "(signal piles up just below 0)")
    ap.add_argument("--egamma-cut", type=float, default=None)
    ap.add_argument("--primariness-cut", type=float, default=None)
    ap.add_argument("--mu-cut", type=float, default=None)
    ap.add_argument("--vtxmu-cut", type=float, default=None)
    ap.add_argument("--elconf-lf-cut", type=float, default=None)
    ap.add_argument("--egamma-lf-cut", type=float, default=None)
    ap.add_argument("--mu-lf-cut", type=float, default=None)
    ap.add_argument("--vtxmu-lf-cut", type=float, default=None)
    ap.add_argument("--nphoton-max", type=int, default=None,
                    help="keep events with <= this many reco photons at the vertex")
    ap.add_argument("--ngoodphoton-max", type=int, default=None,
                    help=f"same, counting only photons with cosmic-BDT score "
                         f">= {SHOWER_BDT_WP} (sharper pi0 veto)")
    ap.add_argument("--nnovtxphoton-max", type=int, default=None,
                    help=f"keep events with <= this many VERTEX-LESS photons "
                         f"(NoVtx score >= {NOVTX_BDT_WP})")
    ap.add_argument("--objectness-cut", type=float, default=None)
    ap.add_argument("--vtxscore-cut", type=float, default=None)
    ap.add_argument("--vtxfraccosmic-cut", type=float, default=None)
    ap.add_argument("--ext-scale", type=float, default=None,
                    help="OVERRIDE the per-event EXT weight (default: from the "
                         "sample set recorded in the tables)")
    ap.add_argument("--ext-rescale", type=float, default=1.0,
                    help="MULTIPLY the EXT weight in effect by this factor, "
                         "e.g. 1.3 to try a data-driven correction on top of "
                         "the spill-based normalization (default 1.0)")
    return ap


def cuts_from_args(args):
    """argparse Namespace -> {cut name: value}, dropping the unset ones."""
    raw = {
        "flashchi2": getattr(args, "flashchi2_cut", None),
        "elconf": getattr(args, "elconf_cut", None),
        "el": getattr(args, "el_cut", None),
        "egamma": getattr(args, "egamma_cut", None),
        "primariness": getattr(args, "primariness_cut", None),
        "mu": getattr(args, "mu_cut", None),
        "vtxmu": getattr(args, "vtxmu_cut", None),
        "elconf_lf": getattr(args, "elconf_lf_cut", None),
        "egamma_lf": getattr(args, "egamma_lf_cut", None),
        "mu_lf": getattr(args, "mu_lf_cut", None),
        "vtxmu_lf": getattr(args, "vtxmu_lf_cut", None),
        "nphoton_max": getattr(args, "nphoton_max", None),
        "ngoodphoton_max": getattr(args, "ngoodphoton_max", None),
        "nnovtxphoton_max": getattr(args, "nnovtxphoton_max", None),
        "objectness": getattr(args, "objectness_cut", None),
        "vtxscore": getattr(args, "vtxscore_cut", None),
        "vtxfraccosmic": getattr(args, "vtxfraccosmic_cut", None),
    }
    # flash: -1 is the documented "disable" sentinel
    if raw["flashchi2"] is not None and raw["flashchi2"] < 0:
        raw["flashchi2"] = None
    cuts = {k: v for k, v in raw.items() if v is not None}
    lo = getattr(args, "flash_lo", None)
    if lo is not None:
        cuts["flash_lo"] = lo
    return cuts


def uses_shower_bdt(cuts):
    return any(k in SHOWER_BDT_CUTS for k in cuts)


def cut_label(cuts):
    """Human-readable one-line summary of the active selection."""
    if not cuts:
        return "no cut"
    bits = []
    for name, val in cuts.items():
        if name == "flash_lo":
            bits.append(f"log10(flashchi2)>{val:g}")
            continue
        if name == "flashchi2":
            bits.append(f"log10(flashchi2)<{val:g}")
            continue
        key, direction = CUT_SPECS[name]
        sym = {"above": ">", "below": "<", "max": "<="}[direction]
        bits.append(f"{key}{sym}{val:g}")
    return " & ".join(bits)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def cut_mask(tab, cuts, d=None):
    """Mask of the cut flags alone (no base selection)."""
    d = derive(tab) if d is None else d
    m = np.ones(len(tab["sel"]), bool)
    for name, val in cuts.items():
        if name == "flash_lo":
            m &= np.isfinite(d["logchi2"]) & (d["logchi2"] > val)
            continue
        key, direction = CUT_SPECS[name]
        v = d[key]
        if direction == "above":
            m &= np.isfinite(v) & (v > val)
        elif direction == "below":
            if name in VETO_CUTS:
                # NaN = no other particle at the vertex = signal-like = keep
                m &= ~(np.isfinite(v) & (v >= val))
            else:
                m &= np.isfinite(v) & (v < val)
        elif direction == "max":
            m &= v <= val
    return m


def base(tab, cuts=None, apply_cuts=True, d=None):
    """The canonical selection mask.

    `sel` (reco nu-vtx in FV + >=1 primary electron shower) plus a finite
    observable, plus every cut in `cuts` when `apply_cuts`. Pass
    `apply_cuts=False` to get the pre-cut selection (e.g. for the flash-chi2
    distribution the cut is being chosen from).
    """
    m = np.asarray(tab["sel"]).astype(bool) & np.isfinite(_col(tab, "reco_ele_E"))
    if apply_cuts and cuts:
        m &= cut_mask(tab, cuts, d=d)
    return m


# --------------------------------------------------------------------------
# Truth categorisation of the reco'd electron candidate
# --------------------------------------------------------------------------
# The category is COMPUTED by nue_cc_analysis.py (it needs the per-event
# trueSimPart tables) and stored in the table as `bg_cat`; this module only
# owns the label/colour vocabulary so every plotter agrees on it.
#
# Ordering note: `cosmic` is tested FIRST when assigning (see
# nue_cc_analysis.py:categorize), unlike single_photon/photon_bdt_study.py
# which assigns the PID classes and then overwrites with cosmic. Same result,
# stated explicitly.
TRUTH_CATS = [
    "nu e (primary)",      # 0  SIGNAL: the neutrino's primary electron
    "nu e (fragment)",     # 1  EM daughter of that primary (shower split)
    "nu e (secondary)",    # 2  Michel / delta ray -- NOT signal
    "nu photon",           # 3  pi0 or other gamma mis-ID'd as the electron
    "nu muon",             # 4
    "nu pion",             # 5
    "nu proton",           # 6
    "nu other",            # 7  matched, nu-origin, none of the above
    "cosmic",              # 8  TID<=0 or majority-unlabeled charge
    "unresolved",          # 9  nu-origin but no trueSimPart entry
]
TRUTH_COLORS = ["#d62728", "#ff9896", "#ff7f0e", "#9467bd", "#1f77b4",
                "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f", "#c7c7c7"]

#: categories that count as "found the signal electron" for efficiency
SIGNAL_CATS = (0, 1)

TRUTH_PURITY_MIN = 0.5

#: trueSimPartProcess encoding (extract_truth_sidecar.py:178-179):
#: 0 = nu primary, 1 = Decay, 2 = everything else (delta rays, conversions).
PROC_PRIMARY, PROC_DECAY, PROC_OTHER = 0, 1, 2


def truth_category(tab):
    """Stored per-event category index into TRUTH_CATS; -1 where undefined."""
    n = len(tab["sel"])
    if "bg_cat" not in tab:
        return np.full(n, -1, np.int64)
    return np.asarray(tab["bg_cat"]).astype(np.int64)


def truth_cat_yields(tab, mask, weights=None):
    """{category label: weighted yield} over `mask`."""
    cat = truth_category(tab)
    w = np.asarray(tab["w"], float) if weights is None else weights
    out = {}
    for i, name in enumerate(TRUTH_CATS):
        m = mask & (cat == i)
        if m.any():
            out[name] = float(w[m].sum())
    return out


# --------------------------------------------------------------------------
# Stacking
# --------------------------------------------------------------------------
def components(tabs, cuts, apply_cuts=True):
    """(masks, weights) for the 4 stacked components, in COMPONENTS order.

    `tabs` is {"nue","bnb","ext"} of loaded tables. True nu_e CC is VETOED in
    the bnb sample so the intrinsic-nue sample is the only signal source.
    """
    nue, bnb, ext = tabs["nue"], tabs["bnb"], tabs["ext"]
    sbdt = uses_shower_bdt(cuts)
    mn = base(nue, cuts, apply_cuts)
    mb = base(bnb, cuts, apply_cuts) & ~np.asarray(bnb["is_nuecc"]).astype(bool)
    numucc = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
    nc = mb & (np.asarray(bnb["ccnc"]) == 1)
    me = base(ext, cuts, apply_cuts) & ext_hygiene_mask(ext, shower_bdt_used=sbdt)
    masks = [mn, numucc, nc, me]
    wts = [np.asarray(nue["w"], float)[mn],
           np.asarray(bnb["w"], float)[numucc],
           np.asarray(bnb["w"], float)[nc],
           np.full(int(me.sum()), ext_scale(shower_bdt_used=sbdt))]
    return masks, wts


def data_mask(dat, cuts, apply_cuts=True):
    return base(dat, cuts, apply_cuts)


def summarize(tabs, cuts, dat=None, pot=DEFAULT_POT, stream=None):
    """Print the standard purity / efficiency summary. Returns a dict."""
    import sys
    out = stream or sys.stdout
    masks, wts = components(tabs, cuts)
    pred = float(sum(w.sum() for w in wts))
    nue = tabs["nue"]
    sig = np.asarray(nue["is_nuecc_fv"]).astype(bool)
    w = np.asarray(nue["w"], float)
    sel_sig = masks[0] & sig
    sig_sel, sig_all = float(w[sel_sig].sum()), float(w[sig].sum())
    eff = sig_sel / sig_all if sig_all > 0 else float("nan")
    purity = float(wts[0].sum()) / pred if pred > 0 else float("nan")
    print(f"\n== SELECTION SUMMARY ({cut_label(cuts)}) ==", file=out)
    for c, wv in zip(COMPONENTS, wts):
        print(f"  {c:22s} {wv.sum():9.2f}", file=out)
    print(f"  {'TOTAL pred':22s} {pred:9.2f}", file=out)
    res = dict(pred=pred, purity=purity, eff=eff,
               yields={c: float(wv.sum()) for c, wv in zip(COMPONENTS, wts)})
    if dat is not None:
        nd = int(data_mask(dat, cuts).sum())
        res["data"] = nd
        res["data_over_pred"] = nd / pred if pred > 0 else float("nan")
        print(f"  {'bnb5e19 data':22s} {nd:9d}   "
              f"(data/pred {res['data_over_pred']:.2f})", file=out)
    print(f"  purity (nueCC/pred)      = {purity:.3f}", file=out)
    print(f"  efficiency (sel/true-FV) = {eff:.3f}  "
          f"[{sig_sel:.1f} / {sig_all:.1f} true nueCC-FV]", file=out)
    print("  (LANTERN benchmark: 55% eff / 90% purity)", file=out)
    return res


def load_tables(nue_npz, bnb_npz, ext_npz, data_npz=None, ext_scale_override=None,
                ext_rescale=1.0):
    """Load the per-sample tables into {"nue","bnb","ext"[,"data"]} and
    activate the sample set they were built from.

    Each table records its `sample_set` (nue_cc_analysis.py --sample-set).
    Mixing sets would put run-1 data against run-3 MC -- exactly the
    disagreement source the run-1 set exists to remove -- so it is an error.
    Tables that predate the tag are taken to be run3_cew6.
    """
    tabs = {"nue": dict(np.load(nue_npz)),
            "bnb": dict(np.load(bnb_npz)),
            "ext": dict(np.load(ext_npz))}
    if data_npz is not None:
        tabs["data"] = dict(np.load(data_npz))
    sets = {k: (str(v["sample_set"]) if "sample_set" in v else DEFAULT_SAMPLE_SET)
            for k, v in tabs.items()}
    if len(set(sets.values())) != 1:
        raise ValueError(f"tables come from different sample sets: {sets}")
    set_sample_set(next(iter(sets.values())), ext_scale_override, ext_rescale)
    print(f">>> sample set: {active_label()}  |  EXT per-event weight "
          f"{ext_scale():.4f}")
    return tabs
