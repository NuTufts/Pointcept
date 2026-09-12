"""Flash light-yield calibration table keyed by (sample kind, run period).

ADDITIVE to dead_channels.py: the legacy spec "auto" still delegates to
dead_channels.resolve_gamma_scale (run-period table, bit-identical), so nothing
produced so far changes. New specs make the data/MC split explicit:

    "auto"        -> dead_channels.GAMMA_SCALE_BY_PERIOD[run_period(run)]   (legacy)
    "auto:data"   -> GAMMA_SCALE_TABLE[("data", run_period(run))]
    "auto:mc"     -> GAMMA_SCALE_TABLE[("mc",   run_period(run))]
    "table"       -> GAMMA_SCALE_TABLE[(kind,   run_period(run))]  (kind required)
    "<float>"     -> that multiplier, forced

An unmeasured cell raises KeyError (never silently 1.0). Every cell carries its
provenance: the flashmodel_calib result JSON it came from, the chain it was
measured on and the date. Cells are written ONLY by
lartpc/larformer_analysis/flashmodel_calib/scripts/make_table.py from results
that passed the closure test (PROTOCOL.md section 7).

The scale multiplies GAMMA_BEAM_REF (5.25) to give gamma_eff (PE per unit of
de-double-counted comb charge x PhotonLib visibility).
"""
from .dead_channels import run_period, resolve_gamma_scale as _legacy_resolve

GAMMA_BEAM_REF = 5.25

# (kind, period) -> dict(scale, err, result, chain, date). Empty until a cell
# passes closure; see flashmodel_calib/CALIBRATION_LOG.md for pending values.
GAMMA_SCALE_TABLE = {
    "chain": "s1ep2p8cew6",
    "entries": {
        ('data', 1): {'scale': 0.5449, 'err': 0.0061, 'N': 255, 'result': 'lartpc/larformer_analysis/flashmodel_calib/results/s1ep2p8cew6/bnb5e19_calib3k__muon_clu_anyb_L120.json', 'chain': 's1ep2p8cew6', 'date': '2026-09-11', 'sample': 'bnb5e19_calib3k', 'arm': 'muon'},
        ('data', 3): {'scale': 0.4281, 'err': 0.0057, 'N': 275, 'result': 'lartpc/larformer_analysis/flashmodel_calib/results/s1ep2p8cew6/extbnb200k_calib3k__muon_clu_anyb_L120.json', 'chain': 's1ep2p8cew6', 'date': '2026-09-11', 'sample': 'extbnb200k_calib3k', 'arm': 'muon'},
        ('mc', 1): {'scale': 0.8576, 'err': 0.0217, 'N': 75, 'result': 'lartpc/larformer_analysis/flashmodel_calib/results/s1ep2p8cew6/run1ovl_calib6k__muon_clu_anyb_L120_nuq.json', 'chain': 's1ep2p8cew6', 'date': '2026-09-11', 'sample': 'run1ovl_calib6k', 'arm': 'muon'},
        ('mc', 3): {'scale': 0.8325, 'err': 0.0253, 'N': 62, 'result': 'lartpc/larformer_analysis/flashmodel_calib/results/s1ep2p8cew6/mcoverlay67k_calib3k__muon_clu_anyb_L120_nuq.json', 'chain': 's1ep2p8cew6', 'date': '2026-09-11', 'sample': 'mcoverlay67k_calib3k', 'arm': 'muon'},
    },
}

# In-time beam-flash window [us] per (kind, period), measured from the
# producer-0 flash-time histograms (flashmodel_calib, 2026-09-11).
# Beam-on and beam-off (EXT) data have DIFFERENT windows in the same period,
# so an optional third key element names the trigger: "bnb" (beam-on, the
# default for kind='data') or "ext". The reco chain passes FLASH_WINDOW
# explicitly for EXT samples.
FLASH_WINDOW_US = {
    ("data", 1): (2.8, 5.0),            # bnb5e19 beam-on
    ("data", 1, "ext"): (3.2, 5.4),     # run-1 EXT (measured 2026-09-12, 1500 files)
    ("data", 3): (3.2, 5.4),            # run-3 EXT (beam-on run 3 not measured)
    ("data", 3, "ext"): (3.2, 5.4),
    ("mc", 3): (3.6, 5.2),
    ("mc", 1): (3.6, 5.2),      # measured on the run-1 overlay pilot (2026-09-12)
}

KINDS = ("data", "mc")


def detect_kind(msp_path):
    """'mc' if the merged_sp file carries at least one truth particle, else
    'data'. (Data-mode conversions still write an EMPTY mc_particle_tree group,
    so the group's presence is not a discriminator.)"""
    import h5py
    with h5py.File(msp_path, "r") as f:
        mt = f["entry_0"].get("mc_particle_tree")
        if mt is None or "trackid" not in mt:
            return "data"
        return "mc" if mt["trackid"].shape[0] > 0 else "data"


def lookup(kind, run):
    key = (str(kind), run_period(run))
    ent = GAMMA_SCALE_TABLE["entries"].get(key)
    if ent is None:
        raise KeyError(f"no calibrated gamma scale for {key}; measured cells: "
                       f"{sorted(GAMMA_SCALE_TABLE['entries'])} -- pass an explicit "
                       f"--gamma-run-scale <float> or calibrate the cell first")
    return float(ent["scale"])


def resolve(spec, run, kind=None):
    """Multiplier on GAMMA_BEAM_REF for this event. See module docstring."""
    if spec is None or spec == "auto":
        return float(_legacy_resolve("auto", run))
    if spec in ("auto:data", "auto:mc"):
        return lookup(spec.split(":")[1], run)
    if spec == "table":
        if kind not in KINDS:
            raise ValueError("spec 'table' needs kind='data'|'mc' (use --sample-kind)")
        return lookup(kind, run)
    return float(spec)


def check_spec_vs_kind(spec, kind, allow_mismatch=False):
    """Refuse an MC spec on a data file and vice versa."""
    if spec in ("auto:data", "auto:mc") and kind in KINDS:
        want = spec.split(":")[1]
        if want != kind and not allow_mismatch:
            raise ValueError(f"--gamma-run-scale {spec} on a {kind} sample "
                             "(pass --allow-kind-mismatch to override)")
    return True


def flash_window(kind, run, trigger=None):
    p = run_period(run)
    if trigger:
        w = FLASH_WINDOW_US.get((str(kind), p, str(trigger)))
        if w is not None:
            return w
    return FLASH_WINDOW_US.get((str(kind), p))
