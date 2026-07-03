"""Event-level interaction categorization for LArFormer analysis.

Reads `entry_0/event_truth/` and `entry_0/nu_showers/` from a flashinfo H5
(populated by the extended `prepare_flashinfo_h5.py`) and returns a
category bitmask. An event can belong to multiple categories simultaneously
(e.g., a CC νμ event that contains a π0 — bits 0 AND 2 set).

Bit layout:
    bit 0 = CCNUMU       — CC νμ inclusive (|nu_pdg|==14 and CCNC==0)
    bit 1 = CCNUE        — CC νₑ inclusive (|nu_pdg|==12 and CCNC==0)
    bit 2 = PI0          — at least one final-state primary π0
                            OR at least 2 visible photons from nu showers
                            (the latter catches showers from in-flight π0
                            decay that don't appear as primaries because
                            they're attached to a different parent track)
    bit 3 = SINGLE_VIS_GAMMA
                         — exactly one "visible" nu-origin γ, where visible
                            means DetProfile().E() > vis_gamma_E_thresh_MeV
                            AND start position inside the fiducial volume.
                            This is the science-target topology — the
                            class we want LArFormer selection to improve.
    bit 4 = OTHER        — none of the above (NC interactions without π0
                            or single γ, or non-categorized cases)

All thresholds are kwargs of `categorize_event(...)` so the analysis
script can sweep them without re-extracting flashinfo.

Usage:
    import h5py
    from lartpc.larformer_analysis.slicer_eval.lib.categorize import (
        categorize_event, CATEGORY_NAMES, CCNUMU, CCNUE, PI0,
        SINGLE_VIS_GAMMA, OTHER, has_category,
    )
    with h5py.File(flashinfo_h5, "r") as f:
        mask = categorize_event(f["entry_0"])
        if has_category(mask, SINGLE_VIS_GAMMA):
            ...
"""

import numpy as np

# Category bits.
CCNUMU            = 1 << 0
CCNUE             = 1 << 1
PI0               = 1 << 2
SINGLE_VIS_GAMMA  = 1 << 3
OTHER             = 1 << 4

# Index-aligned with the bit positions so plot code can iterate easily.
CATEGORY_NAMES = ["ccnumu", "ccnue", "pi0", "single_vis_gamma", "other"]
CATEGORY_BITS  = [CCNUMU, CCNUE, PI0, SINGLE_VIS_GAMMA, OTHER]


# Fiducial volume default — slightly inside the active TPC bounds
# (256.35 x 233.0 x 1036.8 cm). 10 cm inset on x and y, 10/1027 on z.
# Configurable per-call.
DEFAULT_FV = dict(
    x_min=10.0, x_max=246.35,
    y_min=-106.5, y_max=106.5,
    z_min=10.0,  z_max=1026.8,
)


def has_category(mask: int, bit: int) -> bool:
    """Convenience: True iff `bit` is set in `mask`."""
    return bool(int(mask) & int(bit))


def _in_fv(xyz: np.ndarray, fv: dict) -> np.ndarray:
    """Bool mask: True for points inside the fiducial-volume box."""
    return (
        (xyz[:, 0] >= fv["x_min"]) & (xyz[:, 0] <= fv["x_max"]) &
        (xyz[:, 1] >= fv["y_min"]) & (xyz[:, 1] <= fv["y_max"]) &
        (xyz[:, 2] >= fv["z_min"]) & (xyz[:, 2] <= fv["z_max"])
    )


def count_visible_nu_gammas(
    entry_grp,
    vis_gamma_E_thresh_MeV: float = 35.0,
    fv: dict = None,
) -> int:
    """Count nu-origin γ showers that are 'visible': PDG==22 with
    DetProfile().E() above threshold AND start position inside the
    fiducial volume.

    DetProfile().E() (MeV) is the standard 'visible energy' proxy — it's
    the shower energy that landed inside the active TPC. The FV cut is
    redundant with that for showers fully contained, but excludes
    showers that started outside the active volume even though some
    energy leaked in.

    Returns 0 when no `nu_showers` group is present (legacy flashinfo
    or non-MC files).
    """
    if "nu_showers" not in entry_grp:
        return 0
    g = entry_grp["nu_showers"]
    pdg = g["pdg"][:]
    if pdg.size == 0:
        return 0
    dep = g["detprofile_E_MeV"][:]
    xyz = g["start_xyz_cm"][:]
    fv = fv if fv is not None else DEFAULT_FV
    in_fv = _in_fv(xyz, fv)
    is_vis = (pdg == 22) & (dep > vis_gamma_E_thresh_MeV) & in_fv
    return int(is_vis.sum())


def count_primary_pi0(entry_grp) -> int:
    """Count final-state primary π0s in event_truth/primary_pdg."""
    if "event_truth" not in entry_grp:
        return 0
    et = entry_grp["event_truth"]
    if "primary_pdg" not in et:
        return 0
    pdg = et["primary_pdg"][:]
    return int((pdg == 111).sum())


def categorize_event(
    entry_grp,
    vis_gamma_E_thresh_MeV: float = 35.0,
    fv: dict = None,
) -> int:
    """Compute the category bitmask for one event.

    Args:
        entry_grp: h5py group at `flashinfo[entry_0]`.
        vis_gamma_E_thresh_MeV: DetProfile-energy threshold for a γ to
            count as 'visible'. 35 MeV is a common Pandora cut.
        fv: fiducial-volume dict (see DEFAULT_FV).

    Returns: int bitmask. Always sets exactly one of {CCNUMU, CCNUE,
    OTHER}-or-NC plus optional PI0 / SINGLE_VIS_GAMMA modifiers.
    Topology bits and CC-flavor bits are independent (so a CC νμ event
    with a single visible γ has CCNUMU | SINGLE_VIS_GAMMA set).
    """
    mask = 0
    has_nu = False
    nu_pdg = 0
    ccnc = -1
    if "event_truth" in entry_grp:
        et = entry_grp["event_truth"]
        has_nu = bool(et.attrs.get("has_neutrino", False))
        nu_pdg = int(et.attrs.get("nu_pdg", 0))
        ccnc = int(et.attrs.get("ccnc", -1))

    is_cc = has_nu and ccnc == 0
    if is_cc and abs(nu_pdg) == 14:
        mask |= CCNUMU
    if is_cc and abs(nu_pdg) == 12:
        mask |= CCNUE

    # Topology bits: independent of CC/NC.
    n_pi0 = count_primary_pi0(entry_grp)
    n_vis_g = count_visible_nu_gammas(
        entry_grp,
        vis_gamma_E_thresh_MeV=vis_gamma_E_thresh_MeV, fv=fv,
    )
    # PI0: a primary π0 in the mctruth particle list, OR ≥2 visible
    # nu-γ showers (catches π0s that show up only as their two γ
    # daughters in the mcshower list).
    if n_pi0 >= 1 or n_vis_g >= 2:
        mask |= PI0
    # SINGLE_VIS_GAMMA: exactly one visible nu-γ AND no primary π0.
    # The primary-π0 exclusion prevents miscategorizing a π0 event that
    # happens to have one γ outside FV / below threshold as a 1-γ event.
    if n_vis_g == 1 and n_pi0 == 0:
        mask |= SINGLE_VIS_GAMMA

    # OTHER: nothing above (NC events without π0 / γ, or non-MC files).
    if mask == 0:
        mask |= OTHER
    return mask


def category_str(mask: int) -> str:
    """Pretty-print: e.g. 'ccnumu|pi0'. 'none' if no bits set."""
    parts = [name for name, bit in zip(CATEGORY_NAMES, CATEGORY_BITS)
             if mask & bit]
    return "|".join(parts) if parts else "none"
