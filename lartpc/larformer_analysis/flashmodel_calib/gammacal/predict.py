"""Flash prediction at the frozen reference gamma, on CPU.

The production flash model (lartpc/flashmatch/model, MicroBooNE = the legacy
`predict_slice_pe` bit-for-bit): PhotonLib visibility x charge, drift-corrected
by the flash t0, times gamma x readout factor (1.0 for the beam producer). We
use gamma = GAMMA_BEAM_REF for BOTH producers so the result is `pred_ref`.
This calibration (GAMMA_BEAM_REF, run-period tables) is MicroBooNE-specific.
"""
import numpy as np

from lartpc.flashmatch.model import make_flash_model
from lartpc.flashmatch.saturation import _OPDET_POS

from . import GAMMA_BEAM_REF

PMT_YZ = np.asarray(_OPDET_POS, np.float64)[:, 1:3]


_MODELS = {}


def _model(photonlib, device):
    key = (photonlib, device)
    if key not in _MODELS:
        _MODELS[key] = make_flash_model("uboone", photonlib_cache=photonlib, device=device)
    return _MODELS[key]


def predict_pe_ref(points_cm, q_comb, t0_us, device="cpu", photonlib=None):
    """(32,) float32 predicted PE at GAMMA_BEAM_REF for one point set."""
    pts = np.ascontiguousarray(np.asarray(points_cm, np.float32))
    q = np.ascontiguousarray(np.asarray(q_comb, np.float32))
    if pts.shape[0] == 0:
        return np.zeros(32, np.float32)
    return _model(photonlib, device).predict_legacy(
        pts, q, None, 1, float(t0_us), producer_id=0,
        gamma_by_producer=(GAMMA_BEAM_REF, GAMMA_BEAM_REF))[0]


def cos_sim(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else np.nan


def centroid_yz(pe, live):
    w = np.clip(np.asarray(pe, np.float64), 0, None) * live
    s = w.sum()
    if s <= 0:
        return np.nan, np.nan
    return float(PMT_YZ[:, 0] @ w / s), float(PMT_YZ[:, 1] @ w / s)
