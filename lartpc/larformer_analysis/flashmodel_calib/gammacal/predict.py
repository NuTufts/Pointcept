"""Flash prediction at the frozen reference gamma, on CPU.

`predict_slice_pe` (lartpc/flashmatch/flash_predict.py) is the production
function: PhotonLib visibility x charge, drift-corrected by the flash t0,
times gamma x readout factor (1.0 for the beam producer). We call it with
gamma = GAMMA_BEAM_REF for BOTH producers so the result is `pred_ref`.
"""
import numpy as np

from lartpc.flashmatch.flash_predict import predict_slice_pe
from lartpc.flashmatch.saturation import _OPDET_POS

from . import GAMMA_BEAM_REF

PMT_YZ = np.asarray(_OPDET_POS, np.float64)[:, 1:3]


def predict_pe_ref(points_cm, q_comb, t0_us, device="cpu", photonlib=None):
    """(32,) float32 predicted PE at GAMMA_BEAM_REF for one point set."""
    pts = np.ascontiguousarray(np.asarray(points_cm, np.float32))
    q = np.ascontiguousarray(np.asarray(q_comb, np.float32))
    if pts.shape[0] == 0:
        return np.zeros(32, np.float32)
    return predict_slice_pe(pts, q, float(t0_us), producer_id=0,
                            gamma_by_producer=(GAMMA_BEAM_REF, GAMMA_BEAM_REF),
                            photonlib_cache=photonlib, device=device)


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
