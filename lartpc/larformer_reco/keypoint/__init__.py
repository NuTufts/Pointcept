"""Score-field keypoint reconstruction for the LArFormer cascade (v1).

PyTorch port of larflow's ``KeypointReco`` greedy peak-finder, operating on the
attempt-2 keypoint cascade's dense score maps. See ``../keypoint_reco_spec.md``.
"""
from .keypoint_reco import (
    KeypointRecoTorch, KeypointRecoParams, Keypoint, best_nu_vertex,
)
from . import gaussian_fit, io

__all__ = [
    "KeypointRecoTorch", "KeypointRecoParams", "Keypoint", "best_nu_vertex",
    "gaussian_fit", "io",
]
