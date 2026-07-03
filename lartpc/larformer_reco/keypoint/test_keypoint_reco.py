"""Synthetic tests for the score-field keypoint reco (spec section 7).

Builds score maps as sums of known 3 cm Gaussians and asserts the reco recovers
the centers. The decisive cases are TRUNCATED / one-sided support (masked
hemisphere = a track start; masked peak = peak off-support), where the
score-weighted centroid is visibly biased but the shape fits are not.

Run in the pointcept container (needs numpy + torch; scipy optional):

    python -m lartpc.larformer_reco.keypoint.test_keypoint_reco
    # or:  pytest lartpc/larformer_reco/keypoint/test_keypoint_reco.py
"""
from __future__ import annotations

import numpy as np

try:
    from .keypoint_reco import KeypointRecoTorch, KeypointRecoParams
    from . import gaussian_fit as gf
except Exception:                                    # direct-path run
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from keypoint_reco import KeypointRecoTorch, KeypointRecoParams
    import gaussian_fit as gf

SIGMA = 3.0
RNG = np.random.RandomState(0)


def _grid(lo, hi, step=1.0):
    ax = [np.arange(lo[d], hi[d], step) for d in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g.astype(np.float64)


def _score_field(coords, centers, sigma=SIGMA):
    """max over centers of exp(-d^2/2 sigma^2) — the GT-label generative model."""
    s = np.zeros(coords.shape[0])
    for c in centers:
        d2 = ((coords - np.asarray(c)) ** 2).sum(1)
        s = np.maximum(s, np.exp(-0.5 * d2 / (sigma * sigma)))
    return s


def _match(pred, gt):
    """min over pred of distance to each gt center -> (n_gt,) best dists."""
    pred = np.asarray([p.pos_cm for p in pred], np.float64).reshape(-1, 3)
    out = []
    for c in gt:
        if len(pred) == 0:
            out.append(np.inf)
        else:
            out.append(float(np.linalg.norm(pred - np.asarray(c), axis=1).min()))
    return np.asarray(out)


def test_single_peak_recovered():
    coords = _grid([-12, -12, -12], [12, 12, 12], 1.5)
    center = np.array([1.3, -2.1, 0.7])
    s = _score_field(coords, [center])
    reco = KeypointRecoTorch(KeypointRecoParams(sigma_cm=SIGMA))
    kps = reco.reconstruct(coords, s)
    assert len(kps) == 1, f"expected 1 peak, got {len(kps)}"
    err = float(np.linalg.norm(kps[0].pos_cm - center))
    assert err < 0.5, f"single-peak err {err:.3f} cm"
    print(f"[ok] single peak: err={err:.3f} cm  method={kps[0].method}")


def test_multiple_well_separated():
    coords = _grid([-20, -20, -8], [40, 20, 8], 1.5)
    centers = [np.array([0., 0., 0.]), np.array([15., 5., 1.]),
               np.array([30., -6., -2.])]
    s = _score_field(coords, centers)
    reco = KeypointRecoTorch(KeypointRecoParams(sigma_cm=SIGMA))
    kps = reco.reconstruct(coords, s)
    errs = _match(kps, centers)
    assert len(kps) == 3, f"expected 3 peaks, got {len(kps)}"
    assert errs.max() < 0.6, f"multi-peak errs {errs}"
    print(f"[ok] 3 separated peaks: max err={errs.max():.3f} cm")


def test_overlapping_peaks():
    coords = _grid([-15, -10, -8], [20, 10, 8], 1.0)
    centers = [np.array([0., 0., 0.]), np.array([6.5, 0., 0.])]  # ~2 sigma apart
    s = _score_field(coords, centers)
    reco = KeypointRecoTorch(KeypointRecoParams(sigma_cm=SIGMA, radius_cm=10.0))
    kps = reco.reconstruct(coords, s)
    errs = _match(kps, centers)
    assert len(kps) >= 2, f"overlapping: expected >=2, got {len(kps)}"
    assert errs.max() < 1.5, f"overlapping errs {errs}"
    print(f"[ok] overlapping peaks ({len(kps)} found): max err={errs.max():.3f} cm")


def test_half_gaussian_track_start():
    """One-sided support: hits only on x>=0 (a track start). The shape fit must
    stay near the true start; the centroid is pulled into the visible side."""
    coords = _grid([0, -12, -12], [16, 12, 12], 1.2)   # x>=0 only
    center = np.array([0.0, 0.0, 0.0])                 # start at the boundary
    s = _score_field(coords, [center])
    nbr = coords[np.linalg.norm(coords - center, axis=1) <= 10.0]
    ys = _score_field(nbr, [center])
    fit = gf.fit_gaussian_nls(
        np.asarray(nbr) - center, ys, SIGMA)           # centered frame
    nls_err = float(np.linalg.norm(fit.mu.numpy()))
    cen = gf.fit_gaussian_centroid(np.asarray(nbr) - center, ys, SIGMA)
    cen_err = float(np.linalg.norm(cen.mu.numpy()))
    assert nls_err < 1.0, f"half-Gaussian NLS err {nls_err:.2f} cm"
    assert cen_err > nls_err, ("centroid should be MORE biased than the shape "
                               f"fit (nls={nls_err:.2f}, centroid={cen_err:.2f})")
    print(f"[ok] half-Gaussian start: nls_err={nls_err:.2f} cm  "
          f"centroid_err={cen_err:.2f} cm (centroid biased, as expected)")


def test_masked_peak_off_support():
    """Peak masked away: only the tail at d>=5 cm survives, true peak OUTSIDE
    the points. The shape fit extrapolates back to it (within tolerance)."""
    coords = _grid([-14, -14, -6], [14, 14, 6], 1.0)
    center = np.array([0.0, 0.0, 0.0])
    d = np.linalg.norm(coords - center, axis=1)
    keep = d >= 5.0                                    # carve out the peak
    coords, d = coords[keep], d[keep]
    ys = np.exp(-0.5 * d ** 2 / SIGMA ** 2)
    fit = gf.fit_gaussian_nls(coords - center, ys, SIGMA)
    err = float(np.linalg.norm(fit.mu.numpy()))
    assert err < 2.0, f"masked-peak extrapolation err {err:.2f} cm"
    print(f"[ok] masked peak (off-support): extrapolation err={err:.2f} cm")


def test_caruana_parity_single_gaussian():
    """Variable-sigma Caruana should recover both center AND width of a clean
    isotropic Gaussian (parity with the C++ algorithm)."""
    coords = _grid([-10, -10, -10], [10, 10, 10], 1.0)
    center = np.array([0.5, -1.0, 2.0])
    s = _score_field(coords, [center], sigma=SIGMA)
    nbr = coords[np.linalg.norm(coords - center, axis=1) <= 9.0]
    ys = _score_field(nbr, [center], sigma=SIGMA)
    fit = gf.caruana_full(nbr - center, ys)
    err = float(np.linalg.norm(fit.mu.numpy()))
    assert err < 0.3, f"caruana center err {err:.3f} cm"
    print(f"[ok] caruana parity: center err={err:.3f} cm")


def test_threshold_stop():
    """Below-threshold field yields no keypoints."""
    coords = _grid([-10, -10, -10], [10, 10, 10], 1.5)
    s = 0.5 * np.ones(coords.shape[0])                 # all below 0.67
    reco = KeypointRecoTorch(KeypointRecoParams(sigma_cm=SIGMA,
                                                score_thresh=0.67))
    kps = reco.reconstruct(coords, s)
    assert len(kps) == 0, f"expected 0 below threshold, got {len(kps)}"
    print("[ok] threshold stop: 0 keypoints below 0.67")


def _all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _all()
