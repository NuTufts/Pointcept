"""Read the cascade ``--save-score-maps`` H5 and write reco keypoints.

Input format — written by
``tools/larformer/run_larformer_keypoint2_cascade_inference.py --save-score-maps``
(see ``_write_event_h5`` there)::

    score_maps/<head>/coords_cm   (M,3)   token positions, detector cm
    score_maps/<head>/score       (M,)    sigmoid score in [0,1]
    score_maps/<head>.attrs.level          e.g. "spacepoint" | "ptv3_dec2"
    score_maps/<head>.attrs.kp_types       int list (GT types the head targets)
    slice/coord_cm                (N,3)   nu-slice spacepoints (= nu head coords)
    nu_vertex_cm                  (3,)    existing single-centroid decode
    gt_nu_vertex_cm               (3,)    GT (sim only)
    gt_keypoints/pos_cm,type,trackid       all GT mckeypoints (sim only)

This module is intentionally dependency-light (numpy + h5py only) so the
``reco/`` package stays runnable outside the full pointcept environment.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import h5py
except Exception:                                    # pragma: no cover
    h5py = None


def load_score_maps(path: str) -> Dict:
    """Load score maps + GT/context from one inference H5.

    Returns a dict::

        {"score_maps": {name: {coords_cm, score, level, kp_types}},
         "slice_coord_cm": (N,3) | None,
         "nu_vertex_cm": (3,) | None,            # existing decode, for cross-check
         "gt_nu_vertex_cm": (3,) | None,
         "gt_keypoints": {"pos_cm": (K,3), "type": (K,)} | None,
         "attrs": {...}}                          # run/subrun/event/src_file
    """
    if h5py is None:                                  # pragma: no cover
        raise RuntimeError("h5py is required to read score-map H5 files")
    with h5py.File(path, "r") as f:
        if "score_maps" not in f:
            raise KeyError(
                f"{path}: no 'score_maps' group — re-run inference with "
                "--save-score-maps")
        score_maps = {}
        for name in f["score_maps"]:
            g = f["score_maps"][name]
            score_maps[name] = dict(
                coords_cm=np.asarray(g["coords_cm"][:], np.float32),
                score=np.asarray(g["score"][:], np.float32).reshape(-1),
                level=str(g.attrs.get("level", "")),
                kp_types=[int(t) for t in np.asarray(
                    g.attrs.get("kp_types", [])).reshape(-1)])
        out = dict(score_maps=score_maps)
        out["slice_coord_cm"] = (np.asarray(f["slice/coord_cm"][:], np.float32)
                                 if "slice" in f and "coord_cm" in f["slice"]
                                 else None)
        out["nu_vertex_cm"] = (np.asarray(f["nu_vertex_cm"][:], np.float32)
                               if "nu_vertex_cm" in f else None)
        out["gt_nu_vertex_cm"] = (
            np.asarray(f["gt_nu_vertex_cm"][:], np.float32)
            if "gt_nu_vertex_cm" in f else None)
        gt = None
        if "gt_keypoints" in f and "pos_cm" in f["gt_keypoints"]:
            gt = dict(
                pos_cm=np.asarray(f["gt_keypoints"]["pos_cm"][:], np.float32
                                  ).reshape(-1, 3),
                type=np.asarray(f["gt_keypoints"]["type"][:], np.int64
                                ).reshape(-1))
        out["gt_keypoints"] = gt
        attrs = {}
        for k in ("src_file", "run", "subrun", "event"):
            if k in f.attrs:
                v = f.attrs[k]
                attrs[k] = (v.decode() if isinstance(v, bytes) else
                            (int(v) if k != "src_file" else str(v)))
        out["attrs"] = attrs
    return out


def _kp_array(keypoints: List) -> Dict[str, np.ndarray]:
    """Stack a list of ``Keypoint`` into parallel arrays for H5 storage."""
    if not keypoints:
        return dict(pos=np.zeros((0, 3), np.float32),
                    score=np.zeros(0, np.float32),
                    rmse=np.zeros(0, np.float32), rsqr=np.zeros(0, np.float32),
                    n_support=np.zeros(0, np.int32),
                    extrapolated=np.zeros(0, np.int8))
    return dict(
        pos=np.stack([k.pos_cm for k in keypoints]).astype(np.float32),
        score=np.asarray([k.peak_score for k in keypoints], np.float32),
        rmse=np.asarray([k.fit_rmse for k in keypoints], np.float32),
        rsqr=np.asarray([k.fit_rsqr for k in keypoints], np.float32),
        n_support=np.asarray([k.n_support for k in keypoints], np.int32),
        extrapolated=np.asarray([int(k.extrapolated) for k in keypoints],
                                np.int8))


def write_reco_h5(path: str, event_result: Dict[str, dict],
                  best_nu, loaded: Dict, params) -> None:
    """Write reco keypoints (+ copied GT for eval) to a new per-event H5.

    Schema mirrors ``../keypoint_reco_spec.md`` section 6.1, generalised to one
    group per score-map head plus a convenience ``reco/nu_vertex_cm``.
    """
    if h5py is None:                                  # pragma: no cover
        raise RuntimeError("h5py is required to write reco H5 files")
    with h5py.File(path, "w") as f:
        for k, v in (loaded.get("attrs") or {}).items():
            f.attrs[k] = v
        f.attrs["radius_cm"] = params.radius_cm
        f.attrs["sigma_cm"] = params.sigma_cm
        f.attrs["score_thresh"] = params.score_thresh
        f.attrs["fit_method"] = params.fit_method

        g_reco = f.create_group("reco")
        if best_nu is not None:
            g_reco.create_dataset("nu_vertex_cm", data=best_nu.pos_cm)
            g_reco.attrs["nu_vertex_score"] = float(best_nu.peak_score)
        else:
            g_reco.create_dataset("nu_vertex_cm",
                                  data=np.full(3, np.nan, np.float32))

        for name, r in event_result.items():
            gg = g_reco.create_group(name)
            gg.attrs["level"] = r.get("level", "")
            gg.attrs["kp_types"] = np.asarray(r.get("kp_types", []), np.int32)
            gg.attrs["is_nu"] = bool(r.get("is_nu", False))
            arr = _kp_array(r["keypoints"])
            for key, val in arr.items():
                gg.create_dataset(key, data=val,
                                  compression="gzip" if val.size else None)

        # carry the existing decode + GT through for downstream eval/plots
        if loaded.get("nu_vertex_cm") is not None:
            f.create_dataset("nu_vertex_cm_decode",
                             data=loaded["nu_vertex_cm"])
        if loaded.get("gt_nu_vertex_cm") is not None:
            f.create_dataset("gt_nu_vertex_cm", data=loaded["gt_nu_vertex_cm"])
        gt = loaded.get("gt_keypoints")
        if gt is not None:
            gk = f.create_group("gt_keypoints")
            gk.create_dataset("pos_cm", data=gt["pos_cm"], compression="gzip"
                              if gt["pos_cm"].size else None)
            gk.create_dataset("type", data=gt["type"], compression="gzip"
                              if gt["type"].size else None)
