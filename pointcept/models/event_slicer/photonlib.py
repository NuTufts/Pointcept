"""GPU lookup-table flash predictor for the MicroBooNE event slicer.

Wraps the densified UBPhotonLib visibility table (built once by
``lartpc/data_prep/labels/build_photonlib_cache.py``) as a torch module so the
forward pass for "given a cluster of spacepoints, predict its (32,) PE flash"
is a few index_select + segment-sum ops on the GPU.

Conventions (all in detector TPC cm; PE indexed by OpDet 0..31):

    voxel grid is in *cryostat* coordinates; we shift by
    ``cryo_origin_tpc_cm`` to convert TPC pos -> grid pos:
        grid_pos = (pos - cryo_origin_tpc_cm) / voxel_len_cm
    nearest-neighbor:   vid = floor(grid_pos)
    trilinear:          8 corner voxels, weights from frac parts
    out-of-bounds:      zero contribution (corners that fall outside
                         [0, nvoxels_dim) contribute 0)

    Predicted PE on OpDet j from a single spacepoint with q_emitted photons
    emitted isotropically at position r is:
        PE_j(r) = q_emitted * visibility[v(r), j]
    For a cluster, sum over its spacepoints.

    Drift correction for a candidate flash at t_flash (us, relative to TPC
    trigger): the *true* emission x is
        x_true = x_apparent - v_drift * t_flash
    (sign convention from prepare_flashinfo_h5.py).

The LUT visibility is purely geometric+scattering fraction (photons reaching
the OpDet / photons emitted). PMT quantum efficiency and the
charge->emitted-photon scale factor γ are external — the caller multiplies
its charge-derived ``q_emitted`` by them before calling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class PhotonLibMeta:
    nvoxels_dim: np.ndarray         # (3,) int64
    voxel_len_cm: np.ndarray        # (3,) float64
    cryo_origin_tpc_cm: np.ndarray  # (3,) float64
    n_opdets: int                   # 32

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.nvoxels_dim))


def _load_meta_from_npz(npz) -> PhotonLibMeta:
    return PhotonLibMeta(
        nvoxels_dim=np.asarray(npz["nvoxels_dim"]).astype(np.int64),
        voxel_len_cm=np.asarray(npz["voxel_len_cm"]).astype(np.float64),
        cryo_origin_tpc_cm=np.asarray(npz["cryo_origin_tpc_cm"]).astype(np.float64),
        n_opdets=int(npz["n_opdets"]),
    )


def select_charge_y_with_uv_fallback(pixval: torch.Tensor) -> torch.Tensor:
    """Per-spacepoint scalar charge.

    Y-plane ``pixval[:, 2]`` is the most reliable calorimetric proxy. When
    Y is exactly zero (dead channel / non-readout pixel), fall back to the
    average of U (plane 0) and V (plane 1).
    """
    if pixval.ndim != 2 or pixval.shape[-1] != 3:
        raise ValueError(f"pixval must be (N, 3); got {tuple(pixval.shape)}")
    q = pixval[:, 2].clone()
    fb = q == 0
    if fb.any():
        q[fb] = 0.5 * (pixval[fb, 0] + pixval[fb, 1])
    return q


class PhotonLibLookup(nn.Module):
    """Visibility LUT + flash prediction on GPU.

    Buffers (registered, so .to(device) / .cuda() / .half() do the right
    thing):
        visibility (Nx, Ny, Nz, n_opdets)  float32  (or fp16 with fp16=True)
        cryo_origin_tpc                    (3,)     float32
        voxel_len_cm                       (3,)     float32
        nvoxels_dim                        (3,)     int64

    Methods:
        voxelize_floor(pos)     -> (vc, valid)   (N, 3) int64, (N,) bool
        visibility_nn(pos)      -> (N, n_opdets)
        visibility_trilinear(pos) -> (N, n_opdets)
        predict_flash(pos, q_emitted, cluster_id, n_clusters)
                                -> (n_clusters, n_opdets)
        predict_flash_pairs(pos, q_emitted, cluster_id,
                            pair_cluster_idx, pair_flash_t0_us,
                            v_drift_cm_per_us)
                                -> (B, n_opdets)
    """

    def __init__(self, cache_path: str, fp16: bool = False,
                 use_trilinear: bool = True):
        super().__init__()
        npz = np.load(cache_path)
        meta = _load_meta_from_npz(npz)
        vis = np.asarray(npz["visibility"]).astype(
            np.float16 if fp16 else np.float32
        )
        self.use_trilinear = bool(use_trilinear)
        self.meta = meta

        # Keep the table 4D — we use vectorized advanced indexing rather than
        # a flat strides formula, which avoids any C-order/F-order confusion.
        self.register_buffer("vis_table",
                             torch.from_numpy(vis).contiguous())  # (Nx, Ny, Nz, n_opdets)
        self.register_buffer("cryo_origin_tpc",
                             torch.tensor(meta.cryo_origin_tpc_cm,
                                          dtype=torch.float32))
        self.register_buffer("voxel_len_cm",
                             torch.tensor(meta.voxel_len_cm,
                                          dtype=torch.float32))
        self.register_buffer("nvoxels_dim",
                             torch.tensor(meta.nvoxels_dim, dtype=torch.int64))

    @property
    def n_opdets(self) -> int:
        return int(self.meta.n_opdets)

    # ------------------------------------------------------------------
    # Voxelization
    # ------------------------------------------------------------------

    def _grid_pos(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos.to(self.cryo_origin_tpc.dtype)
                - self.cryo_origin_tpc) / self.voxel_len_cm

    def voxelize_floor(self, pos: torch.Tensor):
        """(N, 3) -> (vc (N, 3) int64, valid (N,) bool). True floor (not the
        C++ cast-then-divide quirk; the difference is sub-cm at voxel
        boundaries and trilinear masks it anyway)."""
        g = self._grid_pos(pos)
        vc = torch.floor(g).to(torch.int64)
        nv = self.nvoxels_dim
        valid = (vc[:, 0] >= 0) & (vc[:, 0] < nv[0]) & \
                (vc[:, 1] >= 0) & (vc[:, 1] < nv[1]) & \
                (vc[:, 2] >= 0) & (vc[:, 2] < nv[2])
        return vc, valid

    # ------------------------------------------------------------------
    # Visibility lookup (4D advanced indexing: vis_table[vx, vy, vz, :])
    # ------------------------------------------------------------------

    def visibility_nn(self, pos: torch.Tensor) -> torch.Tensor:
        vc, valid = self.voxelize_floor(pos)
        nv = self.nvoxels_dim
        vc_safe = torch.minimum(torch.clamp_min(vc, 0),
                                (nv - 1)[None, :].expand_as(vc))
        vis = self.vis_table[vc_safe[:, 0], vc_safe[:, 1], vc_safe[:, 2]]  # (N, 32)
        vis = vis * valid.to(vis.dtype).unsqueeze(-1)
        return vis

    def visibility_trilinear(self, pos: torch.Tensor) -> torch.Tensor:
        g = self._grid_pos(pos)
        base = torch.floor(g).to(torch.int64)              # (N, 3)
        frac = (g - base.to(g.dtype)).clamp(0.0, 1.0)      # (N, 3)
        nv = self.nvoxels_dim
        out = torch.zeros(
            (pos.shape[0], self.n_opdets),
            dtype=self.vis_table.dtype, device=pos.device,
        )
        for di in (0, 1):
            wx = (1.0 - frac[:, 0]) if di == 0 else frac[:, 0]
            for dj in (0, 1):
                wy = (1.0 - frac[:, 1]) if dj == 0 else frac[:, 1]
                for dk in (0, 1):
                    wz = (1.0 - frac[:, 2]) if dk == 0 else frac[:, 2]
                    vx = base[:, 0] + di
                    vy = base[:, 1] + dj
                    vz = base[:, 2] + dk
                    valid = ((vx >= 0) & (vx < nv[0]) &
                             (vy >= 0) & (vy < nv[1]) &
                             (vz >= 0) & (vz < nv[2]))
                    vx_s = torch.clamp(vx, 0, int(nv[0]) - 1)
                    vy_s = torch.clamp(vy, 0, int(nv[1]) - 1)
                    vz_s = torch.clamp(vz, 0, int(nv[2]) - 1)
                    vis = self.vis_table[vx_s, vy_s, vz_s]   # (N, n_opdets)
                    w = (wx * wy * wz).to(vis.dtype)
                    w = w * valid.to(vis.dtype)
                    out = out + w.unsqueeze(-1) * vis
        return out

    def visibility(self, pos: torch.Tensor) -> torch.Tensor:
        if self.use_trilinear:
            return self.visibility_trilinear(pos)
        return self.visibility_nn(pos)

    # ------------------------------------------------------------------
    # Flash prediction
    # ------------------------------------------------------------------

    def predict_flash(
        self,
        pos: torch.Tensor,
        q_emitted: torch.Tensor,
        cluster_id: torch.Tensor,
        n_clusters: int,
    ) -> torch.Tensor:
        """Per-cluster predicted PE.

        Args:
            pos:         (N, 3)  TPC cm. Caller does any drift correction.
            q_emitted:   (N,)    photons emitted per spacepoint
                                 (= γ · selected_charge).
            cluster_id:  (N,)    int64 in [0, n_clusters).
            n_clusters:  int.
        Returns:
            (n_clusters, n_opdets) predicted PE.
        """
        vis = self.visibility(pos)  # (N, n_opdets)
        contrib = q_emitted.to(vis.dtype).unsqueeze(-1) * vis  # (N, n_opdets)
        out = torch.zeros(
            (n_clusters, self.n_opdets),
            dtype=vis.dtype, device=pos.device,
        )
        out.index_add_(0, cluster_id.to(torch.int64), contrib)
        return out

    def predict_flash_pairs(
        self,
        pos: torch.Tensor,
        q_emitted: torch.Tensor,
        cluster_id: torch.Tensor,
        pair_cluster_idx: torch.Tensor,
        pair_flash_t0_us: torch.Tensor,
        v_drift_cm_per_us: float = 0.1098,
    ) -> torch.Tensor:
        """Cost-matrix variant: one predicted PE vector per (cluster, flash) pair.

        Implements the drift correction
            x_true = x_apparent - v_drift * t_flash
        once per pair, shifting all of that cluster's spacepoints.

        Args:
            pos:                (N, 3)  TPC cm.
            q_emitted:          (N,).
            cluster_id:         (N,)    int64 in [0, n_clusters).
            pair_cluster_idx:   (B,)    int64 — which cluster for each pair.
            pair_flash_t0_us:   (B,)    float — flash time (us, vs trigger).
            v_drift_cm_per_us:  scalar.
        Returns:
            (B, n_opdets) predicted PE.
        """
        B = int(pair_cluster_idx.shape[0])
        out = torch.zeros(
            (B, self.n_opdets),
            dtype=self.vis_table.dtype, device=pos.device,
        )
        x_hat = torch.tensor([1.0, 0.0, 0.0], dtype=pos.dtype, device=pos.device)
        for b in range(B):
            c = int(pair_cluster_idx[b])
            mask = cluster_id == c
            if not mask.any():
                continue
            dx = float(v_drift_cm_per_us * float(pair_flash_t0_us[b]))
            shifted = pos[mask] - dx * x_hat
            vis = self.visibility(shifted)
            out[b] = (q_emitted[mask].to(vis.dtype).unsqueeze(-1) * vis).sum(dim=0)
        return out
