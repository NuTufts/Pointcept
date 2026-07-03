"""Validate the GPU PhotonLibLookup against the C++ UBPhotonLib and on the
canonical nu slice.

Test 1 (parity): for N=300 random TPC positions inside the cryostat, the
    torch ``visibility_trilinear`` should match the C++
    ``UBPhotonLib::getVisibilityTrilinear`` for each OpDet within a small
    relative tolerance.

Test 2 (smoke): take the nu slice from
    merged_bnb_nu_pi0filter_corsika_fileno00001_entry000000.h5, compute
    q_emitted per spacepoint using the Y-with-UV-fallback rule, run
    PhotonLibLookup.predict_flash(...), and compare to the observed beam
    flash (32,) PE vector via cosine similarity and a per-OpDet ratio.

Test 3 (throughput): time predict_flash on a synthetic batch and
    predict_flash_pairs for several (cluster, flash) candidates.
"""

import os
import sys
import time

# Import h5py before ROOT/PyROOT to avoid an HDF5 ABI clash in the container
# (ROOT pulls in system HDF5 which is incompatible with conda's h5py).
import h5py  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from pointcept.models.event_slicer.photonlib import (  # noqa: E402
    PhotonLibLookup, select_charge_y_with_uv_fallback,
)
from slice_labels import compute_slice_labels  # noqa: E402


CACHE = "/home/twongjirad/working/larbys/gen2/container_u22/Pointcept/lartpc_data_prep/dat/photonlib_v6_70kV.npz"
MERGED_H5 = ("/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/000/000/"
             "merged_bnb_nu_pi0filter_corsika_fileno00001_entry000000.h5")
FLASH_H5 = "/tmp/flashinfo_test_entry000000.h5"


def test_parity_vs_cpp(pl, n_test=300, rng=None):
    print("=== test 1: trilinear parity vs C++ UBPhotonLib ===")
    from ROOT import std
    from ublarcvapp import ublarcvapp
    cpp = ublarcvapp.ubphotonlib.UBPhotonLib.getPhotonLib()

    rng = rng or np.random.default_rng(0)
    nv = pl.meta.nvoxels_dim
    vox = pl.meta.voxel_len_cm
    orig = pl.meta.cryo_origin_tpc_cm
    mins = orig + vox     # interior — one voxel away from each edge
    maxs = orig + (nv - 1) * vox

    positions = rng.uniform(mins, maxs, size=(n_test, 3)).astype(np.float32)
    pt_pos = torch.from_numpy(positions).to(pl.vis_table.device)
    py_vis = pl.visibility_trilinear(pt_pos).cpu().numpy()  # (N, 32)

    cpp_vis = np.zeros_like(py_vis)
    pos_v = std.vector("float")(3)
    for i in range(n_test):
        pos_v[0], pos_v[1], pos_v[2] = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
        for j in range(pl.n_opdets):
            cpp_vis[i, j] = cpp.getVisibilityTrilinear(pos_v, j)

    abs_err = np.abs(py_vis - cpp_vis)
    den = np.maximum(np.abs(cpp_vis), 1e-12)
    rel_err = abs_err / den
    n_strict = int(((abs_err < 1e-7) | (rel_err < 1e-5)).sum())
    total = py_vis.size
    print(f"  n_points={n_test}, n_opdets={pl.n_opdets}, comparisons={total}")
    print(f"  matching cells (abs<1e-7 or rel<1e-5): {n_strict}/{total} "
          f"= {100*n_strict/total:.2f}%")
    print(f"  max abs err over matched cells: {abs_err.max():.2e}")
    print(f"  median rel err on non-zero cells: "
          f"{np.median(rel_err[cpp_vis > 1e-10]):.2e}")
    return n_strict == total


def test_nu_slice_prediction(pl):
    print()
    print("=== test 2: nu slice prediction vs observed beam flash ===")
    import h5py
    with h5py.File(MERGED_H5, "r") as mh:
        e = mh["entry_0"]
        td = e["triplet_data"]
        pos = td["pos"][:].astype(np.float32)
        pixval = td["pixval"][:].astype(np.float32)
        sinfo = compute_slice_labels(
            e["mc_particle_tree"], td["trackid"][:], td["hasmatch"][:],
        )
    # Find nu slice
    nu_idx = int(np.where(sinfo["primary_origin"] == 1)[0][0])
    nu_slice_id = int(sinfo["primary_trackid"][nu_idx])
    mask = sinfo["slice_id"] == nu_slice_id
    print(f"  nu slice_id={nu_slice_id}, n_points={int(mask.sum())}")

    # Observed PE for the matched beam flash
    with h5py.File(FLASH_H5, "r") as fh:
        e = fh["entry_0"]
        fl = e["flashes"]
        sl = e["slice_flash_matches"]
        sid = sl["slice_id"][:]
        idx = int(np.where(sid == nu_slice_id)[0][0])
        matched_idx = int(sl["matched_flash_idx"][idx])
        obs_pe = fl["pe"][matched_idx]
        obs_total = float(fl["total_pe"][matched_idx])
        flash_time_us = float(fl["time_us"][matched_idx])
    print(f"  matched flash idx={matched_idx}, total observed PE={obs_total:.2f}, "
          f"flash time_us={flash_time_us:.3f}")

    # Predict (with γ = 1 first; we report the ratio)
    device = pl.vis_table.device
    pos_t = torch.from_numpy(pos[mask]).to(device)
    q = select_charge_y_with_uv_fallback(
        torch.from_numpy(pixval[mask])
    ).to(device)
    cid = torch.zeros(int(mask.sum()), dtype=torch.int64, device=device)
    pred_unit = pl.predict_flash(pos_t, q, cid, n_clusters=1)[0].cpu().numpy()
    pred_total = float(pred_unit.sum())

    # Drift correction
    v_drift = 0.1098
    pos_corrected = pos_t.clone()
    pos_corrected[:, 0] = pos_corrected[:, 0] - v_drift * flash_time_us
    pred_corrected = pl.predict_flash(
        pos_corrected, q, cid, n_clusters=1
    )[0].cpu().numpy()

    def cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        return float((a * b).sum() / (na * nb + 1e-12))

    cos_uncorr = cosine(pred_unit, obs_pe)
    cos_corr = cosine(pred_corrected, obs_pe)
    gamma_est = obs_total / max(pred_total, 1e-12)
    print(f"  uncorrected predicted ΣPE (γ=1): {pred_total:.3e}")
    print(f"  cosine(pred_uncorr, observed): {cos_uncorr:.4f}")
    print(f"  cosine(pred_drift_corr, observed): {cos_corr:.4f}")
    print(f"  implied γ for ΣPE match: γ ≈ {gamma_est:.3e}  (photons / ADC)")
    # Print per-OpDet top-5
    pred_scaled = pred_corrected * gamma_est
    order_obs = np.argsort(obs_pe)[::-1][:8]
    print("  top OpDets (observed → scaled predicted):")
    for j in order_obs:
        print(f"    OpDet {int(j):>2d}: obs={obs_pe[j]:7.2f}  pred={pred_scaled[j]:7.2f}  "
              f"ratio={pred_scaled[j] / max(obs_pe[j], 1e-9):.2f}")
    return cos_corr


def test_throughput(pl):
    print()
    print("=== test 3: throughput ===")
    device = pl.vis_table.device
    n_clusters = 25
    n_per_cluster = 8000
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.0, 256.0, size=(n_clusters * n_per_cluster, 3)).astype(np.float32)
    pos[:, 1] = rng.uniform(-115, 115, size=pos.shape[0])
    pos[:, 2] = rng.uniform(0, 1036, size=pos.shape[0])
    cid = np.repeat(np.arange(n_clusters, dtype=np.int64), n_per_cluster)
    q = rng.uniform(0, 100, size=pos.shape[0]).astype(np.float32)
    pos_t = torch.from_numpy(pos).to(device)
    cid_t = torch.from_numpy(cid).to(device)
    q_t = torch.from_numpy(q).to(device)
    # warm-up
    _ = pl.predict_flash(pos_t, q_t, cid_t, n_clusters)
    t0 = time.time()
    for _ in range(5):
        out = pl.predict_flash(pos_t, q_t, cid_t, n_clusters)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    print(f"  predict_flash  N={pos.shape[0]:,} K={n_clusters}: "
          f"{dt*1000:.1f} ms/call ({pos.shape[0]/dt/1e6:.1f} M points/sec)")

    # pairs
    B = 60
    pair_c = torch.from_numpy(rng.integers(0, n_clusters, size=B).astype(np.int64)).to(device)
    pair_t = torch.from_numpy(rng.uniform(-3000, 3000, size=B).astype(np.float32)).to(device)
    t0 = time.time()
    for _ in range(3):
        out = pl.predict_flash_pairs(pos_t, q_t, cid_t, pair_c, pair_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"  predict_flash_pairs B={B}: {dt*1000:.1f} ms/call")


def main():
    print(f"Loading cache from {CACHE}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}")
    pl = PhotonLibLookup(CACHE, fp16=False, use_trilinear=True).to(device)
    print(f"  visibility tensor: shape={tuple(pl.vis_table.shape)} "
          f"dtype={pl.vis_table.dtype}")

    test_parity_vs_cpp(pl)
    test_nu_slice_prediction(pl)
    test_throughput(pl)


if __name__ == "__main__":
    main()
