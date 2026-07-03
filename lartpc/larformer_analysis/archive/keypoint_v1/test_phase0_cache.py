"""Phase 0 cache-path acceptance test.

Loads an augmented stage-12 cache event through
LArFormerStage12CacheDataset(emit_keypoints=True) and verifies the same
properties as the raw-path test: `kpscores` (N,6) in [0,1] aligned to the
cache's coord, per-type peaks near GT keypoints, per-instance endpoints, and a
nu vertex. Run AFTER augmenting the cache with
tools/larformer/augment_stage12_cache_keypoints.py.

    python lartpc/larformer_analysis/archive/keypoint_v1/test_phase0_cache.py <cache_root>
"""

import glob
import sys

import numpy as np

from pointcept.datasets.larformer_stage12_cache import (
    LArFormerStage12CacheDataset,
)
from pointcept.datasets.larformer import larformer_collate
from lartpc.data_prep.labels.keypoint_labels import KEYPOINT_TYPE_NAMES


def main():
    cache_root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kpcache_test"
    files = sorted(glob.glob(f"{cache_root}/**/*.h5", recursive=True))
    if not files:
        raise SystemExit(f"no cache .h5 under {cache_root}")
    print(f"[cache] {len(files)} cache file(s) under {cache_root}")

    ds = LArFormerStage12CacheDataset(
        split="train",
        data_root=cache_root,
        source_set_filter="union",   # widest set, deterministic
        emit_keypoints=True,
        keypoint_sigma_cm=3.0,
        min_spacepoints=1,
    )
    print(f"[cache] dataset len = {len(ds)}")

    samples = []
    for i in range(min(len(ds), 4)):
        s = ds[i]
        samples.append(s)
        n = s["coord_norm"].shape[0]
        kps = s["kpscores"]
        assert kps.shape == (n, 6), f"kpscores {kps.shape} != ({n},6)"
        assert kps.min() >= 0 and kps.max() <= 1 + 1e-5

        coord_cm = s["coord"]
        kp_pos_norm = s["mckeypoints_pos_norm"]
        kp_type = s["mckeypoints_type"]
        kp_pos_cm = kp_pos_norm * ds.coord_scale + ds.coord_center

        print(f"\n[cache] --- sample {i}: {s['name']} ---")
        print(f"  n_sp={n}  n_kp={kp_pos_norm.shape[0]}  "
              f"n_gt={s['n_gt_instances']}  nu_vtx={s['nu_vertex_coord_norm'].shape[0]}")
        for t in range(6):
            if int((kp_type == t).sum()) == 0 or kps[:, t].max() == 0:
                continue
            top = int(kps[:, t].argmax())
            d = np.linalg.norm(kp_pos_cm[kp_type == t] - coord_cm[top],
                               axis=1).min()
            ok = d < 5.0 * ds.keypoint_sigma_cm
            print(f"  peak[{KEYPOINT_TYPE_NAMES[t]}]: {kps[top,t]:.3f} "
                  f"d={d:.2f}cm {'OK' if ok else 'FAR'}")
            assert ok
        n_end = sum(1 for g in s["gt_instances"] if g.get("has_end"))
        print(f"  GT with track-end: {n_end}/{len(s['gt_instances'])}")
        for g in s["gt_instances"]:
            assert "end_coord_norm" in g and "has_end" in g

    batch = larformer_collate(samples)
    total = int(batch["offset"][-1].item())
    assert batch["kpscores"].shape == (total, 6)
    assert "nu_vertex_coord_norm_per_event" in batch
    print(f"\n[cache] collate OK: kpscores {tuple(batch['kpscores'].shape)}")
    print("\n[cache] PASS")


if __name__ == "__main__":
    main()
