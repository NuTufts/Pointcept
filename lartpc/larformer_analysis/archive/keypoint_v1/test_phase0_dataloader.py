"""Phase 0 acceptance smoke test for the LArFormer keypoint data path.

Loads a raw merged_h5 event through LArFormerDataset(emit_keypoints=True) and
verifies:
  1. `kpscores` is per-SP, shape (N, 6), values in [0, 1], aligned to coord_norm.
  2. The dense score peaks co-locate with the GT `mckeypoints/pos` (the SP with
     the highest score for a type is close to a keypoint of that type).
  3. GT instances carry `end_coord_norm` / `has_end`; the event carries
     `nu_vertex_coord_norm`.
  4. The collate produces a flat `kpscores` sliceable by `offset`, and the
     per-event mckeypoints lists.

Run inside the pointcept container env:
    python lartpc/larformer_analysis/archive/keypoint_v1/test_phase0_dataloader.py \
        [/path/to/merged_event.h5]
"""

import glob
import sys

import numpy as np

from pointcept.datasets.larformer import LArFormerDataset, larformer_collate
from lartpc.data_prep.labels.keypoint_labels import KEYPOINT_TYPE_NAMES

DEFAULT_GLOB = ("/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/"
                "merged_h5/000/000/*.h5")


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob(DEFAULT_GLOB))[:2]
    if not paths:
        raise SystemExit(f"no input files (looked at {DEFAULT_GLOB})")
    print(f"[phase0] testing {len(paths)} file(s)")

    # Write a one-line list so get_data_list picks it up; data_root irrelevant
    # because paths are absolute.
    list_file = "/tmp/_phase0_keypoint_list.txt"
    with open(list_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

    ds = LArFormerDataset(
        split="train",
        data_root="/",
        data_list_file=list_file,
        gt_source="particle",
        emit_keypoints=True,
        keypoint_sigma_cm=3.0,
        max_spacepoints=None,
        # deterministic threshold for the test
        lm_score_aug_low=0.15, lm_score_aug_high=0.15,
    )
    print(f"[phase0] dataset len = {len(ds)}")

    samples = []
    for i in range(len(ds)):
        s = ds[i]
        samples.append(s)

        kps = s["kpscores"]
        cn = s["coord_norm"]
        n = cn.shape[0]
        assert kps.shape == (n, 6), f"kpscores shape {kps.shape} != ({n}, 6)"
        assert kps.dtype == np.float32
        assert kps.min() >= 0.0 and kps.max() <= 1.0 + 1e-5, \
            f"kpscores out of [0,1]: {kps.min()}..{kps.max()}"

        coord_cm = s["coord"]
        kp_pos = s["mckeypoints_pos_norm"]
        kp_type = s["mckeypoints_type"]
        # de-normalize keypoints to cm for the distance check
        center = ds.coord_center
        scale = ds.coord_scale
        kp_pos_cm = kp_pos * scale + center

        print(f"\n[phase0] --- sample {i}: {s['name']} ---")
        print(f"  n_spacepoints = {n}   n_keypoints = {kp_pos.shape[0]}   "
              f"n_gt_instances = {s['n_gt_instances']}")
        print(f"  kpscores nonzero per type: ", end="")
        for t in range(6):
            nz = int((kps[:, t] > 0).sum())
            print(f"{KEYPOINT_TYPE_NAMES[t]}={nz}", end="  ")
        print()

        # Peak co-location: for each type present, the top-score SP should be
        # within a few sigma of a GT keypoint of that type.
        for t in range(6):
            has_kp = int((kp_type == t).sum())
            if has_kp == 0 or kps[:, t].max() == 0:
                continue
            top = int(kps[:, t].argmax())
            kpt_cm = kp_pos_cm[kp_type == t]
            d = np.linalg.norm(kpt_cm - coord_cm[top], axis=1).min()
            ok = d < 5.0 * ds.keypoint_sigma_cm
            print(f"  peak[{KEYPOINT_TYPE_NAMES[t]}]: top-score "
                  f"{kps[top, t]:.3f} at d={d:.2f}cm to nearest GT  "
                  f"{'OK' if ok else 'FAR'}")
            assert ok, f"type {t} peak {d:.1f}cm from any GT keypoint"

        # nu vertex + endpoints
        nuv = s["nu_vertex_coord_norm"]
        print(f"  nu_vertex_coord_norm: {nuv.shape[0]} vertex(es)")
        n_end = sum(1 for g in s["gt_instances"] if g.get("has_end"))
        print(f"  GT instances with track-end: {n_end}/{len(s['gt_instances'])}")
        for g in s["gt_instances"][:4]:
            assert "end_coord_norm" in g and "has_end" in g, \
                "gt instance missing end_coord_norm/has_end"
            print(f"    tid={g.get('primary_trackid')} pid={g.get('pid')} "
                  f"class={g.get('class_id')} has_end={g['has_end']} "
                  f"start={np.round(g['origin_coord_norm'], 3)} "
                  f"end={np.round(g['end_coord_norm'], 3)}")

    # Collate check
    batch = larformer_collate(samples)
    total_n = int(batch["offset"][-1].item())
    assert batch["kpscores"].shape == (total_n, 6), \
        f"collated kpscores {tuple(batch['kpscores'].shape)} != ({total_n}, 6)"
    assert "mckeypoints_pos_norm_per_event" in batch
    assert len(batch["nu_vertex_coord_norm_per_event"]) == len(samples)
    # per-event slice of flat kpscores must match the per-sample array
    prev = 0
    for i, off in enumerate(batch["offset"].tolist()):
        sl = batch["kpscores"][prev:off].numpy()
        assert np.allclose(sl, samples[i]["kpscores"]), \
            f"flat kpscores slice {i} != per-sample kpscores"
        prev = off
    print(f"\n[phase0] collate OK: flat kpscores {tuple(batch['kpscores'].shape)}, "
          f"per-event slices match, mckeypoints nested for "
          f"{len(samples)} event(s)")
    print("\n[phase0] PASS")


if __name__ == "__main__":
    main()
