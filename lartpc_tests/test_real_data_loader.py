"""
Test script for loading real (data-only) LArTPC files with the dataset loader.

Tests that:
1. data_only=True works on real extbnb data (no truth fields)
2. data_only=False still works on simulation data (backward compat)
3. data_only="auto" correctly detects both file types
4. BiasedSphereCrop works with data_only (no nu_vertices available)
5. The full pretrain transform pipeline works end-to-end on real data

Usage:
    python test_real_data_loader.py
"""

import os
import sys
import numpy as np

import pointcept
from pointcept.datasets import LArTPCDataset
from pointcept.datasets.transform import BiasedSphereCrop, GridSample, Collect

# ============================================================================
# Test file paths — update these if files move
# ============================================================================
REAL_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "lartpc_data_prep",
    "extbnb",
    "pointceptdata_merged_dlreco_00deb37b-c348-4dd4-b886-2b2aa166b24f_entry000000.h5",
)

SIM_DATA_FILE = (
    "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/"
    "v2_expandedclasses/bnb_nu_chargedpiplus_corsika/000/000/"
    "pointceptdata_dlmerged_coriska_bnb_nu_chargedpiplus_fileno000001_entry000000.h5"
)

# Parameters matching the extbnb config
GRID_SIZE = 0.25
COORD_SCALE = 1.0
WIRE_SCALE = 1.0 / 3456.0
BIASED_SPHERECROP_RADIUS = 20.0
MAX_POINTS_SPHERECROP = 20480
MIN_POINTS_SPHERECROP = 4096

passed = 0
failed = 0


def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1


# ============================================================================
# Helper: write a temp file list containing a single path
# ============================================================================
def _write_tmp_filelist(path):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(path + "\n")
    tmp.close()
    return tmp.name


# ============================================================================
# Test 1: data_only=True on real data — raw get_data()
# ============================================================================
def test_real_data_only_true():
    print("\n=== Test 1: data_only=True on real extbnb data ===")

    if not os.path.isfile(REAL_DATA_FILE):
        print(f"  SKIP: real data file not found: {REAL_DATA_FILE}")
        return

    flist = _write_tmp_filelist(REAL_DATA_FILE)
    try:
        ds = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
        )

        data = ds.get_data(0)

        check("coord" in data, "data dict has 'coord'")
        check("strength" in data, "data dict has 'strength'")
        check("color" in data, "data dict has 'color'")
        check("segment" in data, "data dict has 'segment'")
        check("instance" in data, "data dict has 'instance'")
        check("nu_vertices" in data, "data dict has 'nu_vertices'")
        check("origin" in data, "data dict has 'origin'")

        N = data["coord"].shape[0]
        check(N > 0, f"coord has {N} points (> 0)")
        check(data["coord"].shape == (N, 3), f"coord shape is (N, 3)")
        check(data["strength"].shape[0] == N, "strength has N rows")
        check(data["color"].shape == (N, 3), "color (wire_feat) shape is (N, 3)")

        # Dummy truth values
        check(np.all(data["segment"] == -1), "segment is all -1 (no truth)")
        check(np.all(data["instance"] == -1), "instance is all -1 (no truth)")
        check(np.all(data["origin"] == 0), "origin is all 0 (no truth)")
        check(data["nu_vertices"].shape == (0, 3), "nu_vertices is empty (0, 3)")
        check(data["segment_counts"].shape[1] == 0, "segment_counts has 0 classes")
    finally:
        os.unlink(flist)


# ============================================================================
# Test 2: data_only=False on simulation data — backward compatibility
# ============================================================================
def test_sim_data_only_false():
    print("\n=== Test 2: data_only=False on simulation data (backward compat) ===")

    if not os.path.isfile(SIM_DATA_FILE):
        print(f"  SKIP: sim data file not found: {SIM_DATA_FILE}")
        return

    flist = _write_tmp_filelist(SIM_DATA_FILE)
    try:
        ds = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=False,
        )

        data = ds.get_data(0)

        N = data["coord"].shape[0]
        check(N > 0, f"coord has {N} points (> 0)")

        # Truth should be loaded
        check(not np.all(data["segment"] == -1), "segment has real labels (not all -1)")
        check(not np.all(data["origin"] == 0), "origin has real values (not all 0)")
        check(data["nu_vertices"].shape[1] == 3, "nu_vertices has 3 columns")
        check(data["segment_counts"].shape[1] > 0, "segment_counts has >0 classes")
    finally:
        os.unlink(flist)


# ============================================================================
# Test 3: data_only="auto" on both file types
# ============================================================================
def test_auto_detect():
    print("\n=== Test 3: data_only='auto' detection ===")

    # Real data
    if os.path.isfile(REAL_DATA_FILE):
        flist = _write_tmp_filelist(REAL_DATA_FILE)
        try:
            ds = LArTPCDataset(
                coord_scale=COORD_SCALE,
                data_list_file=flist,
                include_ghosts=True,
                exclude_other=False,
                label_mode="ssnet",
                data_only="auto",
            )
            data = ds.get_data(0)
            check(np.all(data["segment"] == -1),
                  "auto-detect real data -> segment is all -1")
            check(data["nu_vertices"].shape[0] == 0,
                  "auto-detect real data -> nu_vertices is empty")
        finally:
            os.unlink(flist)
    else:
        print(f"  SKIP real: file not found: {REAL_DATA_FILE}")

    # Sim data
    if os.path.isfile(SIM_DATA_FILE):
        flist = _write_tmp_filelist(SIM_DATA_FILE)
        try:
            ds = LArTPCDataset(
                coord_scale=COORD_SCALE,
                data_list_file=flist,
                include_ghosts=True,
                exclude_other=False,
                label_mode="ssnet",
                data_only="auto",
            )
            data = ds.get_data(0)
            check(not np.all(data["segment"] == -1),
                  "auto-detect sim data -> segment has real labels")
            check(data["nu_vertices"].shape[1] == 3,
                  "auto-detect sim data -> nu_vertices has data")
        finally:
            os.unlink(flist)
    else:
        print(f"  SKIP sim: file not found: {SIM_DATA_FILE}")


# ============================================================================
# Test 4: BiasedSphereCrop with real data (no nu_vertices)
# ============================================================================
def test_biased_spherecrop_real_data():
    print("\n=== Test 4: BiasedSphereCrop on real data (no anchor points) ===")

    if not os.path.isfile(REAL_DATA_FILE):
        print(f"  SKIP: real data file not found: {REAL_DATA_FILE}")
        return

    flist = _write_tmp_filelist(REAL_DATA_FILE)
    try:
        transform = [
            dict(
                type="GridSample",
                grid_size=GRID_SIZE,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(
                type="BiasedSphereCrop",
                anchor_points_key=None,
                anchor_pdf_key=None,
                radius=BIASED_SPHERECROP_RADIUS,
                point_max=MAX_POINTS_SPHERECROP,
                point_min=MIN_POINTS_SPHERECROP,
                prob_random=1.0,
                max_retries=100,
                fallback_to_random=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                offset_keys_dict=dict(),
                feat_keys=("strength", "color"),
            ),
        ]

        ds = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
            transform=transform,
        )

        # Load through the full pipeline (dataset.__getitem__ applies transforms)
        data = ds[0]

        check("coord" in data, "transformed data has 'coord'")
        check("feat" in data, "transformed data has 'feat'")
        check("segment" in data, "transformed data has 'segment'")

        n_pts = data["coord"].shape[0]
        check(n_pts <= MAX_POINTS_SPHERECROP,
              f"points ({n_pts}) <= point_max ({MAX_POINTS_SPHERECROP})")
        check(n_pts >= MIN_POINTS_SPHERECROP,
              f"points ({n_pts}) >= point_min ({MIN_POINTS_SPHERECROP})")

        # feat should be strength(3) + color(3) = 6 channels
        check(data["feat"].shape[1] == 6,
              f"feat has 6 channels (got {data['feat'].shape[1]})")
    finally:
        os.unlink(flist)


# ============================================================================
# Test 5: Full pretrain transform pipeline on real data
# ============================================================================
def test_full_pretrain_pipeline():
    print("\n=== Test 5: Full pretrain transform pipeline on real data ===")

    if not os.path.isfile(REAL_DATA_FILE):
        print(f"  SKIP: real data file not found: {REAL_DATA_FILE}")
        return

    coord_scale_norm = 1036.0 * 3**0.5 / 2.0 / 5.0
    scaled_grid_size = GRID_SIZE / coord_scale_norm
    max_points_per_view = 20480

    transform = [
        dict(
            type="GridSample",
            grid_size=GRID_SIZE,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(
            type="BiasedSphereCrop",
            anchor_points_key=None,
            anchor_pdf_key=None,
            radius=BIASED_SPHERECROP_RADIUS,
            point_max=MAX_POINTS_SPHERECROP,
            point_min=MIN_POINTS_SPHERECROP,
            prob_random=1.0,
            max_retries=100,
            fallback_to_random=True,
        ),
        dict(type="NormalizeCoord", center=[125.0, 0.0, 1036.0 / 2.0], scale=coord_scale_norm),
        dict(
            type="LogTransform",
            min_val=0.01,
            max_val=1000.0,
            log=True,
            keys=("strength",),
        ),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
        dict(
            type="MultiViewGenerator",
            view_keys=("coord", "origin_coord", "strength"),
            global_view_num=2,
            global_view_scale=(0.4, 1.0),
            local_view_num=6,
            local_view_scale=(0.1, 0.4),
            global_shared_transform=[
                dict(
                    type="MultiplicativeRandomJitter",
                    sigma=0.05,
                    clip=0.05,
                    keys=("strength"),
                    p=0.8,
                ),
            ],
            global_transform=[
                dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
                dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
                dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
                dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
                dict(type="RandomJitter", sigma=scaled_grid_size / 4, clip=scaled_grid_size, keys=("coord",)),
            ],
            local_transform=[
                dict(type="CenterShift", apply_z=False, axes=("x", "y", "z")),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
                dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
                dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
                dict(type="RandomFlip", p=0.5, axes=("x", "y", "z")),
                dict(type="RandomJitter", sigma=scaled_grid_size / 4, clip=scaled_grid_size, keys=("coord",)),
            ],
            max_size=max_points_per_view,
        ),
        dict(type="ToTensor"),
        dict(type="Update", keys_dict={"grid_size": scaled_grid_size}),
        dict(
            type="Collect",
            keys=(
                "global_origin_coord",
                "global_coord",
                "global_offset",
                "local_origin_coord",
                "local_coord",
                "local_offset",
                "grid_size",
                "name",
            ),
            offset_keys_dict=dict(),
            global_feat_keys=("global_coord", "global_strength"),
            local_feat_keys=("local_coord", "local_strength"),
        ),
    ]

    flist = _write_tmp_filelist(REAL_DATA_FILE)
    try:
        ds = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
            transform=transform,
        )

        data = ds[0]

        check("global_coord" in data, "output has 'global_coord'")
        check("global_feat" in data, "output has 'global_feat'")
        check("local_coord" in data, "output has 'local_coord'")
        check("local_feat" in data, "output has 'local_feat'")
        check("global_offset" in data, "output has 'global_offset'")
        check("local_offset" in data, "output has 'local_offset'")

        # global_feat should have 6 channels (3 coord + 3 strength)
        check(data["global_feat"].shape[1] == 6,
              f"global_feat has 6 channels (got {data['global_feat'].shape[1]})")
        check(data["local_feat"].shape[1] == 6,
              f"local_feat has 6 channels (got {data['local_feat'].shape[1]})")

        # global_offset should have 2 entries (2 global views)
        check(len(data["global_offset"]) == 2,
              f"global_offset has 2 views (got {len(data['global_offset'])})")
        # local_offset should have 6 entries (6 local views)
        check(len(data["local_offset"]) == 6,
              f"local_offset has 6 views (got {len(data['local_offset'])})")

        print(f"  global_coord shape: {data['global_coord'].shape}")
        print(f"  global_feat shape:  {data['global_feat'].shape}")
        print(f"  local_coord shape:  {data['local_coord'].shape}")
        print(f"  local_feat shape:   {data['local_feat'].shape}")
    finally:
        os.unlink(flist)


# ============================================================================
# Test 6: true_points_only and drop_cosmics are no-ops when data_only=True
# ============================================================================
def test_truth_filters_noop():
    print("\n=== Test 6: true_points_only/drop_cosmics are no-ops with data_only=True ===")

    if not os.path.isfile(REAL_DATA_FILE):
        print(f"  SKIP: real data file not found: {REAL_DATA_FILE}")
        return

    flist = _write_tmp_filelist(REAL_DATA_FILE)
    try:
        # Load baseline
        ds_base = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
        )
        data_base = ds_base.get_data(0)
        n_base = data_base["coord"].shape[0]

        # Load with true_points_only=True (should be ignored for data_only)
        ds_tpo = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
            true_points_only=True,
        )
        data_tpo = ds_tpo.get_data(0)
        n_tpo = data_tpo["coord"].shape[0]

        check(n_tpo == n_base,
              f"true_points_only=True is no-op: {n_tpo} == {n_base} points")

        # Load with drop_cosmics=True, prob=1.0 (should be ignored for data_only)
        ds_dc = LArTPCDataset(
            coord_scale=COORD_SCALE,
            data_list_file=flist,
            include_ghosts=True,
            exclude_other=False,
            label_mode="ssnet",
            data_only=True,
            drop_cosmics=True,
            drop_cosmics_prob=1.0,
        )
        data_dc = ds_dc.get_data(0)
        n_dc = data_dc["coord"].shape[0]

        check(n_dc == n_base,
              f"drop_cosmics=True is no-op: {n_dc} == {n_base} points")
    finally:
        os.unlink(flist)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Test: Real (data-only) LArTPC file loading")
    print("=" * 70)

    test_real_data_only_true()
    test_sim_data_only_false()
    test_auto_detect()
    test_biased_spherecrop_real_data()
    test_full_pretrain_pipeline()
    test_truth_filters_noop()

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
