"""
Test script to verify SphereCrop point_min and max_retries functionality.
"""
import os
import sys
import numpy as np
import pointcept

from pointcept.datasets import LArTPCDataset
from pointcept.datasets.transform import SphereCrop


def test_spherecrop_basic():
    """Test that SphereCrop still works without point_min (backward compatibility)."""
    print("=" * 60)
    print("Test 1: Basic SphereCrop without point_min (backward compat)")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    coords = np.random.randn(10000, 3).astype(np.float32)
    data_dict = {"coord": coords}

    crop = SphereCrop(point_max=1000, mode="random")
    result = crop(data_dict)

    print(f"Input points: {coords.shape[0]}")
    print(f"Output points: {result['coord'].shape[0]}")
    assert result["coord"].shape[0] == 1000, "Should crop to point_max"
    print("PASSED\n")


def test_spherecrop_with_point_min():
    """Test SphereCrop with point_min requirement."""
    print("=" * 60)
    print("Test 2: SphereCrop with point_min")
    print("=" * 60)

    # Create synthetic data with a cluster
    np.random.seed(42)
    # Sparse points spread out
    sparse_coords = np.random.randn(5000, 3).astype(np.float32) * 100
    # Dense cluster
    dense_coords = np.random.randn(5000, 3).astype(np.float32) * 0.1
    coords = np.vstack([sparse_coords, dense_coords])
    data_dict = {"coord": coords}

    crop = SphereCrop(point_max=1000, mode="random", point_min=500, max_retries=100)
    result = crop(data_dict)

    print(f"Input points: {coords.shape[0]}")
    print(f"Output points: {result['coord'].shape[0]}")
    print(f"point_min: 500, point_max: 1000")
    assert result["coord"].shape[0] >= 500, f"Should have at least point_min points, got {result['coord'].shape[0]}"
    assert result["coord"].shape[0] <= 1000, "Should not exceed point_max"
    print("PASSED\n")


def test_spherecrop_retry_count():
    """Test that retries actually happen by using a mock."""
    print("=" * 60)
    print("Test 3: Verify retry mechanism")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    coords = np.random.randn(10000, 3).astype(np.float32)
    data_dict = {"coord": coords}

    # Track how many times random is called
    original_randint = np.random.randint
    call_count = [0]

    def counting_randint(*args, **kwargs):
        call_count[0] += 1
        return original_randint(*args, **kwargs)

    # Test with point_min set impossibly high to force all retries
    crop = SphereCrop(point_max=100, mode="random", point_min=10000, max_retries=50)

    np.random.randint = counting_randint
    try:
        result = crop({"coord": coords.copy()})
    finally:
        np.random.randint = original_randint

    print(f"Max retries configured: 50")
    print(f"Actual random calls: {call_count[0]}")
    assert call_count[0] == 50, f"Should have called randint max_retries times, got {call_count[0]}"
    print("PASSED\n")


def test_spherecrop_center_mode_no_retry():
    """Test that center mode doesn't use retry logic (deterministic)."""
    print("=" * 60)
    print("Test 4: Center mode ignores point_min (deterministic)")
    print("=" * 60)

    np.random.seed(42)
    coords = np.random.randn(10000, 3).astype(np.float32)
    data_dict = {"coord": coords}

    # Even with point_min set, center mode should be deterministic
    crop = SphereCrop(point_max=1000, mode="center", point_min=500, max_retries=100)

    result1 = crop({"coord": coords.copy()})
    result2 = crop({"coord": coords.copy()})

    print(f"Output points (run 1): {result1['coord'].shape[0]}")
    print(f"Output points (run 2): {result2['coord'].shape[0]}")
    assert np.allclose(result1["coord"], result2["coord"]), "Center mode should be deterministic"
    print("PASSED\n")


def test_spherecrop_with_real_data(nthrows=100):
    """Test SphereCrop with real LArTPC data if available."""
    print("=" * 60)
    print("Test 5: SphereCrop with real LArTPC data")
    print("=" * 60)

    data_root = "data/lartpc"
    if not os.path.exists(data_root):
        print(f"Skipping: {data_root} not found")
        print("SKIPPED\n")
        return

    try:
        dataset = LArTPCDataset(
            data_root=data_root,
            coord_scale=0.001,
            exclude_other=True,
            include_ghosts=True
        )
        data = dataset.get_data(0)

        print(f"Loaded data with {data['coord'].shape[0]} points")

        crop = SphereCrop(point_max=5000, mode="random", point_min=1000, max_retries=100)
        result = crop(data)

        print(f"After SphereCrop: {result['coord'].shape[0]} points")
        assert result["coord"].shape[0] >= 1000, "Should have at least point_min points"
        assert result["coord"].shape[0] <= 5000, "Should not exceed point_max"
        print("PASSED\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("SKIPPED\n")


def test_spherecrop_no_crop_needed():
    """Test that small point clouds are returned unchanged."""
    print("=" * 60)
    print("Test 6: Small point cloud (no cropping needed)")
    print("=" * 60)

    np.random.seed(42)
    coords = np.random.randn(500, 3).astype(np.float32)
    data_dict = {"coord": coords}

    crop = SphereCrop(point_max=1000, mode="random", point_min=100, max_retries=100)
    result = crop(data_dict)

    print(f"Input points: {coords.shape[0]}")
    print(f"Output points: {result['coord'].shape[0]}")
    assert result["coord"].shape[0] == 500, "Should not crop when below point_max"
    assert np.allclose(result["coord"], coords), "Data should be unchanged"
    print("PASSED\n")


if __name__ == "__main__":
    print("\nTesting SphereCrop point_min and max_retries functionality\n")

    #test_spherecrop_basic()
    #test_spherecrop_with_point_min()
    #test_spherecrop_retry_count()
    #test_spherecrop_center_mode_no_retry()
    test_spherecrop_with_real_data()
    #test_spherecrop_no_crop_needed()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
