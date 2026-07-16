"""
Gate tests for the P05B.5 asinh input-scaling changes
(lartpc/pretraining_studies/input_dist_study/P05B5_IMPLEMENTATION_HANDOFF.md §3).

CPU-only; run inside the container at Tufts:

  apptainer exec --bind /cluster:/cluster \
      /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif \
      python3 lartpc_tests/test_strength_transforms.py

Test 5 (end-to-end default preservation) compares the P05B.1 pipeline output
against a golden file that must be produced from the PRE-CHANGE code:

  git stash / checkout <merge-base of p05b5-asinh-input>   # pre-change tree
  apptainer exec ... python3 lartpc_tests/test_strength_transforms.py --make-golden
  git checkout p05b5-asinh-input                           # post-change tree
  apptainer exec ... python3 lartpc_tests/test_strength_transforms.py

The golden (lartpc_tests/golden/p05b1_pipeline_golden.npz) is gitignored
(*.npz): each reviewer regenerates it at their own merge-base, which is the
point — the test then proves the branch did not change P05B.1 behavior.
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_DIR = os.path.join(REPO, "configs", "lartpc", "p05")
GOLDEN_PATH = os.path.join(REPO, "lartpc_tests", "golden", "p05b1_pipeline_golden.npz")
DIAG1K_LIST = os.path.join(REPO, "lartpc", "filelists", "h5list_v3_mc_diag1k_tufts.txt")

B1_CONFIG = os.path.join(CONFIG_DIR, "pretrain-sonata-p05b1-mc-noghost-freerot.py")
B5_CONFIG = os.path.join(CONFIG_DIR, "pretrain-sonata-p05b5-mc-noghost-asinh.py")
B6_CONFIG = os.path.join(CONFIG_DIR, "pretrain-sonata-p05b6-mc-noghost-asinh-jitter005.py")
PROBE_ASINH_CONFIG = os.path.join(
    CONFIG_DIR, "linearprobe-sonata-p05-mc-noghost-asinh-tufts.py")

SEED = 190716  # fixed seed for every stochastic test below

# P05 pipeline constants (handoff §1)
MIN_VAL = 0.01
MAX_VAL = 1000.0
ASINH_SCALE = 50.0


# =============================================================================
# Inline reference implementations of the CURRENT (pre-change) formulas.
# Copied from handoff §1 / the pre-change transform.py — NOT from the
# refactored code, so test 1 detects any behavior drift in the refactor.
# =============================================================================
def ref_log_transform(x, min_val, max_val):
    y0 = np.log10(min_val)
    y1 = np.log10(max_val + min_val)
    return 2 * (np.log10(x + min_val) - y0) / (y1 - y0) - 1


def ref_linear_transform(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1


def ref_legacy_jitter(y, sigma, clip, p, log_space):
    """Replicates the pre-change MultiplicativeRandomJitter draw-for-draw."""
    if random.random() > p:
        return y
    noise = np.clip(np.random.randn(*y.shape) * sigma, -clip, clip)
    if log_space:
        return y + np.log10(1.0 + noise)
    return y * (1.0 + noise)


# Closed forms for the NEW modes (handoff §2.1/§2.2), written independently.
def ref_asinh_transform(x, scale, max_val):
    denom = np.arcsinh(max_val / scale)
    return 2 * np.arcsinh(np.clip(x, 0, max_val) / scale) / denom - 1


def ref_asinh_inverse(y, scale, max_val):
    denom = np.arcsinh(max_val / scale)
    return scale * np.sinh((y + 1) / 2 * denom)


def ref_scaled_log_inverse(y, min_val, max_val):
    denom = np.log10(max_val + min_val) - np.log10(min_val)
    return min_val * (np.power(10.0, (y + 1) * denom / 2) - 1)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


# =============================================================================
# 1. Regression / default-preservation (bit-identical, handoff §3.1)
# =============================================================================
def test_1_default_preservation():
    from pointcept.datasets.transform import LogTransform, MultiplicativeRandomJitter

    rng = np.random.RandomState(SEED)
    for dtype in (np.float32, np.float64):
        # loader-like strengths: clip(pixval,0,1000)+0.01, incl. exact 0 & clip
        pix = rng.uniform(0, 1200, size=(4096, 3)).astype(dtype)
        pix[:16] = 0.0
        pix[16:32] = 1200.0
        x = (np.clip(pix, 0, 1000.0) + 0.01).astype(dtype)

        for log in (True, False):
            t = LogTransform(min_val=MIN_VAL, max_val=MAX_VAL, log=log,
                             keys=("strength",))
            got = t({"strength": x.copy()})["strength"]
            want = (ref_log_transform(x.copy(), MIN_VAL, MAX_VAL) if log
                    else ref_linear_transform(x.copy(), MIN_VAL, MAX_VAL))
            assert got.dtype == want.dtype, (got.dtype, want.dtype)
            assert np.array_equal(got, want), \
                f"LogTransform(log={log}) not bit-identical ({dtype})"

        y = ref_log_transform(x, MIN_VAL, MAX_VAL)
        for log_space in (True, False):
            for p in (1.0, 0.5):
                for trial in range(8):
                    t = MultiplicativeRandomJitter(
                        sigma=0.05, clip=0.05, keys=("strength"), p=p,
                        log_space=log_space)
                    seed_all(SEED + trial)
                    got = t({"strength": y.copy()})["strength"]
                    seed_all(SEED + trial)
                    want = ref_legacy_jitter(y.copy(), 0.05, 0.05, p, log_space)
                    assert np.array_equal(got, want), \
                        f"legacy jitter (log_space={log_space}, p={p}) drifted"
    print("PASS  1. legacy LogTransform + MultiplicativeRandomJitter bit-identical")


# =============================================================================
# 2. Asinh correctness (handoff §3.2)
# =============================================================================
def test_2_asinh_correctness():
    import math
    from pointcept.datasets.transform import LogTransform

    t = LogTransform(mode="asinh", asinh_scale=ASINH_SCALE, max_val=MAX_VAL,
                     keys=("strength",))

    # endpoints exact
    ends = t({"strength": np.array([0.0, MAX_VAL])})["strength"]
    assert ends[0] == -1.0 and ends[1] == 1.0, ends

    # values above max_val clip to +1
    over = t({"strength": np.array([1000.01, 2e4, 1e9])})["strength"]
    assert np.all(over == 1.0), over

    # monotonic on [0, max_val]
    xs = np.linspace(0, MAX_VAL, 20001)
    ys = t({"strength": xs.copy()})["strength"]
    assert np.all(np.diff(ys) > 0), "asinh transform not strictly monotonic"
    assert ys.min() >= -1.0 and ys.max() <= 1.0

    # matches the closed form at random points (scalar math.asinh reference)
    rng = np.random.RandomState(SEED)
    xr = rng.uniform(0, 1200, size=257)
    yr = t({"strength": xr.copy()})["strength"]
    denom = math.asinh(MAX_VAL / ASINH_SCALE)
    for xi, yi in zip(xr, yr):
        want = 2 * math.asinh(min(max(xi, 0.0), MAX_VAL) / ASINH_SCALE) / denom - 1
        assert abs(yi - want) < 1e-12, (xi, yi, want)

    # legacy default construction is untouched by the new args
    legacy = LogTransform(min_val=MIN_VAL, max_val=MAX_VAL, log=True,
                          keys=("strength",))
    xs32 = (rng.uniform(0, 1000, size=(512, 3)) + 0.01).astype(np.float32)
    assert np.array_equal(
        legacy({"strength": xs32.copy()})["strength"],
        ref_log_transform(xs32, MIN_VAL, MAX_VAL))
    print("PASS  2. asinh mode: endpoints, clip, monotonicity, closed form")


# =============================================================================
# 3. Exact-jitter round trip (handoff §3.3)
# =============================================================================
def test_3_exact_jitter_round_trip():
    from pointcept.datasets.transform import MultiplicativeRandomJitter

    rng = np.random.RandomState(SEED)
    # underlying strengths incl. clip boundaries: 0, tiny, mid, exactly max,
    # and values whose jittered product exceeds max_val (re-clip path)
    x = np.concatenate([
        np.array([0.0, 1e-6, 0.01, 0.02]),
        rng.uniform(0, MAX_VAL, size=4096),
        np.array([999.0, 999.99, MAX_VAL, MAX_VAL, MAX_VAL]),
    ])

    for space, fwd, inv, sigma in (
        ("scaled_log",
         lambda v: ref_log_transform(v, MIN_VAL, MAX_VAL),
         lambda v: ref_scaled_log_inverse(v, MIN_VAL, MAX_VAL), 0.125),
        ("asinh",
         lambda v: ref_asinh_transform(v, ASINH_SCALE, MAX_VAL),
         lambda v: ref_asinh_inverse(v, ASINH_SCALE, MAX_VAL), 0.125),
        ("scaled_log",
         lambda v: ref_log_transform(v, MIN_VAL, MAX_VAL),
         lambda v: ref_scaled_log_inverse(v, MIN_VAL, MAX_VAL), 0.05),
        ("asinh",
         lambda v: ref_asinh_transform(v, ASINH_SCALE, MAX_VAL),
         lambda v: ref_asinh_inverse(v, ASINH_SCALE, MAX_VAL), 0.05),
    ):
        # inverse really inverts the forward map (float64)
        y = fwd(x)
        x_back = inv(y)
        assert np.allclose(x_back, np.clip(x, 0, MAX_VAL), atol=1e-8), \
            f"{space}: T_inv(T(x)) != x"

        t = MultiplicativeRandomJitter(
            sigma=sigma, clip=sigma, keys=("strength",), p=1.0,
            value_space=space, min_val=MIN_VAL, max_val=MAX_VAL,
            asinh_scale=ASINH_SCALE)
        for trial in range(8):
            seed_all(SEED + trial)
            got = t({"strength": y.copy()})["strength"]
            # replicate the draw, then apply the closed-form composition
            seed_all(SEED + trial)
            assert not (random.random() > 1.0)  # consume the p draw
            n = np.clip(np.random.randn(*y.shape) * sigma, -sigma, sigma)
            want = fwd(np.clip(inv(y) * (1.0 + n), 0, MAX_VAL))
            assert np.max(np.abs(got - want)) <= 1e-6, \
                f"{space} sigma={sigma}: round trip off by " \
                f"{np.max(np.abs(got - want)):.2e}"
            assert got.min() >= -1.0 - 1e-12 and got.max() <= 1.0 + 1e-12, \
                f"{space}: exact jitter escaped [-1, 1]"
    print("PASS  3. value_space jitter == T(clip(T_inv(y)*(1+n))) to <=1e-6")


# =============================================================================
# 4. Codify the legacy amplification (handoff §3.4)
# =============================================================================
def test_4_legacy_amplification():
    from pointcept.datasets.transform import MultiplicativeRandomJitter

    a = 2 / (np.log10(MAX_VAL + MIN_VAL) - np.log10(MIN_VAL))  # ~0.4000
    rng = np.random.RandomState(SEED)
    pix = rng.uniform(0, 1000, size=8192)
    x = np.clip(pix, 0, MAX_VAL) + 0.01           # loader output
    y = ref_log_transform(x, MIN_VAL, MAX_VAL)    # value entering the jitter

    t = MultiplicativeRandomJitter(sigma=0.05, clip=0.05, keys=("strength",),
                                   p=1.0, log_space=True)
    seed_all(SEED)
    y_j = t({"strength": y.copy()})["strength"]
    seed_all(SEED)
    assert not (random.random() > 1.0)
    n = np.clip(np.random.randn(*y.shape) * 0.05, -0.05, 0.05)

    # invert the jittered value back to the underlying (x + min_val) and
    # assert the effective multiplier is (1+n)^(1/a), NOT (1+n)
    u = ref_scaled_log_inverse(y_j, MIN_VAL, MAX_VAL) + MIN_VAL   # x' + min_val
    u0 = x + MIN_VAL                                              # = pixval_c + 0.02
    assert np.allclose(u / u0, np.power(1.0 + n, 1.0 / a), rtol=1e-9), \
        "legacy log_space jitter is no longer the (1+n)^(1/a) amplification"

    # the headline number: nominal +/-5% is effectively +/-13%
    amp = 1.05 ** (1.0 / a)
    assert abs(amp - 1.1297) < 1e-3, amp
    print(f"PASS  4. legacy log_space jitter == x*(1+n)^(1/a), a={a:.6f} "
          f"(nominal 5% -> {100 * (amp - 1):.2f}%)")


# =============================================================================
# 5. Pipeline-level default preservation + asinh in-range (handoff §3.5)
# =============================================================================
def _build_item(config_path):
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(config_path)
    cfg.data.train["data_list_file"] = DIAG1K_LIST
    seed_all(SEED)
    ds = build_dataset(cfg.data.train)
    seed_all(SEED)
    return cfg, ds[0]


def _item_to_arrays(item):
    import torch
    out = {}
    for k, v in sorted(item.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = np.array(str(v))
    return out


def make_golden():
    assert os.path.exists(B1_CONFIG), B1_CONFIG
    _, item = _build_item(B1_CONFIG)
    arrays = _item_to_arrays(item)
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    import subprocess
    rev = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    np.savez_compressed(GOLDEN_PATH, __git_rev__=np.array(rev), **arrays)
    print(f"wrote golden from {B1_CONFIG}\n  at git rev {rev}\n  -> {GOLDEN_PATH}")
    for k, v in arrays.items():
        print(f"  {k}: {v.shape} {v.dtype}")


def _strength_bounds(cfg):
    """[-1,1] plus whatever the config's own jitter can add on top."""
    jitters = [d for d in cfg.transform if d["type"] == "MultiViewGenerator"
               ][0]["global_shared_transform"]
    lo, hi = -1.0, 1.0
    for j in jitters:
        if j["type"] == "MultiplicativeRandomJitter" and j.get("log_space") \
                and not j.get("value_space"):
            lo += np.log10(1.0 - j["clip"])
            hi += np.log10(1.0 + j["clip"])
    return lo - 1e-6, hi + 1e-6


def test_5_pipeline_default_preservation():
    assert os.path.exists(GOLDEN_PATH), (
        f"golden missing: {GOLDEN_PATH}\nGenerate it from the PRE-CHANGE tree "
        f"(see module docstring): python3 {__file__} --make-golden")
    golden = dict(np.load(GOLDEN_PATH, allow_pickle=False))
    golden_rev = str(golden.pop("__git_rev__"))

    cfg1, item1 = _build_item(B1_CONFIG)
    arrays1 = _item_to_arrays(item1)
    assert sorted(arrays1.keys()) == sorted(golden.keys()), (
        sorted(arrays1.keys()), sorted(golden.keys()))
    for k in golden:
        assert arrays1[k].dtype == golden[k].dtype, (k, arrays1[k].dtype, golden[k].dtype)
        assert np.array_equal(arrays1[k], golden[k]), (
            f"P05B.1 pipeline output '{k}' differs from pre-change golden "
            f"({golden_rev}) — default preservation broken")

    # strength channels in range for B.1 (bounds account for the legacy
    # log-space jitter overshooting the nominal [-1,1] by log10(1±clip))
    lo, hi = _strength_bounds(cfg1)
    s1 = arrays1["global_feat"][:, 3:]
    assert s1.min() >= lo and s1.max() <= hi, (s1.min(), s1.max(), lo, hi)

    results = {}
    for name, path in (("P05B.5", B5_CONFIG), ("P05B.6", B6_CONFIG)):
        cfg, item = _build_item(path)
        arrays = _item_to_arrays(item)
        s = arrays["global_feat"][:, 3:]
        # exact value-space jitter must stay strictly inside [-1, 1]
        assert s.min() >= -1.0 - 1e-6 and s.max() <= 1.0 + 1e-6, \
            (name, s.min(), s.max())
        assert arrays["global_feat"].shape[1] == 6
        # same seed + same RNG consumption => identical geometry, different strength
        assert np.array_equal(arrays["global_coord"], arrays1["global_coord"]), \
            f"{name}: coordinate path changed — must differ from B.1 only in strength"
        assert not np.array_equal(s, s1), f"{name}: strength identical to B.1?"
        results[name] = s

    # NOTE: no B.5-vs-B.6 array comparison here — with the same seed the
    # shared p=0.8 jitter gate fires (or skips) identically for both, and
    # when it skips their outputs are legitimately equal. The sigma
    # difference is covered exactly by tests 3 and 6.
    print(f"PASS  5. P05B.1 bit-identical to golden ({golden_rev[:9]}); "
          f"B.5/B.6 strengths in [-1,1], geometry unchanged")


# =============================================================================
# 6. Probe-side consistency (handoff §3.6)
# =============================================================================
def _find_strength_transform(transforms):
    hits = [d for d in transforms if d["type"] == "LogTransform"]
    assert len(hits) == 1, hits
    return hits[0]


def test_6_probe_matches_ssl():
    from pointcept.utils.config import Config

    ssl = Config.fromfile(B5_CONFIG)
    probe = Config.fromfile(PROBE_ASINH_CONFIG)
    ref = _find_strength_transform(ssl.transform)
    assert ref.get("mode") == "asinh", ref
    for tlist, where in (
        (ssl.val_transform, "ssl val"),
        (probe.train_transform, "probe train"),
        (probe.val_transform, "probe val"),
    ):
        got = _find_strength_transform(tlist)
        assert got == ref, f"strength transform mismatch in {where}:\n{got}\nvs\n{ref}"
    # B.6 shares B.5's transform exactly; only the jitter sigma differs
    b6 = Config.fromfile(B6_CONFIG)
    assert _find_strength_transform(b6.transform) == ref

    def jitter_of(cfg):
        mvg = [d for d in cfg.transform if d["type"] == "MultiViewGenerator"][0]
        hits = [d for d in mvg["global_shared_transform"]
                if d["type"] == "MultiplicativeRandomJitter"]
        assert len(hits) == 1, hits
        return hits[0]

    j5, j6 = jitter_of(ssl), jitter_of(b6)
    for j, sigma in ((j5, 0.125), (j6, 0.05)):
        assert j["value_space"] == "asinh" and j["asinh_scale"] == ASINH_SCALE \
            and j["max_val"] == MAX_VAL and j["sigma"] == sigma \
            and j["clip"] == sigma and j["p"] == 0.8, j
    # the probe must NOT jitter strengths
    for tlist in (probe.train_transform, probe.val_transform):
        assert not any(d["type"] == "MultiplicativeRandomJitter" for d in tlist)
    assert probe.model["freeze_backbone"] is True
    print("PASS  6. asinh probe/B.6 transforms match P05B.5; jitter sigmas "
          "0.125/0.05 as decided")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-golden", action="store_true",
                    help="write the P05B.1 pipeline golden from the CURRENT "
                         "tree (run this at the pre-change merge-base only)")
    args = ap.parse_args()
    if args.make_golden:
        make_golden()
        return
    test_1_default_preservation()
    test_2_asinh_correctness()
    test_3_exact_jitter_round_trip()
    test_4_legacy_amplification()
    test_5_pipeline_default_preservation()
    test_6_probe_matches_ssl()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
