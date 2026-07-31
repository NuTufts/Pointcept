"""
Event-level bootstrap CIs and same-domain null calibration (the P0.2
utility). Events are the resampling unit -- points within an event are
strongly correlated, so point-level resampling would understate the CI.

Usage pattern (see run_domain_study.py):

    from domain_metrics import pad
    from bootstrap import bootstrap_ci, null_split
    res = bootstrap_ci(pad, X_mc, X_data, n_boot=200, seed=0)
    # res["pad_linear"] = {"value":..., "lo":..., "hi":..., "boot_std":...}
    null = null_split(pad, X_mc, seed=0)   # MC-vs-MC half split baseline
"""
import numpy as np


def _resample(X, rng):
    return X[rng.integers(0, len(X), size=len(X))]


def bootstrap_ci(metric_fn, X_a, X_b, n_boot=200, ci=0.95, seed=0, **kw):
    """Percentile bootstrap CI for every scalar the metric returns.

    metric_fn(X_a, X_b, **kw) -> dict of floats. Each side is resampled
    with replacement independently (stratified two-sample bootstrap).
    Non-finite or non-scalar entries pass through with value only.
    """
    point = metric_fn(X_a, X_b, **kw)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k, v in point.items() if np.isscalar(v)}
    for _ in range(n_boot):
        r = metric_fn(_resample(X_a, rng), _resample(X_b, rng), **kw)
        for k in boots:
            boots[k].append(r[k])
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for k, v in point.items():
        if k in boots and len(boots[k]):
            b = np.asarray(boots[k], dtype=float)
            out[k] = {
                "value": float(v),
                "lo": float(np.quantile(b, lo_q)),
                "hi": float(np.quantile(b, hi_q)),
                "boot_std": float(b.std()),
            }
        else:
            out[k] = {"value": v}
    return out


def null_split(metric_fn, X, n_splits=10, seed=0, **kw):
    """Same-domain null: repeatedly split one sample in half and apply the
    two-sample metric. The spread calibrates what 'indistinguishable' looks
    like at this sample size. Returns {metric: {mean, std, values}}."""
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n_splits):
        perm = rng.permutation(len(X))
        h = len(X) // 2
        r = metric_fn(X[perm[:h]], X[perm[h:2 * h]], **kw)
        for k, v in r.items():
            if np.isscalar(v):
                acc.setdefault(k, []).append(float(v))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "values": v}
            for k, v in acc.items()}
