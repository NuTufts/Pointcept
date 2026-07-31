"""
Grouped-statistics metrics for PER-POINT domain-gap measurements.

Points within an event are correlated, so every procedure here treats the
EVENT as the exchangeable unit (see POINT_LEVEL_PLAN.md):
  - pad_points: classifier CV with event-grouped folds;
  - mmd2_points: MMD^2 on point subsamples with an event-block
    permutation null;
  - bootstrap_events: percentile bootstrap resampling events, not points.

Inputs are point arrays plus a parallel integer event id per point.
"""
import numpy as np


def _standardize_pca(Xa, Xb, n_pca, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([Xa, Xb]).astype(np.float32)
    sc = StandardScaler().fit(X)
    Xa, Xb = sc.transform(Xa.astype(np.float32)), \
        sc.transform(Xb.astype(np.float32))
    if n_pca is not None and n_pca < Xa.shape[1]:
        p = PCA(n_components=n_pca, random_state=seed).fit(
            np.concatenate([Xa, Xb]))
        Xa, Xb = p.transform(Xa), p.transform(Xb)
    return Xa, Xb


def _grouped_folds(y, groups, n_splits, seed):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    try:
        return list(StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True,
            random_state=seed).split(np.zeros_like(y), y, groups))
    except Exception:  # noqa: BLE001 -- older sklearn
        return list(GroupKFold(n_splits=n_splits).split(
            np.zeros_like(y), y, groups))


def subsample_per_side(X, ev, max_points, rng):
    """Random point subsample (keeps event ids aligned)."""
    if len(X) <= max_points:
        return X, ev
    idx = rng.choice(len(X), max_points, replace=False)
    return X[idx], ev[idx]


def pad_points(Xa, Xb, ev_a, ev_b, n_splits=5, n_pca=64,
               max_points=150000, classifiers=("linear", "histgb"),
               seed=0):
    """Event-grouped point-level classifier two-sample test.

    Returns AUC for a linear (logreg) and a nonlinear (HistGB) classifier,
    both on PCA-reduced standardized features, with event-grouped CV.
    `classifiers` restricts which are run (bootstrap rounds use linear
    only to stay tractable).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    Xa, ev_a = subsample_per_side(Xa, ev_a, max_points, rng)
    Xb, ev_b = subsample_per_side(Xb, ev_b, max_points, rng)
    Xa2, Xb2 = _standardize_pca(Xa, Xb, n_pca, seed)
    X = np.concatenate([Xa2, Xb2])
    y = np.concatenate([np.zeros(len(Xa2)), np.ones(len(Xb2))])
    groups = np.concatenate([ev_a, ev_b + ev_a.max() + 1])

    out = {"n_points_a": int(len(Xa2)), "n_points_b": int(len(Xb2))}
    for name, make in (
        ("linear", lambda: LogisticRegression(max_iter=1000)),
        ("histgb", lambda: HistGradientBoostingClassifier(
            max_iter=150, random_state=seed)),
    ):
        if name not in classifiers:
            continue
        aucs = []
        for tr, te in _grouped_folds(y, groups, n_splits, seed):
            clf = make()
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te],
                                      clf.predict_proba(X[te])[:, 1]))
        out[f"auc_{name}"] = float(np.mean(aucs))
        out[f"auc_{name}_std"] = float(np.std(aucs))
    return out


def _mmd2_from_subsets(Xa, Xb, rng, n_pts, bandwidths=None):
    ia = rng.choice(len(Xa), min(n_pts, len(Xa)), replace=False)
    ib = rng.choice(len(Xb), min(n_pts, len(Xb)), replace=False)
    A, B = Xa[ia], Xb[ib]
    Z = np.concatenate([A, B])
    n = len(A)
    D = (np.sum(Z**2, 1)[:, None] + np.sum(Z**2, 1)[None, :]
         - 2.0 * Z @ Z.T).clip(min=0.0)
    if bandwidths is None:
        med = np.median(D[np.triu_indices_from(D, k=1)])
        med = med if med > 0 else 1.0
        bandwidths = [0.25 * med, med, 4.0 * med]
    K = np.zeros_like(D)
    for s in bandwidths:
        K += np.exp(-D / (2.0 * s))
    K_aa, K_bb, K_ab = K[:n, :n].copy(), K[n:, n:].copy(), K[:n, n:]
    np.fill_diagonal(K_aa, 0.0)
    np.fill_diagonal(K_bb, 0.0)
    m = len(B)
    return (K_aa.sum() / (n * (n - 1)) + K_bb.sum() / (m * (m - 1))
            - 2.0 * K_ab.mean()), bandwidths


def mmd2_points(Xa, Xb, ev_a, ev_b, n_pca=64, n_pts=8000, n_perm=200,
                seed=0):
    """Point-level MMD^2 with an EVENT-BLOCK permutation null.

    Each permutation reassigns whole events to pseudo-domains (preserving
    the two domain sizes in events), then recomputes the statistic on a
    fresh point subsample. Bandwidths are frozen from the observed
    comparison so the null is computed with the same kernel.
    """
    rng = np.random.default_rng(seed)
    Xa2, Xb2 = _standardize_pca(Xa, Xb, n_pca, seed)
    obs, bw = _mmd2_from_subsets(Xa2, Xb2, rng, n_pts)

    # index points by event for block reassignment
    X = np.concatenate([Xa2, Xb2])
    ev = np.concatenate([ev_a, ev_b + ev_a.max() + 1])
    uev = np.unique(ev)
    n_ev_a = len(np.unique(ev_a))
    by_event = {e: np.flatnonzero(ev == e) for e in uev}

    perm_stats = []
    for _ in range(n_perm):
        perm = rng.permutation(uev)
        pa = np.concatenate([by_event[e] for e in perm[:n_ev_a]])
        pb = np.concatenate([by_event[e] for e in perm[n_ev_a:]])
        s, _ = _mmd2_from_subsets(X[pa], X[pb], rng, n_pts, bandwidths=bw)
        perm_stats.append(s)
    perm_stats = np.array(perm_stats)
    return {
        "mmd2": float(obs),
        "mmd2_p": float((np.sum(perm_stats >= obs) + 1) / (n_perm + 1)),
        "mmd2_null_95": float(np.quantile(perm_stats, 0.95)),
        "n_pts_per_side": int(n_pts),
    }


def bootstrap_events(metric_fn, Xa, Xb, ev_a, ev_b, n_boot=50, ci=0.95,
                     seed=0, **kw):
    """Percentile bootstrap resampling EVENTS on each side."""
    rng = np.random.default_rng(seed)
    point = metric_fn(Xa, Xb, ev_a, ev_b, seed=seed, **kw)
    scalars = {k: [] for k, v in point.items() if np.isscalar(v)}

    def _resample(X, ev):
        uev = np.unique(ev)
        take = rng.choice(uev, len(uev), replace=True)
        idx, new_ev, nid = [], [], 0
        for e in take:
            ii = np.flatnonzero(ev == e)
            idx.append(ii)
            new_ev.append(np.full(len(ii), nid))
            nid += 1
        return X[np.concatenate(idx)], np.concatenate(new_ev)

    for b in range(n_boot):
        Ra, ea = _resample(Xa, ev_a)
        Rb, eb = _resample(Xb, ev_b)
        r = metric_fn(Ra, Rb, ea, eb, seed=seed + 1 + b, **kw)
        for k in scalars:
            scalars[k].append(r[k])
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for k, v in point.items():
        if k in scalars and len(scalars[k]):
            arr = np.asarray(scalars[k], dtype=float)
            out[k] = {"value": float(v), "lo": float(np.quantile(arr, lo)),
                      "hi": float(np.quantile(arr, hi)),
                      "boot_std": float(arr.std())}
        else:
            out[k] = {"value": v}
    return out
