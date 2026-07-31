"""
Two-sample and representation-comparison metrics for the domain-shift study.

Pure numpy/sklearn; every function takes plain arrays and returns a dict of
floats so bootstrap.py can resample any of them uniformly. See
domain_shift_study_plan.md section 3 for definitions and pitfalls.

Conventions:
  X_a, X_b : [n_a, d], [n_b, d] event-level embeddings (rows = events).
             Callers should pass RAW pooled features; preprocessing
             (standardize + optional PCA) happens inside via `preprocess`
             fitted on the union, so bootstrap resamples see the same
             treatment.
  hist_a, hist_b : [n_a, P], [n_b, P] per-event prototype count histograms.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
def preprocess(X_a, X_b, n_pca=None):
    """Standardize (union-fitted) and optionally PCA-reduce both samples."""
    X = np.concatenate([X_a, X_b])
    scaler = StandardScaler().fit(X)
    X_a, X_b = scaler.transform(X_a), scaler.transform(X_b)
    if n_pca is not None and n_pca < X_a.shape[1]:
        p = PCA(n_components=n_pca, random_state=0).fit(
            np.concatenate([X_a, X_b]))
        X_a, X_b = p.transform(X_a), p.transform(X_b)
    return X_a, X_b


# --------------------------------------------------------------------------
def pad(X_a, X_b, n_splits=5, n_pca=None, seed=0):
    """Domain-classifier two-sample test (C2ST) + proxy A-distance.

    Returns AUC and accuracy for a linear (logistic regression) and a kNN
    classifier under stratified k-fold CV, and PAD = 2*(2*acc - 1).
    auc ~ 0.5 / pad ~ 0 means the domains are indistinguishable to that
    classifier family. linear ~= knn suggests a linearly-encoded gap.
    """
    X_a, X_b = preprocess(X_a, X_b, n_pca)
    X = np.concatenate([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    out = {}
    for name, clf in (
        ("linear", LogisticRegression(max_iter=2000, C=1.0)),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ):
        aucs, accs = [], []
        for tr, te in StratifiedKFold(
                n_splits=n_splits, shuffle=True,
                random_state=seed).split(X, y):
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
            accs.append(np.mean((prob > 0.5) == y[te]))
        acc = float(np.mean(accs))
        out[f"auc_{name}"] = float(np.mean(aucs))
        out[f"auc_{name}_std"] = float(np.std(aucs))
        out[f"acc_{name}"] = acc
        out[f"pad_{name}"] = 2.0 * (2.0 * acc - 1.0)
    return out


# --------------------------------------------------------------------------
def _sq_dists(X, Y):
    return (
        np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2.0 * X @ Y.T
    ).clip(min=0.0)


def _mmd2_unbiased(K_aa, K_bb, K_ab):
    n, m = K_aa.shape[0], K_bb.shape[0]
    np.fill_diagonal(K_aa, 0.0)
    np.fill_diagonal(K_bb, 0.0)
    return (K_aa.sum() / (n * (n - 1)) + K_bb.sum() / (m * (m - 1))
            - 2.0 * K_ab.mean())

def mmd2(X_a, X_b, n_pca=64, bandwidth_scales=(0.25, 1.0, 4.0),
         n_perm=1000, seed=0):
    """Unbiased multi-kernel (summed RBF) MMD^2 with permutation p-value.

    Bandwidths = median pairwise distance on the union, scaled by
    `bandwidth_scales`. Run on a PCA-reduced space by default (power decays
    in very high d; pass n_pca=None for the full space).
    """
    X_a, X_b = preprocess(X_a, X_b, n_pca)
    Z = np.concatenate([X_a, X_b])
    n = len(X_a)
    D = _sq_dists(Z, Z)
    med = np.median(D[np.triu_indices_from(D, k=1)])
    med = med if med > 0 else 1.0
    K = np.zeros_like(D)
    for s in bandwidth_scales:
        K += np.exp(-D / (2.0 * med * s))

    def stat(perm):
        a, b = perm[:n], perm[n:]
        return _mmd2_unbiased(K[np.ix_(a, a)].copy(),
                              K[np.ix_(b, b)].copy(),
                              K[np.ix_(a, b)])

    obs = stat(np.arange(len(Z)))
    rng = np.random.default_rng(seed)
    perm_stats = np.array(
        [stat(rng.permutation(len(Z))) for _ in range(n_perm)])
    p = float((np.sum(perm_stats >= obs) + 1) / (n_perm + 1))
    return {"mmd2": float(obs), "mmd2_p": p,
            "mmd2_null_95": float(np.quantile(perm_stats, 0.95))}


# --------------------------------------------------------------------------
def proto_jsd(hist_a, hist_b, min_count=10):
    """Prototype-occupancy divergence between domains.

    Aggregates per-event count histograms into two occupancy distributions;
    returns the Jensen-Shannon divergence (base 2, in [0,1]) and the number
    of prototypes exclusive to each domain (>= min_count total points on the
    exclusive side, 0 on the other).
    """
    ca = np.asarray(hist_a).sum(axis=0).astype(np.float64)
    cb = np.asarray(hist_b).sum(axis=0).astype(np.float64)
    pa, pb = ca / ca.sum(), cb / cb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    jsd = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    return {
        "proto_jsd": jsd,
        "proto_active_a": int(np.sum(ca > 0)),
        "proto_active_b": int(np.sum(cb > 0)),
        "proto_excl_a": int(np.sum((ca >= min_count) & (cb == 0))),
        "proto_excl_b": int(np.sum((cb >= min_count) & (ca == 0))),
    }


# --------------------------------------------------------------------------
def cka(F_a, F_b, rbf=False):
    """CKA between two representations of the SAME samples (rows aligned).

    F_a: [n, d1] from model A, F_b: [n, d2] from model B. Linear CKA by
    default (invariant to rotation/isotropic scaling); rbf=True uses an RBF
    kernel with median-heuristic bandwidth. Comparative diagnostic only --
    see plan section 3.4 for caveats.
    """
    F_a = F_a - F_a.mean(0, keepdims=True)
    F_b = F_b - F_b.mean(0, keepdims=True)
    if not rbf:
        num = np.linalg.norm(F_a.T @ F_b, "fro") ** 2
        den = (np.linalg.norm(F_a.T @ F_a, "fro")
               * np.linalg.norm(F_b.T @ F_b, "fro"))
        return {"cka_linear": float(num / den) if den > 0 else float("nan")}

    def _gram(F):
        D = _sq_dists(F, F)
        med = np.median(D[np.triu_indices_from(D, k=1)])
        return np.exp(-D / (2.0 * (med if med > 0 else 1.0)))

    def _center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Ka, Kb = _center(_gram(F_a)), _center(_gram(F_b))
    num = float(np.sum(Ka * Kb))
    den = float(np.sqrt(np.sum(Ka * Ka) * np.sum(Kb * Kb)))
    return {"cka_rbf": num / den if den > 0 else float("nan")}
