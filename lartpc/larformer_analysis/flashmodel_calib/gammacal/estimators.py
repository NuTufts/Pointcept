"""Statistics on per-event ratios r = sum_live obs / sum_live pred_ref."""
import numpy as np


def ratio_stats(r, nboot=2000, seed=7, core_halfwidth_dex=0.15):
    r = np.asarray(r, np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    out = dict(N=int(len(r)))
    if len(r) == 0:
        out.update(median=np.nan, err_boot=np.nan, p16=np.nan, p84=np.nan,
                   peak=np.nan, core_frac=np.nan, core_median=np.nan,
                   mean_log=np.nan, gate_ok=False)
        return out
    med = float(np.median(r))
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(rng.choice(r, len(r), replace=True))
                     for _ in range(nboot)]) if len(r) > 1 else np.array([med])
    lo, hi = np.percentile(r, [16, 84])
    lr = np.log10(r)
    edges = np.arange(-1.5, 1.5001, 0.05)
    h, _ = np.histogram(np.clip(lr, -1.499, 1.499), edges)
    # 3-bin running mean to find the peak robustly
    hs = np.convolve(h, np.ones(3) / 3.0, mode="same")
    pk = 0.5 * (edges[np.argmax(hs)] + edges[np.argmax(hs) + 1])
    core = float(np.mean(np.abs(lr - pk) < core_halfwidth_dex))
    peak = float(10 ** pk)
    # trimmed core median: iterate median -> keep |log r - log m| < halfwidth
    # -> median of the kept. Stable at small N where the histogram peak is not;
    # the gate compares it to the plain median (background pulls them apart).
    m = np.log10(med)
    for _ in range(3):
        keep = np.abs(lr - m) < core_halfwidth_dex
        if keep.sum() < 3:
            break
        m = float(np.median(lr[keep]))
    core_med = float(10 ** m)
    core_frac_med = float(np.mean(np.abs(lr - m) < core_halfwidth_dex))
    out.update(median=med, err_boot=float(boot.std()), p16=float(lo),
               p84=float(hi), peak=peak, core_frac=core_frac_med,
               core_median=core_med, mean_log=float(10 ** lr.mean()),
               gate_ok=bool(core_frac_med >= 0.5
                            and abs(core_med - med) / med < 0.05))
    return out


def neyman_gamma(obs, pred, live, f_sys=0.10, eps=1.0):
    """Per-event closed-form multiplier g on pred minimising the production
    Neyman chi2 sum_i (obs_i - g pred_i)^2 / var_i over live PMTs."""
    o = np.asarray(obs, np.float64)[live]; p = np.asarray(pred, np.float64)[live]
    var = o + (f_sys * o) ** 2 + eps
    d = np.sum(p * p / var)
    return float(np.sum(o * p / var) / d) if d > 0 else np.nan


def pooled_gamma(obs_rows, pred_rows, live_rows, f_sys=0.10, eps=1.0):
    num = den = 0.0
    for o, p, l in zip(obs_rows, pred_rows, live_rows):
        o = np.asarray(o, np.float64)[l]; p = np.asarray(p, np.float64)[l]
        var = o + (f_sys * o) ** 2 + eps
        num += np.sum(o * p / var); den += np.sum(p * p / var)
    return float(num / den) if den > 0 else np.nan


def binned_medians(x, r, edges):
    """[(lo, hi, N, median)] of r in bins of x."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(r)
        rows.append((float(lo), float(hi), int(m.sum()),
                     float(np.median(r[m])) if m.any() else np.nan))
    return rows


def run_groups(runs, r, ngroups=8):
    """Median r in run-number quantile groups (stability within a period)."""
    runs = np.asarray(runs); r = np.asarray(r)
    if len(runs) == 0:
        return []
    qs = np.unique(np.percentile(runs, np.linspace(0, 100, ngroups + 1)))
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (runs >= lo) & (runs <= hi)
        rows.append((int(lo), int(hi), int(m.sum()),
                     float(np.median(r[m])) if m.any() else np.nan))
    return rows
