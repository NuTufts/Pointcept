"""In-time flash choice.

The cascade takes the brightest producer-0 (simpleFlashBeam) flash with NO time
cut, so ~10% of its choices lie outside any beam window. Here the record keeps
every producer-0 flash inside a BROAD window (default 2-7 us) plus the brightest
one, and the estimator applies the sample's real window with an edge margin at
fit time (EXT emulates the beam trigger, so a flash at the window edge can be
partly truncated).
"""
import numpy as np

MAX_FLASHES = 8


def read_flashes(kp):
    """flash/all from an open kp2 h5 -> dict or None."""
    if "flash" not in kp or "all" not in kp["flash"]:
        return None
    fa = kp["flash/all"]
    return {"pe": fa["pe"][()].astype(np.float32),
            "producer_id": fa["producer_id"][()].astype(np.int32),
            "time_us": fa["time_us"][()].astype(np.float32),
            "total_pe": fa["total_pe"][()].astype(np.float32)}


def choose_flash(fl, broad=(2.0, 7.0), min_pe=20.0):
    """Brightest producer-0 flash inside `broad`. Returns (index or -1,
    times[MAX_FLASHES], pes[MAX_FLASHES]) where the arrays list every producer-0
    flash in the broad window (NaN-padded, sorted by time)."""
    times = np.full(MAX_FLASHES, np.nan, np.float32)
    pes = np.full(MAX_FLASHES, np.nan, np.float32)
    if fl is None or fl["pe"].size == 0:
        return -1, times, pes
    sel = np.nonzero((fl["producer_id"] == 0)
                     & (fl["time_us"] >= broad[0]) & (fl["time_us"] <= broad[1]))[0]
    if sel.size == 0:
        return -1, times, pes
    order = sel[np.argsort(fl["time_us"][sel])][:MAX_FLASHES]
    times[:len(order)] = fl["time_us"][order]
    pes[:len(order)] = fl["total_pe"][order]
    best = int(sel[int(np.argmax(fl["total_pe"][sel]))])
    if fl["total_pe"][best] < min_pe:
        return -1, times, pes
    return best, times, pes


def in_window_ok(t, times, pes, window, margin=0.2, min_pe=20.0):
    """Fit-time test: chosen flash time inside window minus margin, and it is
    the ONLY producer-0 flash above min_pe inside the window."""
    lo, hi = window[0] + margin, window[1] - margin
    if not (lo <= t <= hi):
        return False
    m = np.isfinite(times) & (times >= window[0]) & (times <= window[1]) & (pes >= min_pe)
    return int(m.sum()) == 1
