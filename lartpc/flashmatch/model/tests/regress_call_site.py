#!/usr/bin/env python3
"""
Call-site regression for P4: run a LEGACY copy and the CURRENT version of a flash
call site on the same real MicroBooNE inputs and require identical outputs.

    # 1. before editing a call site, snapshot it:
    cp tools/larformer/run_larformer_keypoint2_cascade_inference.py /tmp/legacy_cascade.py
    # 2. after editing:
    python lartpc/flashmatch/model/tests/regress_call_site.py cascade --legacy /tmp/legacy_cascade.py

Call sites:
  cascade   tools/larformer/run_larformer_keypoint2_cascade_inference.py::_flash_slice_table
            inputs: merged_sp events (run-1 overlay: slices from truth = nu union +
            per-track cosmics; run-3 EXT: slices = z bins), the file's observed flashes and
            the cascade's own de-double-counted charge context; option sets incl.
            --flash-window auto and --mask-saturated off.
Compared: every table entry (pred_pe, chi2, oob_frac, chi2_rank, best_query, ...)
with exact equality (NaN == NaN), and all params except the new 'detector' key.
"""
import argparse
import glob
import importlib.util
import os
import sys
import types

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO)

GLOBS = (
    "/cluster/tufts/wongjiradlab/larbys/data/larformer/run1_bnboverlay_half/merged_sp/000/*.h5",
    "/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9_v29e_dl_run3_G1_extbnb/merged_sp/000/00/*.h5",
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, f"keys {sorted(set(a) ^ set(b))}"
        for k in a:
            ok, why = same(a[k], b[k])
            if not ok:
                return False, f"{k}: {why}"
        return True, ""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False, f"shape {a.shape} vs {b.shape}"
        if a.dtype.kind in "fc":
            ok = np.array_equal(a, b, equal_nan=True)
        else:
            ok = np.array_equal(a, b)
        return ok, "" if ok else f"max|diff| {np.nanmax(np.abs(a.astype(float) - b.astype(float))):.3g}"
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, "length"
        for x, y in zip(a, b):
            ok, why = same(x, y)
            if not ok:
                return False, why
        return True, ""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True, ""
    return (a == b), "" if a == b else f"{a!r} vs {b!r}"


def event_inputs(path, nu_sid):
    import h5py
    with h5py.File(path, "r") as f:
        e = f["entry_0"]
        td = e["triplet_data"]
        run = int(e.attrs.get("run", 0))
        pos = td["pos"][()]
        if "hasmatch" in td and (td["hasmatch"][()] == 1).sum() >= 50:
            sel = np.nonzero(td["hasmatch"][()] == 1)[0]
            org = td["origin"][()][sel]
            tid = td["trackid"][()][sel]
            sid = np.full(sel.size, -1, np.int64)
            sid[org == 1] = nu_sid
            cos = org != 1
            _, q = np.unique(tid[cos], return_inverse=True)
            sid[cos] = q
            kind = "mc"
        else:
            sel = np.nonzero(td["pixval"][()][:, 2] > 10)[0][:20000]
            sid = np.clip((pos[sel, 2] // 100).astype(np.int64), 0, 10)
            kind = "data"
    return pos[sel].astype(np.float32), sid, run, kind


def run_cascade(legacy_path, files):
    new = load_module(os.path.join(REPO, "tools/larformer/run_larformer_keypoint2_cascade_inference.py"),
                      "cascade_new")
    old = load_module(legacy_path, "cascade_legacy")
    base = dict(gamma_beam=5.25, dead_opdets="auto", gamma_run_scale="auto", run_override=None,
                allow_kind_mismatch=False, flash_window="off", mask_saturated="auto",
                max_saturated_pmts=4, slice_min_points=20, flash_f_sys=0.10, flash_eps=1.0,
                flash_oob_max=0.05, photonlib=None, detector="uboone", light_yield=None)
    variants = [dict(), dict(flash_window="auto"), dict(mask_saturated="off", flash_f_sys=0.15)]
    n_ok = n = 0
    for path in files:
        flashes, ctx = new._load_msp_flash_charge(path)
        if flashes is None:
            continue
        coord, sid, run, kind = event_inputs(path, new._NU_SID)
        qs = [int(q) for q in np.unique(sid[sid >= 0])]
        p_nu = {q: float(np.random.default_rng(q).uniform()) for q in qs}
        nu_qs = qs[:2]
        for v in variants:
            args = types.SimpleNamespace(**{**base, **v})
            t_new = new._flash_slice_table(coord, sid, p_nu, nu_qs, flashes, ctx, args, run=run, kind=kind)
            t_old = old._flash_slice_table(coord, sid, p_nu, nu_qs, flashes, ctx, args, run=run, kind=kind)
            pn = dict(t_new["params"])
            det = pn.pop("detector", None)
            t_new = {**t_new, "params": pn}
            ok, why = same(t_old, t_new)
            ok = ok and det == "uboone"
            n += 1
            n_ok += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(path)} run {run} ({kind}) {v or 'defaults'}: "
                  f"{len(t_new['label'])} slices, best {t_new['best_query']}, "
                  f"masked {t_new['params'].get('chi2_masked_opdets', '')!r}{'' if ok else ' -- ' + why}")
    print(f"{n_ok}/{n} cascade tables identical")
    return n_ok == n and n > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", choices=["cascade"])
    ap.add_argument("--legacy", required=True, help="path of the pre-edit copy of the call-site file")
    ap.add_argument("--files", type=int, default=5, help="events per sample glob")
    a = ap.parse_args()
    files = [f for g in GLOBS for f in sorted(glob.glob(g))[: a.files]]
    ok = run_cascade(a.legacy, files)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
