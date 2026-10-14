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
  stage3    tools/larformer/run_larformer_stage3_inference.py::flash_recovery_keep
            inputs: the same events; a slicer-like prediction with one query per slice
            (mask logits +-5, every 4th query no-object); K in (1, 3, 50) and
            chi2_max in (500, 1e12). Compared: the returned keep masks.
  analyze   lartpc/larformer_analysis/slicer_eval/analyze_event.py::_compute_pred_slices and
            _compute_gt_baseline. inputs: an `inf` dict built from the events (reco points,
            pixval as raw pixels, tick/wires for share normalization, one query per slice,
            every 4th query no-object, mock p_real); variants: producer 0/1, share_norm
            none/per-slice, v_drift 0.1098/0.112, flash_preal_min None/0.5. Compared: all outputs.
  gammacal  lartpc/larformer_analysis/flashmodel_calib/gammacal/predict.py::predict_pe_ref
            on every slice of the events, flash t0 from the file.
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


def run_stage3(legacy_path, files):
    import torch
    new = load_module(os.path.join(REPO, "tools/larformer/run_larformer_stage3_inference.py"), "stage3_new")
    old = load_module(legacy_path, "stage3_legacy")
    n_ok = n = 0
    for path in files:
        coord, sid, run, kind = event_inputs(path, -5)
        qs = np.unique(sid)
        Q = len(qs)
        logits = torch.full((Q, coord.shape[0]), -5.0)
        for i, q in enumerate(qs):
            logits[i, torch.from_numpy(sid == q)] = 5.0
        C = 4
        cls = torch.zeros(Q, C)
        for i in range(Q):
            cls[i, (C - 1) if i % 4 == 3 else (i % (C - 1))] = 1.0
        ev = {"mask_logits": {"spacepoint": logits}, "class_logits": cls}
        batch = {"coord": torch.from_numpy(coord)}
        for K in (1, 3, 50):
            for chi2_max in (500.0, 1e12):
                kw = dict(no_object_class_id=C - 1, mask_prob_threshold=0.5, gamma_beam=5.25,
                          K=K, chi2_max=chi2_max, oob_max=0.05, gamma_scale_spec="auto")
                a = old.flash_recovery_keep(ev, batch, path, **kw)
                b = new.flash_recovery_keep(ev, batch, path, detector="uboone", **kw)
                ok = (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                n += 1
                n_ok += ok
                kept = -1 if b is None else int(b.sum())
                print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(path)} run {run} K={K} "
                      f"chi2_max={chi2_max:g}: {Q} queries, kept {kept} points")
    print(f"{n_ok}/{n} stage-3 keep masks identical")
    return n_ok == n and n > 0


def _event_arrays(path, nu_sid):
    import h5py
    coord, sid, run, kind = event_inputs(path, nu_sid)
    with h5py.File(path, "r") as f:
        e = f["entry_0"]
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        # map the selected coords back to rows (event_inputs selects rows; recompute the same way)
        if "hasmatch" in td and (td["hasmatch"][()] == 1).sum() >= 50:
            sel = np.nonzero(td["hasmatch"][()] == 1)[0]
        else:
            sel = np.nonzero(td["pixval"][()][:, 2] > 10)[0][:20000]
        cols = {k: td[k][()][sel] for k in ("pixval", "tick", "uwire", "ywire")}
        fl = e["flashes"]
        beam = np.nonzero(fl["producer_id"][()] == 0)[0]
        if beam.size == 0:
            return None
        bi = beam[np.argmax(fl["total_pe"][()][beam])]
        obs = dict(pe_obs=fl["pe"][()][bi].astype(np.float64), t0_us=float(fl["time_us"][()][bi]))
    assert np.array_equal(pos[sel], coord)
    return coord, sid, run, kind, cols, obs


def run_analyze(legacy_path, files):
    new = load_module(os.path.join(REPO, "lartpc/larformer_analysis/slicer_eval/analyze_event.py"), "ae_new")
    old = load_module(legacy_path, "ae_legacy")
    n_ok = n = 0
    for path in files:
        ev = _event_arrays(path, -5)
        if ev is None:
            continue
        coord, sid, run, kind, cols, obs = ev
        qs, qid = np.unique(sid, return_inverse=True)
        Q = len(qs)
        no_obj = 3
        q_cls = np.array([no_obj if i % 4 == 3 else i % 3 for i in range(Q)], np.int32)
        inf = dict(no_object_class_id=no_obj, post_pred_query=qid.astype(np.int64),
                   post_pred_class=q_cls[qid], q_class_argmax=q_cls, post_coord=coord,
                   post_pixval=cols["pixval"].astype(np.float32), post_rawpix=cols["pixval"].astype(np.float32),
                   post_tick=cols["tick"], post_uwire=cols["uwire"], post_ywire=cols["ywire"],
                   post_p_real=np.random.default_rng(1).uniform(size=len(coord)).astype(np.float32))
        gt_nu = (sid == -5) if (sid == -5).any() else (qid == 0)
        thr = np.array([0.0, 0.05, 0.1, 0.2, 0.5])
        for pid in (0, 1):
            for share in ("none", "per-slice"):
                for vd in (0.1098, 0.112):
                    for pmin in (None, 0.5):
                        it = dict(obs, producer_id=pid)
                        common = (inf, it, gt_nu, (5.25 * 0.86, 5.25 * 0.55), None, None, thr, 0.1, 1.0, vd,
                                  "raw_adc", share)
                        a = old._compute_pred_slices(*common, flash_preal_min=pmin)
                        b = new._compute_pred_slices(*common, flash_preal_min=pmin, detector="uboone")
                        ga = old._compute_gt_baseline(*common)
                        gb = new._compute_gt_baseline(*common, detector="uboone")
                        ok1, w1 = same(a, b)
                        ok2, w2 = same(ga, gb)
                        ok = ok1 and ok2
                        n += 1
                        n_ok += ok
                        if not ok or (pid == 0 and share == "none" and vd == 0.1098 and pmin is None):
                            print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(path)} run {run} pid={pid} "
                                  f"share={share} v={vd} preal_min={pmin}: {len(b['query_id'])} slices"
                                  f"{'' if ok else ' -- ' + (w1 or w2)}")
    print(f"{n_ok}/{n} analyze_event outputs identical")
    return n_ok == n and n > 0


def run_gammacal(legacy_path, files):
    sys.path.insert(0, os.path.join(REPO, "lartpc/larformer_analysis/flashmodel_calib"))
    import importlib
    gnew = importlib.import_module("gammacal.predict")
    spec = importlib.util.spec_from_file_location("gammacal.predict_legacy", legacy_path,
                                                  submodule_search_locations=None)
    gold = importlib.util.module_from_spec(spec)
    gold.__package__ = "gammacal"
    spec.loader.exec_module(gold)
    n_ok = n = 0
    for path in files:
        ev = _event_arrays(path, -5)
        if ev is None:
            continue
        coord, sid, run, kind, cols, obs = ev
        q = cols["pixval"][:, 2].astype(np.float32)
        for s_ in np.unique(sid):
            m = sid == s_
            a = gold.predict_pe_ref(coord[m], q[m], obs["t0_us"])
            b = gnew.predict_pe_ref(coord[m], q[m], obs["t0_us"])
            n += 1
            n_ok += np.array_equal(a, b)
    a = gold.predict_pe_ref(coord[:0], q[:0], 0.0); b = gnew.predict_pe_ref(coord[:0], q[:0], 0.0)
    n += 1; n_ok += np.array_equal(a, b)
    print(f"{n_ok}/{n} gammacal predict_pe_ref vectors identical")
    return n_ok == n and n > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", choices=["cascade", "stage3", "analyze", "gammacal"])
    ap.add_argument("--legacy", required=True, help="path of the pre-edit copy of the call-site file")
    ap.add_argument("--files", type=int, default=5, help="events per sample glob")
    a = ap.parse_args()
    files = [f for g in GLOBS for f in sorted(glob.glob(g))[: a.files]]
    ok = dict(cascade=run_cascade, stage3=run_stage3, analyze=run_analyze,
              gammacal=run_gammacal)[a.site](a.legacy, files)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
