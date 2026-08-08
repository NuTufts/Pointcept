"""Per-TRUE-PARTICLE nu-slice completeness at the SLICER stage.

Fills the gap between the slicer_eval per-event metrics (event-level nu
recall/purity) and the full-chain eval_reco_performance metric C (charge
slice-coverage measured after stages 3/4): for each true particle, what
fraction of its spacepoints / charge ends up in a predicted nu-classed
slice, measured directly on Stage-2 output.

Conventions follow lartpc/larformer_reco/eval/eval_reco_performance.py:
  - linkage is by trackid: mc_particle_tree.trackid == triplet_data.trackid
    (the SP truth labels already backtrack to the tree particle, e.g. a
    photon's shower SPs carry the photon's trackid) — no descendant walk.
  - charge numerator/denominator per particle, count-based versions kept
    alongside. NOTE: charge here is the raw Y-plane pixval sum per SP (NOT
    the de-double-counted unique-pixel charge of metric C) — fine for
    A/B between checkpoints, not directly comparable in absolute value.

Per-particle decomposition (denominator = the particle's SPs in the
slicerpred `pre` set, i.e. post-dedup post-lm-filter, pre-deghost):
    frac_kept   = survived stage-1 deghost            (pre/keep)
    frac_nuslc  = kept AND in a nu-classed pred slice (post/pred_class == nu)
so (1 - frac_kept) is deghost loss and (frac_kept - frac_nuslc) is
slicer-attributable loss.

Inputs:
    --inference-dir   directory with slicerpred_*.h5 (searched recursively)
    --manifest-csv    stage-0 manifest (columns incl. merged_h5) used to map
                      each slicerpred file back to its merged H5
    --out             output .npz of per-particle records
    --species         comma-separated pdg list to keep (default 22,11,-11;
                      pass 'all' for every origin==1 particle)
    --nu-class-id     slicer class id for nu (default 0)
    --max-events      cap for quick tests

Output npz arrays (one entry per selected true particle):
    fileno, entry, tid, pid, energy_mev,
    n_true, q_true, n_kept, q_kept, n_nuslc, q_nuslc

Example:
    python3 lartpc/larformer_analysis/slicer_eval/particle_slice_completeness.py \
        --inference-dir <...>/valtest_epoch4/inference/bnb_nu_pi0filter_corsika \
        --manifest-csv  <...>/manifest/bnb_nu_pi0filter_corsika_valtest.csv \
        --out valtest_epoch4_particle_completeness.npz
"""

import argparse
import csv
import os
import re
import sys
from glob import glob

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Same exact-position join tolerance analyze_event.py uses for the
# pre/coord -> triplet_data row match.
JOIN_TOL_CM = 0.05


def load_manifest(path):
    """merged_h5 basename -> (merged_h5 path, fileno, entry)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mh = row["merged_h5"]
            out[os.path.basename(mh)] = (
                mh, int(row["fileno"]), int(row["entry"]),
            )
    return out


def merged_key_from_slicerpred(sp_path):
    base = os.path.basename(sp_path)
    if not base.startswith("slicerpred_"):
        return None
    return base[len("slicerpred_"):]


def process_event(sp_path, merged_path, species, nu_class_id,
                  keep_source="pred"):
    """Returns list of per-particle record tuples (or None on failure).

    keep_source:
        "pred"     — kept = model deghost decision (pre/keep); nu-slice
                     membership from post/pred_class. Normal mode.
        "hasmatch" — LABEL-CEILING mode: kept = truth label
                     (pre/hasmatch == 1); "nuslc" is set equal to kept, so
                     the n_kept/q_kept columns report the completeness a
                     PERFECT deghoster (exactly reproducing its hasmatch
                     training labels) could achieve. No slicer involved.
    """
    with h5py.File(sp_path, "r") as sp:
        pre_coord = sp["pre/coord"][()].astype(np.float64)
        if keep_source == "hasmatch":
            keep = sp["pre/hasmatch"][()].astype(np.int64) == 1
            post_class = None
        elif keep_source == "tau_sweep":
            # DEGHOSTER-ONLY threshold sweep from the saved continuous
            # p_real. Returns per-particle kept charge for every tau in
            # `tau_list` (module-level; set by main). No slicer columns —
            # assessing the slicer at a new tau requires a real rerun.
            p_real = sp["pre/p_real"][()].astype(np.float64)
            hasmatch = sp["pre/hasmatch"][()].astype(np.int64) == 1
            keep = None
            post_class = None
        else:
            keep = sp["pre/keep"][()].astype(bool)
            post_class = sp["post/pred_class"][()]
    if pre_coord.shape[0] == 0:
        return []
    if keep_source == "tau_sweep":
        pre_is_nuslc = None
    elif keep_source == "hasmatch":
        pre_is_nuslc = keep.copy()
    else:
        n_post = int(keep.sum())
        if post_class.shape[0] != n_post:
            raise RuntimeError(
                f"post rows ({post_class.shape[0]}) != keep.sum() ({n_post}) "
                f"— kept-subset ordering assumption violated in {sp_path}"
            )
        # Map kept pre-rows -> post-row index (post arrays are the kept
        # subset in pre order; verified by the length check above).
        pre_is_nuslc = np.zeros(pre_coord.shape[0], dtype=bool)
        pre_is_nuslc[np.where(keep)[0]] = post_class == nu_class_id

    with h5py.File(merged_path, "r") as mh:
        entry_key = list(mh.keys())[0]
        g = mh[entry_key]
        td = g["triplet_data"]
        td_pos = td["pos"][()].astype(np.float64)
        td_tid = td["trackid"][()].astype(np.int64)
        td_pixval = td["pixval"][()]
        mt = g["mc_particle_tree"]
        mt_tid = mt["trackid"][()].astype(np.int64)
        mt_pid = mt["pid"][()].astype(np.int64)
        mt_origin = mt["origin"][()].astype(np.int64)
        mt_e = mt["energy_mev"][()].astype(np.float64)

    # Exact-position join: each pre SP -> its triplet_data row.
    tree = cKDTree(td_pos)
    dist, idx = tree.query(pre_coord, k=1)
    good = dist < JOIN_TOL_CM
    if good.mean() < 0.99:
        print(f"  [warn] {os.path.basename(sp_path)}: join matched only "
              f"{good.mean():.3f} of pre SPs", file=sys.stderr)
    pre_tid = np.where(good, td_tid[idx], -999999)
    # Y-plane (plane 2) pixval as the per-SP charge proxy.
    pre_q = np.where(good, td_pixval[idx, 2], 0.0).astype(np.float64)

    # True-particle selection: nu-origin, requested species.
    sel_particles = mt_origin == 1
    if species is not None:
        sel_particles &= np.isin(mt_pid, species)
    records = []
    if keep_source == "tau_sweep":
        # Per-particle kept charge at each tau + event-level kept-set
        # real/ghost charge (for purity). Record: (tid, pid, e, n_true,
        # q_true, [q_kept(tau)...], [q_kept_ghostset ignored]) — plus one
        # special record per event (tid=-1) carrying total kept real /
        # ghost charge per tau over ALL joined SPs.
        taus = np.asarray(TAU_LIST, dtype=np.float64)
        keep_t = p_real[:, None] > taus[None, :]           # (N, T)
        for j in np.where(sel_particles)[0]:
            tid = int(mt_tid[j])
            m = pre_tid == tid
            n_true = int(m.sum())
            if n_true == 0:
                continue
            q_true = float(pre_q[m].sum())
            q_kept_t = (pre_q[m, None] * keep_t[m]).sum(axis=0)   # (T,)
            records.append((tid, int(mt_pid[j]), float(mt_e[j]),
                            n_true, q_true, q_kept_t))
        # event-level purity accounting over all matched pre SPs
        real = hasmatch
        q_real_kept = (pre_q[real, None] * keep_t[real]).sum(axis=0)
        q_ghost_kept = (pre_q[~real, None] * keep_t[~real]).sum(axis=0)
        records.append((-1, 0, 0.0, 0, 0.0,
                        np.stack([q_real_kept, q_ghost_kept])))
        return records
    for j in np.where(sel_particles)[0]:
        tid = int(mt_tid[j])
        m = pre_tid == tid
        n_true = int(m.sum())
        if n_true == 0:
            continue
        q_true = float(pre_q[m].sum())
        mk = m & keep
        mn = m & pre_is_nuslc
        records.append((
            tid, int(mt_pid[j]), float(mt_e[j]),
            n_true, q_true,
            int(mk.sum()), float(pre_q[mk].sum()),
            int(mn.sum()), float(pre_q[mn].sum()),
        ))
    return records


TAU_LIST = [0.5]


def run_tau_sweep(sp_files, manifest, species, args):
    taus = np.asarray(TAU_LIST, dtype=np.float64)
    T = len(taus)
    pid_l, e_l, qtrue_l, qkept_l = [], [], [], []
    q_real_kept = np.zeros(T)
    q_ghost_kept = np.zeros(T)
    n_done = n_skip = 0
    for sp_path in sp_files:
        key = merged_key_from_slicerpred(sp_path)
        row = manifest.get(key) if key else None
        if row is None:
            n_skip += 1
            continue
        try:
            recs = process_event(sp_path, row[0], species, args.nu_class_id,
                                 keep_source="tau_sweep")
        except Exception as ex:
            print(f"  [warn] {os.path.basename(sp_path)}: {ex}",
                  file=sys.stderr)
            n_skip += 1
            continue
        for (tid, pid, e, nt, qt, arr) in recs:
            if tid == -1:
                q_real_kept += arr[0]
                q_ghost_kept += arr[1]
            else:
                pid_l.append(pid); e_l.append(e)
                qtrue_l.append(qt); qkept_l.append(arr)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  ...{n_done} events")
    pid_a = np.asarray(pid_l); qtrue_a = np.asarray(qtrue_l)
    qkept_a = np.asarray(qkept_l)                     # (Np, T)
    np.savez(args.out, tau=taus, pid=pid_a, energy_mev=np.asarray(e_l),
             q_true=qtrue_a, q_kept_tau=qkept_a,
             q_real_kept_total=q_real_kept, q_ghost_kept_total=q_ghost_kept)
    print(f"tau-sweep done: {n_done} events, {n_skip} skipped, "
          f"{len(pid_a)} particles -> {args.out}")
    hdr = "  tau   " + "".join(f"{t:7.2f}" for t in taus)
    print(hdr)
    for pdg, name in ((22, "gamma"), (11, "e-"), (-11, "e+")):
        s = pid_a == pdg
        if not s.any():
            continue
        cov = qkept_a[s].sum(axis=0) / max(qtrue_a[s].sum(), 1e-9)
        print(f"  {name:5s} completeness " +
              "".join(f"{c:7.3f}" for c in cov))
        low = ((qkept_a[s] / np.maximum(qtrue_a[s, None], 1e-9)) < 0.10)
        print(f"  {name:5s} frac<0.10    " +
              "".join(f"{v:7.3f}" for v in low.mean(axis=0)))
    purity = q_real_kept / np.maximum(q_real_kept + q_ghost_kept, 1e-9)
    print("  kept-set real-charge purity " +
          "".join(f"{v:7.3f}" for v in purity))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inference-dir", required=True)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", default="22,11,-11",
                    help="comma-separated pdg list, or 'all'")
    ap.add_argument("--nu-class-id", type=int, default=0)
    ap.add_argument("--keep-source", default="pred",
                    choices=("pred", "hasmatch", "tau_sweep"),
                    # tau_sweep: deghoster-only q_kept(tau) per particle from
                    # saved pre/p_real; use with --tau-list.
                    help="'pred' = model deghost+slicer decisions; "
                         "'hasmatch' = truth-label ceiling (see "
                         "process_event docstring)")
    ap.add_argument("--tau-list",
                    default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
                    help="comma-separated deghost thresholds for tau_sweep")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    global TAU_LIST
    TAU_LIST = [float(x) for x in args.tau_list.split(",")]

    species = (None if args.species.strip().lower() == "all"
               else np.array([int(x) for x in args.species.split(",")],
                             dtype=np.int64))
    manifest = load_manifest(args.manifest_csv)
    sp_files = sorted(
        glob(os.path.join(args.inference_dir, "**", "slicerpred_*.h5"),
             recursive=True))
    if args.max_events:
        sp_files = sp_files[: args.max_events]
    if not sp_files:
        sys.exit(f"no slicerpred_*.h5 under {args.inference_dir}")
    print(f"{len(sp_files)} slicerpred files; manifest has "
          f"{len(manifest)} merged files")

    if args.keep_source == "tau_sweep":
        run_tau_sweep(sp_files, manifest, species, args)
        return

    cols = {k: [] for k in
            ("fileno", "entry", "tid", "pid", "energy_mev",
             "n_true", "q_true", "n_kept", "q_kept", "n_nuslc", "q_nuslc")}
    n_done = n_skip = 0
    for sp_path in sp_files:
        key = merged_key_from_slicerpred(sp_path)
        row = manifest.get(key) if key else None
        if row is None:
            n_skip += 1
            continue
        merged_path, fileno, entry = row
        try:
            recs = process_event(sp_path, merged_path, species,
                                 args.nu_class_id,
                                 keep_source=args.keep_source)
        except Exception as ex:
            print(f"  [warn] {os.path.basename(sp_path)}: {ex}",
                  file=sys.stderr)
            n_skip += 1
            continue
        for (tid, pid, e, nt, qt, nk, qk, nn, qn) in recs:
            cols["fileno"].append(fileno)
            cols["entry"].append(entry)
            cols["tid"].append(tid)
            cols["pid"].append(pid)
            cols["energy_mev"].append(e)
            cols["n_true"].append(nt)
            cols["q_true"].append(qt)
            cols["n_kept"].append(nk)
            cols["q_kept"].append(qk)
            cols["n_nuslc"].append(nn)
            cols["q_nuslc"].append(qn)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  ...{n_done} events, {len(cols['tid'])} particles")

    out = {k: np.asarray(v) for k, v in cols.items()}
    np.savez(args.out, **out)
    print(f"done: {n_done} events processed, {n_skip} skipped, "
          f"{len(out['tid'])} particle records -> {args.out}")

    # Quick stdout summary per species (charge-based).
    q_true = out["q_true"]; q_nu = out["q_nuslc"]; q_kept = out["q_kept"]
    for pdg, name in ((22, "gamma"), (11, "e-"), (-11, "e+")):
        s = out["pid"] == pdg
        if not s.any():
            continue
        cov = q_nu[s].sum() / max(q_true[s].sum(), 1e-9)
        kept = q_kept[s].sum() / max(q_true[s].sum(), 1e-9)
        lowcov = float(np.mean(
            (q_nu[s] / np.maximum(q_true[s], 1e-9)) < 0.10))
        print(f"  {name:5s}: n={int(s.sum()):5d}  charge-completeness "
              f"(aggregate) nu-slice={cov:.3f}  post-deghost={kept:.3f}  "
              f"frac(particles cov<0.10)={lowcov:.3f}")


if __name__ == "__main__":
    main()
