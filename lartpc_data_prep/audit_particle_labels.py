"""Audit `compute_particle_labels` at scale on a production sample.

Two design checks this script answers:

  (1) Does the SP-side `triplet_data/trackid` field expose sub-threshold
      track granularity, or is it already aggregated to top-level tracks?
      → Count events with merge cases firing (n_member > 1 for some
      surviving slice), and tabulate what gets merged into what.

  (2) Does the KE-threshold table produce per-event particle GT that
      looks "right" — typical multiplicity, no pathological 1-SP
      micro-instances, reasonable nu-SP coverage, typical KE per class?
      → Per-event + per-particle stats dumped as CSVs.

Outputs (all under `--output-dir`):

  summary.txt           human-readable headline numbers + flagged outliers
  per_event.csv         one row per event with the key metrics
  per_particle.csv      one row per surviving particle (PDG, KE, n_sp,
                        n_member, run/sub/evt) — useful for distributions
  merge_cases.csv       one row per (host_tid, merged_tid) pair, only
                        when n_member > 1. Empty file = check (1) said
                        "data-prep already aggregates; merge path is a
                        safety net not exercised in practice."

Pure h5py + numpy + std lib (no torch). Runs in the pointcept_cuml
container or any Python 3 env with h5py + numpy. Single-threaded by
default; pass `--workers N` to use multiprocessing if the dataset is
large.

Usage on the cluster:

  python lartpc_data_prep/audit_particle_labels.py \\
      --h5-list /cluster/.../h5list_mcall_lantern_valtest.txt \\
      --output-dir /cluster/.../audit_particle_labels_pi0_valtest/ \\
      --workers 16 \\
      [--max-events 5000]            # cap for a smoke run
      [--ke-thresh-other 60]
      [--nu-origin 1]
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from lartpc_data_prep.slice_labels import (  # noqa: E402
    GHOST_SLICE_ID,
    compute_particle_labels,
)


PDG_NAMES = {
    11: "e-",   -11: "e+",
    13: "mu-",  -13: "mu+",
    22: "gamma",
    111: "pi0", 211: "pi+", -211: "pi-",
    2212: "p",  2112: "n",
    321:  "K+", -321: "K-",
    130: "K0L",
}


def _pdg_name(p):
    return PDG_NAMES.get(int(p), f"PDG:{int(p)}")


# ----------------------------------------------------------------------------
# Per-event analysis
# ----------------------------------------------------------------------------

def analyze_one(merged_h5,
                nu_origin=1,
                ke_thresh_other=60.0):
    """Return a dict with the per-event + per-particle + merge-case rows
    for one merged H5. On error returns {'error': str}.
    """
    out = dict(merged_h5=merged_h5)
    try:
        with h5py.File(merged_h5, "r") as f:
            e0 = f["entry_0"]
            if int(e0.attrs.get("oversized", 0)):
                return dict(merged_h5=merged_h5, error="oversized placeholder")
            run    = int(e0.attrs.get("run", -1))
            subrun = int(e0.attrs.get("subrun", -1))
            event  = int(e0.attrs.get("event", -1))
            mpt = e0["mc_particle_tree"]
            td  = e0["triplet_data"]
            sp_trackid  = td["trackid"][:]
            sp_origin   = td["origin"][:].astype(np.int64)
            sp_hasmatch = (td["hasmatch"][:].astype(np.int64)
                           if "hasmatch" in td else None)
            mc_tids = mpt["trackid"][:].astype(np.int64)
            mc_pids = mpt["pid"][:].astype(np.int64)
            mc_origin_arr = mpt["origin"][:].astype(np.int64)
            mc_ke   = mpt["energy_mev"][:].astype(np.float32)
            tid_to_idx = {int(t): i for i, t in enumerate(mc_tids)}
            pi = compute_particle_labels(
                mpt, sp_trackid, sp_hasmatch,
                nu_origin=nu_origin,
                other_ke_threshold=ke_thresh_other,
            )
    except Exception as exc:
        return dict(merged_h5=merged_h5,
                    error=f"{type(exc).__name__}: {exc}")

    real = (sp_hasmatch != 0) if sp_hasmatch is not None else np.ones_like(
        sp_trackid, dtype=bool,
    )
    nu_real = real & (sp_origin == nu_origin)
    cos_real = real & (sp_origin == 2)

    # Raw counts
    n_sp_total       = int(sp_trackid.shape[0])
    n_sp_real        = int(real.sum())
    n_sp_nu          = int(nu_real.sum())
    n_sp_cosmic      = int(cos_real.sum())
    n_sp_nu_to_part  = int(((pi["slice_id"] != GHOST_SLICE_ID) & nu_real).sum())
    n_sp_nu_ghost    = n_sp_nu - n_sp_nu_to_part
    nu_sp_coverage   = (n_sp_nu_to_part / n_sp_nu) if n_sp_nu > 0 else float("nan")
    n_unique_sp_tids_nu = int(len(np.unique(sp_trackid[nu_real])))
    n_particles      = int(len(pi["primary_trackid"]))
    n_mc_tracks      = int(mc_tids.shape[0])
    n_mc_nu_tracks   = int((mc_origin_arr == nu_origin).sum())

    # Per-class counts
    pdg_count = Counter(int(p) for p in pi["primary_pid"])

    # Per-particle rows
    per_particle_rows = []
    for k, sid in enumerate(pi["primary_trackid"]):
        members = pi["slice_member_trackids"][k]
        pdg = int(pi["primary_pid"][k])
        ke  = float(pi["primary_ke_MeV"][k])
        n_sp_slice = int(pi["primary_n_spacepoints"][k])
        per_particle_rows.append(dict(
            run=run, subrun=subrun, event=event,
            file=os.path.basename(merged_h5),
            qid=k,
            host_tid=int(sid),
            pdg=pdg,
            pdg_name=_pdg_name(pdg),
            host_ke_MeV=ke,
            n_sp=n_sp_slice,
            n_member=len(members),
        ))

    # Merge cases: any slice with > 1 members. Emit one row per
    # (host, member_other_than_host).
    merge_rows = []
    n_slices_with_merge = 0
    n_total_merged_tids = 0
    for k, sid in enumerate(pi["primary_trackid"]):
        members = pi["slice_member_trackids"][k]
        if len(members) <= 1:
            continue
        n_slices_with_merge += 1
        host_pdg = int(pi["primary_pid"][k])
        host_ke  = float(pi["primary_ke_MeV"][k])
        for m in members:
            if int(m) == int(sid):
                continue
            n_total_merged_tids += 1
            idx = tid_to_idx.get(int(m))
            mpdg = int(mc_pids[idx]) if idx is not None else 0
            mke  = float(mc_ke[idx])  if idx is not None else float("nan")
            morigin = int(mc_origin_arr[idx]) if idx is not None else -1
            merge_rows.append(dict(
                run=run, subrun=subrun, event=event,
                file=os.path.basename(merged_h5),
                host_tid=int(sid),
                host_pdg=host_pdg, host_pdg_name=_pdg_name(host_pdg),
                host_ke_MeV=host_ke,
                merged_tid=int(m),
                merged_pdg=mpdg, merged_pdg_name=_pdg_name(mpdg),
                merged_ke_MeV=mke,
                merged_origin=morigin,
            ))

    # Per-event row
    per_event = dict(
        run=run, subrun=subrun, event=event,
        file=os.path.basename(merged_h5),
        n_sp_total=n_sp_total,
        n_sp_real=n_sp_real,
        n_sp_nu=n_sp_nu,
        n_sp_cosmic=n_sp_cosmic,
        n_sp_nu_assigned_to_particle=n_sp_nu_to_part,
        n_sp_nu_ghost=n_sp_nu_ghost,
        nu_sp_coverage_frac=nu_sp_coverage,
        n_unique_sp_trackids_nu=n_unique_sp_tids_nu,
        n_particles=n_particles,
        n_mc_tracks=n_mc_tracks,
        n_mc_nu_tracks=n_mc_nu_tracks,
        n_slices_with_merge=n_slices_with_merge,
        n_total_merged_subtracks=n_total_merged_tids,
        # Per-class counts
        n_e=pdg_count.get(11, 0) + pdg_count.get(-11, 0),
        n_gamma=pdg_count.get(22, 0),
        n_mu=pdg_count.get(13, 0) + pdg_count.get(-13, 0),
        n_pi_charged=pdg_count.get(211, 0) + pdg_count.get(-211, 0),
        n_proton=pdg_count.get(2212, 0),
        n_neutron=pdg_count.get(2112, 0),
        n_other=sum(c for p, c in pdg_count.items()
                    if p not in (11, -11, 22, 13, -13, 211, -211, 2212, 2112)),
    )

    return dict(
        merged_h5=merged_h5,
        per_event=per_event,
        per_particle=per_particle_rows,
        merges=merge_rows,
        error=None,
    )


# ----------------------------------------------------------------------------
# Worker (for multiprocessing)
# ----------------------------------------------------------------------------

_WORKER_KWARGS = {}


def _init_worker(kwargs):
    global _WORKER_KWARGS
    _WORKER_KWARGS = kwargs


def _work(path):
    return analyze_one(path, **_WORKER_KWARGS)


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def _gather_paths(args):
    paths = []
    if args.h5_paths:
        paths.extend(args.h5_paths)
    if args.h5_list:
        with open(args.h5_list, "r") as f:
            for raw in f:
                p = raw.strip()
                if p and not p.startswith("#"):
                    paths.append(p)
    if args.h5_dir:
        paths.extend(sorted(glob.glob(
            os.path.join(args.h5_dir, "**", "merged_*.h5"), recursive=True,
        )))
    # Dedup + filter to existing
    seen = set(); deduped = []
    for p in paths:
        if p in seen: continue
        seen.add(p)
        if os.path.exists(p):
            deduped.append(p)
        else:
            sys.stderr.write(f"[warn] missing: {p}\n")
    if args.max_events is not None and args.max_events > 0:
        deduped = deduped[:args.max_events]
    return deduped


def _summarize(per_events, per_particles, merges, out_path,
               args):
    """Write summary.txt with headline numbers + flagged outliers."""
    lines = []
    lines.append(f"# audit_particle_labels summary")
    lines.append(f"# nu_origin = {args.nu_origin}")
    lines.append(f"# ke_thresh_other = {args.ke_thresh_other}")
    lines.append(f"# n_events_processed = {len(per_events)}")
    lines.append("")

    if not per_events:
        lines.append("(no events processed)")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    # Headline counts
    n_events = len(per_events)
    nu_sp_covs = np.array([r["nu_sp_coverage_frac"] for r in per_events],
                          dtype=np.float64)
    finite_cov = nu_sp_covs[np.isfinite(nu_sp_covs)]
    n_parts_arr = np.array([r["n_particles"] for r in per_events])
    n_merge_evt = sum(1 for r in per_events if r["n_slices_with_merge"] > 0)
    n_zero_part_evt = int((n_parts_arr == 0).sum())
    n_one_part_evt  = int((n_parts_arr == 1).sum())

    lines.append("== Check (1): merge-into-host activity ==")
    lines.append(f"  events with at least one merged slice: "
                 f"{n_merge_evt}/{n_events} ({n_merge_evt/n_events:.2%})")
    lines.append(f"  total merge events / slices / sub-track members:")
    n_total_slices_merge = sum(r["n_slices_with_merge"] for r in per_events)
    n_total_subtracks    = sum(r["n_total_merged_subtracks"] for r in per_events)
    lines.append(f"    n_slices_with_merge   = {n_total_slices_merge}")
    lines.append(f"    n_total_subtrack_merge_pairs = {n_total_subtracks}")
    if n_total_subtracks > 0:
        lines.append("  per-(host_pdg → merged_pdg) top pairs:")
        pair_counts = Counter(
            (_pdg_name(m["host_pdg"]), _pdg_name(m["merged_pdg"])) for m in merges
        )
        for (hp, mp_), c in pair_counts.most_common(10):
            lines.append(f"    {hp} ← {mp_}  count={c}")
    else:
        lines.append("  (no merges fired — data-prep aggregates trackids; "
                     "merge logic is a safety net.)")
    lines.append("")

    lines.append("== Check (2): GT instance topology + coverage ==")
    if len(finite_cov):
        lines.append(f"  nu_sp_coverage (= n_sp_nu_assigned / n_sp_nu): "
                     f"mean={float(finite_cov.mean()):.3f}  "
                     f"med={float(np.median(finite_cov)):.3f}  "
                     f"p10={float(np.percentile(finite_cov, 10)):.3f}  "
                     f"p90={float(np.percentile(finite_cov, 90)):.3f}")
    lines.append(f"  n_particles per event: "
                 f"mean={float(n_parts_arr.mean()):.2f}  "
                 f"med={float(np.median(n_parts_arr)):.1f}  "
                 f"p10={float(np.percentile(n_parts_arr, 10)):.0f}  "
                 f"p90={float(np.percentile(n_parts_arr, 90)):.0f}")
    lines.append(f"  events with 0 particles: {n_zero_part_evt} "
                 f"({n_zero_part_evt/n_events:.2%})")
    lines.append(f"  events with 1 particle:  {n_one_part_evt} "
                 f"({n_one_part_evt/n_events:.2%})")
    lines.append("")

    # Per-PDG instance stats
    lines.append("  per-PDG instance count (mean per event):")
    for col in ("n_e", "n_gamma", "n_mu", "n_pi_charged",
                "n_proton", "n_neutron", "n_other"):
        arr = np.array([r[col] for r in per_events])
        lines.append(f"    {col:>14s}  mean={float(arr.mean()):.2f}  "
                     f"max={int(arr.max())}  events_with_>=1={int((arr > 0).sum())}")
    lines.append("")

    # Per-particle stats
    if per_particles:
        n_sp_arr = np.array([r["n_sp"] for r in per_particles])
        ke_arr   = np.array([r["host_ke_MeV"] for r in per_particles])
        n_micro_1 = int((n_sp_arr == 1).sum())
        n_micro_5 = int((n_sp_arr < 5).sum())
        n_micro_10 = int((n_sp_arr < 10).sum())
        lines.append("== Per-particle distributions ==")
        lines.append(f"  n_sp per particle: "
                     f"mean={float(n_sp_arr.mean()):.1f}  "
                     f"med={float(np.median(n_sp_arr)):.1f}  "
                     f"p10={float(np.percentile(n_sp_arr, 10)):.0f}  "
                     f"p90={float(np.percentile(n_sp_arr, 90)):.0f}")
        lines.append(f"  particles with n_sp == 1:  "
                     f"{n_micro_1}  ({n_micro_1/len(per_particles):.2%})")
        lines.append(f"  particles with n_sp <  5:  "
                     f"{n_micro_5}  ({n_micro_5/len(per_particles):.2%})")
        lines.append(f"  particles with n_sp < 10:  "
                     f"{n_micro_10} ({n_micro_10/len(per_particles):.2%})")
        lines.append(f"  KE per particle (MeV): "
                     f"mean={float(ke_arr.mean()):.1f}  "
                     f"med={float(np.median(ke_arr)):.1f}  "
                     f"p10={float(np.percentile(ke_arr, 10)):.1f}  "
                     f"p90={float(np.percentile(ke_arr, 90)):.1f}")
        lines.append("  per-PDG KE distribution (host_ke_MeV, median ± IQR; n_sp median):")
        per_pdg = defaultdict(list)
        per_pdg_n_sp = defaultdict(list)
        for r in per_particles:
            per_pdg[r["pdg_name"]].append(r["host_ke_MeV"])
            per_pdg_n_sp[r["pdg_name"]].append(r["n_sp"])
        for name in sorted(per_pdg.keys(), key=lambda n: -len(per_pdg[n])):
            vs = np.array(per_pdg[name])
            nsp = np.array(per_pdg_n_sp[name])
            lines.append(
                f"    {name:>6s}  n={len(vs):>6d}  "
                f"KE med={float(np.median(vs)):>7.1f}  "
                f"p25={float(np.percentile(vs, 25)):>6.1f}  "
                f"p75={float(np.percentile(vs, 75)):>6.1f}  "
                f"n_sp med={float(np.median(nsp)):>5.1f}"
            )
        lines.append("")

    lines.append("== Outlier events (flagged for inspection) ==")
    # 0-particle
    if n_zero_part_evt > 0:
        flagged = [r for r in per_events if r["n_particles"] == 0]
        lines.append(f"  events with 0 surviving particles (first 10 of {len(flagged)}):")
        for r in flagged[:10]:
            lines.append(f"    {r['file']}  run/sub/evt=({r['run']},{r['subrun']},{r['event']})  "
                         f"n_sp_nu={r['n_sp_nu']}  n_mc_nu_tracks={r['n_mc_nu_tracks']}")
    # Low coverage
    low_cov_thresh = 0.50
    low_cov = [r for r in per_events
               if r["n_sp_nu"] > 0 and r["nu_sp_coverage_frac"] < low_cov_thresh]
    if low_cov:
        lines.append(f"  events with nu_sp_coverage < {low_cov_thresh} "
                     f"(first 10 of {len(low_cov)}):")
        for r in sorted(low_cov, key=lambda x: x["nu_sp_coverage_frac"])[:10]:
            lines.append(f"    {r['file']}  run/sub/evt=({r['run']},{r['subrun']},{r['event']})  "
                         f"cov={r['nu_sp_coverage_frac']:.2f}  "
                         f"n_sp_nu={r['n_sp_nu']}  n_particles={r['n_particles']}")
    # High multiplicity (top 1%)
    if n_events >= 100:
        cutoff = int(np.percentile(n_parts_arr, 99))
        hi = sorted([r for r in per_events if r["n_particles"] >= cutoff],
                    key=lambda x: -x["n_particles"])
        lines.append(f"  highest-multiplicity events (top {min(10, len(hi))} "
                     f"at n_particles >= {cutoff}):")
        for r in hi[:10]:
            lines.append(f"    {r['file']}  run/sub/evt=({r['run']},{r['subrun']},{r['event']})  "
                         f"n_particles={r['n_particles']}  n_sp_nu={r['n_sp_nu']}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--h5-list",   default=None, help="text file of merged_h5 paths")
    ap.add_argument("--h5-dir",    default=None, help="recursive glob root")
    ap.add_argument("--h5-paths",  nargs="*",     help="positional merged_h5 paths")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-events", type=int, default=None,
                    help="stop after this many events (smoke runs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="multiprocessing workers (default 1 = serial)")
    ap.add_argument("--nu-origin", type=int, default=1)
    ap.add_argument("--ke-thresh-other", type=float, default=60.0,
                    help="KE threshold for PDGs not in the default table")
    ap.add_argument("--progress-every", type=int, default=500,
                    help="print progress every N events")
    args = ap.parse_args()

    if not (args.h5_list or args.h5_dir or args.h5_paths):
        ap.error("provide --h5-list, --h5-dir, or positional --h5-paths")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = _gather_paths(args)
    if not paths:
        sys.exit("no input H5 paths found")
    print(f"[audit] processing {len(paths)} merged H5 files "
          f"(workers={args.workers})")

    kw = dict(nu_origin=args.nu_origin,
              ke_thresh_other=args.ke_thresh_other)

    per_events    = []
    per_particles = []
    merges        = []
    errors        = []
    t0 = time.time()

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(kw,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_work, paths, chunksize=4)):
                if res.get("error"):
                    errors.append(res); continue
                per_events.append(res["per_event"])
                per_particles.extend(res["per_particle"])
                merges.extend(res["merges"])
                if (i + 1) % args.progress_every == 0:
                    dt = time.time() - t0
                    print(f"  [{i+1}/{len(paths)}]  "
                          f"{(i+1)/dt:.1f} ev/s  "
                          f"per_events={len(per_events)}  "
                          f"per_particles={len(per_particles)}  "
                          f"merges={len(merges)}  errors={len(errors)}")
    else:
        for i, p in enumerate(paths):
            res = analyze_one(p, **kw)
            if res.get("error"):
                errors.append(res); continue
            per_events.append(res["per_event"])
            per_particles.extend(res["per_particle"])
            merges.extend(res["merges"])
            if (i + 1) % args.progress_every == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(paths)}]  "
                      f"{(i+1)/dt:.1f} ev/s  "
                      f"per_events={len(per_events)}  "
                      f"per_particles={len(per_particles)}  "
                      f"merges={len(merges)}  errors={len(errors)}")

    dt = time.time() - t0
    print(f"[audit] done in {dt:.1f}s  "
          f"({len(per_events)} ok, {len(errors)} errors)")

    # Write outputs
    per_event_fields = [
        "run","subrun","event","file",
        "n_sp_total","n_sp_real","n_sp_nu","n_sp_cosmic",
        "n_sp_nu_assigned_to_particle","n_sp_nu_ghost","nu_sp_coverage_frac",
        "n_unique_sp_trackids_nu","n_particles",
        "n_mc_tracks","n_mc_nu_tracks",
        "n_slices_with_merge","n_total_merged_subtracks",
        "n_e","n_gamma","n_mu","n_pi_charged","n_proton","n_neutron","n_other",
    ]
    per_particle_fields = [
        "run","subrun","event","file","qid",
        "host_tid","pdg","pdg_name","host_ke_MeV","n_sp","n_member",
    ]
    merge_fields = [
        "run","subrun","event","file",
        "host_tid","host_pdg","host_pdg_name","host_ke_MeV",
        "merged_tid","merged_pdg","merged_pdg_name","merged_ke_MeV",
        "merged_origin",
    ]
    _write_csv(per_events,    os.path.join(args.output_dir, "per_event.csv"),
               per_event_fields)
    _write_csv(per_particles, os.path.join(args.output_dir, "per_particle.csv"),
               per_particle_fields)
    _write_csv(merges,        os.path.join(args.output_dir, "merge_cases.csv"),
               merge_fields)

    # Error log
    if errors:
        with open(os.path.join(args.output_dir, "errors.txt"), "w") as f:
            for r in errors:
                f.write(f"{r['merged_h5']}\t{r['error']}\n")

    # Summary
    _summarize(per_events, per_particles, merges,
               os.path.join(args.output_dir, "summary.txt"), args)
    print(f"  wrote {args.output_dir}/{{summary.txt, per_event.csv, "
          f"per_particle.csv, merge_cases.csv}}"
          + (", errors.txt" if errors else ""))


if __name__ == "__main__":
    main()
