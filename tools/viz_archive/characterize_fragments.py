"""
Phase 1 fragment-characterization for the shower-clustering Mask2Former design.

See pointcept/docs/reference/shower_clustering_design.md §5 (Phase 1) for context. The
diagnostics here decide:
    - whether to keep the voxel scale (yes if non-shower context matters)
    - whether to keep spacepoint refinement (depends on fragment IoU CDF)
    - initial voxel size (5 cm guess; orphan→fragment distance informs)
    - whether DBSCAN params need revision

Inputs: merged HDF5 files from Step 4 of the pipeline (output of
ub_showerorigin_reco/ubshowerorginreco/merge_reco_truth_showerorigin.py).
Each merged H5 contains one event under entry_0 with:
    triplet_data/pos       (N, 3) cm
    triplet_data/trackid   (N,)   per-spacepoint true particle id
    triplet_data/pid       (N,)   per-spacepoint true PDG
    triplet_data/origin    (N,)   per-spacepoint true origin label
    triplet_data/hasmatch  (N,)   1 if matched to non-ghost truth point
    shower_fragments/{trackid, pid, type, istrunk, startpt, originpt,
                      pointindices_flat, pointindices_counts, ...}

Outputs (under --out-dir):
    summary.h5    aggregated arrays + per-event compact records
    summary.txt   human-readable distribution summary (also printed)
    plots/*.png   distribution histograms, headline IoU CDF

Usage:
    python characterize_fragments.py \\
        --inputs '/path/to/workdir*/merged_*.h5' \\
        --out-dir pointcept/exp/shower_clustering/phase1_diagnostics

    python characterize_fragments.py \\
        --input-list merged_h5_list.txt \\
        --out-dir pointcept/exp/shower_clustering/phase1_diagnostics \\
        --max-events 5000
"""
import argparse
import glob
import os
import sys
import traceback
from collections import defaultdict

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


SHOWER_PIDS = (11, -11, 22)
TYPE_NAMES = {0: "inside", 1: "outside", 2: "on_track", 3: "ghost", 4: "true_track"}
ORPHAN_SAMPLE_CAP = 2000


def iter_events(h5_path):
    with h5py.File(h5_path, "r") as f:
        for entry_name in sorted(f.keys()):
            entry = f[entry_name]
            if "triplet_data" not in entry or "shower_fragments" not in entry:
                continue
            yield entry_name, entry


def _per_spacepoint_trackid(triplet, n_sp):
    if "trackid" not in triplet:
        return None
    return triplet["trackid"][:].astype(np.int64)


def _build_descendant_sets(entry):
    """
    Read mc_particle_tree and return (children_map, has_tree).

    children_map[parent_trackid] = list of child_trackids.
    Returns (None, False) if the tree is missing (old-schema files).
    """
    if "mc_particle_tree" not in entry:
        return None, False
    mpt = entry["mc_particle_tree"]
    if "trackid" not in mpt or "parent_trackid" not in mpt:
        return None, False
    tids = mpt["trackid"][:].astype(np.int64)
    parents = mpt["parent_trackid"][:].astype(np.int64)
    children = defaultdict(list)
    for tid, par in zip(tids, parents):
        children[int(par)].append(int(tid))
    return children, True


def _descendants_of(root_tid, children_map):
    """BFS over children_map starting at root_tid (inclusive)."""
    out = set()
    stack = [int(root_tid)]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(children_map.get(cur, []))
    return out


def _orphan_to_fragment_distances(pos, orphan_mask, in_any_frag, rng):
    if not orphan_mask.any() or not in_any_frag.any() or not HAVE_SCIPY:
        return np.array([], dtype=np.float32)
    orphan_idx = np.where(orphan_mask)[0]
    if len(orphan_idx) > ORPHAN_SAMPLE_CAP:
        orphan_idx = rng.choice(orphan_idx, ORPHAN_SAMPLE_CAP, replace=False)
    tree = cKDTree(pos[in_any_frag])
    d, _ = tree.query(pos[orphan_idx], k=1)
    return d.astype(np.float32)


def _dominant_pid(pids):
    if len(pids) == 0:
        return -1
    vals, counts = np.unique(pids, return_counts=True)
    return int(vals[np.argmax(counts)])


def analyze_event(entry, rng):
    triplet = entry["triplet_data"]
    sf = entry["shower_fragments"]

    pos = triplet["pos"][:].astype(np.float32)
    n_sp = pos.shape[0]
    sp_trackid = _per_spacepoint_trackid(triplet, n_sp)
    if sp_trackid is None:
        raise RuntimeError("triplet_data/trackid missing — old merged H5; regenerate")
    sp_pid = triplet["pid"][:].astype(np.int64) if "pid" in triplet else \
        np.full(n_sp, 0, dtype=np.int64)

    num_frags = int(sf.attrs.get("num_fragments", 0))
    frag_indices_per_frag = []
    frag_trackids = []
    if num_frags > 0:
        flat = sf["pointindices_flat"][:].astype(np.int64)
        counts = sf["pointindices_counts"][:].astype(np.int64)
        if "trackid" in sf:
            f_tids = sf["trackid"][:].astype(np.int64)
        else:
            f_tids = np.full(num_frags, -1, dtype=np.int64)
        offset = 0
        for fi in range(num_frags):
            n = int(counts[fi])
            indices = flat[offset:offset + n]
            offset += n
            indices = indices[(indices >= 0) & (indices < n_sp)]
            frag_indices_per_frag.append(indices)
            frag_trackids.append(int(f_tids[fi]))

    frag_union_by_tid = defaultdict(set)
    frag_count_by_tid = defaultdict(int)
    for fi, tid in enumerate(frag_trackids):
        frag_union_by_tid[tid].update(frag_indices_per_frag[fi].tolist())
        frag_count_by_tid[tid] += 1

    sp_in_shower_pid = np.isin(sp_pid, SHOWER_PIDS)
    sp_in_track_pid = (~sp_in_shower_pid) & (sp_pid != 0)
    sp_matched = sp_trackid > 0

    sp_in_shower = sp_in_shower_pid & sp_matched
    sp_in_track = sp_in_track_pid & sp_matched

    children_map, have_tree = _build_descendant_sets(entry)

    per_instance = []
    if have_tree:
        # Trunk-descendant GT instances:
        # For each unique non-(-1) trackid that appears in shower_fragments,
        # walk mc_particle_tree to gather all descendant trackids; truth set
        # is per-spacepoint trackids matching any descendant.
        unique_frag_tids = sorted(
            t for t in set(frag_trackids) if t > 0
        )
        for tid in unique_frag_tids:
            descendants = _descendants_of(tid, children_map)
            desc_arr = np.fromiter(descendants, dtype=np.int64,
                                   count=len(descendants))
            truth_idx = np.where(np.isin(sp_trackid, desc_arr))[0]
            truth_set = set(truth_idx.tolist())
            frag_set = frag_union_by_tid.get(int(tid), set())
            inter = truth_set & frag_set
            union = truth_set | frag_set
            iou = len(inter) / max(1, len(union))
            coverage = len(inter) / max(1, len(truth_set))
            dom_pid = _dominant_pid(sp_pid[truth_idx]) if len(truth_idx) else -1
            per_instance.append((
                int(tid), dom_pid, len(truth_set), len(frag_set), len(inter),
                float(iou), float(coverage),
                int(frag_count_by_tid.get(int(tid), 0)),
            ))
    else:
        # Legacy per-trackid IoU (no mc_particle_tree available).
        # Caveat: granular per-Geant4-trackid; biased low by plurality voting.
        unique_shower_tids = np.unique(sp_trackid[sp_in_shower]) \
            if sp_in_shower.any() else np.array([], dtype=np.int64)
        for tid in unique_shower_tids:
            truth_idx = np.where(sp_trackid == int(tid))[0]
            truth_set = set(truth_idx.tolist())
            frag_set = frag_union_by_tid.get(int(tid), set())
            inter = truth_set & frag_set
            union = truth_set | frag_set
            iou = len(inter) / max(1, len(union))
            coverage = len(inter) / max(1, len(truth_set))
            dom_pid = _dominant_pid(sp_pid[truth_idx])
            per_instance.append((
                int(tid), dom_pid, len(truth_set), len(frag_set), len(inter),
                float(iou), float(coverage),
                int(frag_count_by_tid.get(int(tid), 0)),
            ))

    in_any_frag = np.zeros(n_sp, dtype=bool)
    for indices in frag_indices_per_frag:
        in_any_frag[indices] = True

    orphan_shower_mask = sp_in_shower & ~in_any_frag
    orphan_distances = _orphan_to_fragment_distances(
        pos, orphan_shower_mask, in_any_frag, rng,
    )

    # SSNet false-negative diagnostic: for orphan shower-pid spacepoints,
    # what label did SSNet give them? Ideally most should be shower labels.
    # Heavy presence of track-class labels confirms the FN hypothesis.
    ssnet_orphan_labels = np.array([], dtype=np.int64)
    if "ssnet_label" in triplet and orphan_shower_mask.any():
        ssnet_orphan_labels = triplet["ssnet_label"][:].astype(np.int64)[
            orphan_shower_mask
        ]

    frag_purities = []
    for fi, indices in enumerate(frag_indices_per_frag):
        tid = frag_trackids[fi]
        if tid <= 0 or len(indices) == 0:
            continue
        purity = float((sp_trackid[indices] == tid).sum() / len(indices))
        frag_purities.append(purity)

    if n_sp > 0:
        bbox_extent = (pos.max(axis=0) - pos.min(axis=0)).astype(np.float32)
    else:
        bbox_extent = np.zeros(3, dtype=np.float32)

    return {
        "instance_def": "trunk_descendant" if have_tree else "per_trackid_legacy",
        "n_spacepoints": n_sp,
        "n_shower_spacepoints": int(sp_in_shower.sum()),
        "n_track_spacepoints": int(sp_in_track.sum()),
        "n_unmatched_spacepoints": int((~sp_matched).sum()),
        "n_fragments": num_frags,
        "n_shower_instances": int(len(per_instance)),
        "n_track_instances": int(len(np.unique(sp_trackid[sp_in_track]))
                                 if sp_in_track.any() else 0),
        "n_orphan_shower_spacepoints": int(orphan_shower_mask.sum()),
        "frag_point_counts": np.array(
            [len(idx) for idx in frag_indices_per_frag], dtype=np.int64),
        "frag_trackids": np.array(frag_trackids, dtype=np.int64),
        "frag_purities": np.array(frag_purities, dtype=np.float32),
        "per_instance": per_instance,
        "orphan_distances": orphan_distances,
        "ssnet_orphan_labels": ssnet_orphan_labels,
        "bbox_extent": bbox_extent,
    }


def aggregate(events_diag):
    keys_per_event = (
        "n_spacepoints", "n_shower_spacepoints", "n_track_spacepoints",
        "n_unmatched_spacepoints", "n_fragments", "n_shower_instances",
        "n_track_instances", "n_orphan_shower_spacepoints",
    )
    agg = {k: [] for k in keys_per_event}
    agg["frag_point_counts"] = []
    agg["frag_purities"] = []
    agg["instance_iou"] = []
    agg["instance_coverage"] = []
    agg["instance_n_truth_points"] = []
    agg["instance_n_assigned_points"] = []
    agg["instance_n_fragments"] = []
    agg["instance_pid"] = []
    agg["orphan_distances"] = []
    agg["ssnet_orphan_labels"] = []
    for axis in ("x", "y", "z"):
        agg[f"bbox_extent_{axis}"] = []
    agg["instance_def_modes"] = []

    for d in events_diag:
        for k in keys_per_event:
            agg[k].append(d[k])
        if d["frag_point_counts"].size:
            agg["frag_point_counts"].extend(d["frag_point_counts"].tolist())
        if d["frag_purities"].size:
            agg["frag_purities"].extend(d["frag_purities"].tolist())
        for tid, pid, ntp, nap, ni, iou, cov, nf in d["per_instance"]:
            agg["instance_iou"].append(iou)
            agg["instance_coverage"].append(cov)
            agg["instance_n_truth_points"].append(ntp)
            agg["instance_n_assigned_points"].append(nap)
            agg["instance_n_fragments"].append(nf)
            agg["instance_pid"].append(pid)
        if d["orphan_distances"].size:
            agg["orphan_distances"].extend(d["orphan_distances"].tolist())
        if d.get("ssnet_orphan_labels") is not None and d["ssnet_orphan_labels"].size:
            agg["ssnet_orphan_labels"].extend(d["ssnet_orphan_labels"].tolist())
        for ax_i, axis in enumerate(("x", "y", "z")):
            agg[f"bbox_extent_{axis}"].append(float(d["bbox_extent"][ax_i]))
        agg["instance_def_modes"].append(d.get("instance_def", "unknown"))

    out = {}
    for k, v in agg.items():
        if k == "instance_def_modes":
            out[k] = np.asarray(v, dtype=object)
        else:
            out[k] = np.asarray(v)
    return out


def _hist(arr, name, xlabel, plot_dir, bins=50, log_y=False, log_x=False):
    if len(arr) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if log_x and (arr > 0).any():
        positive = arr[arr > 0]
        edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), bins)
        ax.hist(positive, bins=edges)
        ax.set_xscale("log")
    else:
        ax.hist(arr, bins=bins)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=120)
    plt.close(fig)


def make_plots(agg, out_dir):
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    _hist(agg["n_spacepoints"], "spacepoints_per_event",
          "spacepoints / event", plot_dir, log_x=True)
    _hist(agg["n_fragments"], "fragments_per_event",
          "fragments / event", plot_dir)
    _hist(agg["n_shower_instances"], "shower_instances_per_event",
          "true shower instances / event", plot_dir)
    _hist(agg["n_track_instances"], "track_instances_per_event",
          "true track instances / event", plot_dir)
    _hist(agg["frag_point_counts"], "points_per_fragment",
          "points / fragment", plot_dir, log_x=True)
    _hist(agg["frag_purities"], "fragment_purity",
          "fragment plurality purity (assigned-tid match rate)", plot_dir)
    _hist(agg["instance_n_truth_points"], "truth_points_per_instance",
          "truth spacepoints / shower", plot_dir, log_x=True)
    _hist(agg["instance_iou"], "instance_iou",
          "fragment-union vs truth IoU", plot_dir)
    _hist(agg["instance_coverage"], "instance_coverage",
          "|fragments ∩ truth| / |truth|", plot_dir)
    _hist(agg["instance_n_fragments"], "fragments_per_instance",
          "fragments per true shower instance", plot_dir)
    _hist(agg["orphan_distances"], "orphan_distance_to_fragment",
          "orphan shower SP → nearest in-fragment SP [cm]", plot_dir)
    for axis in ("x", "y", "z"):
        _hist(agg[f"bbox_extent_{axis}"], f"bbox_extent_{axis}",
              f"event bounding-box extent {axis} [cm]", plot_dir)

    if len(agg["instance_iou"]) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        sorted_iou = np.sort(agg["instance_iou"])
        cdf = np.linspace(0, 1, len(sorted_iou))
        ax.plot(sorted_iou, cdf, lw=2)
        ax.set_xlabel("IoU(fragment union, per-spacepoint truth)")
        ax.set_ylabel("CDF over true shower instances")
        ax.set_title("Fragment-truth IoU CDF [Phase 1 headline]")
        ax.axvline(0.9, color="g", ls="--", alpha=0.6,
                   label="0.9: pooling alone may suffice")
        ax.axvline(0.7, color="orange", ls="--", alpha=0.6,
                   label="0.7: multi-scale needed")
        ax.axvline(0.5, color="r", ls="--", alpha=0.6,
                   label="0.5: revisit DBSCAN")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "_HEADLINE_iou_cdf.png"), dpi=120)
        plt.close(fig)


def write_summary_h5(events_diag, agg, out_dir):
    out_path = os.path.join(out_dir, "summary.h5")
    instance_dt = np.dtype([
        ("trackid", "i8"), ("pid", "i4"),
        ("n_truth_points", "i8"), ("n_assigned_points", "i8"),
        ("n_intersection", "i8"),
        ("iou", "f4"), ("coverage", "f4"),
        ("n_fragments", "i4"),
    ])
    with h5py.File(out_path, "w") as f:
        agg_grp = f.create_group("aggregate")
        for k, v in agg.items():
            if k == "instance_def_modes":
                # Object dtype — store unique values as attrs
                modes, counts = np.unique(v, return_counts=True)
                for m, c in zip(modes, counts):
                    agg_grp.attrs[f"mode_{m}"] = int(c)
                continue
            agg_grp.create_dataset(k, data=v)

        ev_grp = f.create_group("events")
        scalar_keys = (
            "n_spacepoints", "n_shower_spacepoints", "n_track_spacepoints",
            "n_unmatched_spacepoints", "n_fragments", "n_shower_instances",
            "n_track_instances", "n_orphan_shower_spacepoints",
        )
        for i, d in enumerate(events_diag):
            eg = ev_grp.create_group(f"event_{i:06d}")
            for k in scalar_keys:
                eg.attrs[k] = d[k]
            eg.create_dataset("frag_point_counts", data=d["frag_point_counts"])
            eg.create_dataset("frag_trackids", data=d["frag_trackids"])
            eg.create_dataset("frag_purities", data=d["frag_purities"])
            eg.create_dataset("bbox_extent", data=d["bbox_extent"])
            if d["per_instance"]:
                arr = np.array(d["per_instance"], dtype=instance_dt)
                eg.create_dataset("per_instance", data=arr)


def _stats_line(name, arr, fmt="{:.2f}"):
    if len(arr) == 0:
        return f"{name:42s} (empty)"
    return (f"{name:42s} median={fmt.format(np.median(arr))}  "
            f"mean={fmt.format(np.mean(arr))}  "
            f"p05={fmt.format(np.percentile(arr, 5))}  "
            f"p95={fmt.format(np.percentile(arr, 95))}")


def write_summary_txt(agg, n_events, n_files, out_dir):
    lines = []

    def p(s):
        lines.append(s)
        print(s)

    p("=" * 78)
    p("Phase 1 Fragment Characterization — Summary")
    p("=" * 78)
    p(f"Files processed:                  {n_files}")
    p(f"Events processed:                 {n_events}")
    if "instance_def_modes" in agg:
        modes, counts = np.unique(agg["instance_def_modes"], return_counts=True)
        for m, c in zip(modes, counts):
            p(f"  instance_def = {str(m):<30s} {int(c)} events")
    p("")
    p("--- Event-level distributions ---")
    p(_stats_line("Spacepoints per event:", agg["n_spacepoints"], "{:.0f}"))
    p(_stats_line("  Shower-pid spacepoints/event:",
                  agg["n_shower_spacepoints"], "{:.0f}"))
    p(_stats_line("  Track-pid spacepoints/event:",
                  agg["n_track_spacepoints"], "{:.0f}"))
    p(_stats_line("  Unmatched spacepoints/event:",
                  agg["n_unmatched_spacepoints"], "{:.0f}"))
    p(_stats_line("Fragments per event:", agg["n_fragments"], "{:.0f}"))
    p(_stats_line("True shower instances/event:",
                  agg["n_shower_instances"], "{:.0f}"))
    p(_stats_line("True track instances/event:",
                  agg["n_track_instances"], "{:.0f}"))
    p(_stats_line("Orphan shower SPs / event:",
                  agg["n_orphan_shower_spacepoints"], "{:.0f}"))
    p("")
    p("--- Per-instance / per-fragment ---")
    p(_stats_line("Points per fragment:", agg["frag_point_counts"], "{:.0f}"))
    p(_stats_line("Fragment plurality purity:", agg["frag_purities"], "{:.3f}"))
    p(_stats_line("Points per true shower instance:",
                  agg["instance_n_truth_points"], "{:.0f}"))
    p(_stats_line("Fragments per true shower instance:",
                  agg["instance_n_fragments"], "{:.1f}"))
    p("")
    p("--- HEADLINE: fragment-truth IoU per true shower instance ---")
    p(_stats_line("IoU (all instances):", agg["instance_iou"], "{:.3f}"))
    p(_stats_line("Coverage (all instances):",
                  agg["instance_coverage"], "{:.3f}"))
    if len(agg["instance_iou"]):
        n_truth = agg["instance_n_truth_points"]
        ious = agg["instance_iou"]
        covs = agg["instance_coverage"]
        nfrags = agg["instance_n_fragments"]
        size_buckets = [
            ("[1-19] (DBSCAN-invisible by design)", (n_truth >= 1) & (n_truth < 20)),
            ("[20-99]  (small)", (n_truth >= 20) & (n_truth < 100)),
            ("[100-499] (medium)", (n_truth >= 100) & (n_truth < 500)),
            ("[500+]    (large)", n_truth >= 500),
        ]
        p("")
        p("  Stratified by truth-instance point count:")
        p(f"  {'bucket':<40s} {'count':>7s} {'frac>=0frag':>12s} "
          f"{'IoU med':>8s} {'IoU p95':>8s} {'cov med':>8s}")
        for label, mask in size_buckets:
            n = int(mask.sum())
            if n == 0:
                p(f"  {label:<40s} {0:>7d}  (no instances)")
                continue
            frac_has_frag = float((nfrags[mask] >= 1).mean())
            iou_med = float(np.median(ious[mask]))
            iou_p95 = float(np.percentile(ious[mask], 95))
            cov_med = float(np.median(covs[mask]))
            p(f"  {label:<40s} {n:>7d}  {frac_has_frag:>12.3f}  "
              f"{iou_med:>8.3f}  {iou_p95:>8.3f}  {cov_med:>8.3f}")
        p("")
        p("  Significant-only (truth points >= 20):")
        sig = n_truth >= 20
        if sig.any():
            for t in (0.5, 0.7, 0.8, 0.9):
                frac = float((ious[sig] >= t).mean())
                p(f"    fraction with IoU >= {t:.2f}: {frac:.3f}")
    p("")
    p("--- Voxel-size diagnostic: orphan-shower-SP → fragment distance ---")
    p(_stats_line("Distance [cm]:", agg["orphan_distances"], "{:.2f}"))
    p("")
    p("--- SSNet false-negative diagnostic ---")
    if len(agg.get("ssnet_orphan_labels", [])):
        labels, counts = np.unique(agg["ssnet_orphan_labels"], return_counts=True)
        total = counts.sum()
        p(f"  SSNet labels of orphan shower-pid spacepoints (n={int(total)}):")
        for lab, c in sorted(zip(labels.tolist(), counts.tolist()),
                             key=lambda x: -x[1]):
            p(f"    label={lab:>3d}  n={c:>9d}  ({100.0*c/total:5.1f}%)")
        p("  (track-class labels here = SSNet false-negatives,")
        p("   recoverable only via voxel-scale tokens in the new model)")
    else:
        p("  (no ssnet_label data in input H5)")
    p("")
    p("--- Event spatial extent ---")
    p(_stats_line("BBox extent X [cm]:", agg["bbox_extent_x"], "{:.1f}"))
    p(_stats_line("BBox extent Y [cm]:", agg["bbox_extent_y"], "{:.1f}"))
    p(_stats_line("BBox extent Z [cm]:", agg["bbox_extent_z"], "{:.1f}"))
    p("")
    p("=" * 78)
    p("Decision guidance (cf. shower_clustering_design.md §4):")
    if len(agg["instance_iou"]):
        n_truth = agg["instance_n_truth_points"]
        sig = n_truth >= 20
        if sig.any():
            med_sig = float(np.median(agg["instance_iou"][sig]))
            p(f"  Median IoU (significant >=20pts): {med_sig:.3f}")
            if med_sig >= 0.9:
                p("    >= 0.9 : fragment-pooling alone may suffice for")
                p("             significant showers. Spacepoint refinement")
                p("             still needed for sub-20-pt instances.")
            elif med_sig >= 0.7:
                p("    in [0.7, 0.9) : multi-scale design (current default)")
                p("                    well motivated.")
            elif med_sig >= 0.4:
                p("    in [0.4, 0.7) : DBSCAN is significantly under-clustering")
                p("                    even significant showers. Spacepoint")
                p("                    refinement is critical; consider")
                p("                    revisiting DBSCAN eps.")
            else:
                p("    < 0.4  : DBSCAN coverage is poor. Revisit DBSCAN")
                p("             parameters before committing to architecture.")
        # GT instance size policy hint
        n_per_event_significant = (
            agg["instance_n_truth_points"] >= 20
        ).sum() / max(1, len(agg["n_spacepoints"]))
        n_per_event_all = len(agg["instance_n_truth_points"]) / max(
            1, len(agg["n_spacepoints"])
        )
        p(f"  Avg true shower instances/event: all={n_per_event_all:.1f}, "
          f"significant(>=20pts)={n_per_event_significant:.1f}")
        p("    -> design: N=64 queries comfortably covers significant set;")
        p("       sub-20-pt instances are below DBSCAN resolution and likely")
        p("       must be either (a) merged into parent shower's GT or")
        p("       (b) excluded from GT entirely.")
    if len(agg["orphan_distances"]):
        p(f"  Orphan→fragment distance p95 = "
          f"{np.percentile(agg['orphan_distances'], 95):.1f} cm")
        p("           Pick voxel size at or below this so the model can see")
        p("           orphan spacepoints in voxels adjacent to fragment voxels.")
    p("=" * 78)

    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="+",
                   help="Glob(s) for merged H5 files")
    g.add_argument("--input-list",
                   help="Text file listing merged H5 paths (one per line)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for plots and summary H5")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Cap on total events processed (for fast dev iter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    files = []
    if args.inputs:
        for pat in args.inputs:
            files.extend(sorted(glob.glob(pat)))
    else:
        with open(args.input_list) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(line)
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("ERROR: no input files found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Phase 1] {len(files)} merged H5 files → {args.out_dir}")
    if not HAVE_SCIPY:
        print("[warn] scipy not available — orphan-distance diagnostic skipped",
              file=sys.stderr)

    events_diag = []
    n_failed = 0
    for fi, fpath in enumerate(files):
        try:
            for _, entry in iter_events(fpath):
                d = analyze_event(entry, rng)
                events_diag.append(d)
                if args.max_events and len(events_diag) >= args.max_events:
                    break
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  [warn] {os.path.basename(fpath)}: {e}", file=sys.stderr)
                if n_failed == 5:
                    print("  [warn] further file errors suppressed",
                          file=sys.stderr)
            continue
        if args.max_events and len(events_diag) >= args.max_events:
            break
        if (fi + 1) % 50 == 0:
            print(f"  ... {fi+1}/{len(files)} files, "
                  f"{len(events_diag)} events, {n_failed} failed")

    if not events_diag:
        print("ERROR: no events processed", file=sys.stderr)
        sys.exit(2)

    print(f"[Phase 1] {len(events_diag)} events processed "
          f"({n_failed} files failed). Aggregating...")
    agg = aggregate(events_diag)
    print("[Phase 1] Writing plots...")
    make_plots(agg, args.out_dir)
    print("[Phase 1] Writing summary H5...")
    write_summary_h5(events_diag, agg, args.out_dir)
    print("[Phase 1] Writing text summary...\n")
    write_summary_txt(agg, len(events_diag), len(files), args.out_dir)
    print(f"\n[Phase 1] Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(3)
