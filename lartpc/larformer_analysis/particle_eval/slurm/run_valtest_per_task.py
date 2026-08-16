"""SLURM-array driver: Stage-3 particle-segmenter inference + per-event
analysis on the val (or test) split.

Two input modes (chosen via --input-mode):

  cached       Enumerate Stage-1+2 cache events (matches what the
               in-training evaluator sees). Each task gets a contiguous
               slice [task_id*stride : (task_id+1)*stride] of the
               sorted event list, runs Stage-3 inference once on the
               whole slice via `tools/larformer/run_larformer_stage3_inference.py
               --input-mode cached --cache-file-list <list>`, then loops
               `analyze_event.py` per resulting `stage3pred_*.h5`.

  full-cascade Read filenos from a Stage-0 rerun_lines file + manifest CSV
               (same shape as the slicer's val driver). Each task gets a
               slice of filenos, builds a per-task inputlist of merged_h5
               files, runs the cascade end-to-end.

Outputs (per task):
    OUTPUT_DIR/inference/<TAG>/stage3pred_*.h5
    OUTPUT_DIR/analysis/<TAG>/perevent_<stem>.h5
    OUTPUT_DIR/_tasklists/<TAG>/task<NNN>.txt

Per-event idempotency: a `perevent_*.h5` (or `skipped_*.h5`) marker
already on disk for a given (stem) causes that event's analysis to be
skipped. Delete the marker to force re-analysis.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


# ---------------------------------------------------------------------------
# Helpers shared between modes
# ---------------------------------------------------------------------------

def _run(cmd, log_prefix=""):
    """Stream a subprocess to our stdout, fail loudly on nonzero exit."""
    print(f"{log_prefix}+ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(
            f"{log_prefix}command exited with code {rc}: {' '.join(cmd)}"
        )


def _enumerate_cache_files(cache_dir: str, split: str) -> list[str]:
    """Find all .h5 cache event files under <cache_dir>/<split>."""
    root = os.path.join(cache_dir, split)
    if not os.path.isdir(root):
        sys.exit(f"cache dir not found: {root}")
    out = []
    for r, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".h5"):
                out.append(os.path.join(r, fn))
    return sorted(out)


def _read_rerun_linenos(path: str) -> list[int]:
    out = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(int(s))
    return out


def _read_manifest_by_fileno(path: str) -> dict:
    by_fn = defaultdict(list)
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                fn = int(row["fileno"])
            except (KeyError, ValueError):
                continue
            by_fn[fn].append(row)
    return by_fn


def _analyze_one(stage3pred_path, analyze_dir, model_tag, log_prefix):
    """Call analyze_event.py for a single stage3pred file. The analyzer
    writes either perevent_<stem>.h5 or skipped_<stem>.h5 to
    `analyze_dir`. Skip if either already exists."""
    stem = os.path.splitext(os.path.basename(stage3pred_path))[0]
    # stage3pred_<base>.h5  →  base
    if stem.startswith("stage3pred_"):
        base = stem[len("stage3pred_"):]
    else:
        base = stem
    perevent = os.path.join(analyze_dir, f"perevent_{base}.h5")
    skipped = os.path.join(analyze_dir, f"skipped_{base}.h5")
    if os.path.exists(perevent) or os.path.exists(skipped):
        print(f"{log_prefix}already done: {os.path.basename(stage3pred_path)}")
        return
    # new repo layout first, pre-reorg location as fallback
    candidates = [
        os.path.join(REPO_ROOT, "lartpc", "larformer_analysis",
                     "particle_eval", "analyze_event.py"),
        os.path.join(REPO_ROOT, "lartpc_data_prep",
                     "larformer_particle_analysis", "analyze_event.py"),
    ]
    analyze_script = next((p for p in candidates if os.path.exists(p)), None)
    if analyze_script is None:
        sys.exit(f"missing analyzer; tried: {candidates}")
    _run([
        sys.executable, analyze_script,
        "--stage3pred-h5", stage3pred_path,
        "--output-dir",    analyze_dir,
        "--model-tag",     model_tag,
    ], log_prefix=log_prefix)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_cached(args):
    """Cached-input mode: slice cache events by task_id, run Stage-3
    inference once on the slice, loop the analyzer per event."""
    all_files = _enumerate_cache_files(args.cache_dir, args.split)
    if not all_files:
        sys.exit(f"no cache events found under {args.cache_dir}/{args.split}")
    n_total = len(all_files)
    start = args.task_id * args.stride
    end = min(start + args.stride, n_total)
    if start >= n_total:
        sys.exit(
            f"task-id {args.task_id} stride {args.stride} yields "
            f"start={start} >= n_events={n_total} — out of range"
        )
    chunk = all_files[start:end]
    print(f"[task {args.task_id}] cached-mode chunk=[{start}:{end}) "
          f"({len(chunk)} events of {n_total} total)")

    infer_dir = os.path.join(args.output_dir, "inference", args.tag)
    analyze_dir = os.path.join(args.output_dir, "analysis", args.tag)
    list_dir = os.path.join(args.output_dir, "_tasklists", args.tag)
    for d in (infer_dir, analyze_dir, list_dir):
        os.makedirs(d, exist_ok=True)

    # Per-task input list: skip cache files whose stage3pred_*.h5 already
    # exists so partial reruns don't redo work.
    inputlist_path = os.path.join(list_dir, f"task{args.task_id:06d}.txt")
    n_pending = 0
    n_already = 0
    pending_stage3pred_paths = []
    expected_stage3pred_paths = []
    with open(inputlist_path, "w") as fh:
        for p in chunk:
            stem = os.path.splitext(os.path.basename(p))[0]
            stage3pred = os.path.join(infer_dir, f"stage3pred_{stem}.h5")
            expected_stage3pred_paths.append(stage3pred)
            if os.path.exists(stage3pred):
                n_already += 1
            else:
                fh.write(p + "\n")
                pending_stage3pred_paths.append(stage3pred)
                n_pending += 1
    print(f"[task {args.task_id}] inputlist {inputlist_path}: "
          f"{n_pending} pending, {n_already} already done "
          f"(total {len(chunk)})")

    # ---- Stage-3 inference -------------------------------------------
    if not args.skip_inference and n_pending > 0:
        # Reorganized layout first (tools/larformer/), legacy flat tools/
        # as fallback (same fix as slicer_eval's driver, 2026-08-03).
        _cands = [
            os.path.join(REPO_ROOT, "tools", "larformer",
                         "run_larformer_stage3_inference.py"),
            os.path.join(REPO_ROOT, "tools",
                         "run_larformer_stage3_inference.py"),
        ]
        infer_script = next((c for c in _cands if os.path.exists(c)), None)
        if infer_script is None:
            sys.exit(f"missing inference script; tried: {_cands}")
        _run([
            sys.executable, infer_script,
            "--config",               args.model_config,
            "--weights",              args.model_weights,
            "--output-dir",           infer_dir,
            "--input-mode",           "cached",
            "--cache-dir",            args.cache_dir,
            "--cache-file-list",      inputlist_path,
            "--split",                args.split,
            "--class-prob-threshold", str(args.class_prob_threshold),
        ], log_prefix=f"[task {args.task_id}][infer] ")
    elif args.skip_inference:
        print(f"[task {args.task_id}] --skip-inference set; expecting "
              f"stage3pred_*.h5 to already exist under {infer_dir}")

    # ---- Per-event analysis -----------------------------------------
    if not args.skip_analysis:
        n_missing = 0
        for stage3pred in expected_stage3pred_paths:
            if not os.path.exists(stage3pred):
                sys.stderr.write(
                    f"[task {args.task_id}][analyze] missing inference "
                    f"output: {stage3pred}\n"
                )
                n_missing += 1
                continue
            _analyze_one(
                stage3pred, analyze_dir, args.model_tag,
                log_prefix=f"[task {args.task_id}][analyze] ",
            )
        if n_missing:
            sys.stderr.write(
                f"[task {args.task_id}] WARNING: {n_missing} stage3pred "
                f"files missing for analysis\n"
            )
    else:
        print(f"[task {args.task_id}] --skip-analysis set; done")

    print(f"[task {args.task_id}] DONE  tag={args.tag}  "
          f"chunk=[{start}:{end})  n_events={len(chunk)}")


def run_full_cascade(args):
    """Full-cascade mode: slice filenos by task_id, run cascade end-to-end
    on the matching merged_h5 events, loop the analyzer per event."""
    if args.rerun_lines_file is None or args.manifest_csv is None:
        sys.exit("--input-mode full-cascade requires --rerun-lines-file "
                 "and --manifest-csv")

    linenos = _read_rerun_linenos(args.rerun_lines_file)
    start = args.task_id * args.stride
    end = min(start + args.stride, len(linenos))
    if start >= len(linenos):
        sys.exit(
            f"task-id {args.task_id} stride {args.stride} yields "
            f"start={start} >= n_linenos={len(linenos)} — out of range"
        )
    chunk_filenos = linenos[start:end]
    print(f"[task {args.task_id}] full-cascade chunk=[{start}:{end}) "
          f"filenos={chunk_filenos}")

    by_fn = _read_manifest_by_fileno(args.manifest_csv)
    rows = []
    missing_filenos = []
    for fn in chunk_filenos:
        sub = by_fn.get(fn, [])
        if not sub:
            missing_filenos.append(fn)
        else:
            rows.extend(sub)
    if missing_filenos:
        sys.stderr.write(
            f"[task {args.task_id}] WARNING: no manifest rows for "
            f"{len(missing_filenos)} filenos (e.g. {missing_filenos[:5]})\n"
        )
    if not rows:
        sys.exit(f"[task {args.task_id}] no events to process; nothing to do")

    infer_dir = os.path.join(args.output_dir, "inference", args.tag)
    analyze_dir = os.path.join(args.output_dir, "analysis", args.tag)
    list_dir = os.path.join(args.output_dir, "_tasklists", args.tag)
    for d in (infer_dir, analyze_dir, list_dir):
        os.makedirs(d, exist_ok=True)

    # Build the inputlist (merged_h5 paths), skipping rows whose
    # stage3pred output already exists.
    inputlist_path = os.path.join(list_dir, f"task{args.task_id:06d}.txt")
    n_pending = 0
    n_already = 0
    n_missing_disk = 0
    expected_stage3pred_paths = []
    with open(inputlist_path, "w") as fh:
        for r in rows:
            p = r["merged_h5"]
            if not os.path.exists(p):
                n_missing_disk += 1
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            stage3pred = os.path.join(infer_dir, f"stage3pred_{stem}.h5")
            expected_stage3pred_paths.append(stage3pred)
            if os.path.exists(stage3pred):
                n_already += 1
                continue
            fh.write(p + "\n")
            n_pending += 1
    print(f"[task {args.task_id}] inputlist {inputlist_path}: "
          f"{n_pending} pending, {n_already} already done, "
          f"{n_missing_disk} missing on disk (total {len(rows)})")

    # ---- Stage-3 inference (end-to-end cascade) ----------------------
    if not args.skip_inference and n_pending > 0:
        # Reorganized layout first (tools/larformer/), legacy flat tools/
        # as fallback (same fix as slicer_eval's driver, 2026-08-03).
        _cands = [
            os.path.join(REPO_ROOT, "tools", "larformer",
                         "run_larformer_stage3_inference.py"),
            os.path.join(REPO_ROOT, "tools",
                         "run_larformer_stage3_inference.py"),
        ]
        infer_script = next((c for c in _cands if os.path.exists(c)), None)
        if infer_script is None:
            sys.exit(f"missing inference script; tried: {_cands}")
        _run([
            sys.executable, infer_script,
            "--config",               args.model_config,
            "--weights",              args.model_weights,
            "--output-dir",           infer_dir,
            "--input-mode",           "full-cascade",
            "--input-list",           inputlist_path,
            "--split",                args.split,
            "--class-prob-threshold", str(args.class_prob_threshold),
        ], log_prefix=f"[task {args.task_id}][infer] ")
    elif args.skip_inference:
        print(f"[task {args.task_id}] --skip-inference set; expecting "
              f"stage3pred_*.h5 to already exist under {infer_dir}")

    # ---- Per-event analysis ------------------------------------------
    if not args.skip_analysis:
        n_missing = 0
        for stage3pred in expected_stage3pred_paths:
            if not os.path.exists(stage3pred):
                sys.stderr.write(
                    f"[task {args.task_id}][analyze] missing inference "
                    f"output: {stage3pred}\n"
                )
                n_missing += 1
                continue
            _analyze_one(
                stage3pred, analyze_dir, args.model_tag,
                log_prefix=f"[task {args.task_id}][analyze] ",
            )
        if n_missing:
            sys.stderr.write(
                f"[task {args.task_id}] WARNING: {n_missing} stage3pred "
                f"files missing for analysis\n"
            )
    else:
        print(f"[task {args.task_id}] --skip-analysis set; done")

    print(f"[task {args.task_id}] DONE  tag={args.tag}  "
          f"filenos={chunk_filenos}  n_events={len(rows)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--tag",                  required=True)
    ap.add_argument("--model-config",         required=True)
    ap.add_argument("--model-weights",        required=True)
    ap.add_argument("--model-tag",            required=True)
    ap.add_argument("--output-dir",           required=True)
    ap.add_argument("--input-mode",           default="cached",
                    choices=("cached", "full-cascade"))
    ap.add_argument("--task-id",              required=True, type=int)
    ap.add_argument("--stride",               type=int, default=1)
    ap.add_argument("--class-prob-threshold", type=float, default=0.0)
    ap.add_argument("--split",                default="val")
    # cached-mode args
    ap.add_argument("--cache-dir",            default=None,
                    help="(cached mode) Stage-1+2 cache root.")
    # full-cascade-mode args
    ap.add_argument("--rerun-lines-file",     default=None,
                    help="(full-cascade mode) Stage-0 rerun_lines/<TAG>.txt")
    ap.add_argument("--manifest-csv",         default=None,
                    help="(full-cascade mode) Stage-0 manifest/<TAG>.csv")
    # debug knobs
    ap.add_argument("--skip-inference",       action="store_true")
    ap.add_argument("--skip-analysis",        action="store_true")
    args = ap.parse_args()

    if args.task_id < 0:
        sys.exit(f"task-id must be >= 0; got {args.task_id}")
    if args.stride <= 0:
        sys.exit(f"--stride must be >= 1; got {args.stride}")

    if args.input_mode == "cached":
        if args.cache_dir is None:
            sys.exit("--input-mode cached requires --cache-dir")
        run_cached(args)
    elif args.input_mode == "full-cascade":
        run_full_cascade(args)
    else:  # pragma: no cover
        raise RuntimeError(args.input_mode)


if __name__ == "__main__":
    main()
