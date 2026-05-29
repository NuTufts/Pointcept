"""SLURM-array driver: inference + per-event analysis on the val+test set.

One SLURM task processes a CHUNK of `--stride` consecutive filenos from
a Stage-0 rerun_lines file:
  - selects all manifest rows for those filenos
  - writes a single combined inputlist of merged H5s
  - runs `tools/run_slicer_inference.py` ONCE for the chunk (amortizes
    the model-load + warm-up cost across multiple filenos)
  - loops `analyze_event.py` per surviving event

The chunk this task owns is `rerun_lines[task_id*stride : (task_id+1)*stride]`,
naturally clipped at the end of the list.

Designed to run INSIDE the pointcept_cuml container; called from the
bash submit wrapper. Per-task isolation: each task writes its inference
+ analysis outputs to TAG-keyed subdirectories of OUTPUT_DIR.

Inputs (all required, via CLI):
  --tag               <TAG>
  --rerun-lines-file  Stage 0 rerun_lines/<TAG>.txt (one fileno per line)
  --manifest-csv      Stage 0 manifest/<TAG>.csv
  --model-config      path to LArFormer config
  --model-weights     path to checkpoint .pth
  --model-tag         short identifier embedded in perevent H5 (analysis side)
  --output-dir        per-task outputs go to
                          OUTPUT_DIR/inference/<TAG>/slicerpred_*.h5
                          OUTPUT_DIR/analysis/<TAG>/perevent_*.h5
                          OUTPUT_DIR/_inputlists/<TAG>/task<N>.txt
  --task-id           SLURM_ARRAY_TASK_ID (0-indexed)
  --stride            filenos per task (default 1)
  [--gamma-beam / --gamma-cosmic] forwarded to analyze_event
  [--skip-inference]  skip step A; assume the slicerpred_*.h5 already exist
                      (useful when re-running just the analyzer)
  [--skip-analysis]   skip step B
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _read_rerun_linenos(path):
    out = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(int(s))
    return out


def _read_manifest_by_fileno(path):
    """Return {fileno: list of dicts}, one per (fileno, entry) row."""
    by_fn = defaultdict(list)
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                fn = int(row["fileno"])
            except (KeyError, ValueError):
                continue
            by_fn[fn].append(row)
    return by_fn


def _run(cmd, log_prefix=""):
    """Stream a subprocess to our stdout, fail loudly on nonzero exit."""
    print(f"{log_prefix}+ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(
            f"{log_prefix}command exited with code {rc}: {' '.join(cmd)}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--tag",               required=True)
    ap.add_argument("--rerun-lines-file",  required=True)
    ap.add_argument("--manifest-csv",      required=True)
    ap.add_argument("--model-config",      required=True)
    ap.add_argument("--model-weights",     required=True)
    ap.add_argument("--model-tag",         required=True)
    ap.add_argument("--output-dir",        required=True)
    ap.add_argument("--task-id",           required=True, type=int)
    ap.add_argument("--stride",            type=int, default=1,
                    help="Filenos per task (default 1). The chunk this "
                         "task owns is rerun_lines[task_id*stride : "
                         "(task_id+1)*stride].")
    ap.add_argument("--gamma-beam",        type=float, default=1.0)
    ap.add_argument("--gamma-cosmic",      type=float, default=1.0)
    ap.add_argument("--skip-inference",    action="store_true")
    ap.add_argument("--skip-analysis",     action="store_true")
    args = ap.parse_args()

    # ---- 1. Resolve which filenos this task owns ---------------------
    linenos = _read_rerun_linenos(args.rerun_lines_file)
    if args.task_id < 0:
        sys.exit(f"task-id must be >= 0; got {args.task_id}")
    if args.stride <= 0:
        sys.exit(f"--stride must be >= 1; got {args.stride}")
    start = args.task_id * args.stride
    end   = min(start + args.stride, len(linenos))
    if start >= len(linenos):
        sys.exit(
            f"task-id {args.task_id} stride {args.stride} yields start={start} "
            f">= n_linenos={len(linenos)} — task is out of range for "
            f"{args.rerun_lines_file}"
        )
    chunk_filenos = linenos[start:end]
    print(f"[task {args.task_id}] tag={args.tag}  stride={args.stride}  "
          f"chunk=[{start}:{end})  filenos={chunk_filenos}")

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
            f"{len(missing_filenos)} filenos in chunk "
            f"(e.g. {missing_filenos[:5]}) — manifest_csv may be out "
            f"of sync with rerun-lines file\n"
        )
    if not rows:
        sys.exit(
            f"[task {args.task_id}] no events found for any fileno in chunk; "
            f"nothing to do"
        )
    print(f"[task {args.task_id}] {len(rows)} events across "
          f"{len(chunk_filenos) - len(missing_filenos)} filenos")

    # ---- 2. Set up output dirs + combined inputlist for this task ----
    infer_dir   = os.path.join(args.output_dir, "inference", args.tag)
    analyze_dir = os.path.join(args.output_dir, "analysis",  args.tag)
    list_dir    = os.path.join(args.output_dir, "_inputlists", args.tag)
    for d in (infer_dir, analyze_dir, list_dir):
        os.makedirs(d, exist_ok=True)

    inputlist_path = os.path.join(list_dir, f"task{args.task_id:06d}.txt")
    missing = []
    with open(inputlist_path, "w") as fh:
        for r in rows:
            p = r["merged_h5"]
            if not os.path.exists(p):
                missing.append(p)
                continue
            fh.write(p + "\n")
    if missing:
        sys.stderr.write(
            f"[task {args.task_id}] WARNING: {len(missing)} merged H5 files "
            f"not on disk; first: {missing[0]}\n"
        )
    print(f"[task {args.task_id}] wrote inputlist {inputlist_path} "
          f"({len(rows) - len(missing)} files)")

    # ---- 3. Inference ------------------------------------------------
    if not args.skip_inference:
        infer_script = os.path.join(REPO_ROOT, "tools", "run_slicer_inference.py")
        if not os.path.exists(infer_script):
            sys.exit(f"missing inference script: {infer_script}")
        _run([
            sys.executable, infer_script,
            "--config",     args.model_config,
            "--weights",    args.model_weights,
            "--input-list", inputlist_path,
            "--output-dir", infer_dir,
            "--split",      "val",
        ], log_prefix=f"[task {args.task_id}][infer] ")
    else:
        print(f"[task {args.task_id}] --skip-inference set; expecting "
              f"slicerpred_*.h5 to already exist under {infer_dir}")

    # ---- 4. Per-event analysis ---------------------------------------
    if not args.skip_analysis:
        analyze_script = os.path.join(
            REPO_ROOT, "lartpc_data_prep", "larformer_analysis",
            "analyze_event.py",
        )
        if not os.path.exists(analyze_script):
            sys.exit(f"missing analyzer: {analyze_script}")
        for r in rows:
            merged_p  = r["merged_h5"]
            flash_p   = r["flashinfo_h5"]
            entry     = int(r["entry"])
            # Inference writes `slicerpred_<merged_basename>.h5`.
            base = os.path.basename(merged_p)
            stem, _ = os.path.splitext(base)
            slicerpred_p = os.path.join(infer_dir, f"slicerpred_{stem}.h5")
            if not os.path.exists(slicerpred_p):
                sys.stderr.write(
                    f"[task {args.task_id}][analyze] missing inference "
                    f"output for entry {entry}: {slicerpred_p}\n"
                )
                continue
            if not os.path.exists(flash_p):
                sys.stderr.write(
                    f"[task {args.task_id}][analyze] missing flashinfo for "
                    f"entry {entry}: {flash_p}\n"
                )
                continue
            _run([
                sys.executable, analyze_script,
                "--merged-h5",    merged_p,
                "--flashinfo-h5", flash_p,
                "--inference-h5", slicerpred_p,
                "--output-h5",    analyze_dir + os.sep,   # auto-derives filename
                "--model-tag",    args.model_tag,
                "--gamma-beam",   str(args.gamma_beam),
                "--gamma-cosmic", str(args.gamma_cosmic),
            ], log_prefix=f"[task {args.task_id}][analyze entry={entry}] ")
    else:
        print(f"[task {args.task_id}] --skip-analysis set; done")

    print(f"[task {args.task_id}] DONE  tag={args.tag}  "
          f"filenos={chunk_filenos}  n_events={len(rows)}")


if __name__ == "__main__":
    main()
