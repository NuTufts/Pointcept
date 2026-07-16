#!/usr/bin/env python3
"""
Probe-sweep launcher (RUNS AT TUFTS): one linear-probe SLURM job per
(run, snapshot) pair that hasn't been probed yet.

Scans <repo>/sonata/p05/*/snapshot/snapshot_iter*_img*.pth (synced from
Isambard by sync_from_isambard.sh), picks the matching probe config per run
(see PROBE_MAP), and submits submit_probe_tufts.sbatch with the frozen probe
budget. Idempotent: a .submitted marker in each probe dir prevents
duplicates; rerun after each sync to pick up new snapshots.

Usage:
  python3 launch_probe_sweep.py --repo <REPO> [--dry-run] [--max-submit N]
      [--runs P05B.1 P05C.5 ...] [--budget-opts "epoch=1 eval_epoch=1"]

The default budget-opts are a PLACEHOLDER until the calibration step in
ORCHESTRATION.md freezes them — pass the frozen values explicitly.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

# run_id prefix -> (probe config, extra --options tokens)
# ORDER MATTERS: first matching prefix wins.
PROBE_MAP = [
    ("P05B.4", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-sumcharge-tufts.py", []),
    # P05B.5/P05B.6 (asinh variants): uncomment once the p05b5-asinh-input
    # branch is merged and the asinh probe config exists.
    # ("P05B.5", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-asinh-tufts.py", []),
    # ("P05B.6", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-asinh-tufts.py", []),
    ("P05C.1", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-tufts.py",
     ["model.backbone.head_num_prototypes=2048"]),
    ("P05C.3", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-tufts.py",
     ["model.backbone.head_num_prototypes=8192"]),
    ("P05", "configs/lartpc/p05/linearprobe-sonata-p05-mc-noghost-tufts.py", []),
]

SNAP_RE = re.compile(r"snapshot_iter(\d+)_img(\d+)\.pth$")


def probe_config_for(run_id):
    for prefix, cfg, extra in PROBE_MAP:
        if run_id.startswith(prefix):
            return cfg, list(extra)
    raise KeyError(run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get(
        "LOCAL_REPO",
        "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept"))
    ap.add_argument("--runs", nargs="*", default=None,
                    help="restrict to these run_ids (default: all found)")
    ap.add_argument("--budget-opts", default="epoch=1 eval_epoch=1",
                    help="frozen probe budget as --options tokens "
                         "(set from the calibration step!)")
    ap.add_argument("--max-submit", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sbatch_script = os.path.join(here, "submit_probe_tufts.sbatch")
    budget_tokens = args.budget_opts.split()

    pending, skipped = [], 0
    for snap_dir in sorted(glob.glob(os.path.join(args.repo, "sonata/p05/*/snapshot"))):
        run_id = os.path.basename(os.path.dirname(snap_dir))
        if args.runs and run_id not in args.runs:
            continue
        try:
            cfg, extra = probe_config_for(run_id)
        except KeyError:
            print(f"WARN no probe mapping for {run_id}; skipping "
                  f"(add it to PROBE_MAP)", file=sys.stderr)
            continue
        if not os.path.isfile(os.path.join(args.repo, cfg)):
            print(f"WARN probe config missing for {run_id}: {cfg}", file=sys.stderr)
            continue
        for snap in sorted(glob.glob(os.path.join(snap_dir, "snapshot_*.pth"))):
            m = SNAP_RE.search(os.path.basename(snap))
            if not m:
                continue
            img = int(m.group(2))
            save_rel = f"exp/probes/{run_id}/img{img:09d}"
            probe_dir = os.path.join(args.repo, save_rel)
            if os.path.exists(os.path.join(probe_dir, ".submitted")):
                skipped += 1
                continue
            pending.append((run_id, img, cfg, snap, save_rel, extra))

    pending.sort(key=lambda t: (t[0], t[1]))
    print(f"{len(pending)} pending probes ({skipped} already submitted)")
    n = 0
    for run_id, img, cfg, snap, save_rel, extra in pending:
        if n >= args.max_submit:
            print(f"reached --max-submit={args.max_submit}; rerun for the rest")
            break
        cmd = ["sbatch", "--parsable",
               f"--job-name=probe_{run_id.split('-')[0]}_{img//1000}k",
               sbatch_script, cfg, snap, save_rel] + budget_tokens + extra
        if args.dry_run:
            print("DRY:", " ".join(cmd))
            continue
        probe_dir = os.path.join(args.repo, save_rel)
        os.makedirs(probe_dir, exist_ok=True)
        jobid = subprocess.check_output(cmd, cwd=here, text=True).strip()
        with open(os.path.join(probe_dir, ".submitted"), "w") as f:
            f.write(f"{jobid}\n{snap}\n{cfg}\n{' '.join(budget_tokens + extra)}\n")
        print(f"submitted {run_id} img={img:,}: job {jobid}")
        n += 1


if __name__ == "__main__":
    main()
