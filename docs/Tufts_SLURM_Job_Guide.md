# Submitting & Monitoring SLURM Jobs on the Tufts Cluster

A practical guide for **running batch jobs on Tufts**, written so an agent (or a
new person) can submit a job array, watch it, and collect results without guessing
partitions, modules, or conventions. Grounded in the submit scripts already in
these repos:

- `ub_showerorigin_reco/scripts/submit_showerorigin_reco.sh` (apptainer, array, 3-level output hash)
- `pointcept/lartpc/larformer_analysis/particle_eval/slurm/submit_valtest.sh` (self-resubmitting array, GPU)
- `gen2ntuple/tufts_submit_ntuple_job_example.sh` (singularity, CPU array)

---

## 0. The two clusters (read this first)

Tufts HPC has **two login environments**, and a job must be submitted from the
login node that can see its target partition:

| Cluster | Login node | Reach |
|---------|-----------|-------|
| **New** (default) | `login.cluster.tufts.edu` (the usual `ssh tufts`) | `batch`, `preempt`, `gpu` (incl. **A100**) |
| **Old** | `login.pax.tufts.edu` | `batch`, `preempt`, `gpu` (A100/H200/H100/L40S), `wongjiradlab`, plus the lantern CVMFS containers |

> **Verified June 2025 from `login-p01.pax.tufts.edu`:** the `gpu` partition
> (A100/H200/H100/L40S) is fully visible and submittable from this old-cluster
> login node — `sbatch --partition=gpu --gres=gpu:a100:1` works here, and
> `sbatch`/`squeue`/`sacct` are all available. So an agent on this node can submit
> GPU jobs directly. The **`wongjiradlab` partition currently has 0 nodes** (empty)
> and there is **no `p100` gres anywhere** right now — requesting CPUs there fails
> with "More processors requested than permitted." Use `gpu`/`a100`. If the
> wongjiradlab P100s return, they become the priority fallback again.

---

## 1. Choosing a partition

| Need | Partition / flags |
|------|-------------------|
| CPU-only (ntuple making, analysis, Stage-A convert) | `--partition=batch` (or `preempt` for backfill) |
| **GPU, preferred** | `--partition=gpu --gres=gpu:a100:1` (most A100 availability; submittable from `login.pax.tufts.edu`) |
| **GPU, priority fallback** | `--partition=wongjiradlab --gres=gpu:p100:1` *(our priority P100s — but currently 0 nodes; see the verified note above)* |

Rules of thumb: request **1 GPU** (`--gres=gpu:a100:1`) for cascade/inference; the
LArFormer cascade and shower-origin models need CUDA (CPU mode does not work for
inference). Keep `--cpus-per-task` modest (1-8) and `--mem-per-cpu` 4000-8000.

---

## 2. Containers & modules

Pick the module that matches the container you call. **Submit on a bare node and
call the container from inside the job** (do not `module load` *inside* the
container):

| Container | Module on bare node | Used for |
|-----------|---------------------|----------|
| `…/larbys-container/pointcept_cuml.sif` | `module load apptainer/1.4.0` (or `1.2.4-suid`) | Pointcept / LArFormer / ntuple reading |
| `…/singularity_minkowskiengine_u20.04…sif` | `module load singularity/3.5.3` (or `4.3.x`) | gen2ntuple maker (`ubdl` env) |
| lantern CVMFS container | `module load apptainer/1.2.4-suid` + `cvmfs_config probe uboone.opensciencegrid.org` | Step-1 SSNet/LArMatch (old cluster only) |

Standard bind: `--bind /cluster:/cluster` (or the finer
`/cluster/tufts/wongjiradlabnu:…,/cluster/tufts/wongjiradlab:…`). Inside the
pointcept container, `source …/ubdl/setenv_pointcept_container.sh` to get ROOT/CUDA
on the path.

---

## 3. The job-array idiom (the standard pattern here)

Every batch driver in these repos maps a SLURM **array task** to a contiguous
block of input-list lines:

```
lineno = OFFSET + STRIDE * SLURM_ARRAY_TASK_ID + i      # i = 1 .. STRIDE
```

So `--array=0-99` with `STRIDE=100` processes lines 1..10000 of the input list,
100 lines per task. Pick `--array` so that `OFFSET + STRIDE*max_task_id` covers the
list. Outputs are written to a **3-level hashed directory**
`OUTPUT_DIR/<lineno/1000>/<lineno/100>/` to keep any one directory under ~100 files;
status checkers reconstruct the path from the line number.

### Minimal CPU array template

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/myjob.%A_%a.log
#SBATCH --error=logs/myjob.%A_%a.err
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-99

mkdir -p logs
module load apptainer/1.4.0
CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
apptainer exec --bind /cluster:/cluster ${CONTAINER} bash -c "
  source /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh >/dev/null 2>&1
  python3 my_worker.py --task-id ${SLURM_ARRAY_TASK_ID} --stride 100
"
```

### GPU array template (A100)

Change the SBATCH header to:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1-00:00:00
```

For the **wongjiradlab P100 fallback**, use `--partition=wongjiradlab
--gres=gpu:p100:1` and submit from `login.pax.tufts.edu`.

> Several drivers here (e.g. `run_larformer_wconfig.sh`, `submit_valtest.sh`)
> **self-resubmit**: you `source`/`sbatch` them with a config and they re-exec
> themselves under `sbatch`, reading the `#SBATCH` knobs from the config. Check the
> script header before adding your own `sbatch` wrapper.

---

## 4. Submitting

```bash
sbatch my_submit.sh                 # plain array
sbatch --array=0-4 my_submit.sh     # override array at submit time
# config-driven drivers in these repos:
sbatch lartpc/larformer_analysis/particle_eval/slurm/submit_valtest.sh <conf>
```

For a **capped first pass**, submit a small array (`--array=0-4`) before scaling up.

---

## 5. Monitoring

```bash
squeue -u $USER                       # your queued/running jobs
squeue -u $USER -t RUNNING            # only running
squeue -j <JOBID>                     # one job / array
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode   # finished-job accounting
scontrol show job <JOBID>             # full detail (why pending, node, etc.)
scancel <JOBID>                       # cancel; scancel <JOBID>_<taskid> for one array task
```

- Array states: each task is `<JOBID>_<taskid>`. `PD` = pending, `R` = running,
  `CG` = completing. A pending reason of `(Priority)`/`(Resources)` means it's
  waiting for a slot; `(QOSMaxJobsPerUserLimit)` means you hit a cap.
- **Watch progress** by tailing the per-task logs (`logs/myjob.<A>_<a>.log`) and
  by counting output files in the hashed `OUTPUT_DIR`.
- A repo convention for completion checking is a `check_status.sh <tag>` script
  (see `ub_showerorigin_reco/scripts/check_status.sh`) that reconstructs expected
  output paths from the input list and reports done/missing, optionally writing a
  `--rerun` list of failed line numbers to resubmit as a new `--array`.

### Agent loop for "submit → wait → collect"

1. Submit (or hand the script to the user if it targets `wongjiradlab`).
2. Poll `squeue -u $USER -j <JOBID>` periodically until empty; for long jobs poll
   at a cadence matched to the job time (minutes, not seconds).
3. Run the completion check (`check_status.sh` or count files in `OUTPUT_DIR`).
4. Resubmit the rerun list if any tasks failed; then proceed to analysis.

---

## 6. Common failure modes

| Symptom | Likely cause / fix |
|---------|--------------------|
| Job pending forever on `gpu` | A100s busy → fall back to `wongjiradlab` P100 (old cluster) |
| `Invalid partition` at submit | wrong login node — `wongjiradlab` only visible from `login.pax.tufts.edu` |
| `CUDA`/device errors in logs | landed on a CPU partition, or container lacks GPU bind; ensure `--gres` is set |
| `apptainer: command not found` in job | forgot `module load apptainer/...` on the bare node |
| ROOT `ModuleNotFoundError` inside container | forgot `source …/setenv_pointcept_container.sh` |
| Out-of-memory (`MaxRSS` near limit in `sacct`) | raise `--mem-per-cpu` or lower `STRIDE` |
| lantern CVMFS container won't start | run on old cluster + `cvmfs_config probe uboone.opensciencegrid.org` first |
