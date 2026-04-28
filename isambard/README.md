# Isambard transfer scripts

Scripts for moving LArTPC training/validation datasets from the Tufts cluster
to the Isambard AI cluster (`u6jo.aip2.isambard`, project area `/projects/u6jo`).

The transfer is run as a SLURM job on Tufts and an extraction script on
Isambard. Tarballs are zstd-compressed and sha256-checksummed, and the Tufts
job is fully resumable and can chain itself across the 12-hour Isambard
certificate window.

## Files in this directory

| File | Purpose |
|---|---|
| `tar_and_transfer.py` | Worker: groups a file list by leaf directory, builds one `.tar.zst` per leaf, sha256s it, rsyncs it to Isambard, and updates a resumable manifest. |
| `submit_transfer.sh` | SLURM wrapper for the `batch` partition. Optional `--chain` flag self-submits a successor job until the manifest is complete. |
| `unpack_on_isambard.sh` | Run on Isambard: verifies all sha256s and extracts every tarball into `/projects/u6jo/datasets/`, preserving the original directory structure. |
| `clifton` | Vendor binary used to refresh the Isambard SSH certificate (~12h validity). |

## Datasets queued for transfer

The file lists below are the canonical inputs to `submit_transfer.sh`. They
are large (~10⁵–10⁶ entries each); the scripts handle that by batching files
into one tarball per leaf directory.

- **`pretrain-sonata-v7-extbnb-larmatch`** — larmatch-deghosted cosmic data:
  `/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt`
- **`pretrain-sonata-v6-extbnb`** — full no-larmatch cosmic data:
  `/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/extbnb_g1_g2_100_filenames_only.txt`
- **lartpc v6 simulated** — full simulated data set used for the v6 model
  (no larmatch info for deghosting):
  `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_trainsplit.txt`

## Source/destination layout

The scripts strip a configurable **source prefix** from each absolute path in
the file list. Whatever remains becomes both the path inside the tarball and
the location under the Isambard datasets root.

Example with `--source-prefix /cluster/tufts/wongjiradlab/larbys/data/`:

```
Tufts (source)
  /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000/*.h5

Tufts staging (tarball lives here after creation)
  /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000.tar.zst

Isambard staging (after rsync)
  /projects/u6jo/staging/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000.tar.zst

Isambard datasets (after unpack)
  /projects/u6jo/datasets/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000/*.h5
```

The 2-level subdirectory structure produced by SimChTripletLabelMaker (e.g.
`000/000`, `000/001`, …, `001/010`, …) becomes one tarball per leaf — typically
a few hundred to a few thousand `.h5` files per tarball, and ~O(1k) tarballs
per dataset.

## How the Tufts side works

`tar_and_transfer.py`:

1. Reads the file list and groups paths by leaf directory.
2. For each leaf directory not already in the manifest:
   a. Writes a temp file list of the relative paths in that group.
   b. Runs `tar -I 'zstd -3 -T<cpus>' -C <source_prefix> -cf <tarball> -T <list>`.
      Only files in the input list are included — extras in the directory are ignored.
   c. Computes sha256 of the tarball and updates `<list_stem>.sha256sums.txt`
      (standard `sha256sum` format, sorted, atomic rewrite).
   d. `rsync -a --partial --mkpath <tarball> u6jo.aip2.isambard:/projects/u6jo/staging/<rel>`.
   e. rsyncs the updated checksums file alongside.
   f. Appends the leaf-dir to `<list_stem>.manifest.txt`.
3. After the loop, if every leaf-dir is now in the manifest and no failures
   occurred, writes `<list_stem>.complete` (used by the chain wrapper to
   terminate).

State files (all under `<staging-dir>/`):

| File | Role |
|---|---|
| `<stem>.manifest.txt` | One line per completed leaf-dir. Re-runs skip these. |
| `<stem>.sha256sums.txt` | `<digest>  <tarball_rel>` per line, in `sha256sum -c` format. |
| `<stem>.complete` | Sentinel — present iff all work for this list is done. |
| `<stem>.chain_count` | Internal counter used by `--chain` (deleted on completion). |
| `<stem>.stop_chain` | If you `touch` this, the next end-of-job refuses to chain. |

`<stem>` is the file list basename without extension, so multiple datasets
can share one staging directory without clashing.

## How the Isambard side works

`unpack_on_isambard.sh`:

1. Globs every `*.sha256sums.txt` under the staging dir.
2. Runs `sha256sum -c` against each — aborts on any mismatch.
3. Extracts each tarball with `tar -I zstd -xf <tarball> -C <dest_root>`.
   Because tarballs were created with relative paths, this lands files at
   `<dest_root>/<original_rel_path>`.
4. Records each extracted tarball in `<staging>/extracted.txt` so re-runs skip them.

## Workflow

### 0. One-time: log in to Isambard once

```bash
ssh u6jo.aip2.isambard
```

The SSH config / cert dance is handled by the vendor `clifton` binary in this
folder. The cert is good for ~12 hours.

### 1. Refresh the cert

On the Tufts login node:

```bash
cd /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/isambard
./clifton ...           # follow vendor instructions
```

Re-run `clifton` again at any point during a chained job (roughly daily) so
the next chained successor inherits a valid cert when it starts.

### 2. Smoke-test on a few directories

Before launching a multi-day chain, run a tiny submission to exercise tar,
sha256, rsync, and the chain submit path:

```bash
sbatch submit_transfer.sh --chain --max-chain 1 \
  /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
  /cluster/tufts/wongjiradlab/larbys/data/ \
  --limit 2
```

`--limit 2` stops `tar_and_transfer.py` after two tarballs. `--max-chain 1`
caps the chain to a single job. Inspect the SLURM logs and the staging
directory contents before scaling up.

### 3. Submit the full chain

```bash
sbatch submit_transfer.sh --chain --max-chain 30 \
  /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
  /cluster/tufts/wongjiradlab/larbys/data/
```

What happens:

- The job starts, ssh-probes Isambard, then runs `tar_and_transfer.py` for up
  to 11h30 (matching `--time=11:30:00`).
- Near its end the job submits a successor with
  `sbatch --dependency=afterany:<this_job_id>` and exits.
- The successor sits in the queue and starts within seconds of the predecessor
  finishing, **using whatever clifton cert is current at that moment**.
- This repeats until either `<stem>.complete` is written or `--max-chain` is
  hit.

### 4. Monitor

```bash
squeue -u $USER                                       # see the chain in the queue
tail -f isambard/logs/isambard_xfer-<jobid>.out       # watch progress
ls /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging/
wc -l <staging>/<stem>.manifest.txt                   # count completed leafs
```

### 5. Stop the chain (optional)

```bash
touch <staging>/<list_stem>.stop_chain
```

The currently running job finishes its work then declines to submit a
successor. The active job is **not** killed.

To kill the active job too, `scancel <jobid>` and the dependency chain
collapses (since `afterany` waits for a terminal state, the queued successor
will then run — `scancel` it as well if you want to stop completely).

### 6. Unpack on Isambard

```bash
ssh u6jo.aip2.isambard
cd /projects/u6jo
bash <path-to-this-repo>/isambard/unpack_on_isambard.sh /projects/u6jo/staging /projects/u6jo/datasets
```

Or simply:

```bash
bash unpack_on_isambard.sh    # uses defaults: staging=/projects/u6jo/staging, dest=/projects/u6jo/datasets
```

The script verifies every `*.sha256sums.txt` first; if any tarball is corrupt
or truncated, it aborts before extracting anything.

### 7. Clean up Tufts staging

Once unpack on Isambard succeeds for a dataset, the local tarballs under
`/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging/` are
safe to delete. The manifest, checksums, and `.complete` sentinel are also
disposable at that point.

## Tunables

`submit_transfer.sh` flags (everything before the file list):

| Flag | Default | Meaning |
|---|---|---|
| `--chain` | off | Submit a chained successor at end of job. |
| `--max-chain N` | 20 | Cap on chain length. |

`tar_and_transfer.py` flags (everything after the source prefix is forwarded
through):

| Flag | Default | Meaning |
|---|---|---|
| `--zstd-level` | 3 | zstd compression level. h5 files are mostly incompressible — keep this low. |
| `--zstd-threads` | `$SLURM_CPUS_PER_TASK` (8) | `zstd -T` parameter. |
| `--limit N` | 0 | Stop after N tarballs (testing). When set, suppresses the `.complete` sentinel. |
| `--dry-run` | off | Print actions, write nothing. |
| `--skip-rsync` | off | Build tarballs and checksums locally only. |
| `--manifest PATH` | `<staging>/<stem>.manifest.txt` | Override manifest path. |
| `--checksums PATH` | `<staging>/<stem>.sha256sums.txt` | Override checksums path. |

Environment variables read by `submit_transfer.sh`:

| Var | Default |
|---|---|
| `ISAMBARD_STAGING_DIR` | `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging` |
| `ISAMBARD_REMOTE_HOST` | `u6jo.aip2.isambard` |
| `ISAMBARD_REMOTE_STAGING_DIR` | `/projects/u6jo/staging` |
| `PYTHON` | `python3` |

SLURM resources are set in the `#SBATCH` block at the top of
`submit_transfer.sh`:

- partition `batch`, time `11:30:00`, 1 node, 8 CPUs, 8 GB RAM.
- The 11h30 wall is intentionally <12h to stay inside the cert window with a
  margin for the queue + ssh probe.

## Failure modes and recovery

| Symptom | Cause | Action |
|---|---|---|
| Job exits 2 immediately with `ssh to u6jo.aip2.isambard failed` | Expired clifton cert. | Re-run `clifton`, resubmit. The chain breaks at this point on purpose. |
| Some `[i/N] FAILED rsync ...` lines mid-job; chain continues | Transient network or remote disk issue. | Next chained job retries that leaf-dir (it's not in the manifest yet). |
| `sha256sum -c` fails on Isambard | Corrupt or truncated tarball. | On Tufts, delete the offending tarball + its line in `<stem>.sha256sums.txt` and the leaf-dir's line in `<stem>.manifest.txt`, then resubmit. |
| Successor never starts | Predecessor was killed externally before reaching the chain block, or `sbatch` itself failed. | Resubmit manually. `<stem>.manifest.txt` makes this idempotent. |
| `<stem>.complete` exists but you want to re-transfer | Manual override. | Delete `<stem>.complete` (and possibly `<stem>.manifest.txt` for a full re-do). |
