# Isambard transfer scripts

Scripts for moving LArTPC training/validation datasets from the Tufts cluster
to the Isambard AI cluster (`u6jo.aip2.isambard`, project area `/projects/u6jo`).

The workflow is split into two phases:

1. **Build** (Tufts side, no remote required): convert each dataset into a
   set of zstd-compressed SquashFS shards (one shard per leaf directory) on
   local staging.
2. **Transfer** (Tufts side, requires Isambard cert): rsync the shards plus
   their sha256 file to Isambard.

Both phases run as SLURM jobs and are fully resumable. Decoupling them lets
the build run during Isambard downtime; transfer fires when the remote is
available. On Isambard, shards are **not extracted** — they are mounted
directly at training time via Apptainer's image-src bind, which avoids the
inode pressure of 300k–500k loose files and skips the 5–10 TB extract step.

## Files in this directory

| File | Phase | Purpose |
|---|---|---|
| `build_squashfs.py` | build | Worker: groups a file list by leaf directory, builds one zstd `.sqsh` per leaf via a hardlink staging tree, sha256s it, updates manifests. |
| `submit_build.sh` | build | SLURM wrapper for `build_squashfs.py`. Optional `--chain` self-resubmits until complete. |
| `transfer_shards.py` | transfer | Worker: rsyncs each shard listed in the build's checksums file to Isambard staging. |
| `submit_transfer_shards.sh` | transfer | SLURM wrapper for `transfer_shards.py`. Cert probe + optional `--chain`. |
| `verify_shards_on_isambard.sh` | verify | Run on Isambard: `sha256sum -c` every shard listed in `<staging>/*.sha256sums.txt`. No extraction. |
| `clifton` | transfer | Vendor binary used to refresh the Isambard SSH certificate (~12h validity). |

> Deprecated (kept for reference, no longer documented here): `tar_and_transfer.py`,
> `submit_transfer.sh`, `unpack_on_isambard.sh`. The tarball-then-extract pipeline
> they implement has been superseded by the SquashFS workflow above.

## Datasets queued for transfer

The file lists below are the canonical inputs to `submit_build.sh` /
`submit_transfer_shards.sh`. They are large (~10⁵–10⁶ entries each); the
scripts handle that by sharding files into one `.sqsh` per leaf directory.

- **`pretrain-sonata-v7-extbnb-larmatch`** — larmatch-deghosted cosmic data:
  `/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt`
- **`pretrain-sonata-v6-extbnb`** — full no-larmatch cosmic data:
  `/cluster/tufts/wongjiradlab/hmcgui01/mphys/Pointcept/lartpc_data_prep/extbnb_g1_g2_100_filenames_only.txt`
- **lartpc v6 simulated** — full simulated data set used for the v6 model
  (no larmatch info for deghosting):
  `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_trainsplit.txt`

## Source / destination layout

The scripts strip a configurable **source prefix** from each absolute path in
the file list. Whatever remains becomes both the path inside the shard and
the location under the Isambard staging root.

Example with `--source-prefix /cluster/tufts/wongjiradlab/larbys/data/`:

```
Tufts (source files)
  /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000/*.h5

Tufts staging (sqsh shard lives here after build)
  /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000.sqsh

Isambard staging (after rsync)
  /projects/u6jo/staging/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000.sqsh
```

The 2-level subdirectory structure produced by `SimChTripletLabelMaker`
(e.g. `000/000`, `000/001`, …) becomes one shard per leaf — typically a few
hundred to a few thousand `.h5` files per shard, and ~O(1k) shards per
dataset.

At runtime on Isambard, each shard is mounted directly:

```
apptainer exec --bind /projects/u6jo/staging/<rel>.sqsh:/data:image-src=/,ro \
  <pointcept.sif> python tools/train.py ...
```

Apptainer ships its own `squashfuse_ll` internally, so no `squashfuse`
package is required on Isambard.

## How the build phase works

`build_squashfs.py`:

1. Reads the file list and groups paths by leaf directory.
2. For each leaf directory not in the manifest:
   a. Cleans the per-list staging scratch dir (`.<stem>.stage_tmp`).
   b. **Hardlinks** each listed file into a tree mirroring the source
      layout (no data copy; same-filesystem only — see EXDEV in failure modes).
   c. Runs `mksquashfs <stage> <shard>.sqsh -comp zstd -noI -noD ...`. The
      `-noI -noD` flags skip metadata + data-block compression; `.h5` files
      are already compressed so this avoids wasted CPU and yields shard
      sizes ≈ source sizes.
   d. Computes sha256 of the shard and updates `<stem>.sha256sums.txt`
      (sha256sum format, sorted, atomic rewrite).
   e. Removes the staging scratch dir.
   f. Appends the leaf-dir to `<stem>.manifest.txt`.
3. After the loop, if every leaf-dir is now in the manifest and no failures
   occurred, writes `<stem>.complete` (used by the chain wrapper to terminate).

## How the transfer phase works

`transfer_shards.py`:

1. Reads the build's `<stem>.sha256sums.txt` as the authoritative shard list.
2. For each shard not in `<stem>.transfer_manifest.txt`:
   a. `rsync -a --partial --mkpath <local>.sqsh u6jo.aip2.isambard:/projects/u6jo/staging/<rel>.sqsh`.
   b. Append the shard rel-path to the transfer manifest.
3. After the loop the checksums file is rsynced alongside (always, when any
   shards moved) so the remote can verify them.
4. If every shard is in the transfer manifest and no failures occurred,
   writes `<stem>.transfer_complete`.

The build manifest and transfer manifest are independent: rebuilding a
shard locally doesn't reset its transfer state, and re-transferring doesn't
trigger a rebuild.

## State files (under `<staging-dir>/`)

| File | Phase | Role |
|---|---|---|
| `<stem>.manifest.txt` | build | One leaf-dir per line. Re-runs of `build_squashfs.py` skip these. |
| `<stem>.sha256sums.txt` | build | `<digest>  <shard_rel>` per line, in `sha256sum -c` format. |
| `<stem>.complete` | build | Sentinel — present iff the build phase is fully done. |
| `<stem>.transfer_manifest.txt` | transfer | One shard rel-path per line. Re-runs of `transfer_shards.py` skip these. |
| `<stem>.transfer_complete` | transfer | Sentinel — present iff the transfer phase is fully done. |
| `<stem>.chain_count` | both | Internal counter for the `--chain` wrapper. |
| `<stem>.transfer_chain_count` | transfer | Internal counter for the transfer chain. |
| `<stem>.stop_chain` | both | If you `touch` this, the next end-of-job in either phase refuses to chain. |

`<stem>` is the file list basename without extension, so multiple datasets
can share one staging directory without clashing.

## Workflow

### 0. One-time: log in to Isambard once

```bash
ssh u6jo.aip2.isambard
```

The SSH config / cert dance is handled by the vendor `clifton` binary in
this folder. The cert is good for ~12 hours.

### 1. Build the shards on Tufts (no Isambard cert required)

```bash
cd /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/isambard

# Smoke-test on a few shards first
sbatch submit_build.sh \
  /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
  /cluster/tufts/wongjiradlab/larbys/data/ \
  --limit 3
```

After confirming the smoke-test produced sane shards (manifest + checksums
populated, `.sqsh` files listable with `unsquashfs -l`), launch the full
chained build:

```bash
sbatch submit_build.sh --chain --max-chain 5 \
  /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
  /cluster/tufts/wongjiradlab/larbys/data/
```

### 2. Refresh the Isambard cert

On the Tufts login node:

```bash
cd /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/isambard
./clifton ...           # follow vendor instructions
```

Re-run `clifton` again at any point during a chained transfer (roughly
daily) so the next chained successor inherits a valid cert when it starts.

### 3. Transfer the shards to Isambard

```bash
sbatch submit_transfer_shards.sh --chain --max-chain 30 \
  /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt
```

What happens:

- The job ssh-probes Isambard, then runs `transfer_shards.py` for up to
  11h30 (matching `--time=11:30:00`).
- Near its end the job submits a successor with
  `sbatch --dependency=afterany:<this_job_id>` and exits.
- The successor sits in the queue and starts within seconds of the
  predecessor finishing, **using whatever clifton cert is current at that moment**.
- This repeats until either `<stem>.transfer_complete` is written or
  `--max-chain` is hit.

### 4. Monitor

```bash
squeue -u $USER                                       # see the chain in the queue
tail -f isambard/logs/sqsh_build-<jobid>.out          # build progress
tail -f isambard/logs/sqsh_xfer-<jobid>.out           # transfer progress
wc -l <staging>/<stem>.manifest.txt                   # built shards
wc -l <staging>/<stem>.transfer_manifest.txt          # transferred shards
```

### 5. Stop the chain (optional)

```bash
touch <staging>/<stem>.stop_chain
```

The currently running job finishes its work then declines to submit a
successor. The active job is **not** killed. The same stop file halts both
build and transfer chains.

### 6. Verify on Isambard

Verification is read-only — no extraction. Copy
`verify_shards_on_isambard.sh` to Isambard (or run from a shared filesystem
if your project area mounts the Tufts script copy) and point it at the
staging dir:

```bash
ssh u6jo.aip2.isambard
bash <path>/verify_shards_on_isambard.sh /projects/u6jo/staging
```

The script discovers every `<staging>/*.sha256sums.txt` and runs
`sha256sum -c` against each, reporting per-dataset pass/fail counts. Exit
codes: `0` = all OK, `2` = at least one shard failed verification, `1` =
setup error (missing staging dir, no checksums files).

With 5–10 TB of shards this is read-IO bound and can take several hours;
run it in screen/tmux. If a shard fails, see Failure modes below.

### 7. Use the dataset

In your training launcher on Isambard, mount each shard via Apptainer:

```bash
apptainer exec --nv \
  --bind /projects/u6jo/staging/<dataset>.sqsh:/data:image-src=/,ro \
  <pointcept.sif> \
  python tools/train.py --config-file configs/lartpc/<config>.py ...
```

The `LArTPCDataset` data loader reads through the mountpoint with no code
changes — its `data_list_file` should reference paths under `/data/...`
(or whatever bind mountpoint you choose). For datasets split across
multiple shards, use multiple `--bind` flags or merge the mountpoints with
`mergerfs` / overlayfs.

### 8. Clean up

Once verification on Isambard succeeds, the local `.sqsh` shards in the
Tufts staging dir are safe to delete. The manifests, checksums, and
`.complete` / `.transfer_complete` sentinels are also disposable at that
point.

## Tunables

`submit_build.sh` / `submit_transfer_shards.sh` flags (everything before
the file list):

| Flag | Default | Meaning |
|---|---|---|
| `--chain` | off | Submit a chained successor at end of job. |
| `--max-chain N` | 10 (build), 20 (xfer) | Cap on chain length. |

`build_squashfs.py` flags (forwarded through `submit_build.sh`):

| Flag | Default | Meaning |
|---|---|---|
| `--processors N` | `$SLURM_CPUS_PER_TASK` (8) | `mksquashfs -processors`. |
| `--zstd-level N` | 1 | zstd metadata compression level. h5 data is already compressed and `-noD` skips data compression entirely, so a low level is fine. |
| `--block-size SIZE` | 128K | `mksquashfs -b` block size. |
| `--no-hardlinks` | off | Copy files into the staging tree instead of hardlinking. Use only when source and staging are on different filesystems (slower; doubles I/O). |
| `--limit N` | 0 | Stop after N shards. When set, suppresses the `.complete` sentinel. |
| `--dry-run` | off | Print actions, write nothing. |

`transfer_shards.py` flags (forwarded through `submit_transfer_shards.sh`):

| Flag | Default | Meaning |
|---|---|---|
| `--limit N` | 0 | Stop after N transfers. |
| `--dry-run` | off | Print rsync commands without running them. |

Environment variables read by both submit scripts:

| Var | Default |
|---|---|
| `ISAMBARD_STAGING_DIR` | `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging` |
| `ISAMBARD_REMOTE_HOST` | `u6jo.aip2.isambard` |
| `ISAMBARD_REMOTE_STAGING_DIR` | `/projects/u6jo/staging` |
| `PYTHON` | `python3` |

SLURM resources are set in the `#SBATCH` block at the top of each submit
script:

- `submit_build.sh`: partition `batch`, time `2-00:00:00`, 1 node, 8 CPUs, 8 GB RAM.
- `submit_transfer_shards.sh`: partition `batch`, time `11:30:00`, 1 node, 4 CPUs, 4 GB RAM.
  The 11h30 wall is intentionally <12h to stay inside the cert window with
  margin for the queue + ssh probe.

## Failure modes and recovery

| Symptom | Cause | Action |
|---|---|---|
| `build_squashfs.py`: `[Errno 18] Invalid cross-device link` | `--source-prefix` and `--staging-dir` are on different filesystems. | Move staging onto the same FS as the source data, or pass `--no-hardlinks` (slower; data is copied into the staging tree). |
| `submit_*` job exits 2 immediately with `mksquashfs not in PATH on this node` | Compute node lacks the `squashfs-tools` install. | Submit to a partition / node where `mksquashfs` is available. On Tufts it's at `/usr/sbin/mksquashfs`. |
| `submit_transfer_shards.sh` exits 2 with `ssh to u6jo.aip2.isambard failed` | Expired clifton cert. | Re-run `clifton`, resubmit. The chain breaks at this point on purpose. |
| Some `[i/N] FAILED rsync ...` lines mid-job; chain continues | Transient network or remote disk issue. | Next chained job retries that shard (it's not in the transfer manifest yet). |
| `sha256sum -c` fails on Isambard | Corrupt or truncated shard. | On Tufts, delete the offending `.sqsh`, the matching line in `<stem>.sha256sums.txt`, the build manifest line, and the transfer manifest line, then resubmit `submit_build.sh` and `submit_transfer_shards.sh`. |
| Successor never starts | Predecessor was killed externally before reaching the chain block, or `sbatch` itself failed. | Resubmit manually. The manifests make this idempotent. |
| `<stem>.complete` or `<stem>.transfer_complete` exists but you want to redo work | Manual override. | Delete the sentinel (and possibly the relevant manifest for a full re-do). |
