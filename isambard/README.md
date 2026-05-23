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
| `merge_shards_on_isambard.sh` | merge (optional) | Run on Isambard as a SLURM job: `squashfuse`-mount every shard on the bare host, then `mksquashfs.static` them into a single combined `.sqsh` so training only needs one bind. Uses Isambard's documented `mksquashfs.static` + `squashfuse` by default. |
| `squashfs_tools.def` | merge (optional, fallback) | Apptainer def for a ~10 MB alpine + `squashfs-tools` sidecar container. Only needed if Isambard's host-side `mksquashfs.static` / `squashfuse` are not on `$PATH` on compute nodes; see the merge section's sidecar fallback subsection. |
| `test_dataset_read.py` | post-merge test | Python entrypoint that builds the real `LArTPCDataset` / `LArFormerDataset` from a pointcept config and reads N batches through `torch.utils.data.DataLoader`. Reports per-batch shapes / dtypes / stats and aggregate samples/sec + MiB/sec. |
| `test_dataset_read.sh` | post-merge test | Apptainer-exec wrapper for `test_dataset_read.py`. Binds the combined `.sqsh` at `/data` inside `pointcept_cuml.sif`. Runs as both `bash` (login / interactive, no allocation) and `sbatch` (queued worker). |
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

For instructions on ssh and clifton for the Isambard cluster go [here](https://docs.isambard.ac.uk/user-documentation/guides/login/).

```
./clifton auth --identity /path/to/ssh_key
```

It's the public key ending in `.pub`.

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
multiple shards you have three options: multiple `--bind` flags, a union
mount (`mergerfs` / overlayfs) over per-shard `squashfuse_ll` mounts, or a
one-time merge into a single `.sqsh` — see [Optional: merge shards into a
single image on Isambard](#optional-merge-shards-into-a-single-image-on-isambard)
below. At 696 shards per dataset the merge is the recommended path because
binding ~700 squashfs images per training job costs 3–7 GB of FUSE daemon
RSS and tens of seconds of cold-start time per job.

### 8. Clean up

Once verification on Isambard succeeds, the local `.sqsh` shards in the
Tufts staging dir are safe to delete. The manifests, checksums, and
`.complete` / `.transfer_complete` sentinels are also disposable at that
point.

## Optional: merge shards into a single image on Isambard

After transfer and verification (steps 3 and 6 above), the dataset lives on
Isambard as ~700 `.sqsh` files. Training can mount each shard individually,
but at this scale that means ~700 `squashfuse_ll` daemons per training job,
each costing ~5–10 MB RSS, plus tens of seconds of FUSE-setup time at every
container start. [merge_shards_on_isambard.sh](merge_shards_on_isambard.sh)
collapses all shards into one combined `.sqsh` so each training job mounts
exactly one image.

### Tooling

Isambard's user documentation calls out two binaries for working with
SquashFS: `mksquashfs.static` (build the image) and `squashfuse` (mount
it). The merge script uses these directly on the bare host — no apptainer
container needed for the merge itself. If those binaries are missing on
the partition where you run the job (e.g. behind a `module load` you
haven't sourced), the script aborts in preflight; see the
[Sidecar container fallback](#sidecar-container-fallback) subsection
below for what to do then.

### What the script does

The merge is split into two phases to stay under Isambard's per-user FUSE
mount cap (`mount_max=1000` by default in `/etc/fuse3.conf`, but other
node workloads count against the same total — so we conservatively keep
concurrent mounts below 100).

1. Reads `$SHARDLIST` (a text file of shard paths on Isambard, e.g.
   [isambard_shardlist.txt](isambard_shardlist.txt)).
2. Preflights: tools on `$PATH`, every shard exists, `df` shows ≥2.2× the
   total shard byte count available in `$OUTPUT_DIR` (peak disk during
   phase 2 is intermediates + emerging final).
3. Sets up cleanup discipline (see [Cleanup discipline](#cleanup-discipline)
   below) and pre-cleans any stale FUSE mounts left under `$MOUNT_ROOT`
   by a prior crashed job.
4. **Phase 1.** Processes shards in batches of `$BATCH_SIZE` (default 64):
   for each batch, `squashfuse`-mounts the batch's shards under
   `$MOUNT_ROOT/phase1_batch_<N>/<rel>` with `-o subdir=/<rel>` (the
   `subdir` is load-bearing: each shard's internal layout mirrors its full
   relative path, so without `subdir` the unified view would be
   `$MOUNT_ROOT/<rel>/<rel>/file.h5` — doubled). Then runs
   `mksquashfs.static` on that mount tree, writing
   `$INTERMEDIATES_DIR/intermediate_<NNNN>.sqsh`, and unmounts the batch.
   Mounts within a batch run in parallel via `xargs -P 8`. Repeat until
   all batches done — typically ~12 intermediates of ~620 GB each for 708
   shards / 7.4 TB.
5. **Phase 2.** If more than one intermediate was produced,
   `squashfuse`-mounts every intermediate under
   `$MOUNT_ROOT/phase2/inter_<NNNN>` and runs `mksquashfs.static` with all
   mount points as multiple sources (`mksquashfs A B C ... dest.sqsh`).
   `mksquashfs` merges same-named directories across sources; the
   intermediates' leaf-dir paths are disjoint by construction, so no
   filename conflicts arise. If only one intermediate exists, it's
   `mv`'d into place — no phase 2 needed.
6. `mksquashfs` flags used in both phases: `-comp $COMPRESSOR -noI -noD
   -b 128K`. The source shards were built with `-comp zstd
   -Xcompression-level 1`, but Isambard's `mksquashfs.static` only ships
   `gzip` and `lz4` — so the combined image uses `gzip` by default.
   With `-noI -noD` only metadata tables are compressed (data blocks stay
   raw, matching the source shards), so the size difference between the
   two compressors is negligible.
7. Optionally writes a `.sha256` (skip with `SKIP_SHA256=1`) and runs the
   `.h5` file-count sanity check (skipped if `unsquashfs` isn't on PATH).
8. Deletes `$INTERMEDIATES_DIR` unless `KEEP_INTERMEDIATES=1`. On exit
   (normal, error, or signal) releases every FUSE mount under
   `$MOUNT_ROOT` and removes the directory.

**Resume support.** If `intermediate_<NNNN>.sqsh` already exists at the
start of batch N, that batch is skipped. If the final `$OUTPUT_SQSH`
already exists, phase 2 is skipped entirely. So a job that times out
partway through phase 1 picks up where it left off on resubmit. To force
a rebuild of a batch, delete its intermediate; to redo phase 2, delete
the final and resubmit.

### Why this is option C, not just "do A/B with one bind"

Three options exist for serving N shards to a training job:

- **A — host-side per-shard FUSE mounts unioned with mergerfs:** no new
  file, but ~700 stale `squashfuse` mounts per crashed job pile up on
  the node (the kernel marks them `ENOTCONN` until rebooted) and you depend
  on perfect cleanup discipline.
- **B — apptainer-owned per-shard mounts (`--mount type=squashfs` repeated):**
  apptainer places the mounts in the container's mount namespace, so
  SIGKILL of the job cleanly releases them when the namespace refcount
  drops. Safer against leaked mounts, but the live-job cost (~700 daemons,
  ~3–7 GB RSS, 30 s–2 min cold start) is paid every training job.
- **C — pre-merge into one image:** spend disk and CPU once to produce
  `combined.sqsh`; thereafter every training job mounts a single image
  with one daemon and ~10 MB RSS. Trade-off: ~7.4 TB of additional disk
  while both the shards and the combined image coexist, and any future
  shard rebuild requires a re-merge.

The merge job itself is essentially "Option A but one-shot": 696
`squashfuse` mounts go up, mksquashfs runs against them, mounts come
down. The cleanup discipline below is the price of A — the merge script
pays it carefully because the merge runs once; you do not want to pay it
every training job.

### Cleanup discipline

The script mounts under `$MOUNT_ROOT` (defaults to
`/tmp/merge_mounts.$SLURM_JOB_ID`). `/tmp` is node-local on Isambard's
compute nodes. SLURM does not reclaim it at job end, so the trap-based
cleanup and the pre-cleanup at job start (described below) are what
keep stale mounts from accumulating.

Three release paths:

- **EXIT trap.** Catches normal exit and most error paths. `awk`s
  `/proc/mounts` for everything under `$MOUNT_ROOT` whose fs type matches
  `squashfs|fuse`, sorts deepest-first to avoid EBUSY on nested mounts,
  and `fusermount -uz`s each. Then `rm -rf $MOUNT_ROOT`.
- **TERM/INT/USR1 trap.** Catches SLURM's pre-timeout SIGTERM (sent
  before the wall-clock SIGKILL by 30 s on most sites), Ctrl-C, and the
  custom USR1 from `#SBATCH --signal=B:USR1@120` (notifies us 2 min
  before TIMEOUT). Calls `cleanup_mounts` then exits 1.
- **Pre-cleanup at job start.** Before mounting anything, the script
  runs the same cleanup over `$MOUNT_ROOT`. If a prior crash left mounts
  there (possible if `$MOUNT_ROOT` is overridden to non-SLURM-TMPDIR
  storage, or if SLURM Epilog didn't run), this clears them before we
  reuse the directory.

SIGKILL cannot be trapped. If the job is killed by OOM-killer or
`scancel --signal=KILL`, mountpoints remain on the compute node under
`/tmp/merge_mounts.<jobid>/` until either the next job using this
script runs pre-cleanup, or the node reboots, or you ssh in and run
`awk '$2 ~ "^/tmp/merge_mounts" && $3 ~ /fuse/ {print $2}' /proc/mounts | sort -r | xargs -r -n1 fusermount -uz`.
The damage is node-local — `/projects/u6jo` is never affected.

#### A note on `$SLURM_TMPDIR`

You might expect `$SLURM_TMPDIR` to be the natural mount root since
SLURM cleans it up automatically. **Do not use it on Isambard.** That
path is set up by SLURM with mount-propagation semantics
(`PrivateTmp` / a private bind mount) that make user-space FUSE mounts
invisible in `/proc/mounts` to the script's own bash, even though
`squashfuse` exits 0 and the daemon appears to start. Symptom: the
script logs `Mounted 0 shards` and aborts with `expected N mounts, got 0`,
with no errors in the `.err` file. We verified this empirically:
the same parallel mount loop succeeds on `/tmp` and silently produces
zero visible mounts on `$SLURM_TMPDIR`. The script's default avoids
this pitfall; only set `MOUNT_ROOT=$SLURM_TMPDIR/...` if you've
confirmed your site doesn't have this configuration.

### Resources and runtime

`mksquashfs -noI -noD` is essentially a streaming copy with metadata
bookkeeping: read 7.4 TB once through FUSE, write 7.4 TB once. Reads and
writes overlap, so wall-clock ≈ max(read, write).

Observed on Isambard (5-shard / 27 GB smoke test): mksquashfs averaged
**2.1 GB/s**. With the two-phase batched merge, total I/O roughly doubles
(write all intermediates, then read them back during phase 2), so
extrapolated runtime is ~2 h for the merge itself on 7.4 TB. The same
smoke test showed `sha256sum` at **380 MB/s** (single-threaded), which
extrapolates to **~5 h** on the full image — dominating the wall clock if
enabled. Set `SKIP_SHA256=1` to skip and compute later if needed.

| Stage | Smoke test (27 GB / 5 shards) | Full run (7.4 TB / 708 shards) projected |
|---|---|---|
| `squashfuse` mount loop (per batch) | <1 s | ~5 s × 12 batches |
| `mksquashfs.static` phase 1 (build intermediates) | 13 s | ~1 h total |
| `mksquashfs.static` phase 2 (merge ~12 intermediates) | 0 s (single-batch shortcut) | ~1 h |
| `sha256sum` (single-threaded) | 71 s | ~5 h (skippable) |
| **Total** | 84 s | **~2 h** (`SKIP_SHA256=1`) / **~7 h** (with sha256) |

Peak disk during phase 2: roughly 2 × the shard total (intermediates +
emerging final). After phase 2 + cleanup, only the final remains. With
`KEEP_INTERMEDIATES=1`, steady-state is ~2 × the shard total.

The script defaults to `--time=08:00:00`, `--mem=64G`,
`--cpus-per-task=16` — time is deliberately overcommitted. Phase 1 is
fully resumable (one intermediate per batch), so a job that times out
during phase 1 can be re-submitted and will pick up at the next missing
batch. Phase 2 is one big `mksquashfs` call with no resume; if it times
out, delete the partial `$OUTPUT_SQSH` and re-run (intermediates are
preserved).

**Do not run on the Isambard login node.** Sustained 7.4 TB read+write on
shared storage will get flagged by admins, the ~10–30 GB RAM peak during
mksquashfs metadata-table construction may exceed login limits, and a
multi-hour run is at risk of being reaped by idle / session timeouts.

### Workflow

1. **Smoke-test on a handful of shards first.** This catches path-layout
   surprises (`subdir=/<rel>` assumption) before you burn 4 hours:

   ```bash
   ssh u6jo.aip2.isambard
   cd /projects/u6jo/staging   # or wherever the scripts live on Isambard
   head -5 isambard_shardlist.txt > /tmp/test_shards.txt
   SHARDLIST=/tmp/test_shards.txt OUTPUT_NAME=combined_test \
       bash merge_shards_on_isambard.sh
   unsquashfs -l /projects/u6jo/staging/combined_test.sqsh | head -30
   ```

   Confirm paths look like `squashfs-root/ub_on_tufts/.../000/000/file.h5`,
   not doubled (`.../000/000/000/000/file.h5`). If the layout is wrong,
   inspect a single shard with `unsquashfs -l <shard>.sqsh` and adjust
   the `subdir=` argument inside the script accordingly.

2. **Full merge job.** Edit the SLURM directives to fill in the partition /
   account / qos that your Isambard project uses (look for
   `# TODO Isambard:` in the script), then:

   ```bash
   sbatch merge_shards_on_isambard.sh
   ```

3. **Verify.** Confirm the `.h5` file count printed at the end of the log
   matches the source count:

   ```bash
   wc -l <stem>.sha256sums.txt           # one line per shard
   unsquashfs -l combined_*.sqsh | grep -c '\.h5$'
   ```

   The two numbers should agree once you account for the shard-count vs.
   file-count distinction (the source has ~500k–1M files across the
   shards; the combined `.sqsh` should match the source file count).

4. **Switch the training launcher to one bind:**

   ```bash
   apptainer exec --nv \
     --bind /projects/u6jo/staging/combined_<dataset>.sqsh:/data:image-src=/,ro \
     <pointcept.sif> \
     python tools/train.py --config-file configs/lartpc/<config>.py ...
   ```

5. **(Optional) reclaim disk by deleting the per-shard files.** Once the
   combined image is verified and you've confirmed training works, the
   ~700 individual `.sqsh` files on Isambard can be deleted. Keep the
   `<stem>.sha256sums.txt` and `<stem>.transfer_manifest.txt` around for
   provenance and for rebuilds.

### Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `STAGING_DIR` | `/projects/u6jo/staging` | Root the shardlist paths are relative to. Stripping this prefix yields the per-shard `<rel>` used in `subdir=`. |
| `SHARDLIST` | `$STAGING_DIR/isambard_shardlist.txt` | One absolute shard path per line. |
| `OUTPUT_DIR` | `$STAGING_DIR` | Where the combined `.sqsh` and `.sha256` are written. |
| `OUTPUT_NAME` | `combined_pretrain-sonata-v7-extbnb-larmatch` | Basename for the output (without `.sqsh`). Refusing-to-overwrite is enforced. |
| `MKSQUASHFS` | `mksquashfs.static` | Image-builder binary. Override if Isambard names it differently or requires a `module load`. |
| `UNSQUASHFS` | `unsquashfs.static` | Optional. Only used for the final `.h5` file-count sanity check; if not on `$PATH` the check is skipped (not an error). |
| `SQUASHFUSE` | `squashfuse` | Mount tool. `squashfuse_ll` works too if available. |
| `COMPRESSOR` | `gzip` | Compressor passed as `mksquashfs -comp`. Isambard's `mksquashfs.static` only supports `gzip` and `lz4` (no `zstd`). With `-noI -noD` only metadata tables are compressed, so the choice barely affects output size. |
| `BATCH_SIZE` | `64` | Shards per phase-1 intermediate. Keep below the per-user `mount_max` (default 1000 on Isambard, but shared with apptainer / other workloads — 64 is a safe ceiling). Larger batches = fewer intermediates = faster phase 2, but more concurrent mounts during phase 1. |
| `INTERMEDIATES_DIR` | `$OUTPUT_DIR/${OUTPUT_NAME}.intermediates` | Where per-batch `intermediate_NNNN.sqsh` files are written during phase 1. Override if you want them on a different volume. |
| `KEEP_INTERMEDIATES` | `0` | `1` keeps `$INTERMEDIATES_DIR` after a successful phase 2. Useful for debugging or if you want to redo phase 2 with different merge flags without rebuilding from shards. |
| `PROCESSORS` | `$SLURM_CPUS_PER_TASK` (16) | `mksquashfs -processors`. |
| `MOUNT_PARALLEL` | 8 | `xargs -P` for the mount loop. |
| `MOUNT_ROOT` | `/tmp/merge_mounts.$SLURM_JOB_ID` | Where shards get mounted. Do NOT point this at `$SLURM_TMPDIR` on Isambard — see "A note on `$SLURM_TMPDIR`" below. |
| `SKIP_SHA256` | `0` | Set to `1` to skip the post-merge `sha256sum` step. `sha256sum` is single-threaded and dominates wall-clock at ~5 h on a 7.4 TB image (vs. ~1 h for the merge itself, measured at ~2.1 GB/s on Isambard). Compute later out-of-band if needed. |

### Sidecar container fallback

If `mksquashfs.static` or `squashfuse` is not on `$PATH` on Isambard
compute nodes, [squashfs_tools.def](squashfs_tools.def) builds a tiny
alpine + `squashfs-tools` container (~10 MB) that provides them. Build
on Tufts and rsync to Isambard:

```bash
# On Tufts:
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/isambard
apptainer build --fakeroot squashfs_tools.sif squashfs_tools.def
apptainer exec squashfs_tools.sif mksquashfs -version    # sanity check
rsync -aP squashfs_tools.sif \
    u6jo.aip2.isambard:/projects/u6jo/containers/squashfs_tools.sif
```

If `--fakeroot` is unavailable on your Tufts shell, alternatives are
(a) a node where it works, (b) `apptainer build --remote`, or (c) build
on Isambard if their apptainer install supports `--fakeroot`.

To use the sidecar from the merge script, override the tool env vars so
each invocation goes through `apptainer exec`:

```bash
SIF=/projects/u6jo/containers/squashfs_tools.sif
MKSQUASHFS="apptainer exec $SIF mksquashfs" \
UNSQUASHFS="apptainer exec $SIF unsquashfs" \
SQUASHFUSE="apptainer exec $SIF squashfuse" \
    sbatch merge_shards_on_isambard.sh
```

Note: with this fallback, the FUSE mounts live in the host namespace
(not in any apptainer container's namespace) because `squashfuse` is
invoked as a one-shot child of the launcher shell. So the cleanup
discipline above still applies — the apptainer wrapper just provides
the binaries, not lifecycle management.

### Failure modes

| Symptom | Cause | Action |
|---|---|---|
| Smoke-test produces `$MOUNT_ROOT/<rel>/<rel>/...` (doubled paths) | Internal shard layout differs from the `build_squashfs.py` hardlink-tree convention. | Inspect with `unsquashfs -l <one>.sqsh`. Either drop `subdir=/$rel` from the `mount_one` function (if files are at shard root) or adjust the path. |
| `mksquashfs.static: command not found` | Isambard's squashfs tools are behind a `module load`, or named differently. | Source the module in your `~/.bashrc` or in the script, or set `MKSQUASHFS=...` to the absolute path. |
| `squashfuse: command not found` | Same as above. | Set `SQUASHFUSE=...`. Or fall back to the sidecar (above). |
| `mksquashfs.static: Compressor "X" is not supported!` | Isambard's `mksquashfs.static` is a minimal build with limited compressor support (typically just `gzip` and `lz4`). | Override `COMPRESSOR=gzip` (the default) or `COMPRESSOR=lz4`. Do not change the source shards' compression — only the combined output is affected. |
| `fusermount: failed to unmount: Device or resource busy` during cleanup | A process inside the mount tree still has open fds. | The trap uses `-uz` (lazy unmount) which sidesteps EBUSY by detaching the mountpoint and letting the kernel finish cleanup when fds close. If you see this, it means the lazy path also failed — investigate manually with `lsof`. |
| Job hits wall clock before completion | Underestimated throughput, or shared-FS contention. | Output is unusable; mksquashfs has no resume. Delete the partial `.sqsh`, raise `--time=`, resubmit. The USR1 trap should have caught the pre-timeout and unmounted; verify in the log. |
| `df`-based preflight refuses to start | Less than 1.05× total shard size free in `$OUTPUT_DIR`. | Free space, or set `$OUTPUT_DIR` to a different filesystem with capacity. Don't bypass — running out of space mid-merge wastes hours. |
| Stale mounts left after SIGKILL | The one signal that cannot be trapped. | The next job's pre-cleanup (same script, same `MOUNT_ROOT`) will release them. To clean now: ssh to the compute node and `awk '$2 ~ "^<root>" && $3 ~ /fuse/ {print $2}' /proc/mounts \| sort -r \| xargs -r -n1 fusermount -uz`. |
| `Mounted 0 shards` with no errors in `.err` | `MOUNT_ROOT` was set to `$SLURM_TMPDIR/...` on a site (Isambard) where that path has `PrivateTmp`-style propagation that hides FUSE mounts. | Unset `MOUNT_ROOT`, or set it explicitly to a `/tmp/...` path. See "A note on `$SLURM_TMPDIR`" above. |
| `fusermount3: too many FUSE filesystems mounted; mount_max=N can be set in /etc/fuse3.conf` | The per-user FUSE mount cap (default 1000) has been hit. Other workloads (apptainer, other users) count against the same cap on shared compute nodes. | Lower `BATCH_SIZE` (e.g. 32) so phase 1 holds fewer concurrent mounts. Verify the cap with `cat /etc/fuse3.conf`. If repeatedly bumping up against it, ask Isambard support to raise `mount_max` system-wide. |
| Phase 1 timed out partway | Underestimated throughput, or shared-FS contention. | Just resubmit — intermediates already written are kept and skipped on resume. The job continues from the next missing batch. |
| Phase 2 timed out / produced a broken `$OUTPUT_SQSH` | Phase 2 is one big `mksquashfs` call with no resume. | Delete `$OUTPUT_SQSH` and resubmit. Phase 1 is skipped via the intermediate-exists checks; phase 2 starts fresh. |

## Testing dataset reads from the combined image

After the smoke-test (or full) merge produces a `combined_*.sqsh`,
exercise it end-to-end by reading batches through the same dataset
classes training uses — `LArTPCDataset` / `LArFormerDataset` from
[pointcept/datasets/lartpc.py](../pointcept/datasets/lartpc.py) and
[pointcept/datasets/larformer.py](../pointcept/datasets/larformer.py).
This validates three things at once: the apptainer bind / mount of the
`.sqsh` works, the HDF5 read path survives FUSE, and `entry_0`
attributes (`run`, `subrun`, `event`) propagated through the merge.

The driver pair is [test_dataset_read.sh](test_dataset_read.sh) (wrapper
that handles apptainer + binds + SLURM) and
[test_dataset_read.py](test_dataset_read.py) (the in-container Python
that builds the dataset and pulls batches).

### Workflow

**1. Build an Isambard-side data list.** The training-config list files
(absolute Tufts paths) won't resolve inside the container. Build one
that points at paths under the in-container mount point (default
`/data`). Easiest is to mount the combined `.sqsh` with `squashfuse`,
`find` for `.h5` files, and rewrite the prefix:

```bash
mkdir -p /tmp/check
squashfuse /projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch-test.sqsh /tmp/check
find /tmp/check -name '*.h5' | head -20 \
    | sed 's|^/tmp/check|/data|' \
    > /projects/u6jo/datasets/test_list_isambard.txt
fusermount -u /tmp/check
wc -l /projects/u6jo/datasets/test_list_isambard.txt
```

Result: a list whose every line is e.g.
`/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/000/000/foo.h5`.
For the full combined image, drop `head -20`.

**2. Smoke-test on the login node** (no allocation, no GPU):

```bash
cd ~/ubpointcept/pointcept/isambard    # or wherever you cloned pointcept on Isambard
DATA_LIST_FILE=/projects/u6jo/datasets/test_list_isambard.txt \
    bash test_dataset_read.sh
```

You should see: configuration echoed, `entry_0` attributes for the first
HDF5 file, dataset built, 4 batches read with per-batch tensor shapes,
and an aggregate summary (samples/sec, MiB/sec, per-batch median / p95
latency).

Failure modes here flag specific problems:

| Symptom | Likely cause |
|---|---|
| `ERROR: first listed file does not exist (mount problem?)` | The bind isn't working or the data list's paths don't match the in-container layout. Re-check `$DATA_ROOT` and the `subdir=` trick used at merge time. |
| `WARNING: 'entry_0' not in <path>` | H5 file predates the Step-2/3 attribute additions (or merge corrupted things). |
| `ImportError: No module named pointcept` | `POINTCEPT_DIR` is wrong; the bind to `/pointcept` didn't land on a tree containing `pointcept/pointcept/...`. |
| Hangs in `build_dataset` | The dataset's `__init__` is recursively scanning the mount because `data_list_file` got ignored. Confirm the override worked by checking the printed `transforms : ...` line — if the dataset built, the list was found. |

**3. Multi-worker scaling sweep.** Once basic reads work, measure where
the FUSE single-thread bottleneck bites:

```bash
mkdir -p logs
for W in 0 1 2 4 8; do
    DATA_LIST_FILE=/projects/u6jo/datasets/test_list_isambard.txt \
    NUM_WORKERS=$W NUM_BATCHES=32 \
        bash test_dataset_read.sh 2>&1 | tee logs/scan-w${W}.log
done
grep -H 'samples / sec' logs/scan-w*.log
```

Expected shape: 0→1 is "free" (workers fork off and stream
concurrently with the main process), 1→2 near-linear, 2→4 diminishing
returns, 4→8 flat or worse. The plateau is the per-mount FUSE single-
thread saturation point. If you need to scale past it, the workaround
is mounting the same `.sqsh` at multiple mountpoints and splitting
workers across them (or building the per-shard `--mount` apptainer
arrangement from earlier as option B) — but only worth the complexity
if the plateau actually constrains your training step.

**4. Realistic check with transforms.** Once basic reads are healthy,
flip on the full config transform pipeline to exercise every code path
training will hit:

```bash
DATA_LIST_FILE=/projects/u6jo/datasets/test_list_isambard.txt \
APPLY_TRANSFORMS=1 NUM_BATCHES=8 \
    bash test_dataset_read.sh
```

**5. Queued worker run.** Same script works as `sbatch` — the SLURM
directives are honored:

```bash
DATA_LIST_FILE=/projects/u6jo/datasets/test_list_isambard.txt \
    sbatch test_dataset_read.sh
```

Defaults inside the script: `--time=00:30:00`, `--cpus-per-task=8`,
`--mem=16G`, `--gres=gpu:1`. Drop the `--gres` if your project's queue
allows pure-CPU jobs and you want a cheaper / faster-queued test.

### Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DATA_LIST_FILE` | *(required)* | Isambard-side list with paths that resolve under `$DATA_ROOT` inside the container. |
| `COMBINED_SQSH` | `/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch-test.sqsh` | The merged image to mount. Default is the smoke-test image; override for the full one. |
| `CONTAINER` | `/projects/u6jo/containers/pointcept_cuml.sif` | The training container. The test scripts run inside this. |
| `DATA_ROOT` | `/data` | In-container mount point for `$COMBINED_SQSH`. The list paths must start with this. |
| `POINTCEPT_DIR` | `$HOME/ubpointcept/pointcept` | Host path to the cloned pointcept repo on Isambard. Bound at `/pointcept` inside the container. |
| `CONFIG_FILE` | `$POINTCEPT_DIR/configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py` | Pointcept config; the wrapper translates this to its in-container path. Point at the larformer config to test that dataset instead. |
| `SPLIT` | `train` | Which `cfg.data.<split>` block to read. |
| `NUM_BATCHES` | `4` | How many batches to pull. |
| `BATCH_SIZE` | `2` | Per-batch sample count. |
| `NUM_WORKERS` | `0` | `DataLoader` worker count. `0` = in-process; use this when debugging an FS issue. |
| `APPLY_TRANSFORMS` | `0` | Set `1` to apply the config's transform pipeline. Default off so transform failures don't get blamed on the mount. |
| `GPU` | `auto` | `auto` = enable `--nv` only if `nvidia-smi` is visible (so login-node runs work). `0` = force CPU. `1` = force GPU. |

### A note on FUSE concurrency

`squashfuse` (plain) is single-threaded; `squashfuse_ll` is multi-
threaded. Apptainer ships its own `squashfuse_ll` for `image-src` bind
mounts, but a single mount still has one FUSE process per `.sqsh`.
Multiple DataLoader workers reading the same mount all serialize
through that process's user-space scheduler. Empirically this means
the read throughput curve flattens between 2 and 8 workers depending
on file size and access pattern. The scan in step 3 above tells you
where it flattens for *your* dataset; once you know that, choose
`num_worker` in the training config a bit below the plateau.

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
