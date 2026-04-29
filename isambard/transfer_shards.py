#!/usr/bin/env python3
"""rsync .sqsh shards already produced by build_squashfs.py to Isambard.

Reads the per-list checksums file (`<stem>.sha256sums.txt`) as the
authoritative list of shards to transfer. For each shard not yet present in
the transfer manifest, rsyncs it to the remote staging dir and appends the
shard to a transfer-side manifest so re-runs are idempotent.

Build state (`<stem>.manifest.txt`) is left untouched — this script only
writes `<stem>.transfer_manifest.txt`. That separation lets you re-build a
shard locally without losing the record of which shards were already
transferred.

The remote sha256sums file is rsynced alongside so unpack_on_isambard.sh
(or a sqsh-aware variant) can verify shards on arrival.

The chain-on-cert-window logic lives in submit_transfer_shards.sh, mirroring
submit_transfer.sh.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--file-list", required=True,
                   help="Original file list (used only to derive the per-list "
                        "stem for naming the checksums and manifest files).")
    p.add_argument("--staging-dir", required=True,
                   help="Local directory containing the built .sqsh shards.")
    p.add_argument("--remote-host", default="u6jo.aip2.isambard")
    p.add_argument("--remote-staging-dir", default="/projects/u6jo/staging",
                   help="Remote directory where shards land.")
    p.add_argument("--checksums", default=None,
                   help="Checksums file. Default: <staging-dir>/<stem>.sha256sums.txt")
    p.add_argument("--transfer-manifest", default=None,
                   help="Transfer manifest. Default: <staging-dir>/<stem>.transfer_manifest.txt")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N transfers. 0 = no limit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rsync commands without running them.")
    return p.parse_args()


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_checksums(path):
    """Read an `<sha256>  <rel_path>`-format file. Returns list of
    (digest, rel_path) preserving file order so the transfer is deterministic."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue
            digest, rel = parts
            out.append((digest, rel))
    return out


def load_manifest(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    done.add(line)
    return done


def append_line(path, line):
    with open(path, "a") as f:
        f.write(line + "\n")


def rsync_to_remote(local_path, remote_host, remote_path, dry_run):
    cmd = [
        "rsync",
        "-a",
        "--partial",
        "--mkpath",
        "--info=stats1,progress0",
        local_path,
        f"{remote_host}:{remote_path}",
    ]
    if dry_run:
        log("DRY-RUN rsync: " + " ".join(shlex.quote(c) for c in cmd))
        return
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    staging = Path(args.staging_dir)
    if not staging.is_dir():
        log(f"ERROR: staging-dir does not exist: {staging}")
        sys.exit(1)

    list_stem = os.path.splitext(os.path.basename(args.file_list))[0]
    checksums_path = args.checksums or str(staging / f"{list_stem}.sha256sums.txt")
    manifest_path = args.transfer_manifest or str(staging / f"{list_stem}.transfer_manifest.txt")
    complete_path = str(staging / f"{list_stem}.transfer_complete")

    if not os.path.exists(checksums_path):
        log(f"ERROR: checksums file not found: {checksums_path}")
        log("       Run build_squashfs.py first, or pass --checksums explicitly.")
        sys.exit(1)

    entries = load_checksums(checksums_path)
    done = load_manifest(manifest_path)
    log(f"Checksums: {checksums_path} ({len(entries)} shards)")
    log(f"Transfer manifest: {manifest_path} ({len(done)} already transferred)")
    log(f"Remote: {args.remote_host}:{args.remote_staging_dir}")

    if args.dry_run:
        log("DRY-RUN mode: no rsync will be executed.")

    n_total = len(entries)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    bytes_this_run = 0
    transferred_this_run = set()
    t0 = time.time()

    for i, (digest, shard_rel) in enumerate(entries, start=1):
        if shard_rel in done:
            n_skipped += 1
            if n_skipped <= 3 or n_skipped % 100 == 0:
                log(f"[{i}/{n_total}] SKIP {shard_rel} (already transferred)")
            continue

        local_path = staging / shard_rel
        if not local_path.exists():
            log(f"[{i}/{n_total}] WARN missing local shard: {local_path} (skip)")
            n_failed += 1
            continue

        remote_path = args.remote_staging_dir.rstrip("/") + "/" + shard_rel
        sz = local_path.stat().st_size if not args.dry_run else 0
        log(f"[{i}/{n_total}] rsync {shard_rel} ({sz / (1024**2):.1f} MiB) "
            f"-> {args.remote_host}:{remote_path}")
        try:
            rsync_to_remote(str(local_path), args.remote_host, remote_path, dry_run=args.dry_run)
        except subprocess.CalledProcessError as e:
            log(f"[{i}/{n_total}] FAILED rsync ({shard_rel}): {e}")
            n_failed += 1
            continue

        if not args.dry_run:
            append_line(manifest_path, shard_rel)
            transferred_this_run.add(shard_rel)
            bytes_this_run += sz

        n_done += 1
        elapsed = time.time() - t0
        mb_rate = (bytes_this_run / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        log(f"[{i}/{n_total}] DONE {shard_rel}  (transferred={n_done} skipped={n_skipped} "
            f"failed={n_failed} elapsed={elapsed:.0f}s {mb_rate:.1f} MB/s)")

        if args.limit and n_done >= args.limit:
            log(f"Reached --limit {args.limit}; stopping.")
            break

    # Push the (final) checksums file alongside the shards so the remote
    # verifier can sha256sum -c against it. Always re-push: cheap and keeps
    # the remote fresh after partial runs.
    if not args.dry_run and (n_done > 0 or not args.limit):
        cs_remote = args.remote_staging_dir.rstrip("/") + "/" + os.path.basename(checksums_path)
        try:
            rsync_to_remote(checksums_path, args.remote_host, cs_remote, dry_run=False)
        except subprocess.CalledProcessError as e:
            log(f"WARNING: failed to push checksums file: {e}")

    log(f"Done. transferred={n_done} skipped={n_skipped} failed={n_failed} "
        f"elapsed={time.time() - t0:.0f}s pushed={bytes_this_run / (1024**3):.2f} GiB")

    all_done = done.union(transferred_this_run)
    fully_complete = (
        not args.dry_run
        and n_failed == 0
        and all_done == {rel for _, rel in entries}
        and not args.limit
    )
    if fully_complete:
        Path(complete_path).touch()
        log(f"All {n_total} shards transferred. Wrote sentinel: {complete_path}")

    if n_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
