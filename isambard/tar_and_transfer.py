#!/usr/bin/env python3
"""Tar files from a file list (grouped by leaf directory) and rsync to Isambard.

Workflow per leaf directory:
  1. Collect the subset of files from the input list that live in this directory.
  2. Create a zstd-compressed tarball using paths relative to a configurable
     source prefix, so the remote can extract with the original directory
     structure preserved.
  3. Compute sha256 of the tarball and update a global checksums file.
  4. rsync the tarball into the remote staging area, mirroring the relative
     directory structure so the unpack script can place it correctly.
  5. Append the rel-dir to a manifest so a re-run skips already-done entries.

The script is fully resumable: re-running with the same arguments will skip
any leaf directory present in the manifest.
"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--file-list", required=True,
                   help="Text file with absolute paths, one per line.")
    p.add_argument("--source-prefix", required=True,
                   help="Prefix stripped from each absolute path to derive the "
                        "relative path used inside tarballs and on the remote.")
    p.add_argument("--staging-dir", required=True,
                   help="Local directory where tarballs are written.")
    p.add_argument("--remote-host", default="u6jo.aip2.isambard")
    p.add_argument("--remote-staging-dir", default="/projects/u6jo/staging",
                   help="Remote directory where tarballs land.")
    p.add_argument("--manifest", default=None,
                   help="Manifest file. Default: <staging-dir>/manifest.txt")
    p.add_argument("--checksums", default=None,
                   help="Checksums file. Default: <staging-dir>/sha256sums.txt")
    p.add_argument("--zstd-threads", type=int, default=0,
                   help="zstd -T parameter. 0 = use all available cores.")
    p.add_argument("--zstd-level", type=int, default=3,
                   help="zstd compression level (1..19).")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N tarballs. 0 = no limit. Useful for testing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without creating tarballs or rsyncing.")
    p.add_argument("--skip-rsync", action="store_true",
                   help="Create tarballs and checksums locally but skip rsync.")
    return p.parse_args()


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def group_files_by_leaf(file_list_path, source_prefix):
    """Return dict: rel_dir -> sorted list of relative file paths.

    Both `source_prefix` and each input path are passed through
    os.path.normpath, so quirks like doubled slashes (`/foo//bar`) collapse
    consistently before the prefix check.
    """
    groups = defaultdict(list)
    skipped = 0
    total = 0
    with open(file_list_path) as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total += 1
            path = os.path.normpath(raw)
            if not path.startswith(source_prefix):
                skipped += 1
                if skipped <= 5:
                    log(f"WARNING: path does not start with prefix, skipping: {raw}")
                continue
            rel = path[len(source_prefix):]
            rel_dir = os.path.dirname(rel)
            groups[rel_dir].append(rel)
    if skipped:
        log(f"WARNING: skipped {skipped}/{total} entries that did not match prefix.")
    for k in groups:
        groups[k].sort()
    return groups


def sha256_of_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


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


def load_checksums(path):
    """Load existing sha256sum-format file as dict tarball_rel -> digest."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            # sha256sum format: "<digest>  <path>"
            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue
            digest, name = parts
            out[name] = digest
    return out


def write_checksums(path, checksums):
    """Atomically rewrite the checksums file in sha256sum-compatible format."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for name in sorted(checksums):
            f.write(f"{checksums[name]}  {name}\n")
    os.replace(tmp, path)


def make_tarball(source_prefix, rel_files, tarball_path, zstd_threads, zstd_level, dry_run):
    """Create a zstd-compressed tarball of `rel_files` (paths relative to
    `source_prefix`). The tarball stores those same relative paths."""
    tarball_path = str(tarball_path)
    list_path = tarball_path + ".filelist"
    with open(list_path, "w") as f:
        for rel in rel_files:
            f.write(rel + "\n")
    zstd_cmd = f"zstd -{zstd_level} -T{zstd_threads}"
    cmd = [
        "tar",
        "-I", zstd_cmd,
        "-cf", tarball_path,
        "-C", source_prefix,
        "-T", list_path,
    ]
    if dry_run:
        log("DRY-RUN tar: " + " ".join(shlex.quote(c) for c in cmd))
        os.remove(list_path)
        return
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


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

    # Normalize the prefix the same way input paths get normalized so we
    # don't get spurious mismatches from doubled slashes etc.
    source_prefix = os.path.normpath(args.source_prefix)
    if not source_prefix.endswith("/"):
        source_prefix = source_prefix + "/"
    if not os.path.isdir(source_prefix):
        log(f"ERROR: source-prefix is not a directory: {source_prefix}")
        sys.exit(1)

    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    # Default manifest/checksums per-file-list so multiple datasets can share
    # one staging dir without clashing.
    list_stem = os.path.splitext(os.path.basename(args.file_list))[0]
    manifest_path = args.manifest or str(staging / f"{list_stem}.manifest.txt")
    checksums_path = args.checksums or str(staging / f"{list_stem}.sha256sums.txt")
    complete_path = str(staging / f"{list_stem}.complete")

    done = load_manifest(manifest_path)
    checksums = load_checksums(checksums_path)
    log(f"Manifest: {manifest_path} ({len(done)} entries already done)")
    log(f"Checksums: {checksums_path} ({len(checksums)} entries)")

    log(f"Reading file list: {args.file_list}")
    groups = group_files_by_leaf(args.file_list, source_prefix)
    n_files = sum(len(v) for v in groups.values())
    log(f"Grouped into {len(groups)} leaf directories covering {n_files} files.")

    if args.dry_run:
        log("DRY-RUN mode: no tarballs will be written and no rsyncs performed.")

    n_total = len(groups)
    n_processed = 0
    n_skipped = 0
    n_failed = 0
    processed_this_run = set()
    t0 = time.time()

    for i, rel_dir in enumerate(sorted(groups.keys()), start=1):
        files = groups[rel_dir]
        if rel_dir in done:
            n_skipped += 1
            if n_skipped <= 3 or n_skipped % 100 == 0:
                log(f"[{i}/{n_total}] SKIP {rel_dir} (already in manifest)")
            continue

        tarball_rel = rel_dir + ".tar.zst"
        tarball_path = staging / tarball_rel
        tarball_path.parent.mkdir(parents=True, exist_ok=True)

        log(f"[{i}/{n_total}] Tarring {rel_dir} ({len(files)} files) -> {tarball_path}")
        try:
            make_tarball(
                source_prefix.rstrip("/"),
                files,
                tarball_path,
                zstd_threads=args.zstd_threads,
                zstd_level=args.zstd_level,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as e:
            log(f"[{i}/{n_total}] FAILED tar ({rel_dir}): {e}")
            n_failed += 1
            continue

        if not args.dry_run:
            digest = sha256_of_file(str(tarball_path))
            checksums[tarball_rel] = digest
            write_checksums(checksums_path, checksums)
            log(f"[{i}/{n_total}] sha256={digest[:16]}...  size={tarball_path.stat().st_size:,} bytes")

        if not args.skip_rsync:
            remote_path = args.remote_staging_dir.rstrip("/") + "/" + tarball_rel
            log(f"[{i}/{n_total}] rsync -> {args.remote_host}:{remote_path}")
            try:
                rsync_to_remote(str(tarball_path), args.remote_host, remote_path, dry_run=args.dry_run)
            except subprocess.CalledProcessError as e:
                log(f"[{i}/{n_total}] FAILED rsync ({rel_dir}): {e}")
                n_failed += 1
                continue

            # Push the (updated) checksums file alongside.
            if not args.dry_run:
                cs_remote = args.remote_staging_dir.rstrip("/") + "/" + os.path.basename(checksums_path)
                try:
                    rsync_to_remote(checksums_path, args.remote_host, cs_remote, dry_run=False)
                except subprocess.CalledProcessError as e:
                    log(f"WARNING: failed to push checksums file: {e}")

        if not args.dry_run:
            append_line(manifest_path, rel_dir)
            processed_this_run.add(rel_dir)

        n_processed += 1
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        log(f"[{i}/{n_total}] DONE {rel_dir}  (processed={n_processed} skipped={n_skipped} "
            f"failed={n_failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s)")

        if args.limit and n_processed >= args.limit:
            log(f"Reached --limit {args.limit}; stopping.")
            break

    log(f"Done. processed={n_processed} skipped={n_skipped} failed={n_failed} "
        f"elapsed={time.time() - t0:.0f}s")

    # Sentinel for the chain wrapper: every leaf dir is now in the manifest
    # and nothing failed this run. Don't write it if --limit truncated the run.
    all_done = done.union(processed_this_run)
    fully_complete = (
        not args.dry_run
        and n_failed == 0
        and all_done == set(groups.keys())
        and not args.limit
    )
    if fully_complete:
        Path(complete_path).touch()
        log(f"All {n_total} leaf directories complete. Wrote sentinel: {complete_path}")

    if n_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
