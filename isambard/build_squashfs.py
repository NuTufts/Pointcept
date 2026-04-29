#!/usr/bin/env python3
"""Build SquashFS shards from a file list (grouped by leaf directory).

Build-only counterpart of tar_and_transfer.py: produces .sqsh shards locally
and writes a manifest + sha256 file in the same format as the tar pipeline,
but performs no rsync. Intended for the Isambard workflow where build and
transfer are decoupled — run this now, transfer the .sqsh shards later.

Per shard:
  1. Build a hardlink staging tree mirroring the source layout for the files
     in this shard (no data copy; same-filesystem hardlinks).
  2. Run mksquashfs against the staging root to produce <shard_id>.sqsh
     under the staging dir, mirroring the rel_dir hierarchy.
  3. Compute sha256 of the .sqsh and update the per-list checksums file.
  4. Append the shard id to the manifest so re-runs are idempotent.
  5. Remove the hardlink staging tree.

The on-disk shard naming mirrors tar_and_transfer.py: a leaf-dir
"ub_on_tufts/.../000/000" becomes "ub_on_tufts/.../000/000.sqsh" under the
staging dir, which means the existing chain wrapper, manifest format, and
checksums file format are unchanged.
"""

import argparse
import hashlib
import os
import shlex
import shutil
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
                        "relative path used inside shards.")
    p.add_argument("--staging-dir", required=True,
                   help="Local directory where .sqsh shards are written.")
    p.add_argument("--manifest", default=None,
                   help="Manifest file. Default: <staging-dir>/<list-stem>.manifest.txt")
    p.add_argument("--checksums", default=None,
                   help="Checksums file. Default: <staging-dir>/<list-stem>.sha256sums.txt")
    p.add_argument("--processors", type=int, default=0,
                   help="mksquashfs -processors. 0 = mksquashfs default (all cores).")
    p.add_argument("--zstd-level", type=int, default=1,
                   help="zstd compression level for sqsh metadata (1..22). h5 data "
                        "is incompressible so -noD skips data compression entirely.")
    p.add_argument("--block-size", default="128K",
                   help="mksquashfs -b block size.")
    p.add_argument("--no-hardlinks", action="store_true",
                   help="Copy files into the staging tree instead of hardlinking. "
                        "Use when source and staging are on different filesystems.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N shards. 0 = no limit. Useful for testing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without creating any files.")
    return p.parse_args()


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def group_files_by_leaf(file_list_path, source_prefix):
    """Return dict: rel_dir -> sorted list of relative file paths.

    Both `source_prefix` and each input path are passed through
    os.path.normpath, so quirks like doubled slashes (`/foo//bar`) collapse
    consistently before the prefix check. Identical semantics to
    tar_and_transfer.py.
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
    """Load existing sha256sum-format file as dict shard_rel -> digest."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
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


def build_staging_tree(source_prefix, rel_files, stage_root, use_hardlinks):
    """Create a tree under stage_root containing entries at rel_files,
    hardlinked (or copied) from source_prefix/rel_files.

    Returns the number of files placed.
    """
    n = 0
    for rel in rel_files:
        src = os.path.join(source_prefix, rel)
        dst = os.path.join(stage_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if use_hardlinks:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        n += 1
    return n


def make_squashfs(stage_root, sqsh_path, processors, zstd_level, block_size, dry_run):
    """Run mksquashfs on stage_root -> sqsh_path. Overwrites any existing
    sqsh_path (we use -noappend)."""
    sqsh_path = str(sqsh_path)
    # mksquashfs appends to existing files by default; -noappend forces a
    # fresh build. Be defensive and remove any stale partial output too so
    # mksquashfs doesn't get confused on flags like -reproducible.
    if os.path.exists(sqsh_path):
        os.remove(sqsh_path)
    cmd = [
        "mksquashfs",
        stage_root,
        sqsh_path,
        "-comp", "zstd",
        "-Xcompression-level", str(zstd_level),
        "-b", block_size,
        "-no-xattrs",
        "-noI", "-noD",
        "-noappend",
        "-no-progress",
        "-quiet",
    ]
    if processors and processors > 0:
        cmd += ["-processors", str(processors)]
    if dry_run:
        log("DRY-RUN mksquashfs: " + " ".join(shlex.quote(c) for c in cmd))
        return
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    source_prefix = os.path.normpath(args.source_prefix)
    if not source_prefix.endswith("/"):
        source_prefix = source_prefix + "/"
    if not os.path.isdir(source_prefix):
        log(f"ERROR: source-prefix is not a directory: {source_prefix}")
        sys.exit(1)

    if shutil.which("mksquashfs") is None:
        log("ERROR: mksquashfs not found in PATH.")
        sys.exit(1)

    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    list_stem = os.path.splitext(os.path.basename(args.file_list))[0]
    manifest_path = args.manifest or str(staging / f"{list_stem}.manifest.txt")
    checksums_path = args.checksums or str(staging / f"{list_stem}.sha256sums.txt")
    complete_path = str(staging / f"{list_stem}.complete")
    # Per-list staging scratch dir for hardlink trees. Cleared on entry to
    # each shard build so we never inherit state from a crashed prior run.
    stage_scratch = staging / f".{list_stem}.stage_tmp"

    done = load_manifest(manifest_path)
    checksums = load_checksums(checksums_path)
    log(f"Manifest: {manifest_path} ({len(done)} entries already done)")
    log(f"Checksums: {checksums_path} ({len(checksums)} entries)")

    log(f"Reading file list: {args.file_list}")
    groups = group_files_by_leaf(args.file_list, source_prefix)
    n_files = sum(len(v) for v in groups.values())
    log(f"Grouped into {len(groups)} leaf directories covering {n_files} files.")

    if args.dry_run:
        log("DRY-RUN mode: no shards will be written.")

    n_total = len(groups)
    n_processed = 0
    n_skipped = 0
    n_failed = 0
    processed_this_run = set()
    bytes_this_run = 0
    t0 = time.time()

    for i, rel_dir in enumerate(sorted(groups.keys()), start=1):
        files = groups[rel_dir]
        if rel_dir in done:
            n_skipped += 1
            if n_skipped <= 3 or n_skipped % 100 == 0:
                log(f"[{i}/{n_total}] SKIP {rel_dir} (already in manifest)")
            continue

        shard_rel = rel_dir + ".sqsh"
        shard_path = staging / shard_rel
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        # Always start from a clean scratch so a previous failed shard build
        # cannot leak hardlinks into this one.
        if stage_scratch.exists():
            shutil.rmtree(stage_scratch)
        stage_scratch.mkdir(parents=True)

        log(f"[{i}/{n_total}] Building {rel_dir} ({len(files)} files) -> {shard_path}")

        try:
            try:
                if args.dry_run:
                    log(f"  DRY-RUN: would hardlink {len(files)} files into {stage_scratch}")
                else:
                    n_linked = build_staging_tree(
                        source_prefix.rstrip("/"),
                        files,
                        str(stage_scratch),
                        use_hardlinks=not args.no_hardlinks,
                    )
                    if n_linked != len(files):
                        raise RuntimeError(
                            f"staging tree built {n_linked} files, expected {len(files)}"
                        )
            except OSError as e:
                # EXDEV (cross-device link) is the headline failure mode.
                # Surface a useful hint and keep going so other shards still
                # get a chance — though for EXDEV every shard will hit this.
                log(f"[{i}/{n_total}] FAILED staging ({rel_dir}): {e}")
                if getattr(e, "errno", None) == 18:  # EXDEV
                    log("  Hint: source and staging dir are on different filesystems. "
                        "Re-run with --no-hardlinks or move staging onto the same FS.")
                n_failed += 1
                continue

            try:
                make_squashfs(
                    str(stage_scratch),
                    shard_path,
                    processors=args.processors,
                    zstd_level=args.zstd_level,
                    block_size=args.block_size,
                    dry_run=args.dry_run,
                )
            except subprocess.CalledProcessError as e:
                log(f"[{i}/{n_total}] FAILED mksquashfs ({rel_dir}): {e}")
                # Remove any partial output so the next run starts clean.
                if shard_path.exists():
                    try:
                        shard_path.unlink()
                    except OSError:
                        pass
                n_failed += 1
                continue
        finally:
            if stage_scratch.exists():
                shutil.rmtree(stage_scratch, ignore_errors=True)

        if not args.dry_run:
            if not shard_path.exists() or shard_path.stat().st_size == 0:
                log(f"[{i}/{n_total}] FAILED ({rel_dir}): shard missing or empty after mksquashfs")
                n_failed += 1
                continue
            digest = sha256_of_file(str(shard_path))
            checksums[shard_rel] = digest
            write_checksums(checksums_path, checksums)
            sz = shard_path.stat().st_size
            bytes_this_run += sz
            log(f"[{i}/{n_total}] sha256={digest[:16]}...  size={sz:,} bytes")
            append_line(manifest_path, rel_dir)
            processed_this_run.add(rel_dir)

        n_processed += 1
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        mb_rate = (bytes_this_run / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        log(f"[{i}/{n_total}] DONE {rel_dir}  (processed={n_processed} skipped={n_skipped} "
            f"failed={n_failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s {mb_rate:.0f} MB/s)")

        if args.limit and n_processed >= args.limit:
            log(f"Reached --limit {args.limit}; stopping.")
            break

    # Final cleanup of any leftover scratch (defensive).
    if stage_scratch.exists():
        shutil.rmtree(stage_scratch, ignore_errors=True)

    log(f"Done. processed={n_processed} skipped={n_skipped} failed={n_failed} "
        f"elapsed={time.time() - t0:.0f}s built={bytes_this_run / (1024**3):.2f} GiB")

    all_done = done.union(processed_this_run)
    fully_complete = (
        not args.dry_run
        and n_failed == 0
        and all_done == set(groups.keys())
        and not args.limit
    )
    if fully_complete:
        Path(complete_path).touch()
        log(f"All {n_total} shards complete. Wrote sentinel: {complete_path}")

    if n_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
