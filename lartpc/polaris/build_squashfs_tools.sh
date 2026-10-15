#!/bin/bash
# Build squashfs-tools (mksquashfs / unsquashfs) INSIDE the pointcept container
# and install them under $POL_ASSETS/bin, so stage B can pack per-shard
# squashfs images on Polaris without a container rebuild and without any
# site-installed package (Polaris has no mksquashfs; checked 2026-10-15).
# mksquashfs is pure user space (no kernel module, no privilege), so the binary
# works on login, GPU and Krux nodes alike; run_cascade_shard.sh invokes it
# through pol_exec (it is linked against the container's glibc/zlib).
#
#   source lartpc/polaris/polaris_env.sh
#   bash lartpc/polaris/build_squashfs_tools.sh [install dir, default $POL_ASSETS/bin] [source tarball]
#
# Compressors: gzip always; lz4/xz/zstd only if their headers exist in the
# container (Ubuntu 22.04 pointcept_cuml.sif has zlib only -> gzip). The packer
# picks the compressor from `mksquashfs -help` (POL_PACK_COMP=auto).
set -euo pipefail
VER=4.6.1
SHA=9c4974e07c61547dae14af4ed1f358b7d04618ae194e54d6be72ee126f0d2f53
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! type pol_exec >/dev/null 2>&1; then source "$HERE/polaris_env.sh"; fi
DEST=${1:-$POL_ASSETS/bin}
TARBALL=${2:-}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sqtools.XXXX")
trap 'rm -rf "$WORK"' EXIT
if [ -z "$TARBALL" ]; then
  TARBALL=$WORK/squashfs-tools-$VER.tar.gz
  echo ">>> downloading squashfs-tools $VER"
  curl -sSL -o "$TARBALL" "https://github.com/plougher/squashfs-tools/archive/refs/tags/$VER.tar.gz"
fi
echo "$SHA  $TARBALL" | sha256sum -c - >/dev/null || { echo "ERROR: tarball sha256 mismatch ($TARBALL)" >&2; exit 2; }
tar -xzf "$TARBALL" -C "$WORK"
mkdir -p "$DEST"
# build in the container: gcc + zlib are in the image; extra compressors if headers exist
pol_exec "cd '$WORK/squashfs-tools-$VER/squashfs-tools' && \
  LZ4=\$([ -e /usr/include/lz4.h ] && echo 1 || echo 0); XZ=\$([ -e /usr/include/lzma.h ] && echo 1 || echo 0); ZSTD=\$([ -e /usr/include/zstd.h ] && echo 1 || echo 0); \
  echo \">>> building with GZIP=1 LZ4=\$LZ4 XZ=\$XZ ZSTD=\$ZSTD\"; \
  make -j4 GZIP_SUPPORT=1 LZ4_SUPPORT=\$LZ4 XZ_SUPPORT=\$XZ ZSTD_SUPPORT=\$ZSTD LZO_SUPPORT=0 XATTR_SUPPORT=0 >/dev/null && \
  cp mksquashfs unsquashfs '$DEST/' && '$DEST/mksquashfs' -version | head -1 && \
  '$DEST/mksquashfs' -help 2>&1 | sed -n '/^Compressors available/,/^$/p' | head -8"
echo ">>> installed: $DEST/mksquashfs $DEST/unsquashfs (invoke through pol_exec; polaris_env.sh uses them when POL_PACK_FMT=sqfs)"
