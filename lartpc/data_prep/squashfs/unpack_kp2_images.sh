#!/bin/bash
#SBATCH --job-name=kp2_unpack
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=8G --cpus-per-task=4
#SBATCH --output=logs/data_prep/kp2_unpack.%j.log --error=logs/data_prep/kp2_unpack.%j.log
# Unpack the per-shard keypoint2 images made by the Polaris stage-B pass
# (kp2_<tag>_shardNNN.sqfs, or .tar fallback) into one keypoint2_streams tree:
#   IMGDIR=<dir with images + manifests> DEST=<DATADIR>/keypoint2_streams sbatch unpack_kp2_images.sh
# Verifies every image against its manifest (sha256) first, refuses on any
# mismatch, then unsquashfs -f (shards hold disjoint event indices, so the
# per-shard subtrees merge). Writes $DEST/.unpacked_ok with the h5 count.
set -eu
IMGDIR=${IMGDIR:?image dir}; DEST=${DEST:?keypoint2_streams dir}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$HERE/verify_images.sh" "$IMGDIR" || { echo "ERROR: image verification failed; not unpacking" >&2; exit 2; }
mkdir -p "$DEST"
WANT=0; N=0
for m in "$IMGDIR"/*.manifest; do
  img=${m%.manifest}; N=$((N + 1))
  WANT=$((WANT + $(sed -n 's/.*h5_in_image=\([0-9]*\).*/\1/p' "$m" | head -1)))
  case "$img" in
    *.sqfs) unsquashfs -f -n -d "$DEST" "$img" >/dev/null ;;
    *.tar)  tar -xf "$img" -C "$DEST" ;;
    *) echo "ERROR: unknown image type $img" >&2; exit 2 ;;
  esac
  echo "$(date +%FT%T) unpacked $(basename "$img")"
done
GOT=$(find "$DEST" -name 'keypoint2_event*.h5' | wc -l)
echo ">>> $N images; keypoint2 h5 in tree: $GOT (manifests say $WANT)"
[ "$GOT" -ge "$WANT" ] || { echo "ERROR: tree has fewer files than the manifests" >&2; exit 1; }
echo "images=$N h5=$GOT $(date -Is)" > "$DEST/.unpacked_ok"
