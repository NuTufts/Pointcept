#!/bin/bash
#SBATCH --job-name=sqfs_pack
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=16G --cpus-per-task=8
#SBATCH --output=logs/data_prep/sqfs_pack.%A_%a.log --error=logs/data_prep/sqfs_pack.%A_%a.log
# Pack a merged_sp tree (2-level NNN/NN/ layout, ~10 MB h5 per event) into one
# squashfs image PER TOP-LEVEL DIRECTORY, for transfer to / use on Lustre
# systems (ALCF Polaris) where 10^5 small files are slow to transfer and to
# open. Images are uncompressed-friendly (h5 is already gzip-compressed).
#
#   SRC=<merged_sp dir> OUT=<image dir> SAMPLE=<name> \
#     sbatch --array=0-$((N-1)) lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh
#   (N = number of top-level dirs; array index -> sorted top-level dir)
#
# Per task: <OUT>/<SAMPLE>_merged_sp_<dir>.sqfs + <...>.sqfs.manifest
# (file count, bytes, listing) + <...>.sqfs.ok when the image lists the same
# number of files as the source. Idempotent: a task whose .ok exists exits.
set -eu
SRC=${SRC:?merged_sp dir}
OUT=${OUT:?output dir for images}
SAMPLE=${SAMPLE:?sample name, e.g. bnb5e19}
COMP=${COMP:-lz4}            # lz4: fast to read; h5 payload is already compressed
NPROC=${SLURM_CPUS_PER_TASK:-8}
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT"
mapfile -t DIRS < <(ls -1 "$SRC" | sort)
D=${DIRS[$TASK]:?no top-level dir for array index $TASK}
IMG=$OUT/${SAMPLE}_merged_sp_${D}.sqfs
if [ -e "$IMG.ok" ]; then echo "$IMG already verified; nothing to do"; exit 0; fi
NSRC=$(find "$SRC/$D" -type f -name '*.h5' | wc -l)
echo ">>> $(date) packing $SRC/$D ($NSRC h5, $(du -sh "$SRC/$D" | cut -f1)) -> $IMG (comp=$COMP, $NPROC procs)"
rm -f "$IMG"
# -keep-as-directory: the image root contains the directory "<D>/" so several
# images can be bind-mounted side by side under one merged_sp mount point.
mksquashfs "$SRC/$D" "$IMG" -comp "$COMP" -no-xattrs -noappend -keep-as-directory \
  -processors "$NPROC" -mem 8G -info -progress 2>&1 | grep -vE "^\s*$" | tail -n 5
NIMG=$(unsquashfs -lls "$IMG" 2>/dev/null | grep -c '\.h5$')
BYTES=$(stat -c %s "$IMG")
{ echo "sample=$SAMPLE dir=$D source=$SRC/$D"; echo "image=$IMG bytes=$BYTES comp=$COMP";
  echo "h5_in_source=$NSRC h5_in_image=$NIMG"; echo "created=$(date -Is) host=$(hostname)";
  echo "sha256=$(sha256sum "$IMG" | cut -d' ' -f1)"; } > "$IMG.manifest"
if [ "$NIMG" -eq "$NSRC" ] && [ "$NSRC" -gt 0 ]; then
  touch "$IMG.ok"; echo ">>> $(date) OK $IMG: $NIMG files, $((BYTES/1024/1024/1024)) GiB"
else
  echo ">>> $(date) MISMATCH $IMG: source $NSRC vs image $NIMG"; exit 1
fi
