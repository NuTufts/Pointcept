#!/bin/bash
#SBATCH --job-name=sqfs_pack
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=16G --cpus-per-task=8
#SBATCH --output=logs/data_prep/sqfs_pack.%A_%a.log --error=logs/data_prep/sqfs_pack.%A_%a.log
# Pack a merged_sp tree (~10 MB h5 per event) into squashfs images for transfer
# to / use on Lustre systems (ALCF Polaris) where 10^5 small files are slow to
# transfer and to open. Images are uncompressed-friendly (h5 is already
# gzip-compressed). Two modes:
#
#  (1) per TOP-LEVEL DIRECTORY (legacy; bnb5e19's 2-level NNN/NN/ layout):
#      SRC=<merged_sp dir> OUT=<image dir> SAMPLE=<name> \
#        sbatch --array=0-$((N-1)) lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh
#      (N = number of top-level dirs; array index -> sorted top-level dir)
#      -> <OUT>/<SAMPLE>_merged_sp_<dir>.sqfs, image root = "<dir>/" (-keep-as-directory)
#
#  (2) per FILENO GROUP (2026-10-15; converter output NNN/<flat>, one sample =
#      one or two huge NNN dirs): GROUP_FILENOS=10 packs filenos
#      [k*10, k*10+9] into image k, so a 119-file sample becomes 12 images of
#      ~50 GB that can be transferred / deleted in waves:
#      SRC=.. OUT=.. SAMPLE=.. GROUP_FILENOS=10 sbatch --array=0-11 ...
#      -> <OUT>/<SAMPLE>_merged_sp_g<kk>.sqfs, image root = the NNN/ dirs that
#      hold the group's files (staged with HARD LINKS: no extra disk; OUT must
#      be on the same volume as SRC). mount_args.sh mounts a g<kk> image at
#      <mnt>/g<kk>/ (image-src=/), i.e. <mnt>/g00/000/merged_*.h5; the list
#      (list_merged_sp.py) and the chain are layout-invariant (basename keys).
#
# Per task also: <image>.manifest (file count, bytes, sha256) and <image>.ok
# once the image lists the same number of h5 files as the source. Idempotent:
# a task whose .ok exists exits. Rerun the array to fill gaps.
set -eu
SRC=${SRC:?merged_sp dir}
OUT=${OUT:?output dir for images}
SAMPLE=${SAMPLE:?sample name, e.g. bnb5e19}
COMP=${COMP:-lz4}            # lz4: fast to read; h5 payload is already compressed
NPROC=${SLURM_CPUS_PER_TASK:-8}
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT"
STG=""
if [ -n "${GROUP_FILENOS:-}" ]; then
  LO=$((TASK * GROUP_FILENOS)); HI=$((LO + GROUP_FILENOS - 1))
  G=$(printf 'g%02d' "$TASK")
  IMG=$OUT/${SAMPLE}_merged_sp_${G}.sqfs
  if [ -e "$IMG.ok" ]; then echo "$IMG already verified; nothing to do"; exit 0; fi
  STG=$OUT/.stage_$G; rm -rf "$STG"; mkdir -p "$STG"
  NSRC=0
  for i in $(seq "$LO" "$HI"); do
    ZFN=$(printf '%05d' "$i"); DD=$(printf '%03d' $((i / 100)))
    for f in "$SRC/$DD"/merged_${SAMPLE}_fileno${ZFN}_*.h5; do
      [ -e "$f" ] || continue
      mkdir -p "$STG/$DD"
      ln "$f" "$STG/$DD/" || { echo "ERROR: hard link failed ($f -> $STG): OUT must be on the same volume as SRC" >&2; exit 2; }
      NSRC=$((NSRC + 1))
    done
  done
  [ "$NSRC" -gt 0 ] || { echo "group $G (filenos $LO-$HI): no files; nothing to pack"; rm -rf "$STG"; exit 0; }
  SRCDIR=$STG; KEEP=""; WHAT="group $G filenos $LO-$HI"
else
  mapfile -t DIRS < <(ls -1 "$SRC" | sort)
  D=${DIRS[$TASK]:?no top-level dir for array index $TASK}
  IMG=$OUT/${SAMPLE}_merged_sp_${D}.sqfs
  if [ -e "$IMG.ok" ]; then echo "$IMG already verified; nothing to do"; exit 0; fi
  NSRC=$(find "$SRC/$D" -type f -name '*.h5' | wc -l)
  # -keep-as-directory: the image root contains the directory "<D>/" so several
  # images can be bind-mounted side by side under one merged_sp mount point.
  SRCDIR=$SRC/$D; KEEP="-keep-as-directory"; WHAT="dir $D"
fi
echo ">>> $(date) packing $WHAT ($NSRC h5, $(du -shL "$SRCDIR" | cut -f1)) -> $IMG (comp=$COMP, $NPROC procs)"
rm -f "$IMG"
# shellcheck disable=SC2086
mksquashfs "$SRCDIR" "$IMG" -comp "$COMP" -no-xattrs -noappend $KEEP \
  -processors "$NPROC" -mem 8G -info -progress 2>&1 | grep -vE "^\s*$" | tail -n 5
[ -n "$STG" ] && rm -rf "$STG"
NIMG=$(unsquashfs -lls "$IMG" 2>/dev/null | grep -c '\.h5$')
BYTES=$(stat -c %s "$IMG")
{ echo "sample=$SAMPLE $WHAT source=$SRC"; echo "image=$IMG bytes=$BYTES comp=$COMP";
  echo "h5_in_source=$NSRC h5_in_image=$NIMG"; echo "created=$(date -Is) host=$(hostname)";
  echo "sha256=$(sha256sum "$IMG" | cut -d' ' -f1)"; } > "$IMG.manifest"
if [ "$NIMG" -eq "$NSRC" ] && [ "$NSRC" -gt 0 ]; then
  touch "$IMG.ok"; echo ">>> $(date) OK $IMG: $NIMG files, $((BYTES/1024/1024/1024)) GiB"
else
  echo ">>> $(date) MISMATCH $IMG: source $NSRC vs image $NIMG"; exit 1
fi
