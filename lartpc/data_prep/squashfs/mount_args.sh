#!/bin/bash
# Print the apptainer --bind flags that mount every <SAMPLE>_merged_sp_<dir>.sqfs
# image in a directory under one merged_sp mount point (read-only), so the
# tree looks exactly like the Tufts merged_sp/ directory:
#   apptainer exec --nv $(bash mount_args.sh /eagle/.../bnb5e19_squashfs /data/bnb5e19/merged_sp) pointcept.sif ...
# Then build the list once: list_merged_sp.py --dir /data/bnb5e19/merged_sp --out <list>
IMGDIR=${1:?image dir}; MNT=${2:?mount point inside the container}
for img in "$IMGDIR"/*_merged_sp_*.sqfs; do
  d=$(basename "$img" .sqfs); d=${d##*_merged_sp_}
  printf -- '--bind %s:%s/%s:image-src=/%s,ro ' "$img" "$MNT" "$d" "$d"
done
echo
