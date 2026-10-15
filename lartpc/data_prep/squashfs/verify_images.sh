#!/bin/bash
# Verify a directory of squashfs images against their .manifest files (after a
# Globus transfer, on either side; stdlib tools only):
#   bash verify_images.sh <image dir> [--quick]      (--quick: size only, no sha256)
# Exit 1 if any image is missing, has a different size, or a different sha256.
set -u
DIR=${1:?image dir}; QUICK=${2:-}
rc=0; n=0
for m in "$DIR"/*.manifest; do
  [ -e "$m" ] || { echo "no manifests in $DIR"; exit 1; }
  img=$DIR/$(basename "$m" .manifest); n=$((n + 1))
  want_b=$(sed -n 's/.*bytes=\([0-9]*\).*/\1/p' "$m" | head -1); want_s=$(sed -n 's/^sha256=//p' "$m")
  if [ ! -f "$img" ]; then echo "MISSING  $img"; rc=1; continue; fi
  got_b=$(stat -c %s "$img")
  if [ "$got_b" != "$want_b" ]; then echo "SIZE     $img: $got_b != $want_b"; rc=1; continue; fi
  if [ "$QUICK" = --quick ]; then echo "size ok  $img"; continue; fi
  got_s=$(sha256sum "$img" | cut -d' ' -f1)
  if [ "$got_s" != "$want_s" ]; then echo "SHA256   $img: mismatch"; rc=1; else echo "ok       $img"; fi
done
echo ">>> $n image(s) checked: $([ $rc -eq 0 ] && echo ALL OK || echo FAILURES)"
exit $rc
