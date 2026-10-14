#!/bin/bash
# run_convert.sh -- convert SBND larcv2 files to LArFormer per-event HDF5 inside the pointcept
# container (larcv2 from test/larcv2, built per README section 1).
#
#   bash lartpc/data_prep/sbnd_official/run_convert.sh -o OUTDIR FILE.root [FILE2.root ...] [-- converter options]
#   then: python3 lartpc/data_prep/sbnd_official/validate_converted.py OUTDIR --flash   (in the container)
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CONTAINER=${POINTCEPT_CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
LARCV2_DIR=${LARCV2_DIR:-$HERE/test/larcv2}
OUT=""; FILES=(); EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    -o) OUT=$2; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) FILES+=("$(readlink -f "$1")") ;;
  esac; shift
done
[ -n "$OUT" ] && [ ${#FILES[@]} -gt 0 ] || { echo "usage: $0 -o OUTDIR FILE.root ... [-- options]"; exit 1; }
mkdir -p "$OUT"
apptainer exec -B /cluster:/cluster "$CONTAINER" bash -c "
  source $HERE/test/setenv_larcv2.sh > /dev/null
  source $LARCV2_DIR/configure.sh -q
  cd $REPO
  python3 lartpc/data_prep/sbnd_official/convert_larcv2_to_larformer_h5.py -i ${FILES[*]} -o $(readlink -f "$OUT") ${EXTRA[*]}
" 2>&1 | grep -v "NORMAL\|UserWarning\|_warn("
