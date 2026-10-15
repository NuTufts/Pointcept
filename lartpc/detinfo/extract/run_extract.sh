#!/bin/bash
# run_extract.sh -- extract LArSoft detector info for an SBN experiment into HDF5.
#
# Starts the SL7 container with cvmfs mounted through singcvmfs (EL7 cvmfsexec
# dist), sets up the experiment code at the requested version, and runs
# extract_larsoft_detinfo.py.
#
#   bash run_extract.sh <detector> <version> [qualifiers] [outdir]
#   bash run_extract.sh uboone v10_04_07_13                 # quals default: e26:prof (v08/v09: e17:prof)
#   bash run_extract.sh sbnd v10_14_02                      # -> $DETINFO_ROOT/sbnd/sbndcode_v10_14_02/
#
# Notes (Tufts cluster):
#   * sbndcode v10_14_02 is built for SL7 only (slf7.x86_64.e26).
#   * sbn.opensciencegrid.org must be mounted, or setup_sbnd.sh fails
#     ("Error setting up SBN UPS area").
#   * The first run on a node takes ~10 min (cold node-local cvmfs cache);
#     CPU time is small.
#   * h5py is not in the LArSoft python; a vendored py3.9 manylinux2014 wheel
#     (h5py 3.11.0, unpacked) lives in $PY39_SITE (install: install_py39_h5py.sh).
#   * Extra extractor options via DETINFO_EXTRA_ARGS (e.g. "--skip-photonlib").
#   * Also copies the photon visibility library (~130 MB for SBND) and writes a
#     dense photonlib_<det>_<ver>.h5 next to the main file.

set -e
DET=${1:?detector (sbnd|uboone|icarus)}
VERSION=${2:?experiment code version, e.g. v10_14_02}
QUALS=${3:-}
if [ -z "$QUALS" ]; then
  case $VERSION in v08_*|v09_*) QUALS=e17:prof ;; *) QUALS=e26:prof ;; esac
fi
DETINFO_ROOT=${DETINFO_ROOT:-/cluster/tufts/wongjiradlab/larbys/detinfo}

case $DET in
  sbnd)
    SETUP=/cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh
    PRODUCT=sbndcode
    REPOS=sbn.opensciencegrid.org,sbnd.opensciencegrid.org,larsoft.opensciencegrid.org,fermilab.opensciencegrid.org ;;
  uboone)
    SETUP=/cvmfs/uboone.opensciencegrid.org/products/setup_uboone.sh
    PRODUCT=uboonecode
    REPOS=uboone.opensciencegrid.org,larsoft.opensciencegrid.org,fermilab.opensciencegrid.org ;;
  icarus)
    SETUP=/cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh
    PRODUCT=icaruscode
    REPOS=sbn.opensciencegrid.org,icarus.opensciencegrid.org,larsoft.opensciencegrid.org,fermilab.opensciencegrid.org ;;
  *) echo "unknown detector $DET"; exit 1 ;;
esac
OUTDIR=${4:-$DETINFO_ROOT/$DET/${PRODUCT}_${VERSION}}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CVMFSEXEC_EL7=/cluster/tufts/wongjiradlab/larbys/cvmfs_containers/cvmfsexec-el7
CONTAINER=/cluster/tufts/wongjiradlab/larbys/cvmfs_containers/uboone-devel-sl7-11
PY39_SITE=${PY39_SITE:-/cluster/tufts/wongjiradlab/larbys/detinfo/py39_site}
if [ ! -d "$PY39_SITE/h5py" ]; then
  echo "h5py for the LArSoft python3.9 not found in $PY39_SITE."
  echo "Install it once with:  bash $HERE/install_py39_h5py.sh $PY39_SITE"
  exit 1
fi

mkdir -p "$OUTDIR"
GIT_COMMIT=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
if ! git -C "$HERE" ls-files --error-unmatch extract_larsoft_detinfo.py > /dev/null 2>&1; then
  GIT_COMMIT="${GIT_COMMIT}+untracked"
elif ! git -C "$HERE" diff --quiet HEAD -- extract_larsoft_detinfo.py 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}+dirty"
fi

export SINGCVMFS_REPOSITORIES=$REPOS
export SINGCVMFS_CACHEDIR=/tmp/cvmfs-cache-${USER}-el7
export SINGCVMFS_LOGDIR=$OUTDIR/singcvmfs-logs
mkdir -p "$SINGCVMFS_CACHEDIR" "$SINGCVMFS_LOGDIR"

INNER=$OUTDIR/_run_inner.sh
cat > "$INNER" <<EOF
source $SETUP > /dev/null 2>&1
setup $PRODUCT $VERSION -q $QUALS || { echo "setup $PRODUCT $VERSION -q $QUALS failed"; exit 1; }
export PYTHONPATH=$PY39_SITE:\$PYTHONPATH
export DETINFO_GIT_COMMIT=$GIT_COMMIT
export DETINFO_CONTAINER=$CONTAINER
export DETINFO_QUALS=$QUALS
cd $OUTDIR
python $HERE/extract_larsoft_detinfo.py --detector $DET --outdir $OUTDIR ${DETINFO_EXTRA_ARGS:-} 2>&1 | grep -v '^%MSG\|^Info in <TGeo'
EOF

echo "extracting $DET $PRODUCT $VERSION ($QUALS) -> $OUTDIR"
"$CVMFSEXEC_EL7/singcvmfs" -s exec -B /cluster:/cluster "$CONTAINER" bash "$INNER" 2>&1 | tee "$OUTDIR/extract.log"
rm -f "$INNER"
