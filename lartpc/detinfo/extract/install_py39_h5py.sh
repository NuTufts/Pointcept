#!/bin/bash
# install_py39_h5py.sh -- make h5py importable by the LArSoft (UPS, python 3.9) environment.
#
# The SL7 LArSoft python has numpy but no h5py. The h5py manylinux2014 wheel
# (glibc 2.17, bundles libhdf5) runs there, so we download it with any
# python3+pip that has network access and unpack it into a directory that
# run_extract.sh puts on PYTHONPATH. Nothing is installed into the repo.
#
#   bash install_py39_h5py.sh [target_dir]
#   (default target: /cluster/tufts/wongjiradlab/larbys/detinfo/py39_site)
set -e
TARGET=${1:-/cluster/tufts/wongjiradlab/larbys/detinfo/py39_site}
H5PY_VERSION=${H5PY_VERSION:-3.11.0}   # last line with cp39 manylinux2014 wheels + numpy<2 support
TMP=$(mktemp -d)
python3 -m pip download "h5py==${H5PY_VERSION}" --no-deps --only-binary=:all: \
    --python-version 3.9 --platform manylinux_2_17_x86_64 -d "$TMP"
mkdir -p "$TARGET"
python3 -m zipfile -e "$TMP"/h5py-${H5PY_VERSION}-cp39-*.whl "$TARGET"
rm -rf "$TMP"
echo "h5py ${H5PY_VERSION} unpacked into $TARGET"
