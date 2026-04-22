#!/bin/bash
#
# make_filelist.sh
# Generate a file list of all produced HDF5 files for use with Pointcept training.
#
# Usage: bash make_filelist.sh [output_file]
#

OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_ext_larmatch/extbnb_run3_G1/
OUTFILE=${1:-extbnb_larmatch_filelist.txt}

find ${OUTPUT_DIR} -name "pointceptdata_*.h5" -type f | sort > ${OUTFILE}

nfiles=$(wc -l < ${OUTFILE})
echo "Found ${nfiles} H5 files, written to ${OUTFILE}"
