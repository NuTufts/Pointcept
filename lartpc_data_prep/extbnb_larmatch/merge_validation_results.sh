#!/bin/bash
#
# Merge validation results from all array jobs into a single
# bad files list and a validated good files list.
#
# Run after all submit_validate_for_sonata.sh jobs complete:
#   bash merge_validation_results.sh
#

WORKDIR=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/lartpc_data_prep/extbnb_larmatch
OUTDIR=${WORKDIR}/validation_sonata_results
FILELIST=${WORKDIR}/hdlist_extbnb_larmatch_run3_g1.txt

# Merge all bad file lists
bad_merged=${OUTDIR}/bad_files_all.txt
> ${bad_merged}
for f in ${OUTDIR}/bad_files_*.txt; do
    [ -f "$f" ] && cat "$f" >> ${bad_merged}
done
bad_count=$(wc -l < ${bad_merged})

# Also merge the per-chunk validated file lists
good_merged=${WORKDIR}/hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt
> ${good_merged}
for f in ${OUTDIR}/chunk_*_sonata_validated.txt; do
    [ -f "$f" ] && cat "$f" >> ${good_merged}
done
good_count=$(wc -l < ${good_merged})

total=$(wc -l < ${FILELIST})

echo "========================================"
echo "Validation merge complete"
echo "  Total files:     ${total}"
echo "  Good files:      ${good_count}"
echo "  Bad files:       ${bad_count}"
echo "  Bad files list:  ${bad_merged}"
echo "  Good files list: ${good_merged}"
echo "========================================"
