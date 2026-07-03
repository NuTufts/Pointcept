#!/bin/bash

BNBNU=hdflist_bnbnu_corsika_validated.txt
BNBPIZERO=hdflist_bnb_nu_pi0_corsika_validated.txt
BNBNUE=hdflist_bnbnue_corsika_validated.txt
BNBPIZEROFILTER=hdflist_bnb_nu_pi0filter_corsika_validated.txt
BNBCHARGEDPI=hdflist_bnb_nu_chargedpiplus_corsika_validated.txt

OUTNAME_TAG=$1

cat $BNBNU $BNBPIZERO $BNBNUE $BNBPIZEROFILTER $BNBCHARGEDPI > ${OUTNAME_TAG}_validated.txt
cat ${OUTNAME_TAG}_validated.txt | shuf > ${OUTNAME_TAG}_validated_shuffled.txt