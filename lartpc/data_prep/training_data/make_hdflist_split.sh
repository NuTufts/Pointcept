#!/bin/bash

HDFLIST=$1
NVALTEST_SPLIT=$2
NTEST_SPLIT=$3

let ntot=`cat ${HDFLIST} | wc -l`
echo "NUM IN HDFLIST: ${ntot}"

let ntrain=${ntot}-${NVALTEST_SPLIT}
echo "NUM Train Split: ${ntrain}"

let nval=${NVALTEST_SPLIT}-${NTEST_SPLIT}

trainsplit_name=`echo ${HDFLIST} | sed 's|\.txt|\_trainsplit.txt|g'`
echo "Train split: ${trainsplit_name}"

valtestsplit_name=`echo ${HDFLIST} | sed 's|\.txt|\_valtestsplit.txt|g'`
echo "Val+test split: ${valtestsplit_name}"

valsplit_name=`echo ${HDFLIST} | sed 's|\.txt|\_valsplit.txt|g'`
echo "Val split: ${valsplit_name}"

testsplit_name=`echo ${HDFLIST} | sed 's|\.txt|\_testsplit.txt|g'`
echo "Test split: ${testsplit_name}"


head -n ${ntrain} $HDFLIST > ${trainsplit_name}
echo "Made train split."
echo `cat ${trainsplit_name} | wc -l`

echo "tail -n ${NVALTEST_SPLIT} ${HDFLIST}> ${valtestsplit_name}"
tail -n ${NVALTEST_SPLIT} ${HDFLIST} > ${valtestsplit_name}
echo "Made val+test split"
echo `cat ${valtestsplit_name} | wc -l`


echo "head -n ${nval} ${valtestsplit_name} > ${valsplit_name}"
head -n ${nval} ${valtestsplit_name} > ${valsplit_name}
echo "Made val split"
echo `cat ${valsplit_name} | wc -l`

echo "tail -n ${NTEST_SPLIT} ${valtestsplit_name} > ${testsplit_name}"
tail -n ${NTEST_SPLIT} ${valtestsplit_name} > ${testsplit_name}
echo "Made test split"
echo `cat ${testsplit_name} | wc -l`