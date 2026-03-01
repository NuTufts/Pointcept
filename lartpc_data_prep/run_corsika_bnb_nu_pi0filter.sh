#!/bin/bash

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl
INPUTLIST=${WORKDIR}/inputlists/bnb_nu_pi0filter_corsika.txt
JOBIDLIST=${WORKDIR}/jobidlist_bnbnu_pi0_corsika.txt
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_showerfragments/bnb_nu_pi0filter_corsika/
SCRIPT=process_dlmerged_to_hdf5_event_files.py

OFFSET=0
TAG=coriska_bnb_nu_pi0filter

stride=100
jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${OFFSET}+${stride}*${jobid}")

mkdir -p $WORKDIR
jobworkdir=`printf "%s/workdir/ubprod_on_tufts_${TAG}_jobid_%03d" $WORKDIR $jobid`
mkdir -p $jobworkdir
mkdir -p $OUTPUT_DIR

local_jobdir=`printf /tmp/mlreco_dlmerged2hdf5_${TAG}_jobid%03d $jobid`
rm -rf $local_jobdir
mkdir -p $local_jobdir
cd $local_jobdir
touch log_${TAG}_jobid${jobid}.txt
local_logfile=`echo ${local_jobdir}/log_${TAG}_jobid${jobid}.txt`

# setup container
cd ${UBDL_DIR}
source setenv_pointcept_container.sh

# go back to the job dir
cd $local_jobdir

# copy the job script
cp $WORKDIR/${SCRIPT} .

for i in {1..100}
do

    let lineno=$startline+$i

    # get job request number
    fileno=`sed -n ${lineno}p ${JOBIDLIST}`
    echo "FILE NO: ${fileno}" >> ${local_logfile}

    # get the job file
    inputfile=`sed -n ${fileno}p ${INPUTLIST}`
    echo "inputfile: ${inputfile}" >> ${local_logfile}

    # run job    
    python3 ${SCRIPT} -i ${inputfile} > log_process.txt
    head -n 100 log_process.txt >> ${local_logfile}
    tail -n 100 log_process.txt >> ${local_logfile}
    echo "==== FILES MADE ====" >> ${local_logfile}
    ls -lh *.h5 >> ${local_logfile}

    # copy to output location
    let nsubdir1=$(expr "${fileno}/1000")
    zsubdir1=`printf %03d ${nsubdir1}`

    let nsubdir2=$(expr "${fileno}/100")
    zsubdir2=`printf %03d ${nsubdir2}`

    outfolder=${OUTPUT_DIR}/${zsubdir1}/${zsubdir2}/
    echo "Output folder: ${outfolder}"

    mkdir -p $outfolder

    cp *.h5 ${outfolder}/

    # destroy all root files from this run
    rm *.h5
    
    # remove process log file
    rm log_process.txt

done

# copy log files
cp ${local_logfile} ${jobworkdir}/

# clean up
cd /tmp
rm -r $local_jobdir
