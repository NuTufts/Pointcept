
# clean up previous run
rm output/eval_reco_valdata_all/*.npz
rm logs/eval_reco/*.err
rm logs/eval_reco/*.log

# submit jobs
JID=$(NSHARDS=10 sbatch --parsable --array=0-9 submit_eval_reco_shard.sh)
sbatch --dependency=afterok:${JID} submit_eval_reco_merge.sh