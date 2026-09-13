#!/bin/bash
# Submit run_tier2_tranche.sh (the serial tier2-staging conversion sequencer)
# as a batch job -- login-session cleanup kills nohup/setsid sequencers on
# disconnect (2026-09-12), so long sequencers must run under slurm.
#
#   bash lartpc/data_prep/uboone_official/sbatch_tier2_tranche.sh <specfile> [jobname]
# Prints the job id. Log: logs/data_prep/<jobname>.<jobid>.log
set -eu
SPEC=${1:?specfile}; NAME=${2:-tier2_$(basename "${SPEC%.spec}")}
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
cd "$K"; [ -s "$SPEC" ] || { echo "ERROR: $SPEC missing/empty" >&2; exit 2; }
mkdir -p logs/data_prep
sbatch --parsable --job-name="$NAME" --partition=batch,wongjiradlab --time=2-00:00:00 \
  --mem=4G --cpus-per-task=1 --output="logs/data_prep/$NAME.%j.log" --error="logs/data_prep/$NAME.%j.log" \
  --wrap="bash $K/lartpc/data_prep/uboone_official/run_tier2_tranche.sh $K/$SPEC"
