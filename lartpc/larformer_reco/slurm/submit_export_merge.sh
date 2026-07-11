#!/bin/bash
#
# Merge step for the sharded gen2ntuple export: ROOT hadd of the
# ${OUT%.root}_shardNN.root files written by submit_export_shard.sh into the
# final ${OUT}, entry-count validation, then shard cleanup (KEEP_SHARDS=1 to
# keep them). hadd comes from the ubdl environment sourced inside the
# pointcept container.
#
#   OUT=... sbatch --dependency=afterok:<shardJID> submit_export_merge.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=ntuphadd
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/export/ntuphadd.%j.%N.log
#SBATCH --error=logs/export/ntuphadd.%j.%N.err

set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
RECODIR=${WORKDIR}/lartpc/larformer_reco
TAG=${TAG:-mcc9_bnbnu_overlay_1500_full}
OUT=${OUT:-${RECODIR}/output/${TAG}/dlgen2_larformer_ntuple_${TAG}_67k.root}
KEEP_SHARDS=${KEEP_SHARDS:-0}

SHARDS=( $(ls "${OUT%.root}"_shard*.root | sort) )
echo ">>> hadd ${#SHARDS[@]} shards -> ${OUT}"
mkdir -p "${WORKDIR}/logs/export"

module load apptainer 2>/dev/null || true
apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${UBDL_DIR} && source setenv_pointcept_container.sh >/dev/null 2>&1 && \
  cd ${WORKDIR} && \
  hadd -f ${OUT} ${SHARDS[*]} && \
  python3 -c \"
import ROOT
tot = 0
for s in '${SHARDS[*]}'.split():
    f = ROOT.TFile.Open(s); tot += f.Get('EventTree').GetEntries(); f.Close()
f = ROOT.TFile.Open('${OUT}')
n = f.Get('EventTree').GetEntries(); npot = f.Get('potTree').GetEntries()
f.Close()
assert n == tot, f'entry mismatch: merged {n} != shard sum {tot}'
print(f'>>> merged EventTree entries: {n} (== shard sum), potTree: {npot}')
\"
"
if [ "${KEEP_SHARDS}" != "1" ]; then
  rm -f "${OUT%.root}"_shard*.root
  echo ">>> shard files removed (KEEP_SHARDS=1 to keep)"
fi
echo "DONE merge -> ${OUT}"
