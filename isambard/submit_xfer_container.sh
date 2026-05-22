#!/bin/bash
#
# SLURM wrapper for transfer_shards.py with optional self-chaining.
#
# Transfers .sqsh shards already produced by build_squashfs.py to Isambard.
# Mirrors submit_transfer.sh's cert + chain logic — refresh `clifton` before
# submitting and again partway through, so the chained successor inherits a
# valid cert.
#
# Usage:
#   sbatch submit_transfer_shards.sh [--chain] [--max-chain N] \
#       <file_list> [extra args to transfer_shards.py]
#
# (Note: unlike submit_transfer.sh, no <source_prefix> argument — the shards
# already exist on disk; only the file_list is needed to derive the per-list
# stem used to find the checksums file.)
#
# Example (auto-resubmit until complete):
#   sbatch submit_transfer_shards.sh --chain --max-chain 30 \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt
#
#SBATCH --partition=batch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --job-name=xfer_container
#SBATCH --output=logs/xfer_container-%j.out
#SBATCH --error=logs/xfer_container-%j.err

CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
rsync -av --progress $CONTAINER u6jo.aip2.isambard:/projects/u6jo/containers/