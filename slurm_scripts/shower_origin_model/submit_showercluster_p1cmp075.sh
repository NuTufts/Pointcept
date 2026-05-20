#!/bin/bash

#SBATCH --job-name=showercluster
#SBATCH --output=logs/showercluster.p1cmp075.%j.%N.log
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=24
#SBATCH --time=6-00:00:00
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:4
#SBATCH --error=logs/showercluster.p1cmp075.%j.%N.err

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer/1.2.4-suid

# run job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR}/../ && \
  source setenv_pointcept_only.sh && \
  cd ${WORKDIR} && \
  python tools/train.py --config configs/lartpc/shower-cluster-sonata-v1.py --num-gpus 4"

