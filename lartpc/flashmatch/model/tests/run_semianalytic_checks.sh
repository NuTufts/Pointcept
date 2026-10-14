#!/bin/bash
# run_semianalytic_checks.sh -- validate the python semi-analytic light model against
# the LArSoft reference stored in the detinfo file, and benchmark it.
#
#   bash lartpc/flashmatch/model/tests/run_semianalytic_checks.sh            # CPU: validation + small CPU benchmark (here)
#   bash lartpc/flashmatch/model/tests/run_semianalytic_checks.sh --gpu      # submit validation + 1e5-point GPU benchmark (gpu,preempt)
#   options: --detector sbnd  --n 20000 (CPU benchmark points)  --log-dir DIR (GPU job logs)
#
# Needs the detinfo file with schema >= 1.3 (lartpc/detinfo/extract/run_extract.sh sbnd <version>).
# Runs inside the pointcept container. Results of the reference runs: ../../SEMIANALYTIC_PORT_NOTES.md
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
CONTAINER=${POINTCEPT_CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}
DET=sbnd; N=20000; GPU=0; LOGDIR=/cluster/tufts/wongjiradlab/larbys/detinfo/bench
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) GPU=1 ;;
    --detector) DET=$2; shift ;;
    --n) N=$2; shift ;;
    --log-dir) LOGDIR=$2; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac; shift
done
TEST=lartpc/flashmatch/model/tests/test_semianalytic.py

if [ $GPU -eq 0 ]; then
  cd "$REPO"
  apptainer exec -B /cluster:/cluster "$CONTAINER" \
    python3 $TEST --detector $DET --bench --bench-n $N --device cpu
else
  mkdir -p "$LOGDIR"
  sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=sam_${DET}_bench
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=$LOGDIR/sam_${DET}_bench_%j.log
cd $REPO
nvidia-smi -L
apptainer exec --nv -B /cluster:/cluster $CONTAINER \
  python3 $TEST --detector $DET --bench --bench-n 100000 --device cuda
EOS
  echo "log: $LOGDIR/sam_${DET}_bench_<jobid>.log"
fi
