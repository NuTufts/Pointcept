#!/bin/bash
#SBATCH --job-name=polB_it
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8 --mem=64G --time=24:00:00
#SBATCH --output=logs/inference/tufts_it_stageB.%j.log --error=logs/inference/tufts_it_stageB.%j.log
# ---------------------------------------------------------------------------
# Tufts-side integration test of the Polaris STAGE-B scripts (one A100 here
# stands in for a Polaris node; Tufts paths stand in for Eagle):
#  1. pack the run4ovl_4a smoke merged_sp (12 events) into a g00 image with the
#     fileno-group packer; mount it with mount_args.sh; build the list
#  2. node_launcher.sh: 2 shards x 2 events as 2 processes on ONE GPU
#     (POL_NGPU=1, PPG=2), GPU monitor on, MPS if the node has the binary,
#     node-local output + per-shard squashfs images
#  3. check_cascade --images, unpack_kp2_images.sh, check_kp2_attrs, report
#   IT_ROOT=<scratch dir> sbatch lartpc/polaris/tufts_it_stageB.sbatch
# ---------------------------------------------------------------------------
set -u
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
D=/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a
IT=${IT_ROOT:-$D/polaris_it}; rm -rf "$IT"; mkdir -p "$IT/merged_sp/000" "$IT/squashfs" "$IT/stageB/lists"
SMOKE=$(ls -d "$D"/smoke/*/merged_sp | tail -1)
FAILS=0; step() { echo; echo "=================== $(date +%FT%T) $*"; }
result() { if [ "$1" -eq 0 ]; then echo ">>> PASS: $2"; else echo ">>> FAIL: $2 (rc=$1)"; FAILS=$((FAILS + 1)); fi; }

step "1. pack (fileno-group mode) + mount + list"
for f in "$SMOKE"/*.h5; do ln "$f" "$IT/merged_sp/000/" 2>/dev/null || cp "$f" "$IT/merged_sp/000/"; done
SRC=$IT/merged_sp OUT=$IT/squashfs SAMPLE=run4ovl_4a GROUP_FILENOS=10 SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=4 \
  bash lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh; result $? "pack g00"
cat "$IT"/squashfs/*.manifest
bash lartpc/data_prep/squashfs/verify_images.sh "$IT/squashfs"; result $? "verify_images"
# Tufts stand-ins for the Polaris site values; the sample env supplies the rest
export POL_EAGLE=/cluster/tufts/wongjiradlab/larbys/data/larformer
export POL_REPO=$PWD POL_ASSETS=$POL_EAGLE/polaris_assets
export POL_SIF=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
export POL_IMGDIR=$IT/squashfs POL_DATADIR=$IT/stageB POL_NIMG=1 POL_NEVENTS=12
export POL_LOCAL_ROOT=${TMPDIR:-/tmp}/$USER POL_NGPU=1 POL_GPU_MONITOR=1 POL_THREADS=4
export POL_SAMPLE_ENV=lartpc/polaris/samples/run4ovl_4a.env
mkdir -p "$POL_LOCAL_ROOT"
# shared Tufts GPU nodes: a user-private MPS pipe dir (Polaris nodes are exclusive, /tmp/nvidia-mps is fine there)
export CUDA_MPS_PIPE_DIRECTORY=$POL_LOCAL_ROOT/nvidia-mps-${SLURM_JOB_ID:-x} CUDA_MPS_LOG_DIRECTORY=$POL_LOCAL_ROOT/nvidia-log-${SLURM_JOB_ID:-x}
command -v nvidia-cuda-mps-control >/dev/null 2>&1 && export POL_MPS=1 || { export POL_MPS=0; echo ">>> no nvidia-cuda-mps-control on $(hostname): MPS not exercised"; }
source lartpc/polaris/polaris_env.sh
pol_check_env; result $? "pol_check_env (sample env, 1 image)"
echo ">>> flags: $POL_INF_FLAGS"
LIST=$POL_DATADIR/lists/$POL_LIST_NAME
pol_exec "python3 lartpc/data_prep/uboone_official/list_merged_sp.py --dir $POL_MSP_MNT --out $LIST"; result $? "list build"
NL=$(grep -c . "$LIST"); [ "$NL" -eq 12 ]; result $? "list has 12 paths ($NL): $(head -1 "$LIST")"

step "2. node_launcher: 2 shards x 2 events, 2 processes on GPU 0, node-local output, mps=$POL_MPS"
OUT=$POL_DATADIR/tests/it; mkdir -p "$OUT"
WL=$POL_DATADIR/lists/it.wl; printf '0 0 2\n1 2 2\n' > "$WL"
JOBENV=$POL_DATADIR/it.env
echo "export POL_EAGLE='$POL_EAGLE' POL_REPO='$POL_REPO' POL_ASSETS='$POL_ASSETS' POL_SIF='$POL_SIF' POL_IMGDIR='$POL_IMGDIR' POL_DATADIR='$POL_DATADIR' POL_NIMG=1 POL_NEVENTS=12 POL_LOCAL_ROOT='$POL_LOCAL_ROOT' POL_NGPU=1 POL_GPU_MONITOR=1 POL_THREADS=4 POL_MPS=$POL_MPS POL_SAMPLE_ENV='$POL_SAMPLE_ENV' POL_PACK_FMT='${POL_PACK_FMT:-tar}' POL_SQFS_VIA='${POL_SQFS_VIA:-auto}' POL_PACK_COMP='${POL_PACK_COMP:-auto}' CUDA_MPS_PIPE_DIRECTORY='$CUDA_MPS_PIPE_DIRECTORY' CUDA_MPS_LOG_DIRECTORY='$CUDA_MPS_LOG_DIRECTORY'
export POL_TAG=it POL_OUTDIR='$OUT' POL_LIST='$LIST' POL_LOCAL_OUT='$POL_LOCAL_ROOT/kp2_it_${SLURM_JOB_ID:-x}'
source '$POL_REPO/lartpc/polaris/polaris_env.sh'" > "$JOBENV"
T0=$(date +%s)
bash lartpc/polaris/node_launcher.sh "$JOBENV" "$WL" 2 lartpc/polaris/run_cascade_shard.sh; result $? "node_launcher (2 shards)"
T1=$(date +%s)
ls -la "$OUT" "$OUT/.shards"; cat "$OUT"/*.manifest
tail -5 "$POL_LOGDIR/it/cascade_shard000.log"
head -5 "$POL_LOGDIR"/it/gpumon_*.csv; wc -l "$POL_LOGDIR"/it/gpumon_*.csv

step "3. check_cascade --images, unpack, attrs, report"
python3 lartpc/polaris/check_cascade.py --worklist "$WL" --outdir "$OUT" --images "$OUT" --logdir "$POL_LOGDIR/it" --tag it --throughput; result $? "check_cascade --images"
IMGDIR=$OUT DEST=$IT/keypoint2_streams bash lartpc/data_prep/squashfs/unpack_kp2_images.sh; result $? "unpack_kp2_images"
find "$IT/keypoint2_streams" -name '*.h5' | head
pol_exec "python3 lartpc/polaris/check_kp2_attrs.py --tree $IT/keypoint2_streams --sample 5 --expect-gamma-eff $POL_EXPECT_GAMMA_EFF --expect-spec $POL_EXPECT_SPEC --expect-kind $POL_EXPECT_KIND --expect-window $POL_EXPECT_WINDOW"
result $? "kp2 attrs (gamma_eff $POL_EXPECT_GAMMA_EFF, $POL_EXPECT_SPEC, $POL_EXPECT_KIND, $POL_EXPECT_WINDOW)"
python3 lartpc/polaris/summarize_gpu_util.py --logdir "$POL_LOGDIR/it" --tag it --nodes 1 --ppg 2 --mps "$POL_MPS" --elapsed $((T1 - T0)) \
  --out "$IT/report_it.md" --csv "$IT/util_it.csv"; result $? "summarize_gpu_util"
cat "$IT/report_it.md"; cat "$IT/util_it.csv" | head
echo; [ $FAILS -eq 0 ] && echo ">>> TUFTS IT: ALL PASS" || echo ">>> TUFTS IT: $FAILS FAILED step(s)"
exit $FAILS
