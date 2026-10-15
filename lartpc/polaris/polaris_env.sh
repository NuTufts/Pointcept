# shellcheck shell=bash
# ---------------------------------------------------------------------------
# Site configuration for the LArFormer bnb5e19 chain on Polaris (ALCF).
#
#   source lartpc/polaris/polaris_env.sh        # from the repo root, any node
#
# This is the ONLY file with Eagle paths / allocation names. Every value is
# ${VAR:-default}, so a one-off override is `VAR=... source ...` and a
# permanent one is an edit of the default below. Everything else in
# lartpc/polaris/ reads the POL_* variables and the helper functions defined
# here (pol_exec, pol_check_env). Safe to source repeatedly.
# ---------------------------------------------------------------------------

# ---- site paths -----------------------------------------------------------
# Every path is canonicalized (readlink -f): on Polaris /eagle is a symlink to
# /lus/eagle/projects and only /lus is a real mount inside the container (on the
# login nodes the container's /eagle is an empty auto-created mount-point tree,
# seen 2026-09-13). Stale POL_* exports from an older shell are canonicalized
# too, but start from a fresh shell if in doubt.
pol_canon() { readlink -f "$1" 2>/dev/null || echo "$1"; }
export POL_EAGLE=$(pol_canon "${POL_EAGLE:-/eagle/neutrinoGPU/twongj01}")
# ---- per-sample overrides (2026-10-15) --------------------------------------
# POL_SAMPLE_ENV=lartpc/polaris/samples/<x>.env sets the sample's images, datadir,
# inference flags and expected attrs BEFORE the defaults below (which are the
# original bnb5e19 values, kept so the bnb5e19 cascade/tail scripts still work).
export POL_SAMPLE_ENV=${POL_SAMPLE_ENV:-}
if [ -n "$POL_SAMPLE_ENV" ]; then
  if [ -r "$POL_SAMPLE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$POL_SAMPLE_ENV"
  else
    echo "ERROR: POL_SAMPLE_ENV=$POL_SAMPLE_ENV not readable" >&2
  fi
fi
export POL_SAMPLE=${POL_SAMPLE:-bnb5e19}
export POL_NIMG=${POL_NIMG:-12}                 # expected merged_sp images (0 = any)
export POL_NEVENTS=${POL_NEVENTS:-176302}       # expected list length (0 = any)
export POL_LIST_NAME=${POL_LIST_NAME:-merged_sp_bnb5e19_polaris.txt}
export POL_NEED_TAIL=${POL_NEED_TAIL:-1}        # 0: skip the CPU-tail asset checks
export POL_NGPU=${POL_NGPU:-4}
export POL_EXPECT_GAMMA_EFF=${POL_EXPECT_GAMMA_EFF:-2.86073}
export POL_EXPECT_SPEC=${POL_EXPECT_SPEC:-table}
export POL_EXPECT_KIND=${POL_EXPECT_KIND:-data}
export POL_EXPECT_WINDOW=${POL_EXPECT_WINDOW:-2.8,5.0}
export POL_REPO=$(pol_canon "${POL_REPO:-$POL_EAGLE/larformer_pointcept}")
export POL_ASSETS=$(pol_canon "${POL_ASSETS:-$POL_EAGLE/polaris_assets}")
export POL_SIF=$(pol_canon "${POL_SIF:-$POL_ASSETS/pointcept_cuml.sif}")
# the 12 bnb5e19_merged_sp_NNN.sqfs images (note the squashfs/ leaf)
export POL_IMGDIR=$(pol_canon "${POL_IMGDIR:-$POL_EAGLE/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs}")
# all outputs of this campaign (lists, logs, tests, keypoint2_streams, tail)
export POL_DATADIR=$(pol_canon "${POL_DATADIR:-$POL_EAGLE/data/uboone/bnb5e19_kp2_polaris}")
# where the images are mounted INSIDE the container. The input list is built
# with this prefix, so it must never change once the list exists. Default: a
# host directory (12 empty NNN/ mount points created by pol_exec), which needs
# only bind mounts. A path that does not exist in the image (the README's
# /data/bnb5e19/merged_sp) additionally needs POL_APPTAINER_FLAGS=--writable-tmpfs
# (overlay support) or --fakeroot -- not guaranteed on every node.
export POL_MSP_MNT=$(pol_canon "${POL_MSP_MNT:-$POL_DATADIR/merged_sp_mnt}")
export POL_LOGDIR=$(pol_canon "${POL_LOGDIR:-$POL_DATADIR/logs}")
export PRONGCNN_DIR=$(pol_canon "${PRONGCNN_DIR:-$POL_EAGLE/prongCNN}")

# ---- PBS accounting (ALCF suballocation form: project::subname) -----------
export POL_ACCOUNT=${POL_ACCOUNT:-neutrinoGPU::DetsimGPU}      # production
export POL_DEBUG_ACCOUNT=${POL_DEBUG_ACCOUNT:-neutrinoGPU::debug} # tests
export POL_APPTAINER_FLAGS=${POL_APPTAINER_FLAGS:-}  # e.g. --fakeroot if the site needs it

# ---- runtime environment ---------------------------------------------------
export HDF5_USE_FILE_LOCKING=FALSE      # Lustre: h5py file locking is unreliable
# thread caps for the CPU tail (24-32 processes per 64-thread node). OpenBLAS
# does NOT honour OMP_NUM_THREADS: without OPENBLAS_NUM_THREADS every nu_reco
# process spawned 64 BLAS threads and the node hit its process limit
# (pthread_create / fork "Resource temporarily unavailable", 2026-09-13).
# FORCED, not defaulted: PBS presets OMP_NUM_THREADS=64 on Polaris compute nodes
# (diag_export.sh 2026-09-16: 8 exporters x 64 OpenMP threads = 22 s/event vs
# 0.8 s/event alone). Override with POL_THREADS.
export POL_THREADS=${POL_THREADS:-2}
export OMP_NUM_THREADS=$POL_THREADS
export MKL_NUM_THREADS=$POL_THREADS
export OPENBLAS_NUM_THREADS=$POL_THREADS
export NUMEXPR_NUM_THREADS=$POL_THREADS
export VECLIB_MAXIMUM_THREADS=$POL_THREADS
export PYTHONUNBUFFERED=1
# ---- stage-B (GPU pass) knobs: node-local output + MPS (2026-10-15) ---------
# POL_LOCAL_OUT: per-shard outputs go to this NODE-LOCAL dir (Polaris: /local/scratch,
#   2x1.6 TB NVMe, wiped at job end) and are packed into one squashfs image per
#   shard in $POL_OUTDIR; empty = write per-event files straight to Eagle (legacy).
#   POL_LOCAL_ROOT is bound into the container by pol_exec.
export POL_LOCAL_ROOT=${POL_LOCAL_ROOT:-/local/scratch}
export POL_LOCAL_OUT=${POL_LOCAL_OUT:-}
# POL_PACK_FMT: per-shard image format, tar (default; plain uncompressed tar,
#   Polaris has no site mksquashfs -- checked 2026-10-15) or sqfs. For sqfs
#   run_cascade_shard.sh uses a host mksquashfs if present, else the one built
#   inside the container by lartpc/polaris/build_squashfs_tools.sh into
#   $POL_ASSETS/bin (run through pol_exec), else falls back to tar.
#   POL_SQFS_VIA=auto|host|container forces the route; POL_PACK_COMP=auto|lz4|gzip|xz
#   (auto = lz4 if the tool has it, else gzip level 1). Both formats carry a
#   .manifest (sha256, counts); unpack_kp2_images.sh / mount_args.sh handle both.
export POL_PACK_FMT=${POL_PACK_FMT:-tar}
export POL_SQFS_VIA=${POL_SQFS_VIA:-auto}
export POL_PACK_COMP=${POL_PACK_COMP:-auto}
# POL_MPS=1: node_launcher.sh starts the CUDA MPS control daemon on each node
#   (lartpc/polaris/mps_enable.sh) and the shards share GPUs through it; the
#   pipe dir must be node-local and the same for daemon and clients.
export POL_MPS=${POL_MPS:-0}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
export POL_GPU_MONITOR=${POL_GPU_MONITOR:-0}    # 1: nvidia-smi sampler per node -> $POL_LOGDIR/$POL_TAG/gpumon_<host>.csv
export POL_GPU_MONITOR_INTERVAL=${POL_GPU_MONITOR_INTERVAL:-5}
unset PYTHONSTARTUP   # ALCF sets /etc/pythonstart, absent in the container (harmless warning)

# ---- model assets: set the two roots BEFORE sourcing the shipped env.sh ----
# (its defaults point at a stale /eagle/.../uboone/assets path)
export LARFORMER_KPV2_ROOT=$(pol_canon "${LARFORMER_KPV2_ROOT:-$POL_ASSETS/kpv2_assets}")
export LARFORMER_OLD_REPO=$(pol_canon "${LARFORMER_OLD_REPO:-$POL_ASSETS/oldrepo_assets}")
if [ -r "$POL_ASSETS/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$POL_ASSETS/env.sh"
fi
export POL_PHOTONLIB=${POL_PHOTONLIB:-$LARFORMER_KPV2_ROOT/lartpc/flashmatch/data/photonlib_v6_70kV.npz}

# ---- frozen chain version v2_s1ep2p8cew6 (mirror of submit_extbnb_chain.sh)
export POL_CONFIG=${POL_CONFIG:-configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py}
export LARFORMER_SHOWER_BDT=${LARFORMER_SHOWER_BDT:-$POL_REPO/lartpc/larformer_reco/export/data/shower_cosmic_bdt_cew6.joblib}
export LARFORMER_SHOWER_BDT_NOVTX=${LARFORMER_SHOWER_BDT_NOVTX:-$POL_REPO/lartpc/larformer_reco/export/data/shower_novtx_bdt_cew6.joblib}
export POL_ATTACH_LLR=${POL_ATTACH_LLR:-$POL_REPO/lartpc/larformer_reco/trajfit/data/attachment_llr_tables_s1ep2p8.npz}
export POL_ATTACH_LLR_THR=${POL_ATTACH_LLR_THR:-4.0}
# bnb5e19 = beam-on run-1 DATA: calibrated table gamma (cell (data,1) = 0.5449
# -> gamma_eff 2.861), beam-on window 2.8-5.0 us, no truth, deterministic.
# Do NOT add --no-flash. Same flags as the Tufts production.
# A sample env sets POL_INF_FLAGS_SAMPLE (without --photonlib); the bnb5e19 flags are the default.
export POL_INF_FLAGS_SAMPLE=${POL_INF_FLAGS_SAMPLE:-"--deterministic --save-score-maps --device cuda --output-tree --no-gt --gamma-run-scale table --sample-kind data --flash-window 2.8,5.0"}
export POL_INF_FLAGS=${POL_INF_FLAGS:-"$POL_INF_FLAGS_SAMPLE --photonlib $POL_PHOTONLIB"}
# LArPID: a tag without 'run3' selects LArPID_default_network_weights.pt (run-1 data)
export POL_LARPID_TAG=${POL_LARPID_TAG:-bnb5e19_polaris}

# ---- apptainer ---------------------------------------------------------------
pol_load_apptainer() {
  command -v apptainer >/dev/null 2>&1 && return 0
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
      # shellcheck disable=SC1090
      [ -r "$f" ] && source "$f"
    done
  fi
  module use /soft/modulefiles 2>/dev/null || true
  module load spack-pe-base 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found (tried: module use /soft/modulefiles; module load spack-pe-base apptainer)" >&2
    return 1
  fi
}
pol_load_apptainer || true
# per-node scratch for apptainer (ALCF recommendation); fall back to /tmp
_pol_scratch=/local/scratch
[ -d "$_pol_scratch" ] && [ -w "$_pol_scratch" ] || _pol_scratch=/tmp
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$_pol_scratch/$USER/apptainer-tmp}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$_pol_scratch/$USER/apptainer-cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" 2>/dev/null || true

# the --bind flags that mount the 12 images under $POL_MSP_MNT (read-only);
# the mount-point directories are created on the host when the mount root is
# a host path (they stay empty outside the container)
pol_mount_args() {
  local img d
  if [ -d "$(dirname "$POL_MSP_MNT")" ] || mkdir -p "$POL_MSP_MNT" 2>/dev/null; then
    for img in "$POL_IMGDIR"/*_merged_sp_*.sqfs; do
      d=$(basename "$img" .sqfs); d=${d##*_merged_sp_}
      mkdir -p "$POL_MSP_MNT/$d" 2>/dev/null || true
    done
  fi
  bash "$POL_REPO/lartpc/data_prep/squashfs/mount_args.sh" "$POL_IMGDIR" "$POL_MSP_MNT"
}

# bind the data filesystem root (/lus on Polaris) explicitly only if the site
# apptainer config does not already expose it (a duplicate bind is an error in
# some apptainer versions). Cached in POL_EAGLE_BIND for child processes. The
# check uses a FILE (the sif) rather than a directory: apptainer auto-creates
# directories for mount points, so `test -d` can succeed on an empty tree.
pol_eagle_bind() {
  if [ -z "${POL_EAGLE_BIND+x}" ]; then
    local rest=${POL_EAGLE#/}; local root=/${rest%%/*}
    if apptainer exec "$POL_SIF" test -f "$POL_SIF" 2>/dev/null; then
      export POL_EAGLE_BIND=""
    else
      export POL_EAGLE_BIND="--bind $root:$root"
    fi
  fi
  echo "$POL_EAGLE_BIND"
}

# pol_exec [--nv] <shell command string>
# Runs the command inside the container, in the repo root, with PYTHONPATH set
# and the merged_sp images mounted. Host environment (LARFORMER_*, HDF5_*,
# CUDA_VISIBLE_DEVICES, PRONGCNN_DIR, ...) is inherited by the container.
pol_exec() {
  local nv="" lb=""
  if [ "${1:-}" = "--nv" ]; then nv="--nv"; shift; fi
  local cmd="$*"
  [ -n "$POL_LOCAL_OUT" ] && [ -d "$POL_LOCAL_ROOT" ] && lb="--bind $POL_LOCAL_ROOT:$POL_LOCAL_ROOT"
  # shellcheck disable=SC2046
  apptainer exec $nv $POL_APPTAINER_FLAGS $lb $(pol_eagle_bind) $(pol_mount_args) "$POL_SIF" \
    bash -c "cd '$POL_REPO' && export PYTHONPATH='$POL_REPO' && $cmd"
}

# pol_check_env: assert every asset this campaign needs exists; print them.
pol_check_env() {
  local rc=0 f
  echo ">>> polaris_env: sample=$POL_SAMPLE${POL_SAMPLE_ENV:+ ($POL_SAMPLE_ENV)} repo=$POL_REPO"
  echo "    assets=$POL_ASSETS  sif=$POL_SIF"
  echo "    images=$POL_IMGDIR -> $POL_MSP_MNT"
  echo "    datadir=$POL_DATADIR  prongCNN=$PRONGCNN_DIR"
  echo "    account=$POL_ACCOUNT  debug=$POL_DEBUG_ACCOUNT"
  echo "    config=$POL_CONFIG"
  for f in "$POL_SIF" "$POL_REPO/$POL_CONFIG" \
           "$LARFORMER_BATTERY_SLICER_CKPT" "$LARFORMER_KP_PARTICLE_CKPT" \
           "$LARFORMER_KP_KEYPOINT_CKPT" "$LARFORMER_SONATA_PRETRAIN" \
           "$LARFORMER_KPV2_ROOT/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth" \
           "$POL_PHOTONLIB"; do
    if [ -f "$f" ]; then echo "    ok  $f"; else echo "    MISSING $f"; rc=1; fi
  done
  if [ "$POL_NEED_TAIL" = 1 ]; then
    for f in "$LARFORMER_SHOWER_BDT" "$LARFORMER_SHOWER_BDT_NOVTX" "$POL_ATTACH_LLR" \
             "$POL_REPO/lartpc/larformer_reco/export/lib_wirecell_fiducial_volume.so" \
             "$POL_REPO/lartpc/larformer_reco/trajfit/data/range2ke_lar.npz" \
             "$POL_REPO/lartpc/larformer_reco/trajfit/data/calo_calib.npz" \
             "$PRONGCNN_DIR/checkpoints/LArPID_default_network_weights.pt" \
             "$PRONGCNN_DIR/models/models_instanceNorm_reco_2chan_quadTask.py" \
             "$PRONGCNN_DIR/models/normalization_constants.py"; do
      if [ -f "$f" ]; then echo "    ok  $f"; else echo "    MISSING $f"; rc=1; fi
    done
  fi
  local nimg nman
  nimg=$(ls "$POL_IMGDIR"/*_merged_sp_*.sqfs 2>/dev/null | wc -l)
  nman=$(ls "$POL_IMGDIR"/*_merged_sp_*.sqfs.manifest 2>/dev/null | wc -l)
  if [ "$nimg" -gt 0 ] && { [ "$POL_NIMG" -eq 0 ] || [ "$nimg" -eq "$POL_NIMG" ]; }; then
    echo "    ok  $nimg squashfs images ($nman manifests) in $POL_IMGDIR"
  else
    echo "    ERROR: $nimg squashfs images (expect $POL_NIMG) in $POL_IMGDIR"; rc=1
  fi
  [ "$nman" -eq "$nimg" ] || echo "    WARNING: $nman manifests for $nimg images (verify_images.sh cannot check them all)"
  command -v apptainer >/dev/null 2>&1 && echo "    ok  apptainer $(apptainer --version 2>/dev/null | awk '{print $NF}')" || { echo "    MISSING apptainer"; rc=1; }
  if [ "$POL_NEED_TAIL" != 1 ]; then
    [ $rc -eq 0 ] && echo ">>> pol_check_env: PASS (GPU pass only)" || echo ">>> pol_check_env: FAIL"
    return $rc
  fi
  # python packages the export stage needs that are NOT in pointcept_cuml.sif:
  # they come from the user's ~/.local (pip --user, bound into the container).
  # Tufts production versions: scikit-learn 1.8.0 (the cew6 BDT pickles),
  # uproot 5.7.2, awkward 2.9.0. Install on the far side (inside the container,
  # with the site proxy set):  pip install --user scikit-learn==1.8.0 uproot==5.7.2 awkward==2.9.0
  local pyver
  if pyver=$(apptainer exec $(pol_eagle_bind) "$POL_SIF" python3 -c "import sklearn, uproot, awkward, sys; print(sklearn.__version__, uproot.__version__, awkward.__version__, sklearn.__file__)" 2>/dev/null); then
    echo "    ok  container python: sklearn/uproot/awkward = $pyver"
    case "$pyver" in 1.8.0\ *) ;; *) echo "    WARNING: scikit-learn != 1.8.0 (cew6 shower BDT pickles were made with 1.8.0)";; esac
  else
    echo "    MISSING python packages in the container: scikit-learn / uproot / awkward (pip install --user scikit-learn==1.8.0 uproot==5.7.2 awkward==2.9.0 inside the container)"; rc=1
  fi
  [ $rc -eq 0 ] && echo ">>> pol_check_env: PASS" || echo ">>> pol_check_env: FAIL"
  return $rc
}
