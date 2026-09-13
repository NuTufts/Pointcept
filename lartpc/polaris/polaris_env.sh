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
export POL_EAGLE=${POL_EAGLE:-/eagle/neutrinoGPU/twongj01}
export POL_REPO=${POL_REPO:-$POL_EAGLE/larformer_pointcept}
export POL_ASSETS=${POL_ASSETS:-$POL_EAGLE/polaris_assets}
export POL_SIF=${POL_SIF:-$POL_ASSETS/pointcept_cuml.sif}
# the 12 bnb5e19_merged_sp_NNN.sqfs images (note the squashfs/ leaf)
export POL_IMGDIR=${POL_IMGDIR:-$POL_EAGLE/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs}
# all outputs of this campaign (lists, logs, tests, keypoint2_streams, tail)
export POL_DATADIR=${POL_DATADIR:-$POL_EAGLE/data/uboone/bnb5e19_kp2_polaris}
# where the images are mounted INSIDE the container. The input list is built
# with this prefix, so it must never change once the list exists. Default: a
# host directory (12 empty NNN/ mount points created by pol_exec), which needs
# only bind mounts. A path that does not exist in the image (the README's
# /data/bnb5e19/merged_sp) additionally needs POL_APPTAINER_FLAGS=--writable-tmpfs
# (overlay support) or --fakeroot -- not guaranteed on every node.
export POL_MSP_MNT=${POL_MSP_MNT:-$POL_DATADIR/merged_sp_mnt}
export POL_LOGDIR=${POL_LOGDIR:-$POL_DATADIR/logs}
export PRONGCNN_DIR=${PRONGCNN_DIR:-$POL_EAGLE/prongCNN}

# ---- PBS accounting (ALCF suballocation form: project::subname) -----------
export POL_ACCOUNT=${POL_ACCOUNT:-neutrinoGPU::DetsimGPU}      # production
export POL_DEBUG_ACCOUNT=${POL_DEBUG_ACCOUNT:-neutrinoGPU::debug} # tests
export POL_APPTAINER_FLAGS=${POL_APPTAINER_FLAGS:-}  # e.g. --fakeroot if the site needs it

# ---- runtime environment ---------------------------------------------------
export HDF5_USE_FILE_LOCKING=FALSE      # Lustre: h5py file locking is unreliable
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1

# ---- model assets: set the two roots BEFORE sourcing the shipped env.sh ----
# (its defaults point at a stale /eagle/.../uboone/assets path)
export LARFORMER_KPV2_ROOT=${LARFORMER_KPV2_ROOT:-$POL_ASSETS/kpv2_assets}
export LARFORMER_OLD_REPO=${LARFORMER_OLD_REPO:-$POL_ASSETS/oldrepo_assets}
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
export POL_INF_FLAGS=${POL_INF_FLAGS:-"--deterministic --save-score-maps --device cuda --output-tree --no-gt --gamma-run-scale table --sample-kind data --flash-window 2.8,5.0 --photonlib $POL_PHOTONLIB"}
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

# bind the data filesystem root (/eagle) explicitly only if the site apptainer
# config does not already expose it (a duplicate bind is an error in some
# apptainer versions). Cached in POL_EAGLE_BIND for child processes.
pol_eagle_bind() {
  if [ -z "${POL_EAGLE_BIND+x}" ]; then
    local rest=${POL_EAGLE#/}; local root=/${rest%%/*}
    if apptainer exec "$POL_SIF" test -d "$POL_EAGLE" 2>/dev/null; then
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
  local nv=""
  if [ "${1:-}" = "--nv" ]; then nv="--nv"; shift; fi
  local cmd="$*"
  # shellcheck disable=SC2046
  apptainer exec $nv $POL_APPTAINER_FLAGS $(pol_eagle_bind) $(pol_mount_args) "$POL_SIF" \
    bash -c "cd '$POL_REPO' && export PYTHONPATH='$POL_REPO' && $cmd"
}

# pol_check_env: assert every asset this campaign needs exists; print them.
pol_check_env() {
  local rc=0 f
  echo ">>> polaris_env: repo=$POL_REPO"
  echo "    assets=$POL_ASSETS  sif=$POL_SIF"
  echo "    images=$POL_IMGDIR -> $POL_MSP_MNT"
  echo "    datadir=$POL_DATADIR  prongCNN=$PRONGCNN_DIR"
  echo "    account=$POL_ACCOUNT  debug=$POL_DEBUG_ACCOUNT"
  echo "    config=$POL_CONFIG"
  for f in "$POL_SIF" "$POL_REPO/$POL_CONFIG" \
           "$LARFORMER_BATTERY_SLICER_CKPT" "$LARFORMER_KP_PARTICLE_CKPT" \
           "$LARFORMER_KP_KEYPOINT_CKPT" "$LARFORMER_SONATA_PRETRAIN" \
           "$LARFORMER_KPV2_ROOT/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth" \
           "$POL_PHOTONLIB" "$LARFORMER_SHOWER_BDT" "$LARFORMER_SHOWER_BDT_NOVTX" \
           "$POL_ATTACH_LLR" \
           "$POL_REPO/lartpc/larformer_reco/export/lib_wirecell_fiducial_volume.so" \
           "$POL_REPO/lartpc/larformer_reco/trajfit/data/range2ke_lar.npz" \
           "$POL_REPO/lartpc/larformer_reco/trajfit/data/calo_calib.npz" \
           "$PRONGCNN_DIR/checkpoints/LArPID_default_network_weights.pt" \
           "$PRONGCNN_DIR/models/models_instanceNorm_reco_2chan_quadTask.py" \
           "$PRONGCNN_DIR/models/normalization_constants.py"; do
    if [ -f "$f" ]; then echo "    ok  $f"; else echo "    MISSING $f"; rc=1; fi
  done
  local nimg
  nimg=$(ls "$POL_IMGDIR"/*_merged_sp_*.sqfs 2>/dev/null | wc -l)
  if [ "$nimg" -eq 12 ]; then echo "    ok  12 squashfs images"; else echo "    ERROR: $nimg squashfs images (expect 12) in $POL_IMGDIR"; rc=1; fi
  command -v apptainer >/dev/null 2>&1 && echo "    ok  apptainer $(apptainer --version 2>/dev/null | awk '{print $NF}')" || { echo "    MISSING apptainer"; rc=1; }
  [ $rc -eq 0 ] && echo ">>> pol_check_env: PASS" || echo ">>> pol_check_env: FAIL"
  return $rc
}
