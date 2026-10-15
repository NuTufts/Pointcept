#!/bin/bash
#SBATCH --job-name=stageC_launch
#SBATCH --partition=batch,wongjiradlab --time=24:00:00 --mem=4G --cpus-per-task=1
#SBATCH --output=logs/export/stageC_launch.%j.log --error=logs/export/stageC_launch.%j.log
# STAGE C of the split workflow (docs/reference/Polaris_Split_Workflow.md):
# the keypoint2 cascade ran on Polaris and came back as per-shard squashfs
# images; run the CPU tail at Tufts (regen -> nu_reco -> larpid -> export ->
# hadd) on top of them. Unpacks the images into $DATADIR/keypoint2_streams
# (sbatch) and submits the chain in RESUME_AFTER_INF mode (submit_extbnb_chain.sh
# skips prep + inference and hangs regen on the unpack job).
#
#   SPEC=lartpc/data_prep/uboone_official/tranche_<x>.spec TAG=<tag> IMGDIR=<kp2 image dir at Tufts> \
#   GAMMA_SPEC=<as used on Polaris> FLASH_WINDOW=<as used> WEIGHTS_PKL=<pkl|none> \
#     sbatch --export=ALL lartpc/larformer_reco/slurm/launch_stageC_from_polaris.sh
#
# GAMMA_SPEC/FLASH_WINDOW are only recorded (the GPU pass baked them in); the
# kp2 attrs are asserted against them before anything is submitted. Other
# knobs pass through launch_tranche_chain.sh (NNR, NEXP, EXCLUDE_NODES, ...).
set -eu
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
cd $K
SPEC=${SPEC:?}; TAG=${TAG:?}; IMGDIR=${IMGDIR:?kp2 image dir}
: "${GAMMA_SPEC:?}" "${FLASH_WINDOW:?}"
read -r _S _L OUT_ROOT MODE _T _ST _M _F _LA < <(grep -vE '^\s*(#|$)' "$SPEC" | head -1)
DATADIR=$(dirname "$OUT_ROOT"); KP2=$DATADIR/keypoint2_streams
MSP_LIST=lartpc/larformer_reco/inputlists/merged_sp_${TAG}.txt
NIMG=$(ls "$IMGDIR"/kp2_*_shard*.sqfs "$IMGDIR"/kp2_*_shard*.tar 2>/dev/null | wc -l)
[ "$NIMG" -gt 0 ] || { echo "ERROR: no kp2 images in $IMGDIR" >&2; exit 2; }
NMAN=$(ls "$IMGDIR"/*.manifest 2>/dev/null | wc -l)
[ "$NMAN" -eq "$NIMG" ] || { echo "ERROR: $NIMG images but $NMAN manifests in $IMGDIR (transfer incomplete?)" >&2; exit 2; }
if [ -e "$DATADIR/.chain_submitted" ]; then echo "ERROR: $DATADIR/.chain_submitted exists ($(cat "$DATADIR/.chain_submitted")); remove it to relaunch" >&2; exit 2; fi
if [ -d "$KP2" ] && [ ! -e "$KP2/.unpacked_ok" ] && [ -n "$(ls -A "$KP2" 2>/dev/null)" ]; then
  echo "ERROR: $KP2 exists and is not a verified unpack; move it away first" >&2; exit 2
fi
module load apptainer 2>/dev/null || true
echo ">>> $TAG: $NIMG kp2 images in $IMGDIR -> $KP2 ; DATADIR=$DATADIR mode=$MODE"
# flash attrs baked into the GPU pass must match what this launch records
TMPU=$(mktemp -d "$DATADIR/.attrcheck.XXXX")
IMG0=$(ls "$IMGDIR"/kp2_*_shard*.sqfs "$IMGDIR"/kp2_*_shard*.tar 2>/dev/null | head -1)
if [ -n "$IMG0" ]; then
  mkdir -p "$TMPU/t"
  case "$IMG0" in
    *.sqfs) unsquashfs -n -d "$TMPU/t" "$IMG0" >/dev/null ;;
    *.tar)  tar -xf "$IMG0" -C "$TMPU/t" ;;
  esac
  KIND=$([ "$MODE" = mc ] && echo mc || echo data)
  GEFF=$(python3 -c "print(5.25 * float('$GAMMA_SPEC'))" 2>/dev/null || echo "")
  if [ -n "$GEFF" ]; then
    apptainer exec --bind /cluster:/cluster $C python3 lartpc/polaris/check_kp2_attrs.py --tree "$TMPU/t" --sample 10 \
      --expect-gamma-eff "$GEFF" --expect-spec "$GAMMA_SPEC" --expect-kind "$KIND" --expect-window "$FLASH_WINDOW" \
      || { echo "ERROR: kp2 attrs do not match GAMMA_SPEC=$GAMMA_SPEC FLASH_WINDOW=$FLASH_WINDOW kind=$KIND" >&2; rm -rf "$TMPU"; exit 2; }
  else
    echo ">>> GAMMA_SPEC=$GAMMA_SPEC is not a float: attrs not asserted (check them by hand with check_kp2_attrs.py)"
  fi
fi
rm -rf "$TMPU"
# the merged_sp list the tail resolves kp2 src_file basenames against
apptainer exec --bind /cluster:/cluster $C python3 lartpc/data_prep/uboone_official/list_merged_sp.py --dir "$DATADIR/merged_sp" --out "$MSP_LIST"
echo ">>> $(wc -l < "$MSP_LIST") merged_sp events -> $MSP_LIST"
mkdir -p logs/data_prep logs/export
UNPACK=$(IMGDIR=$IMGDIR DEST=$KP2 sbatch --parsable --export=ALL,IMGDIR=$IMGDIR,DEST=$KP2 --job-name=${TAG}_unpack lartpc/data_prep/squashfs/unpack_kp2_images.sh)
echo ">>> unpack job $UNPACK -> $KP2"
RESUME_AFTER_INF=$UNPACK SPEC=$SPEC TAG=$TAG GAMMA_SPEC=$GAMMA_SPEC FLASH_WINDOW=$FLASH_WINDOW \
  bash lartpc/larformer_reco/slurm/launch_tranche_chain.sh
echo "LAUNCHED stage C for $TAG (tail waits for unpack $UNPACK)"
