#!/bin/bash
# Gather everything the keypoint2 cascade inference needs that is NOT in git
# into one directory for transfer to another cluster (Polaris), preserving the
# relative layout the configs expect, and write env.sh to source there.
#
#   bash lartpc/data_prep/squashfs/stage_inference_assets.sh <out_dir>
#
# On the far side:  source <out_dir>/env.sh   (edit the two roots at the top)
set -eu
OUT=${1:?output dir}
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
OLD=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
mkdir -p "$OUT/kpv2_assets" "$OUT/oldrepo_assets"
copy () { mkdir -p "$(dirname "$2")"; [ -f "$2" ] || cp -v "$1" "$2"; }
# chain version v2_s1ep2p8cew6 (see lartpc/larformer_analysis/model_and_output_file_versions.md)
copy $K/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth \
     $OUT/kpv2_assets/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth
copy $K/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth \
     $OUT/kpv2_assets/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth
copy $K/exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth \
     $OUT/kpv2_assets/exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth
copy $K/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth \
     $OUT/kpv2_assets/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth
copy $K/lartpc/flashmatch/data/photonlib_v6_70kV.npz \
     $OUT/kpv2_assets/lartpc/flashmatch/data/photonlib_v6_70kV.npz
copy $OLD/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth \
     $OUT/oldrepo_assets/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth
copy /cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif $OUT/pointcept_cuml.sif
cat > "$OUT/env.sh" <<'ENV'
# source this on the far side after cloning the repo and copying the assets.
# 1) the repo clone (must contain the assets' kpv2_assets/* files at the same
#    relative paths, or point LARFORMER_KPV2_ROOT at the kpv2_assets dir itself
#    -- the configs only read sonata/, exp/ and lartpc/flashmatch/data/ from it)
export LARFORMER_KPV2_ROOT=${LARFORMER_KPV2_ROOT:-/eagle/neutrinoGPU/twongj01/uboone/assets/kpv2_assets}
# 2) the old-repo assets (sonata pretrain loaded into the backbones at build time)
export LARFORMER_OLD_REPO=${LARFORMER_OLD_REPO:-/eagle/neutrinoGPU/twongj01/uboone/assets/oldrepo_assets}
export LARFORMER_BATTERY_SLICER_CKPT=$LARFORMER_KPV2_ROOT/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth
export LARFORMER_KP_PARTICLE_CKPT=$LARFORMER_KPV2_ROOT/exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth
export LARFORMER_KP_KEYPOINT_CKPT=$LARFORMER_KPV2_ROOT/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth
export LARFORMER_SONATA_PRETRAIN=$LARFORMER_OLD_REPO/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth
# PhotonLib cache: pass --photonlib $LARFORMER_KPV2_ROOT/lartpc/flashmatch/data/photonlib_v6_70kV.npz
# (or copy it into the clone at lartpc/flashmatch/data/, its default location)
ENV
{ echo "staged $(date)"; find "$OUT" -type f ! -name '*.manifest' -exec ls -la {} \; | awk '{printf "%.2f GB %s\n", $5/1e9, $9}'; } | tee "$OUT/MANIFEST.txt"
(cd "$OUT" && find kpv2_assets oldrepo_assets pointcept_cuml.sif -type f | xargs sha256sum > sha256sums.txt) && echo "sha256sums.txt written"
