# merged_sp -> squashfs images (for Polaris / Lustre)

A merged_sp tree is ~10^5 files of ~10 MB. Lustre file systems (ALCF Eagle /
Grand) transfer and open such trees slowly (per-file metadata cost), so the
tree is packed into one squashfs image per top-level directory (`000`, `001`,
...; ~150 GB each for bnb5e19) and the images are bind-mounted read-only into
the Apptainer container, where the files appear at their original paths.

## Pack (Tufts)

    SRC=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v28_wctagger_bnb5e19/merged_sp
    OUT=/cluster/tufts/wongjiradlab/larbys/data/larformer/mcc9_v28_wctagger_bnb5e19/squashfs
    N=$(ls $SRC | wc -l)
    SRC=$SRC OUT=$OUT SAMPLE=bnb5e19 sbatch --array=0-$((N-1)) \
        lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh

Each task writes `<OUT>/bnb5e19_merged_sp_<dir>.sqfs`, a `.manifest` (file
count, bytes, sha256) and a `.ok` marker once the image lists the same number
of h5 files as the source. lz4 compression (h5 payload is already gzip'ed):
fast to build and to read. Idempotent: rerun the array to fill gaps.

## Transfer

Globus, Tufts -> `/eagle/neutrinoGPU/twongj01/uboone/bnb5e19_squashfs/`
(12 files of ~150 GB instead of 176k small ones). Copy the `.manifest`
files too and compare `sha256sum` on the far side. On Eagle, set wide
striping on the destination directory BEFORE the transfer:
`lfs setstripe -c 8 /eagle/neutrinoGPU/twongj01/uboone/bnb5e19_squashfs`.

## Use (Polaris)

    module load apptainer
    IMG=/eagle/neutrinoGPU/twongj01/uboone/bnb5e19_squashfs
    apptainer exec --nv $(bash lartpc/data_prep/squashfs/mount_args.sh $IMG /data/bnb5e19/merged_sp) \
        pointcept_cuml.sif bash -c "cd <repo> && python3 lartpc/data_prep/uboone_official/list_merged_sp.py \
            --dir /data/bnb5e19/merged_sp --out inputlists/merged_sp_bnb5e19_polaris.txt"

`list_merged_sp.py` sorts by the (fileno, entry) basename key, so the list
order and the cascade index linkage are identical to the Tufts production
(only the path prefix differs). Inference then runs exactly as in
`lartpc/larformer_reco/slurm/submit_inference_shard.sh` (Polaris is PBS, so
the sbatch array becomes a PBS job array or a per-node loop): config
`larformer-keypoint2-fullcascade-v6lantern-envslicer.py`, checkpoints
`exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth` and
`exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth`,
PhotonLib cache `lartpc/flashmatch/data/photonlib_v6_70kV.npz`, flags
`--gamma-run-scale table --sample-kind data --flash-window 2.8,5.0
--deterministic --save-score-maps --output-tree` (beam-on run-1 window).
Determinism is per GPU type: A100 outputs differ from the Tufts ones at the
float32 level in the slicer partition.

Reading from the images is sequential large-file I/O on Lustre (one
squashfs block read per h5 open) instead of a metadata lookup per file.
