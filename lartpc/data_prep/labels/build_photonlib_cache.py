"""Convert the MicroBooNE photon-library ROOT TTree to a dense .npz cache.

Source: ${UBLARCVAPP_BASEDIR}/ublarcvapp/UBPhotonLib/dat/uboone_photon_library_v6_70kV_EnhancedExtraTPCVis.root
Tree:   pmtresponse/PhotonLibraryData  (Voxel/I, OpChannel/I, Visibility/F)

The TTree has 33.6M entries; densified into (Nx, Ny, Nz, NOpDets) = (75, 75, 400, 32)
float32, that's 288 MB and trivially GPU-resident. We do this conversion *once*
and store the result alongside grid metadata in an .npz so the runtime path
only ever loads numpy.

Grid metadata mirrors UBPhotonLib.cxx constants:
    cryo box in TPC coords:  origin = cryo_global - tpc_global
    voxel size:              cryo_length / nvoxels_dim
    voxel id formula:        vid = vx + vy * Nx + vz * Nx * Ny    (x-fastest)
    LUT row index:           lib_index = vid * 32 + opchannel

Run from inside the pointcept container with the ubdl env sourced:
    cd ubdl && source setenv_pointcept_container.sh
    cd Pointcept
    python lartpc/data_prep/labels/build_photonlib_cache.py
"""

import argparse
import os
import sys
import time

import numpy as np


# Grid constants (mirror UBPhotonLib.cxx).
TPC_GLOBAL_ORIGIN_CM = np.array([-1.55, -115.53 + 0.5 * (117.47 + 115.53), 0.1],
                                dtype=np.float64)
CRYO_GLOBAL_ORIGIN_CM = np.array([-63.435, -191.61, -92.375], dtype=np.float64)
CRYO_LENGTH_CM = np.array([383.22, 383.22, 1221.75], dtype=np.float64)
NVOXELS_DIM = np.array([75, 75, 400], dtype=np.int64)
N_OPDETS = 32

CRYO_ORIGIN_TPC_CM = CRYO_GLOBAL_ORIGIN_CM - TPC_GLOBAL_ORIGIN_CM
VOXEL_LEN_CM = CRYO_LENGTH_CM / NVOXELS_DIM.astype(np.float64)

DEFAULT_ROOT_PATH = (
    "{ublarcvapp_basedir}/ublarcvapp/UBPhotonLib/dat/"
    "uboone_photon_library_v6_70kV_EnhancedExtraTPCVis.root"
)

DEFAULT_OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dat", "photonlib_v6_70kV.npz",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-file", default=None,
                    help="Path to source ROOT photon library. Defaults to "
                         "$UBLARCVAPP_BASEDIR/.../EnhancedExtraTPCVis.root")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH,
                    help=f"Output .npz path (default: {DEFAULT_OUT_PATH})")
    args = ap.parse_args()

    if args.root_file is None:
        base = os.environ.get("UBLARCVAPP_BASEDIR")
        if not base:
            print("ERROR: $UBLARCVAPP_BASEDIR is not set and --root-file was "
                  "not given. Source ubdl/setenv_pointcept_container.sh first.",
                  file=sys.stderr)
            sys.exit(1)
        args.root_file = DEFAULT_ROOT_PATH.format(ublarcvapp_basedir=base)

    if not os.path.exists(args.root_file):
        print(f"ERROR: ROOT photon library not found at {args.root_file}",
              file=sys.stderr)
        sys.exit(2)

    print(f"Reading ROOT TTree from: {args.root_file}")
    print(f"Output .npz             : {args.out}")
    print(f"Grid: {tuple(NVOXELS_DIM.tolist())} voxels x {N_OPDETS} opdets "
          f"= {int(np.prod(NVOXELS_DIM)) * N_OPDETS:,} cells "
          f"({int(np.prod(NVOXELS_DIM)) * N_OPDETS * 4 / 2**20:.0f} MB float32)")
    print(f"voxel_len_cm     = {VOXEL_LEN_CM}")
    print(f"cryo origin (TPC)= {CRYO_ORIGIN_TPC_CM}")

    import ROOT

    t0 = time.time()
    rdf = ROOT.RDataFrame("pmtresponse/PhotonLibraryData", args.root_file)
    cols = rdf.AsNumpy(["Voxel", "OpChannel", "Visibility"])
    voxel = cols["Voxel"].astype(np.int64)
    opch = cols["OpChannel"].astype(np.int64)
    vis = cols["Visibility"].astype(np.float32)
    print(f"  loaded {len(voxel):,} entries in {time.time() - t0:.1f} s")

    n_voxels = int(np.prod(NVOXELS_DIM))
    if voxel.min() < 0 or voxel.max() >= n_voxels:
        print(f"WARN: voxel range [{voxel.min()}, {voxel.max()}] vs n_voxels={n_voxels}")
    if opch.min() < 0 or opch.max() >= N_OPDETS:
        print(f"WARN: opch range [{opch.min()}, {opch.max()}] vs N_OPDETS={N_OPDETS}")

    t0 = time.time()
    lut = np.zeros((NVOXELS_DIM[0], NVOXELS_DIM[1], NVOXELS_DIM[2], N_OPDETS),
                   dtype=np.float32)
    # Voxel ID -> (vx, vy, vz) inversion (mirror UBPhotonLib::getVoxelCoords)
    nx, ny, _ = NVOXELS_DIM.tolist()
    vx = (voxel % nx).astype(np.int64)
    vy = ((voxel // nx) % ny).astype(np.int64)
    vz = ((voxel // (nx * ny))).astype(np.int64)

    lut[vx, vy, vz, opch] = vis
    nonzero_cells = int((lut != 0).sum())
    fill_frac = nonzero_cells / lut.size
    print(f"  densified in {time.time() - t0:.1f} s; "
          f"non-zero cells: {nonzero_cells:,} ({100*fill_frac:.1f}%)")
    print(f"  vis stats: min={lut[lut > 0].min():.2e} median={np.median(lut[lut > 0]):.2e} "
          f"max={lut.max():.2e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    np.savez_compressed(
        args.out,
        visibility=lut,
        nvoxels_dim=NVOXELS_DIM,
        voxel_len_cm=VOXEL_LEN_CM,
        cryo_origin_tpc_cm=CRYO_ORIGIN_TPC_CM,
        tpc_global_origin_cm=TPC_GLOBAL_ORIGIN_CM,
        cryo_global_origin_cm=CRYO_GLOBAL_ORIGIN_CM,
        cryo_length_cm=CRYO_LENGTH_CM,
        n_opdets=N_OPDETS,
        photonlib_version=np.array(["v6_70kV_EnhancedExtraTPCVis"]),
    )
    sz_mb = os.path.getsize(args.out) / 2**20
    print(f"  wrote {args.out} ({sz_mb:.0f} MB) in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
