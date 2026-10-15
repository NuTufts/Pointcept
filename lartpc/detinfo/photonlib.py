"""
Photon visibility library (photonlib_<det>_<ver>.h5, written by the extractor).

Dense float32 arrays [ix, iy, iz, opchannel] over the PhotonVisibilityService
voxel grid; voxel id = ix + iy*NX + iz*NX*NY (LArSoft sim::PhotonVoxelDef).

    pl = detinfo.load("sbnd").photonlib          # lazy: loads on first use
    vis = pl.GetVisibility(points)               # (N, n_opch), nearest voxel like LArSoft
    pl.GetVoxelID(points)                        # -1 outside the grid
    pl.export_flashmatch_npz(path)               # npz for pointcept PhotonLibLookup

NOTE (SBND): SBND_OpLibOUT_v2.00 only covers voxels OUTSIDE the TPC active
volumes; visibilities inside are zero. In-TPC light in SBND is simulated with
larsim's semi-analytic model (lartpc/flashmatch/model/visibility/semianalytic.py).

NOTE (ICARUS): the library covers cryostat 0 and its 180 PMTs only. LArSoft
serves both cryostats through ICARUSPhotonMappingTransformations (translate a
point in cryostat 1 into cryostat 0, shift library channel i to i + 180);
GetVisibility() applies that mapping (attrs mapping_*) and returns all 360
channels. GetVoxelID() is the plain library voxel of the given coordinates.
"""
import numpy as np


class PhotonLibrary:
    def __init__(self, path, load_reflected=False):
        import h5py
        self.path = path
        with h5py.File(path, "r") as f:
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            self.grid_min = np.asarray(f.attrs["grid_min"], dtype=np.float64)
            self.grid_max = np.asarray(f.attrs["grid_max"], dtype=np.float64)
            self.grid_n = np.asarray(f.attrs["grid_n"], dtype=np.int64)
            self.visibility = f["visibility"][()]
            self.refl_visibility = (f["refl_visibility"][()]
                                    if load_reflected and "refl_visibility" in f else None)
        self.voxel_len = (self.grid_max - self.grid_min) / self.grid_n
        # detector -> library mapping (ICARUSPhotonMappingTransformations); None = identity
        self.mapping = None
        if "mapping_switch_x" in self.attrs:
            self.mapping = dict(
                switch_x=float(self.attrs["mapping_switch_x"]),
                translations=np.asarray(self.attrs["mapping_translations"], dtype=np.float64),
                shifts=np.asarray(self.attrs["mapping_channel_shifts"], dtype=np.int64),
                lib_to_opch=np.asarray(self.attrs["mapping_library_to_opchannel"], dtype=np.int64))

    @property
    def n_opchannels(self):
        """Optical channels returned by GetVisibility (whole detector)."""
        if self.mapping is not None:
            return int(self.attrs.get("n_opdets_detector", self.visibility.shape[-1]))
        return self.visibility.shape[-1]

    def DetectorToLibrary(self, points):
        """(library points (N,3), cryostat index (N,)) under the mapping tool."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.mapping is None:
            return p, np.zeros(len(p), dtype=np.int64)
        cryo = (p[:, 0] > self.mapping["switch_x"]).astype(np.int64)
        return p + self.mapping["translations"][cryo], cryo

    def VoxelIndices(self, points):
        """(N,3) integer voxel indices and (N,) inside mask (PhotonVoxelDef::GetVoxelID)."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.all((p >= self.grid_min) & (p < self.grid_max), axis=1)
        idx = np.floor((p - self.grid_min) / self.voxel_len).astype(np.int64)
        idx = np.clip(idx, 0, self.grid_n - 1)
        return idx, inside

    def GetVoxelID(self, points):
        idx, inside = self.VoxelIndices(points)
        nx, ny, _ = self.grid_n
        vid = idx[:, 0] + idx[:, 1] * nx + idx[:, 2] * nx * ny
        vid = np.where(inside, vid, -1)
        return int(vid[0]) if np.ndim(points) == 1 else vid

    def GetVisibility(self, points, reflected=False):
        """(N, n_opch) visibilities of the voxel containing each point (0 outside)."""
        table = self.refl_visibility if reflected else self.visibility
        if table is None:
            raise ValueError("reflected visibilities not loaded (load_reflected=True)")
        if self.mapping is None:
            idx, inside = self.VoxelIndices(points)
            vis = table[idx[:, 0], idx[:, 1], idx[:, 2]]
            vis[~inside] = 0.0
            return vis[0] if np.ndim(points) == 1 else vis
        lib_pts, cryo = self.DetectorToLibrary(points)
        idx, inside = self.VoxelIndices(lib_pts)
        lib = table[idx[:, 0], idx[:, 1], idx[:, 2]]
        lib[~inside] = 0.0
        m = self.mapping
        vis = np.zeros((len(lib_pts), self.n_opchannels), dtype=lib.dtype)
        nlib = len(m["lib_to_opch"])
        for c in np.unique(cryo):
            rows = np.where(cryo == c)[0]
            vis[np.ix_(rows, m["lib_to_opch"] + m["shifts"][c])] = lib[rows, :nlib]
        return vis[0] if np.ndim(points) == 1 else vis

    def export_flashmatch_npz(self, path):
        """Write the npz layout read by pointcept.models.event_slicer.photonlib.PhotonLibLookup
        (grid origin = grid_min in detector coordinates). For a mapped library (ICARUS)
        this is the raw library: cryostat 0 points and library channels only."""
        np.savez(path, visibility=self.visibility, nvoxels_dim=self.grid_n,
                 voxel_len_cm=self.voxel_len, cryo_origin_tpc_cm=self.grid_min,
                 n_opdets=np.int64(self.visibility.shape[-1]))
        return path
