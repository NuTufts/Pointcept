"""Spacepoint charge for a particle from the source merged_sp file.

Charge convention = the production's (`_dedup_row_charge` in the cascade and
`part_charge` in nu_reco): exact-position match of slice coords into
triplet_data, then calo.dedup_charge over the WHOLE nu-slice point set (so a
pixel shared by two particles is split between them), then per-particle sums
over its point indices. Using the slice-wide dedup keeps the muon charge equal
to the ntuple's track Charge and lets the full-slice prediction reproduce the
stored slices/pred_pe as a pipeline check.
"""
import numpy as np
import h5py

from lartpc.larformer_reco.trajfit.calo import dedup_charge


class MspCharge:
    def __init__(self, msp_path):
        with h5py.File(msp_path, "r") as f:
            td = f["entry_0/triplet_data"]
            pos = td["pos"][()].astype(np.float32)
            self.pix = td["pixval"][()].astype(np.float64)
            self.tick = td["tick"][()].astype(np.int64)
            self.uw = td["uwire"][()].astype(np.int64)
            self.vw = td["vwire"][()].astype(np.int64)
            self.yw = td["ywire"][()].astype(np.int64)
            self._row = {pos[i].tobytes(): i for i in range(len(pos))}
        self.n = len(self._row)

    def rows_for(self, coords):
        c = np.ascontiguousarray(np.asarray(coords, np.float32))
        return np.asarray([self._row.get(c[i].tobytes(), -1)
                           for i in range(len(c))], np.int64)

    def dedup_comb(self, rows):
        """Per-row de-double-counted comb charge, dedup within `rows`
        (rows < 0 -> 0)."""
        q = np.zeros(len(rows), np.float64)
        ok = rows >= 0
        if ok.any():
            r = rows[ok]
            _, qc = dedup_charge(self.pix[r], self.tick[r], self.uw[r],
                                 self.vw[r], self.yw[r])
            q[ok] = qc
        return q


def slice_charge(msp, slice_coords):
    """(q_comb per slice point, n_unmatched) with dedup over the whole slice."""
    rows = msp.rows_for(slice_coords)
    return msp.dedup_comb(rows), int((rows < 0).sum())
