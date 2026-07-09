"""Numpy port of the LArPID prong cropper (FlowTriples
make_cropped_initial_sparse_prong_image_reco + gen2ntuple makeImage).

Builds the model's (6, 512, 512) input from merged_sp sparse plane images:
channels = [p0 prong, p0 context, p1 prong, p1 context, p2 prong, p2 context].

Faithful to the original with two documented substitutions:
- thrumu (cosmic-tag) veto -> SLICE-UNION ACCEPTANCE: context pixels are
  restricted to pixels touched by spacepoints of the ACCEPTED slices (the nu
  union + the flash-match slice), which is what the thrumu veto approximated
  (removing cosmic activity). Prong pixels are the particle's own pixels.
- out-of-crop prong pixels are DROPPED (the original python scattered their
  negative local coords, silently wrapping); only affects prongs whose image
  span exceeds 512 pixels in a plane.

Image conventions (verified against merged_sp files):
  image_data/planeN/coord[:,0] = wire (col, 0..3455)
  image_data/planeN/coord[:,1] = time row = (tick - 2400) / 6 (0..1007)
  image_data/triplet_imgpix_index (N_sp, 4): per-spacepoint pixel index per
  plane (cols 0..2); -1 = none.
Crop bounds per plane (from the C++): if the prong bounding box fits in
512x512, center on the BBOX CENTER; else center on the projection of the 3D
crop point (here: the (row, wire) of the particle spacepoint nearest that
point); clamp to the image.
"""
import numpy as np

PIX_THRESHOLD = 10.0
CROP = 512
N_ROWS, N_COLS = 1008, 3456
MIN_PRONG_PIXELS = 10          # per plane; below on ANY plane -> unclassified


class EventImages:
    """Per-merged_sp-event pixel tables + spacepoint->pixel/plane maps."""

    def __init__(self, msp_entry):
        img = msp_entry["image_data"]
        self.pix = []                       # per plane: (rows, cols, vals)
        for p in range(3):
            g = img[f"plane{p}"]
            coord = g["coord"][()]
            self.pix.append((coord[:, 1].astype(np.int64),
                             coord[:, 0].astype(np.int64),
                             g["feat"][()].astype(np.float32)))
        self.tpi = img["triplet_imgpix_index"][()][:, :3].astype(np.int64)
        td = msp_entry["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        self.tick = td["tick"][()].astype(np.int64)
        self.wires = np.stack([td["uwire"][()], td["vwire"][()],
                               td["ywire"][()]], axis=1).astype(np.int64)
        self._row_of = {pos[i].tobytes(): i for i in range(len(pos))}
        self.pos = pos

    def triplet_rows(self, coords_cm):
        """Map (bit-identical) slice/particle coords -> triplet row indices."""
        c = np.asarray(coords_cm, np.float32)
        rows = [self._row_of.get(c[i].tobytes()) for i in range(len(c))]
        return np.asarray([r for r in rows if r is not None], np.int64)

    def pixels_for(self, trip_rows, plane):
        """(rows, cols, vals) of the plane pixels touched by these spacepoints,
        above PIX_THRESHOLD, deduplicated."""
        if trip_rows.size == 0:
            return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float32),)
        idx = np.unique(self.tpi[trip_rows, plane])
        idx = idx[idx >= 0]
        r, c, v = self.pix[plane]
        r, c, v = r[idx], c[idx], v[idx]
        keep = v >= PIX_THRESHOLD
        return r[keep], c[keep], v[keep]

    def center_rowcol(self, trip_rows, center3d):
        """(row, col-per-plane) of the spacepoint (among trip_rows) nearest to
        the 3D crop point — the projection used when a prong exceeds the crop."""
        if trip_rows.size == 0:
            return None
        d = np.linalg.norm(self.pos[trip_rows]
                           - np.asarray(center3d, np.float32), axis=1)
        i = int(trip_rows[int(np.argmin(d))])
        row = int((self.tick[i] - 2400) // 6)
        return row, [int(self.wires[i, p]) for p in range(3)]


def crop_bounds(prong_pix, center_rc):
    """Per-plane [row0, col0]: bbox-centered when the prong fits, else centered
    on the 3D-point projection; clamped to the image (C++ getRecoImageBounds)."""
    bounds = []
    for p in range(3):
        r, c, _ = prong_pix[p]
        if r.size and (r.max() - r.min()) < CROP and (c.max() - c.min()) < CROP:
            r0 = (int(r.min()) + int(r.max())) // 2 - CROP // 2
            c0 = (int(c.min()) + int(c.max())) // 2 - CROP // 2
        elif center_rc is not None:
            r0 = center_rc[0] - CROP // 2
            c0 = center_rc[1][p] - CROP // 2
        else:
            r0, c0 = 0, 0
        r0 = min(max(r0, 0), N_ROWS - CROP)
        c0 = min(max(c0, 0), N_COLS - CROP)
        bounds.append((r0, c0))
    return bounds


def build_input(prong_pix, ctx_pix, bounds):
    """Scatter into the (6, 512, 512) tensor layout of gen2ntuple makeImage.
    Returns (image float32, n_prong_in_crop per plane)."""
    img = np.zeros((6, CROP, CROP), np.float32)
    n_prong = []
    for p in range(3):
        r0, c0 = bounds[p]
        for ch_off, (r, c, v) in ((0, prong_pix[p]), (1, ctx_pix[p])):
            lr, lc = r - r0, c - c0
            keep = (lr >= 0) & (lr < CROP) & (lc >= 0) & (lc < CROP)
            img[2 * p + ch_off, lr[keep], lc[keep]] = v[keep]
            if ch_off == 0:
                n_prong.append(int(keep.sum()))
    return img, n_prong
