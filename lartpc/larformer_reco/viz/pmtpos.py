"""MicroBooNE PMT positions + opdet<->opchannel map (v12 geometry).

Self-contained copy of ubdl/lardly/lardly/ubdl/pmtpos.py (v12 branch only) so
the visualizer does not depend on lardly being importable in the container.
Positions are returned in the TPC/spacepoint coordinate frame (same frame as
merged_sp triplet_data/pos), so PMT circles overlay directly on the charge.

The 32-element flash PE arrays (merged_sp flashes/pe, cascade flash/observed_pe,
slices/pred_pe) may be indexed by either opdet (Geant4/GDML sort order) or
opchannel (electronics readout). Use getPMTPosByOpDet vs getPMTPosByOpChannel
to test which indexing makes the observed light line up with the reco charge.
"""

# opdet index (Geant4/GDML) -> global (x,y,z) cm, v12 microboone geometry.
# 0-31 are the PMTs; 32-35 are the light-bar paddles (x=-161, ignore for PMTs).
_opdet_pos = {
    0: (-11.4545, -28.625, 990.356), 1: (-11.4175, 27.607, 989.712),
    2: (-11.7755, -56.514, 951.865), 3: (-11.6415, 55.313, 951.861),
    4: (-12.0585, -56.309, 911.939), 5: (-11.8345, 55.822, 911.065),
    6: (-12.1765, -0.722, 865.599), 7: (-12.3045, -0.502, 796.208),
    8: (-12.6045, -56.284, 751.905), 9: (-12.5405, 55.625, 751.884),
    10: (-12.6125, -56.408, 711.274), 11: (-12.6615, 55.8, 711.073),
    12: (-12.6245, -0.051, 664.203), 13: (-12.6515, -0.549, 585.284),
    14: (-12.8735, 55.822, 540.929), 15: (-12.6205, -56.205, 540.616),
    16: (-12.5945, -56.323, 500.221), 17: (-12.9835, 55.771, 500.134),
    18: (-12.6185, -0.875, 453.096), 19: (-13.0855, -0.706, 373.839),
    20: (-12.6485, -57.022, 328.341), 21: (-13.1865, 54.693, 328.212),
    22: (-13.4175, 54.646, 287.976), 23: (-13.0075, -56.261, 287.639),
    24: (-13.1505, -0.829, 242.014), 25: (-13.4415, -0.303, 173.743),
    26: (-13.3965, 55.249, 128.354), 27: (-13.2784, -56.203, 128.18),
    28: (-13.2375, -56.615, 87.8695), 29: (-13.5415, 55.249, 87.7605),
    30: (-13.4345, 27.431, 51.1015), 31: (-13.1525, -28.576, 50.4745),
}

# opdet -> (opchannel, +100, +200, +300); channel 0-31 is "the" OpChannel.
_opdet2opch_first = {
    0: 29, 1: 27, 2: 31, 3: 26, 4: 30, 5: 25, 6: 28, 7: 22, 8: 24, 9: 20,
    10: 23, 11: 19, 12: 21, 13: 16, 14: 14, 15: 18, 16: 17, 17: 13, 18: 15,
    19: 10, 20: 12, 21: 8, 22: 7, 23: 11, 24: 9, 25: 3, 26: 1, 27: 6, 28: 5,
    29: 0, 30: 2, 31: 4,
}
_opch2opdet = {ch: od for od, ch in _opdet2opch_first.items()}

# spacepoint/TPC frame origin (subtract from global to match the charge frame)
_tpc_origin = (-1.825, 0.97, -4.0)


def getPMTPosByOpDet(opdet, in_tpc_coord=True):
    g = _opdet_pos[opdet]
    if in_tpc_coord:
        return [g[i] - _tpc_origin[i] for i in range(3)]
    return list(g)


def getPMTPosByOpChannel(opch, in_tpc_coord=True):
    return getPMTPosByOpDet(_opch2opdet[opch], in_tpc_coord=in_tpc_coord)
