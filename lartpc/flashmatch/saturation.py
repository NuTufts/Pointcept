"""Saturated-PMT ("hole") finder for the in-time flash.

A PMT sitting under a bright flash can saturate: the pulse rails past the 12-bit
ADC ceiling (4096), the baseline restoration undershoots, the ophit's integrated
area goes NEGATIVE, the reconstructed PE collapses to <= 0, and the opflash then
reports exactly 0.0 PE for that channel. The Neyman chi2 charges the CORRECT
slice (obs - pred)^2 / eps ~ pred^2 for that tube -- order 1e6 -- which swamps
the sum, so the chi2 ranking prefers a near-empty slice somewhere else entirely.

Verified on MC run3b overlay RSE (14263, 243, 12177): opdet 17 (opch 14) has an
ophit at the beam time (3.625 us) with amplitude 505 but area -228 -> PE -1.90
-> flash PE 0.0, while its neighbour opch 13 hit amplitude 8106 (2x the ADC
ceiling). The nu slice was the physically-correct one (reco vertex 2.5 cm from
truth, predicted light centroid z=489.6 vs observed 493.7) yet scored chi2 =
1.3e6, 92% of it from that single tube. Masking it: chi2 -> 81.6.

The finder uses the OBSERVED PE only. That matters: the mask is a property of
the event, not of the slice hypothesis, so every slice is scored on the same PMT
subset and the comparison stays fair. A prediction-dependent mask (e.g. "drop
tubes where pred >> obs") would let any slice excuse its own mismatches and is
circular.

`max_masked` caps how many tubes the finder may drop, so a pathological event
cannot mask away most of the array and earn an artificially low chi2. Candidates
are ranked by how anomalous they are (neighbour brightness) and only the worst
`max_masked` are returned. Dead channels are handled separately (see
dead_channels.py) and do not consume the saturation budget.
"""
import numpy as np

N_PMTS = 32

# opdet -> (x, y, z) cm, MicroBooNE v12 geometry. Matches larlite
# larutil::Geometry::GetOpDetPosition -- verified 32/32 against the
# pmt_positions the stage-A converter writes into merged_sp.
# NOTE: only distances between PMTs are used here, so the frame offset
# (global vs TPC/spacepoint) is irrelevant.
_OPDET_POS = np.array([
    (-11.4545, -28.625, 990.356), (-11.4175, 27.607, 989.712),
    (-11.7755, -56.514, 951.865), (-11.6415, 55.313, 951.861),
    (-12.0585, -56.309, 911.939), (-11.8345, 55.822, 911.065),
    (-12.1765, -0.722, 865.599), (-12.3045, -0.502, 796.208),
    (-12.6045, -56.284, 751.905), (-12.5405, 55.625, 751.884),
    (-12.6125, -56.408, 711.274), (-12.6615, 55.8, 711.073),
    (-12.6245, -0.051, 664.203), (-12.6515, -0.549, 585.284),
    (-12.8735, 55.822, 540.929), (-12.6205, -56.205, 540.616),
    (-12.5945, -56.323, 500.221), (-12.9835, 55.771, 500.134),
    (-12.6185, -0.875, 453.096), (-13.0855, -0.706, 373.839),
    (-12.6485, -57.022, 328.341), (-13.1865, 54.693, 328.212),
    (-13.4175, 54.646, 287.976), (-13.0075, -56.261, 287.639),
    (-13.1505, -0.829, 242.014), (-13.4415, -0.303, 173.743),
    (-13.3965, 55.249, 128.354), (-13.2784, -56.203, 128.18),
    (-13.2375, -56.615, 87.8695), (-13.5415, 55.249, 87.7605),
    (-13.4345, 27.431, 51.1015), (-13.1525, -28.576, 50.4745),
], dtype=np.float64)

# defaults tuned on the run3b overlay high-chi2 CC-2gamma sample
HOLE_PE = 5.0        # "reads ~nothing": observed PE at or below this
NEIGH_PE = 100.0     # "surrounded by light": median neighbour PE above this
N_NEIGHBORS = 3      # how many nearest tubes define the neighbourhood
MAX_MASKED = 2       # cap on saturation-masked tubes per event


def _neighbor_index(positions, k, exclude):
    """For each opdet, the k nearest other opdets (excluding `exclude`)."""
    n = positions.shape[0]
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if exclude:
        d[:, list(exclude)] = np.inf
    return np.argsort(d, axis=1)[:, :k]


def find_saturated(pe_obs, dead=(), max_masked=MAX_MASKED, hole_pe=HOLE_PE,
                   neigh_pe=NEIGH_PE, n_neighbors=N_NEIGHBORS, positions=None,
                   return_all=False):
    """Opdets that read ~nothing while their nearest neighbours are bright.

    Args:
        pe_obs: (N_PMTS,) observed PE (opdet-indexed).
        dead: opdets known dead for this run -- skipped as candidates and
            excluded from neighbourhoods (a dead tube must not make its
            neighbour look like a hole, nor hide one).
        max_masked: cap on returned tubes; the most anomalous survive. Use
            None for no cap (diagnostics only).
        hole_pe: candidate if pe_obs <= this.
        neigh_pe: candidate only if the median neighbour PE exceeds this.
        return_all: also return the uncapped candidate list.

    Returns:
        tuple of opdets (sorted), or (capped, all_candidates) if return_all.
    """
    pe = np.nan_to_num(np.asarray(pe_obs, np.float64))
    pos = _OPDET_POS if positions is None else np.asarray(positions, np.float64)
    dead = tuple(int(d) for d in dead if 0 <= int(d) < pe.shape[0])
    nbr = _neighbor_index(pos[:pe.shape[0]], n_neighbors, dead)

    cand = []
    for i in range(pe.shape[0]):
        if i in dead or pe[i] > hole_pe:
            continue
        score = float(np.median(pe[nbr[i]]))
        if score > neigh_pe:
            cand.append((score, i))
    cand.sort(key=lambda t: (-t[0], t[1]))          # most anomalous first
    allc = tuple(sorted(i for _, i in cand))
    keep = cand if max_masked is None else cand[:int(max_masked)]
    out = tuple(sorted(i for _, i in keep))
    return (out, allc) if return_all else out


def masked_opdets(pe_obs, run=None, dead=None, max_masked=MAX_MASKED, **kw):
    """Convenience: dead channels for `run` (or the explicit `dead`) UNION the
    saturation mask. The cap applies to the saturation part only -- dead tubes
    are a known detector fact and always masked."""
    if dead is None:
        if run is None:
            dead = ()
        else:
            from .dead_channels import dead_opdets_for_run
            dead = dead_opdets_for_run(run)
    dead = tuple(int(d) for d in dead)
    sat = find_saturated(pe_obs, dead=dead, max_masked=max_masked, **kw)
    return tuple(sorted(set(dead) | set(sat)))
