# Prototype: neutrino-interaction reconstruction from nu-vertex + tracks

Combine the dense-head **nu-vertex** prediction with the per-particle **track
trajectories** (cluster_fit_stitch output) into a candidate neutrino interaction:
a tree of particles rooted at the vertex, growing outward through secondary
vertices at track ends.

## Inputs (per event, from `keypoint2_out`)
- **Primary vertex candidates** `V0`: from `--vertex-source`:
  - `reco` (default): ranked peaks from the **score-field fitter**
    (`reco.KeypointRecoTorch` — greedy Gaussian peak-finder with NMS peeling, in
    the sibling `reco/` package). This is the right decode — NOT the
    `nu_vertex_cm` score-weighted centroid, which a far spurious cluster drags
    tens of cm off (e.g. event4: centroid 70 cm off, fitter peak 0.4 cm).
  - `pred`: the single `nu_vertex_cm` centroid decode (baseline, for comparison).
  - `gt`: MC-truth vertex (development).
  Candidates are tried in score order (see Iterative seeding).
- **Tracks**: every track-class instance (cls in {mu,pi,p}), each reconstructed to
  a dense ordered centerline by `cluster_fit_stitch`. Each track has two
  **endpoints**; at each endpoint an **initial direction** `u_in` (unit, pointing
  into the body) is measured over the first `seg_cm` (≤3 cm, or the whole track if
  shorter).

## Attachment test (track end e vs vertex V)
A track end attaches to a vertex when **both**:
1. **near**: `gap = |pos_e - V| ≤ d_vertex`;
2. **points back**: the vertex lies on the backward extrapolation of the initial
   direction — perpendicular distance of `V` to the ray `{pos_e - s·u_in, s≥0}` is
   `≤ d_perp`, and `V` is not far *in front* of the track (`(V-pos_e)·(-u_in) ≥
   -front_tol`).

Cost (for choosing among candidates) `= gap + 2·perp`. Lower is better.

## Growth (BFS tree)
```
add V0 as the primary vertex; queue = [V0]
while queue:
    V = queue.pop_front()
    for each UNATTACHED track T:
        pick the end of T with the lowest admissible cost vs V (if any)
    attach those tracks to V (greedy, lowest cost first); for each attached T:
        mark T attached at end e
        far = T's OTHER endpoint  ->  becomes a new vertex V'
              (merged into an existing vertex if within merge_radius)
        enqueue V' if newly created
```
So: primary particles attach at `V0`; each particle's far end becomes a secondary
vertex where the remaining tracks are tested — chaining scatters, decays, and
secondary interactions. A track is attached at most once (one end → one vertex);
its other end seeds the next vertex. Vertices within `merge_radius` coalesce, so a
multi-prong secondary vertex is one node.

## Output
- `vertices`: list of {id, pos, depth, parent_track, attached_track_ids}. depth 0
  = primary; depth d = reached after d particles.
- `tracks`: each with {attached?, attach_vertex, attach_end, far_vertex, depth}.
- `unattached`: tracks not consistent with any reached vertex (candidates for a
  different interaction, cosmic contamination, or a missed link — tune `d_vertex`/
  `d_perp` or widen, mirroring the track-stitch `max_gap` philosophy).

## Parameters (CLI)
`--d-vertex` (endpoint→vertex max, cm), `--d-perp` (back-pointing tolerance, cm),
`--seg-cm` (initial-direction arc length), `--merge-radius` (vertex coalescing),
`--front-tol`, `--vertex-source {pred,gt}`.

## Iterative seeding (over ranked candidates)
The score-field fitter returns several ranked candidates. Each is tried (snap +
grow) and the one building the **best** interaction is kept — objective
`(#tracks attached, primary prong count, peak score)`. A candidate that attaches
nothing is passed over for the next, exactly as proposed. On the dev events the
top-score candidate usually wins outright; the iteration is the safety net for
when the strongest peak is spurious.

## Seed-vertex refinement (`snap`, on by default)
The predicted nu vertex can be tens of cm off (it's a score-weighted centroid).
Before growing, `snap_vertex` moves the seed to the densest **cross-track**
endpoint cluster within `--snap-radius` (a real interaction point = where ≥2
*different* tracks' ends converge; same-track endpoint pairs are excluded so a
single short track can't masquerade as a vertex). On the dev events this recovers
a 22.7 cm-off predicted vertex (snap 27.6 cm → 3-prong primary, matching the
GT-vertex result). A 70 cm-off vertex is beyond range and left alone — that needs
a better vertex *candidate* upstream (score-map clustering), not snapping.

## Known limitations / next steps
- **Neutral-induced secondary vertices are unreachable.** Secondary vertices form
  only at the far end of a *visible* track, so a neutron/photon-induced secondary
  (no parent track) won't be reached — its daughters stay unattached (seen on
  event0: 2 neutron-daughter tracks left unattached). Seeding secondaries from
  clusters of unattached track starts would address this.
- Single primary vertex (highest score). Multi-candidate vertices (cluster the
  dense `score_maps/nu_vertex`) → multiple interactions is a wrapper extension.
- Loose gates (`d_vertex=12`, `d_perp=4`) absorb the ~5-10 cm reconstructed
  track-start scatter; tighten once start resolution improves.
- Greedy BFS attachment (primary-first). A global assignment (Hungarian / tree
  optimization) could resolve ambiguous multi-vertex cases.
- No particle-type logic yet (e.g. require a proton+lepton at the primary); purely
  geometric.
