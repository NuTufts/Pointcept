# LArFormer Reco — Output Data Schema

> **Status: REFERENCE** — schema of the files produced by the larformer_reco
> workflow, as written by the code at commit `19a4477` (2026-07). Verified
> against files produced from the mcc9 bnb-nu-overlay sample.

The workflow produces three file types (plus one optional sidecar):

```
tools/larformer/run_larformer_keypoint2_cascade_inference.py
  → 1. keypoint2_event{i:05d}[_fm]_{ei}.h5      per (event, stream)
  → 2. sliceid_event{i:05d}.h5                  optional sidecar (--save-slice-ids)
lartpc/larformer_reco/scripts/run_nu_reco.py
  → 3. nu_reco_shard{START:07d}.h5              one per CPU shard
lartpc/larformer_reco/eval/eval_reco_performance.py
  → 4. eval_shard*.npz / merged results npz     per-true-particle records
```

## Global conventions

- **Frame/units**: all positions are detector coordinates in **cm**
  (x drift [0,256], y vertical [−117,117], z beam [0,1036]); energies/momenta in
  **MeV** / **MeV/c**; times in **µs** relative to the trigger; charge in
  de-double-counted ADC ("comb" = Y plane, else mean(U,V); see
  `trajfit/calo.py`).
- **Sentinels**: missing 3-vectors are `NaN`; missing indices/ids are `-1`.
- **Streams**: each event can yield up to two reco streams, labeled by the
  `stream` attr — `"nu"` (the slicer's nu-union slice; production path),
  `"flashmatch"` (the best flash-χ² slice when it differs from the nu union),
  or `"nu,flashmatch"` (one file, the nu slice was also the best flash match).
  Files without a `stream` attr predate streams and are `"nu"`.
- **Linkage keys**:
  - `src_file` (attr, all file types) = basename of the parent `merged_sp`
    input file (charge + MC truth).
  - `gt_trackid` ↔ `merged_sp:entry_0/mc_particle_tree/trackid`
    (↔ `triplet_data/trackid`); GEANT4 track ids.
  - `gidx` — `nu_reco` event groups are named `event_{gidx:07d}` where `gidx`
    is the **line index in the keypoint2 list** given to `run_nu_reco.py`.
    The eval uses the same list, so streams must never be mixed in one list
    (use the per-stream lists from `slurm/regen_kp2_list.sh`).
- **Particle class ids** (`cls`, `part_pred_class`):
  `0=e, 1=gamma, 2=mu, 3=pi, 4=p, 5=other` (`7=no_object` internally; never
  emitted for kept instances).

---

## 1. `keypoint2_event{i:05d}_{ei}.h5` / `..._fm_{ei}.h5` — cascade inference

One file per (event, stream). `i` = dataset index in the SORTED input list,
`ei` = in-batch event index (always 0 for the per-event driver). The `_fm_`
infix marks the flashmatch stream.

### File attributes

| attr | type | description |
|---|---|---|
| `src_file` | str | parent merged_sp basename |
| `stream` | str | `"nu"`, `"flashmatch"`, or `"nu,flashmatch"` |
| `slice_label` | str | flashmatch stream only: which slicer query, e.g. `"cosmic05"` (also set in `--all-slices` study outputs) |
| `flash_chi2` | f64 | flashmatch stream only: the chosen slice's Neyman χ² |
| `n_particles` | i64 | number of `particle/{i}` groups |
| `has_gt` | bool | GT matching present (sim; `--no-gt` for data) |
| `has_score_maps` | bool | `score_maps/` present (`--save-score-maps`) |
| `run`,`subrun`,`event` | i64 | event ids (from the input merged_sp `entry_0` attrs when the dataset does not surface them) |

### Datasets

| path | shape, dtype | description |
|---|---|---|
| `slice/coord_cm` | (N,3) f32 | spacepoints of THIS stream's slice, detector cm. All `point_idx` index into this array. |
| `nu_vertex_cm` | (3,) f32 | dense-head nu vertex: score-weighted centroid of spacepoints above `nu_thresh` |
| `gt_nu_vertex_cm` | (3,) f32 | true nu vertex (mckeypoints type 0); NaN without GT |
| `particle/{i}/point_idx` | (n,) i32 | predicted instance's indices into `slice/coord_cm` |
| `particle/{i}/start_cm`, `end_cm` | (3,) f32 | predicted start/end keypoints (end NaN if the end query lost to no-object) |
| `particle/{i}/gt_point_idx` | (m,) i32 | matched GT particle's indices into `slice/coord_cm` (majority per-point trackid match) |
| `particle/{i}/gt_start_cm`, `gt_end_cm` | (3,) f32 | GT visible start / track end (the loss targets), NaN if unavailable |

Per-particle attrs: `cls` (class id), `has_match` (bool), `iou` (f64,
predicted-vs-GT point IoU), `gt_trackid` (i64, −1 if unmatched),
`loose_pass` (bool: instance recovered by the below-threshold single-object
fallback — always enabled for the flashmatch stream, `--loose-fallback` for
the nu stream), `loose_conf` (f64, the fallback's ranking confidence; NaN for
standard instances).

### Flash-match products (all streams; absent with `--no-flash`)

| path | shape, dtype | description |
|---|---|---|
| `flash/observed_pe` | (32,) f32 | the in-time beam flash (producer 0, max total PE), PE per PMT |
| `flash/` attrs | | `time_us`, `total_pe`, `producer_id`, `flash_index` of that flash; `has_beam_flash`; the prediction/χ² parameters `gamma_beam`, `f_sys`, `eps`, `oob_max` |
| `flash/all/{pe,producer_id,total_pe,time_us}` | (Nf,32)/(Nf,) | every flash from the input (`producer_id`: 0=simpleFlashBeam, 1=simpleFlashCosmic) |
| `slices/label` | (S,) bytes | `"nu"` or `"cosmicQQ"` per slice row |
| `slices/query` | (S,) i32 | slicer query index; **−5 = the nu union** |
| `slices/n_points` | (S,) i32 | slice spacepoint count (rows require ≥ `--slice-min-points`) |
| `slices/pred_pe` | (S,32) f32 | PhotonLib-predicted PE per PMT per slice (drift-corrected by the beam-flash t0; NaN without a beam flash/charge) |
| `slices/chi2` | (S,) f32 | Neyman χ² vs `observed_pe` |
| `slices/oob_frac` | (S,) f32 | fraction of the slice outside the TPC post drift-correction |
| `slices/chi2_rank` | (S,) i32 | 1-based χ² rank among rows with `oob_frac ≤ oob_max`; **0 = not ranked**. Rank 1 defines the flashmatch stream. |
| `slices/p_nu` | (S,) f32 | slicer softmax P(nu) (nu row: max over nu queries) |
| `slices/nu_queries/{query,p_nu}` | (Nnu,) | the individual nu-class queries composing the nu union |

The `slices/` table covers the WHOLE event, so streams can be re-ranked
offline (different χ² cuts, top-K) without re-running the GPU inference.

### Diagnostics (`--save-score-maps`)

`score_maps/{nu_vertex,object}/{coords_cm,score}` — dense keypoint-head
sigmoid scores at token positions (attrs: `level`, `kp_types`);
`gt_keypoints/{pos_cm,type,trackid}` — all MC keypoints
(type: 0=nu_vertex, 1=track_start, 2=track_end, 3=shower).

---

## 2. `sliceid_event{i:05d}.h5` — full-event slice-id sidecar (optional)

Written with `--save-slice-ids` (every event, from the same slicer forward the
streams use) or `--slice-ids-only`.

| path | shape, dtype | description |
|---|---|---|
| `full_slice/coord_cm` | (M,3) f32 | ALL input spacepoints (pre-deghost) |
| `full_slice/slice_id` | (M,) i64 | **−2** ghost (deghoster dropped), **−1** kept-unclustered, **−5** nu union, **q ≥ 0** cosmic slice = slicer query index (matches `slices/query` and `cosmicQQ` labels) |
| `full_slice/deghost_p_real` | (M,) f32 | deghoster P(real) |

File attrs: `src_file`, `n_ghost`, `n_unclustered`, `n_nu`, `n_cosmic`,
`n_cosmic_slices`.

---

## 3. `nu_reco_shard{START:07d}.h5` — nu-interaction reconstruction

One file per shard (`START` = shard's first gidx); one group
`event_{gidx:07d}` per keypoint2 file that produced ≥1 interaction.
**This is the final reco product** (the source for the planned ROOT export).

File attrs: `shard_start`, `n_requested`, `n_reco`, `n_skip`, `n_err`.

### Event-group attributes

| attr | description |
|---|---|
| `src_file`, `run`, `subrun`, `event` | copied from the keypoint2 file |
| `stream`, `slice_label`, `flash_chi2` | stream provenance (see conventions) |
| `gt_nu_vertex_cm` | (3,) f32 true nu vertex (NaN without GT) |
| `n_interactions`, `n_particles` | table sizes |

### Vertex tree (flattened over the event's interactions)

| dataset | shape, dtype | description |
|---|---|---|
| `vertices_cm` | (I,3) f32 | PRIMARY vertex per interaction (row = interaction index) |
| `vertices_score` | (I,) f32 | seed nu-vertex-candidate score per interaction (NaN if unavailable). NOTE: interaction row order is reco-iteration order (attachment quality), NOT score order — rank by this column. |
| `vtx_pos_cm` | (V,3) f32 | all vertices, primary + secondary |
| `vtx_interaction` | (V,) i64 | owning interaction index (row of `vertices_cm`) |
| `vtx_depth` | (V,) i64 | 0 = primary, ≥1 = secondary (kink/decay) depth in the tree |
| `vtx_parent_track` | (V,) i64 | particle-table row of the track this vertex hangs off (−1 for primaries) |

### Per-particle table (row-aligned arrays, length `n_particles`)

Tracks first, then ATTACHED showers, per interaction.

| dataset | shape, dtype | description |
|---|---|---|
| `part_interaction` | (P,) i64 | interaction index |
| `part_kind` | (P,) i64 | **0 = track, 1 = shower** |
| `part_pred_class` | (P,) i64 | class id (see conventions) |
| `part_vtx` | (P,) i64 | index into `vtx_*` of the attach vertex (−1 = none, e.g. unassociated shower connection) |
| `part_energy` | (P,) f64 | TOTAL energy E [MeV] (E² = p² + m²_pred-class; showers massless) |
| `part_momentum` | (P,3) f32 | (px,py,pz) [MeV/c] |
| `part_fourvec` | (P,4) f32 | **(E, px, py, pz)** [MeV] |
| `part_direction` | (P,3) f32 | unit direction at the start |
| `part_method` | (P,) i64 | energy method: **0 = range** (Bethe–Bloch CSDA, tracks), **1 = calo** (calibrated charge, showers), −1 unknown |
| `part_length` | (P,) f64 | track polyline length [cm] (NaN for showers) |
| `part_charge` | (P,) f64 | de-double-counted comb charge |
| `part_start_cm` | (P,3) f32 | start point (tracks: first polyline point, oriented to start at the attach vertex; showers: trunk start / connection point) |
| `part_npoly` | (P,) i64 | polyline point count per particle (0 for showers) |
| `part_poly_cm` | (ΣN,3) f32 | fitted track polylines, concatenated; unpack with `np.split(part_poly_cm, np.cumsum(part_npoly)[:-1])` |
| `part_inst_idx` | (P,) i64 | keypoint2 particle index (`particle/{inst_idx}` in the source kp2 file; −1 unknown) — links a reco particle back to its predicted instance (point_idx etc.) |
| `part_gt_trackid` | (P,) i64 | majority-matched GT trackid (−1 unmatched / no GT) |
| `part_true_ke` | (P,) f64 | matched particle's true KE [MeV] (NaN unmatched) |

Note: showers carry no polyline — draw them from `part_start_cm` along
`part_direction`. Unattached showers are not in this table (they remain
visible in the keypoint2 file's `particle/` groups).

---

## 4. Eval records npz (`eval_shard*.npz`, merged results npz)

Flat per-TRUE-particle arrays (denominator = nu-origin, reconstructable
species; `--primaries-only` restricts to nu-vertex primaries; photons
additionally require E_vis > `gamma_min_evis`, default 20 MeV):

`species` (index into `species_names` = e,gamma,mu,pi,p), `true_ke`,
`found_A` (seg ≥70% completeness), `compl`, `found_B` (attached in nu_reco),
`reco_ke` (KE; **0.0 = not attached**), `reco_class`, `had_kp` (event had a
keypoint2 file), `found_C` (charge slice coverage ≥50%), `slice_cov`,
`slice_cov_count`, `q_true` (comb charge of all true points; E_vis ≈
`gamma_evis_calib`·`q_true`), `n_true_sp`, `has_instance`.
Metadata: `species_names`, `stream`, `gamma_min_evis`, `gamma_evis_calib`.
