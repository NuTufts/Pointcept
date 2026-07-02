# Performance Evaluation Spec: Per-Species Reco Efficiency vs KE

## 0. Goal
Measure how well the nu-interaction reco finds each **particle species** as a
function of **true kinetic energy (KE)**, over the 10k validation set. Three
complementary efficiencies (same denominator):

- **Metric A — segmentation/cluster efficiency**: fraction of true nu-origin
  particles for which a predicted instance ("query mask") captured **≥70%** of the
  particle (completeness measured *within the nu slice*).
- **Metric B — attachment + kinematics efficiency**: fraction of true nu-origin
  particles for which a reco particle is **attached to a (primary or secondary)
  vertex and has a reconstructed 4-momentum**. Also produces a **2D reco-vs-true
  KE** scatter/2D-hist per species; a missed particle is assigned reco KE = 0.
- **Metric C — slice-coverage efficiency (upstream diagnostic)**: fraction of true
  nu-origin particles whose **nu slice** (`keypoint2_out:/slice/coord_cm`) captures
  **≥50%** of the particle's **visible ionization**. This is **charge-based**, not
  spacepoint-count-based: `coverage = Q(in-slice true points) / Q(all true points)`,
  where `Q` is the reco's de-double-counted calorimetric charge
  (`trajfit_dev/calo.py:dedup_charge`, `comb` = Y plane else mean(U,V) — the same
  quantity the shower energy reco integrates). Each wire pixel's ADC is split among
  the spacepoints sharing it, so `Q` over a set counts every pixel it touches once;
  denominator = "pass in the GT spacepoints", numerator = "pass in the reco/slice
  spacepoints", each de-double-counted over its own set. Charge (not count) because
  the GT labeller is **over-liberal on the low-charge ionization edges/tails** —
  count-based coverage over-penalises those many low-ADC edge spacepoints (count
  coverage ~0.31 vs charge ~0.71 on the dev electron). Isolates **upstream
  slice/deghost losses** from the reco: if C is high but A/B are low the reco is the
  culprit; if C itself is low the particle never made it into the slice. (The old
  count-based coverage is still saved as `slice_cov_count` for comparison, and the
  charge denominator as `q_true`.)

Linkage adds `triplet_data/trackid == mc_particle_tree.trackid` — verified: for
the valdata samples `triplet_data/trackid` carries the real GEANT nu trackids
(unlike the older overlay dev set where nu deposits were labelled 0). `triplet_truth`
does **not** exist in these files; use `triplet_data` (with per-point `trackid`,
`origin`, `pid`, `pos`). `slice/coord_cm` points are bit-identical members of
`triplet_data/pos` (match dist 0), matched triplet_data→slice so the coverage
numerator and denominator count the same (duplicate-inclusive) objects.

Denominator (all three): number of true, **neutrino-origin** instances of the
species. Per-particle `slice_cov` (the fraction) and `n_true_sp` (the denominator)
are saved so a visibility cut (e.g. `n_true_sp > 0`) can be applied downstream.

---

## 1. Is there enough info? YES — file roles + the (verified) linkage
Three products per event, all needed:

| product | role | key fields |
|---|---|---|
| `merged_sp` | **truth** (denominator) | `entry_0/mc_particle_tree`: `trackid`, `pid`, `energy_mev` (= true **KE**), `origin` (**1=ν, 2=cosmic**, −1=the ν), `parent_trackid`, `process_code` |
| `keypoint2_out` | **segmentation** (metric A) | per `particle/{i}`: `point_idx` (predicted cluster in the slice), `gt_point_idx` (the matched true particle's slice points), `cls` (predicted species), `gt_trackid`, `iou`; root `slice/coord_cm` |
| `nu_reco_shard*.h5` | **reco + kinematics** (metric B) | per event group `event_{gidx}`: `part_gt_trackid`, `part_pred_class`, `part_energy`, `part_momentum`, `part_true_ke`, `part_kind`, `part_interaction`; `vertices_cm`; attr `src_file` |

**The linkage is by trackid, and it is verified consistent:**
`keypoint2.gt_trackid` == `nu_reco.part_gt_trackid` == `mc_particle_tree.trackid`
(same Geant4 trackid). So a true particle `t` (from `mc_particle_tree`) is matched
to the reco by `gt_trackid == t`. Confirmed on the dev set: gt_trackid 1415816 →
mc trk 1415816 = γ, KE 32 MeV; 1416263 → proton, KE 78 MeV; etc.

**Two linkage gotchas (must respect):**
1. **Resolve the parent by `src_file`, per file.** keypoint2 files are named by a
   processing index (`keypoint2_event{gidx}`), NOT by the source; the parent
   `merged_sp` is the root attr `src_file` (basename), matched against the
   merged_sp list. (Using the wrong parent silently breaks everything.)
2. **Do NOT use `triplet_data`'s per-spacepoint `trackid` for ν particles** — it
   labels ν deposits as `0`. Use `gt_trackid` (which carries the real mc trackid)
   for matching. (`slice/coord_cm` IS an exact subset of `triplet_data/pos`, dist
   0, so spatial matching also works if ever needed.)

**Cross-product event key:** a `nu_reco` `event_{gidx}` corresponds to
`keypoint2_list[gidx]`, whose `src_file` gives the `merged_sp`. So iterate one
list and index the others by `gidx` / `src_file`.

---

## 2. Denominator — true nu-origin particles by species
From each event's `merged_sp/mc_particle_tree`, keep particles with **`origin==1`**
(neutrino) whose `pid` maps to a reconstructable species:

`pid -> species`: ±11→e, 22→γ, ±13→μ, ±211→π, 2212→p.
(Exclude pid 111=π⁰ [reconstructed as its two γ showers, not itself], 2112=n
[not directly visible], nuclei, etc. — report their count separately as
"not-reconstructable ν-origin".)

True KE = `energy_mev`. Species and KE define the binning.

**Visibility cut (denominator options — report both):**
- **end-to-end**: all `origin==1` particles of the species (includes upstream
  deghost/slice losses; a particle with no slice points counts as missed).
- **reconstructable**: additionally require the particle be visible in the slice
  (≥ `min_true_slice_pts`, e.g. 10, computed from `gt_point_idx` when a matching
  instance exists; or via the slice→triplet map). This isolates the
  segmenter+reco from upstream losses.

**Primary vs all ν-origin:** default = all `origin==1` (primaries at the ν vertex
+ secondaries at secondary vertices). Optionally restrict to primaries
(`parent_trackid` is the ν / `process_code` primary) — expose as a flag.

**Fully-missed events:** the ~1.5k `merged_sp` with no `keypoint2` output (no ν
slice found) contribute their ν-origin particles as **all-missed** to the
denominator. Iterate the `merged_sp` (or keypoint2 `src_file`) set so these are
included; don't silently drop them.

---

## 3. Metric A — segmentation/cluster efficiency (≥70% completeness)
For a true particle `t`, **completeness** = fraction of `t`'s slice points captured
by a predicted instance. Directly from `keypoint2_out`: the instance with
`gt_trackid == t` has `gt_point_idx` = all of `t`'s slice points, so
`completeness = |point_idx ∩ gt_point_idx| / |gt_point_idx|`. (If `t` is split
across instances, take the max over instances with that `gt_trackid`; if no
instance matches `t`, completeness = 0.)

`t` is **found (A)** if `max completeness ≥ 0.70`. Efficiency_A(species, KE bin) =
found / denominator.

Notes:
- This is completeness **within the nu-slice** (what the segmenter's query mask is
  responsible for). Combined with the end-to-end denominator it also folds in
  deghost/slice losses; with the reconstructable denominator it isolates the
  segmenter. Report both.
- Optionally also record the **predicted-class correctness** (`cls` of the
  matched instance vs true species) — a confusion matrix per species is a cheap,
  valuable by-product.

---

## 4. Metric B — attachment + kinematics efficiency + 2D reco-vs-true KE
`nu_reco_shard` contains **only attached particles with kinematics** (tracks +
attached showers, across all interaction candidates). So a true particle `t` is
**found (B)** if some `nu_reco` row has `part_gt_trackid == t`.

- Efficiency_B(species, KE bin) = found_B / denominator.
- **2D reco-vs-true KE** per species: x = true KE (`energy_mev`), y = **reco KE**
  (see below); missed → y = 0. Plot as a 2D histogram / scatter, split by KE
  range if helpful.
- **reco KE convention** (compare KE to KE): `part_energy` is total E for tracks
  (mass + KE) and calorimetric KE for showers. So
  `reco_KE = part_energy - mass(class)` for tracks (μ 105.66, π 139.57, p 938.27),
  `= part_energy` for showers (e/γ). (`part_method` distinguishes range vs calo;
  `part_true_ke` is the matched mc KE for a cross-check.)
- If `t` matches **multiple** reco rows (fragmented/duplicated), take the one with
  the largest energy (or nearest KE) and flag the multiplicity.

---

## 5. Binning & aggregation
- **KE bins** per species (they span very different ranges): e.g. protons
  0–1000 MeV, showers 0–1000 MeV, μ 0–2000 MeV; use ~20–50 MeV bins near the
  turn-on, coarser above. Make bin edges configurable per species.
- Efficiency = found/denominator per (species, KE bin), with binomial errors
  `sqrt(eff(1-eff)/N)`.
- Save the **per-particle record table** (species, true_KE, foundA, complA,
  foundB, reco_KE, reco_class, method, event key) so plots/rebinning are trivial
  downstream.

---

## 6. Output & plots
- `eval_reco_performance.py` writes `eval_records.npz` (the per-particle table) +
  prints a per-species summary (denominator, eff_A, eff_B in a few coarse KE bins)
  + a class-confusion summary.
- Plots (matplotlib; the script can emit PNGs or leave to a notebook):
  - **Efficiency vs true KE** per species (A and B on the same axes).
  - **2D reco-vs-true KE** per species (with the y=x line; missed pile up at y=0).
  - Optional: completeness distribution; predicted-class confusion matrix.

---

## 7. Suggested code layout
```
lartpc_data_prep/larformer_keypoint_v2/
  eval_reco_performance.py   # this eval (first pass provided)
  performance_eval_spec.md   # this doc
```
The eval reuses `trajfit_dev/trajfit_io.py` only if it needs the slice→triplet map
(not required for the trackid-based path). It is otherwise pure h5py+numpy so it
runs anywhere (no torch/ROOT).

---

## 8. References for background (for the cluster Claude session)
Read these (all under `lartpc_data_prep/larformer_keypoint_v2/` unless noted) to
understand the reconstruction being evaluated:
- **`trajfit_dev/README.md`** — the running log of the whole reco: track fitting,
  vertex finding, shower attachment, momentum, findings & validated numbers. Best
  single starting point.
- **`nu_interaction_spec.md`** — the vertex→track→shower interaction algorithm.
- **`shower_reco_spec.md`** — the 3 shower-direction methods (vertex-biased wins).
- **`trajectory_fitting_brief.md`** — §2 is the authoritative `keypoint2_out` /
  `merged_sp` data schema; §2.4 the particle-class codes.
- **`particle_momentum_spec.md`** — range (Bethe-Bloch) + calorimetry momentum;
  the `nu_reco` `part_*` fields come from here (`assign_momenta`).
- **`docs/shower_fragment_origin_spec.md`** (repo `docs/`) — shower origin/trunk
  truth (`merged_sp/shower_fragments`).
- Code: **`run_nu_reco.py`** (production driver + the `nu_reco_shard` output
  schema in `_write_event`), **`trajfit_dev/nu_interaction.py`** (the reco),
  **`trajfit_dev/particle_momentum.py`** (kinematics + calo calibration),
  **`trajfit_dev/trajfit_io.py`** (loader, slice↔triplet, mc pid/KE).
- Container/run: `run_in_local_pointcept_container.sh` (local) or the apptainer
  sif on the cluster; `submit_nu_reco_shard.sh` produced the shards being
  evaluated. The eval itself is h5py+numpy only.

---

## 9. Open questions / decisions for the new session
1. Denominator: primaries-only vs all ν-origin (default all `origin==1`); and the
   visibility cut (`min_true_slice_pts`) — confirm the values for the headline
   number.
2. Completeness "70%" is vs **slice** points (post-deghost). Acceptable, or do you
   want it vs true energy deposits (needs the slice→triplet→edep path, since
   triplet ν-trackid is 0 — heavier)?
3. Shower species truth: a converted photon's shower matches the **photon**
   (pid 22) trackid — confirm that's the intended "true particle" for γ efficiency
   (vs the conversion e±).
4. π⁰ / neutron handling — report separately as not-reconstructable, or fold π⁰
   into a "2γ found" efficiency?
5. For metric B multiplicity (one true particle → several reco rows across
   interaction candidates), pick largest-energy vs nearest-KE.
