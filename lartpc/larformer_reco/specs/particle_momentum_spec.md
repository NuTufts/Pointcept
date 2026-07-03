# Project Spec: Particle 4-Momentum Assignment for the Nu-Interaction Reco

## 0. Scope & how to use
The nu-interaction reco (`trajfit/nu_interaction.py`) now produces, per event,
one or more **nu-vertex candidates**, each a tree of attached **tracks** and
**showers** (primary at the vertex, secondary at track ends). The last step before
the output is analysis-ready is to assign a **4-momentum to every primary particle
at every vertex candidate** (including the secondary/displaced vertices).

This spec defines two energy estimators and how they combine per particle type:
- **(A) Range-based momentum** for stopping tracks (mu/pi/p) — port of LANTERN
  `ubdl/larflow/larflow/Reco/NuTrackKinematics.{h,cxx}`, with the ROOT range→KE
  splines converted to a **numpy `.npz`** (no ROOT at reco time).
- **(B) Calorimetric energy** from summed wire-plane pixel charge — the primary
  method for **showers** (e/γ), a cross-check for tracks, and the fallback for
  non-stopping (reinteracting) hadrons. Requires **de-double-counting** of pixel
  charge across spacepoints.

Both feed a single per-particle 4-momentum (§6). Direction is shared (§2).
Momenta are assigned per particle **regardless of which candidate is the true nu
vertex** — the analyzer selects the vertex; the kinematics are attached to
whatever tree each particle sits in.

---

## 1. Inputs
Per reconstructed particle we already have (from `nu_interaction.py`):
- **Tracks**: the dense centerline polyline (`poly`), total length, the two
  endpoints, the initial direction near each end, the predicted class (mu/pi/p),
  the point set, and which vertex it attaches to.
- **Showers**: the trunk `start`+`direction` (vertex-biased method), the point
  set, the predicted class (e/γ), and the attachment vertex.
- **Per-point wire-plane charge** from `merged_sp/triplet_data` (the slice points
  are an exact subset — KD-matched in `trajfit_io`): `pixval (N,3)` = ADC at the
  projected pixel in planes (U,V,Y); `tick` (drift row); `uwire,vwire,ywire` (wire
  number per plane). These are **reco** quantities (from the ADC images), so they
  exist on real data, not just MC.
- Truth for calibration/eval (MC only): `mc_particle_tree` (pid, energy_mev =
  kinetic energy, start_pos_sce, parent_trackid, daughters).

---

## 2. Direction (shared by both estimators)
The 4-momentum direction is the particle's direction **at the vertex it attaches
to** (its trunk pointing away from that vertex):
- **Track**: charge-weighted average of the centerline direction over points
  within `R_dir` (≈10 cm, matching LANTERN's `get_trackdir_radius`) of the
  attachment vertex, pointing away from it. Falls back to the near-end `u_in`
  already computed.
- **Shower**: the reconstructed trunk `direction` (already oriented away from the
  vertex).

Unit vector `d̂`. The 4-momentum is `(E, |p|·d̂)`.

---

## 3. (A) Range-based track momentum  (LANTERN NuTrackKinematics port)
### 3.1 Algorithm (per track)
1. **Track length** `L` = sum of centerline segment lengths (`poly`).
2. **KE from range**: `KE = range2KE[hyp](L)` for the mass hypothesis `hyp` given
   by the predicted class (mu→muon, p→proton, pi→pion). Also evaluate the other
   hypotheses for a PID cross-check (LANTERN keeps muon+proton for both).
3. `E = m_hyp + KE`, `|p| = sqrt(E² − m_hyp²)`.
4. 4-momentum `= (E, |p|·d̂)` with `d̂` from §2.
Masses (MeV): μ 105.66, π± 139.57, p 938.27.

### 3.2 The range→KE tables (ROOT → numpy)
LANTERN loads `TSpline3` `sMuonRange2T`, `sProtonRange2T` from
`ubdl/larflow/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root`
(range[cm] → KE[MeV], CSDA in liquid argon).
- **DECIDED**: compute all three tables (μ, π, p) from a small numpy
  **Bethe-Bloch dE/dx + CSDA integrator** in liquid argon (ROOT-free, and it
  yields the pion table the ROOT file lacks). `range(KE) = ∫₀^KE dE/(dE/dx(E))`,
  inverted to `KE(range)` on a dense grid. Save `range2ke_lar.npz` with
  `{muon,proton,pion}_{range,ke}`.
- **Validation** (one-time): read the ROOT `sMuonRange2T`/`sProtonRange2T`
  splines and confirm the Bethe-Bloch μ/p tables agree (target ≲ few %); this is
  the only step that touches ROOT and is not part of the reco.
- **Runtime**: `KE = np.interp(L, range, ke)` — no ROOT, no scipy needed.

### 3.3 Stopping vs non-stopping (containment)
Range→momentum is only valid for a **stopping** particle. **DECIDED**: tag
stopping primarily by **fiducial containment**, not a Bragg test (per-point dE/dx
along the centerline is expected to be too noisy — deferred, to be evaluated
empirically before adding):
- **Stopping** if both endpoints are inside the FV AND the far end is NOT a
  secondary vertex (no daughter tracks/interaction there). → range momentum
  trusted.
- Otherwise (exits the FV, or reinteracts): range is a **lower bound**; defer
  to §5.
Expose `stopping: bool`, `momentum_method ∈ {range, calo, range_lowerbound}`.

---

## 4. (B) Calorimetric energy from summed pixel charge
### 4.1 Why not sum per-spacepoint edep / pixval directly
Multiple 3D spacepoints project to the **same** wire-plane pixel (the 2D→3D
ambiguity), and each carries that pixel's full ADC in `pixval`. Summing `pixval`
over a particle's spacepoints therefore **double-counts** shared pixels (verified:
adjacent spacepoints sharing a Y pixel carry identical `pixval`).

### 4.2 De-double-counting (per plane p ∈ {U,V,Y})
The pixel a spacepoint projects to in plane p is identified by `(tick, wire_p)`
(`wire_p` = u/v/y wire). Over the set of spacepoints being summed (the nu-slice
deghosted points — NOT the raw ghost triplets):
1. group spacepoints by `(tick, wire_p)`; `count_p[i]` = # spacepoints in i's group.
2. per-spacepoint charge share `q_p[i] = pixval[i,p] / count_p[i]`.
3. a particle's plane-p charge `Q_p = Σ_{i∈particle} q_p[i]`.
(Equivalently via `triplet_imgpix_index` pixel indices; `(tick,wire)` is
self-contained in `triplet_data`.)

### 4.3 Per-plane and combined charge
Return `Q_U, Q_V, Q_Y` individually, and a **combined** `Q_comb`:
`Q_comb[i] = Q_Y[i] if pixval[i,Y] > 0 else mean(pixval[i,U], pixval[i,V])`
(applied per spacepoint before summing, with the same de-double-counting), so a
particle with dead/zero Y-wire regions still gets a charge from the induction
planes. Y is the collection plane (best calorimetry); U/V back it up.

### 4.4 Calibration: charge → energy (per particle type)
The conversion from summed charge to energy is empirical and per particle type:
1. On MC, for each reconstructed particle compute `Q_comb` (and per-plane) with
   §4.2–4.3.
2. Plot `Q_comb` vs **true KE** (`mc_particle_tree.energy_mev`, matched by
   `gt_trackid`) per type: e, γ, μ, π, p.
3. Fit the conversion (linear `KE ≈ a·Q_comb + b`, or a low-order/piecewise fit if
   non-linear at low E). Save `calo_calib.npz` with `{type: (a, b, ...)}` +
   residual spread (the calorimetric resolution).
4. Runtime energy `E_calo = calib[type](Q_comb)`.

**Empirical findings (calo.py study on the pi0 merged_sp):**
- The de-double-counting works; γ charge correlates with true KE at **corr 0.80**
  (~53% spread per photon) — the shower-calo approach is sound.
- **Calibrate at the PRIMARY-particle level**, not per Geant trackid: summing per
  trackid makes "e" thousands of sub-shower electron *fragments* (corr only 0.46,
  KE up to GeV). The fit must sum ALL of a shower's charge (primary + EM
  descendants) vs the *primary's* true KE — which is what a reco shower *instance*
  already is. On MC truth, aggregate charge by the primary ancestor (walk the
  parent chain) before fitting.
- **Muons don't calorimeter-calibrate**: in overlay they're through-going cosmics
  (corr 0.05, KE to 190 GeV) — charge ∝ length. Muons → range only (§3);
  calorimetry is not a muon KE estimator.

### 4.5 Use per particle type
- **Showers (e, γ)**: calorimetric is the **primary** energy. `E = E_calo`,
  `|p| = E` (massless-limit for γ; electron mass negligible). 4-mom `(E, E·d̂)`.
- **Tracks (μ, π, p)**: calorimetric is a **cross-check** of the range momentum
  (and the primary for non-stopping hadrons, §5). Report both.

---

## 5. Hadronic reinteraction (the hard case)
### 5.1 The problem
A **stopping** proton/π gets a clean range momentum (§3). A **reinteracting**
primary hadron (hard elastic/inelastic scatter before stopping) has a visible
track length that **under-estimates** its range, because part of its energy leaves
as secondaries (some charged & visible, some **invisible** — neutrons, nuclear
de-excitation, binding energy). So range → a lower bound only.

### 5.2 Established procedures (survey)
There is **no clean single-particle momentum** for a reinteracting hadron. Common
LArTPC practice:
- **Require stopping** (contained + Bragg) for range-based momentum; otherwise flag
  and either drop or report a lower bound. (MicroBooNE proton/π analyses.)
- **Calorimetric visible energy** of the hadron + its charged daughters, with an
  empirical sampling-fraction / missing-energy correction (analogous to EM calo,
  §4). Coarser; biased low by the invisible fraction, which grows with inelasticity
  and neutron multiplicity. (DUNE-style hadronic calorimetry.)
- **Depth / shower-profile** methods for high-energy hadronic showers — not
  reliable at the low energies typical here (the user's intuition is right).

### 5.3 The user's iterative-tree "corrected range" scheme
Walk the true (or reco) interaction tree from leaves to root: for each parent, sum
the KE of its daughter subtrees, convert that summed KE to an equivalent range,
add it to the parent's visible track length, convert the **total** length back to a
KE, assign the parent a 4-momentum; iterate to the root.
- **What it gets right**: it puts the charged daughters' energy back onto the
  parent, correcting the range under-estimate first-order.
- **What it violates** (why it's biased): (1) **invisible energy** — neutrons and
  nuclear breakup are not in any daughter KE, so the sum is incomplete; (2)
  **binding/Q-value** losses at each inelastic vertex; (3) the daughter-KE→range
  conversion assumes a single particle type; (4) reco tree incompleteness (missed
  neutral daughters). Bias is small for near-elastic / charged-daughter-dominated
  scatters, large for inelastic/neutron-rich ones.
- **Verdict**: worth prototyping as an **option** to quantify vs truth, but not the
  default. It is a lower-bound-plus-correction, not an unbiased estimator.

### 5.4 Recommendation (DECIDED)
1. **stopping hadron** → range momentum (§3), trusted.
2. **reinteracting hadron** → **calorimetric visible-energy** estimate is the
   **default** reported momentum, with an empirically-derived "total-ionization →
   initial primary KE" factor per type (derived like §4.4 but fit against true
   **initial** KE, which absorbs the average invisible fraction). Also report the
   range lower bound, flagged. The §5.3 **tree-corrected** estimate is
   **postponed** — implement only if the calorimetric estimate proves extremely
   poor against truth.
3. Always set `momentum_method` and `stopping` so downstream cuts can require
   high-confidence (stopping) momenta only.

Note the invisible-energy floor is irreducible per-particle; at analysis level the
**total** hadronic energy (sum over the hadronic subtree + a neutron correction)
is often better constrained than any single primary hadron's momentum.

---

## 6. Output schema
```python
@dataclass
class Particle4Mom:
    vertex_id: int            # which vertex candidate / tree node it attaches to
    kind: str                 # "track" | "shower"
    pred_class: str           # e|gamma|mu|pi|p
    direction: np.ndarray     # (3,) unit, from §2
    # energy estimators (MeV)
    ke_range: float           # range-based KE (tracks; NaN if n/a)
    ke_calo: float            # calorimetric KE
    energy: float             # chosen E (mass+KE for tracks, E_calo for showers)
    momentum: np.ndarray      # (3,) MeV, = |p|*direction
    fourvec: np.ndarray       # (4,) (E, px, py, pz)
    # calorimetry detail
    charge_U: float; charge_V: float; charge_Y: float; charge_comb: float
    # flags / provenance
    stopping: bool
    momentum_method: str      # "range" | "calo" | "range_lowerbound" | "tree_corrected"
    length_cm: float
    extra: dict               # alt-hypothesis KEs, calib residual, etc.
```
Attached to each `nu_interaction` result: a list of `Particle4Mom`, plus a
per-vertex-candidate **summed visible 4-momentum** (for a total-energy / invariant
handle).

---

## 7. Code & data layout
```
trajfit/
  calo.py             # de-double-counted per-plane charge sums (§4.2-4.3)
  range_momentum.py   # np.interp range->KE; 4-mom for tracks (§3)
  particle_momentum.py# assign_momenta(interactions, merged_sp) -> [Particle4Mom]
  data/
    range2ke_lar.npz  # {muon,proton,pion}_{range,ke}  (§3.2)
    calo_calib.npz    # {type: (a,b,...)} charge->KE   (§4.4)
tools/ (offline, one-time)
  make_range2ke_npz.py  # ROOT/Bethe-Bloch -> range2ke_lar.npz
  fit_calo_calib.py     # MC charge vs true KE -> calo_calib.npz
```
`calo.py` reuses the `trajfit_io` slice→triplet KD-match to get `pixval/tick/wire`
per slice point; grouping by `(tick, wire_p)` is a `np.unique(..., return_inverse)`
+ `np.bincount`.

## 8. Evaluation (vs truth, MC)

**Validated results (pi0 dev set)** — prototype built in `range_momentum.py`,
`calo.py`, `particle_momentum.py`; range tables in `make_range2ke_npz.py`:
- Range tables agree with the ROOT splines to **0.2% (μ), 0.9% (p)**.
- **Range momentum, proton: bias −9%, resolution 19%** (stopping hadrons — clean).
- **Calo, photon: bias −4%, resolution 43%** (primary shower method — works).
  e/γ calibration factors are ~equal (0.021 vs 0.019) → **use one calibration**.
- Range momentum, **π −31% / μ −36% biased low** — the non-stopping cases
  (μ = through-going cosmics, π reinteract); these must route to calo + the
  stopping tag, confirming §3.3/§5.

Metrics to keep reporting:
Per particle type, matched by `gt_trackid` to `mc_particle_tree`:
- **Range momentum** (stopping tracks): reco KE vs true KE — bias & resolution.
- **Calorimetric** (showers, and tracks): `E_calo` vs true KE after calibration —
  resolution, linearity, low-E turn-on.
- **Reinteracting hadrons**: range-lower-bound, calo, and tree-corrected each vs
  true **initial** KE — quantify the bias of each, bin by inelasticity / neutron
  fraction.
- Per-vertex summed visible energy vs true (a neutrino-energy proxy).

## 9. Resolved decisions (from review)
1. **Range tables**: compute μ/π/p CSDA range→KE tables from **Bethe-Bloch in
   numpy** (ROOT-free, gives the pion table) and **validate against the ROOT
   muon/proton splines**. Ship `range2ke_lar.npz`. (§3.2)
2. **Stopping tag**: rely primarily on **fiducial containment** (both endpoints
   inside the FV, and the far end not coincident with a secondary vertex). A
   Bragg / dE/dx-rise test along the centerline is expected to be **noisy** and is
   deferred — evaluate empirically before adding. (§3.3)
3. **Reinteracting hadrons → calorimetric by default** (most defensible; every
   method fits a fudge factor anyway). The §5.3 tree-based estimate is **postponed**
   unless the calorimetric estimate proves extremely poor. (§5.4)
4. **e vs γ calorimetry**: fit **separate** conversions; if the factors come out
   effectively equal, collapse to one. (§4.4)
5. **Recombination/lifetime**: no explicit correction for now — the per-type
   empirical fit absorbs it (a proper correction is substantial extra work). (§4.4)
