# MC–data domain-shift study plan (proposal preparation)

*Created 2026-07-27. Target: preliminary results + a defensible methods story
for the proposal due ~2026-08-27. GPU budget: Tufts H100/A100 nodes at low
priority — every study below is ordered so that frozen-feature (CPU/1-GPU)
work leads and pretraining resumes run in the background.*

Companion docs: `STATUS_AND_HANDOFF.md` (run states),
`microboone_sonata_experiment_plan.md` §P5B (mixture design + analysis
toolbox note), `probe_orchestration/ORCHESTRATION.md` (probe protocol).

---

## 1. Question and framing

**Proposal claim to support:** a foundation model pretrained jointly on
simulation (BNB nu + CORSIKA cosmics + ghosts) and real detector data
(EXTBNB cosmic-only triggers) yields a *shared embedding space* in which the
MC–data gap can be (a) measured with statistical rigor, (b) decomposed by
cause, and (c) propagated into a systematic uncertainty on downstream
selections (e.g. the existing 2-photon / pi0 selection).

Three distinct things get conflated under "domain shift" — the study must
keep them separate:

1. **Detector-response shift** — wire response, noise, recombination, space
   charge, diffusion mismodeling. This is the physics target: it is what a
   "domain-gap systematic" should quantify.
2. **Content/composition shift** — MC events contain a neutrino interaction,
   EXTBNB events do not; cosmic flux/multiplicity and readout-window details
   also differ. This is a *confound*: a domain classifier will happily fire
   on "is there a nu vertex" and report a huge gap that says nothing about
   mismodeling.
3. **Preprocessing shift** — ghost content and LArMatch behavior differ
   between domains (WP1 finding: MC files carry no `larmatch_score`, so the
   v8-era filter silently no-oped on MC). The P5B.1/.2/.3 recipe axis
   exists precisely to expose this.

The P5B runs were designed with this decomposition in mind (P5B.1 raw pairs
with P1A.2+P1A.3; P5B.2 deployment-prep pairs with P1A.1+P1A.4; P5B.3
symmetric-filter pairs with P1A.4+P1A.4b).

## 2. Assets in hand (verified 2026-07-27)

- **Checkpoints:** P5B.1/.2/.3 at ~45% budget, 10 snapshots each, with
  images-seen anchors matched to P1A.2 (MC+ghosts) and P1A.3 (EXTBNB-full);
  full P1A five-cell family at 40–52%; all resume states verified at Tufts.
- **Frozen diagnostic sets:** `h5list_v3_mc_diag1k_tufts.txt` and
  `h5list_v3_extbnb_diag1k_tufts.txt` (1,000 events each, held out of all
  training, hash-frozen). All mixture/EXTBNB train/val lists are now
  remapped to Tufts paths and verified (`filelist_stats_tufts.txt`).
- **Per-point truth on MC:** `triplet_data/origin` ∈ {0 = no-truth/ghost,
  1 = neutrino, 2 = cosmic} plus `pid`, `trackid`, `hasmatch`, and the full
  `mc_particle_tree`. `LArTPCDataset` already loads `origin` and carries it
  through transforms (`label_mode='origin'`, masking machinery at
  `lartpc.py:501`). **This means a cosmic-only *view* of MC events needs no
  new sample production** (§4).
- **Reco-based selections:** LArFormer-based interaction isolation (other
  repo); a working 2-photon selection already applied to both data and MC.
- **Tooling to adapt:** `tools/visualize_sonata_{umap,tsne,pca_rgb}.py`
  (feature extraction scaffolding), `PrototypeUsageLogger` (per-domain
  prototype occupancy), probe orchestration for linear probes.
- **Still owed:** the P0.2 bootstrap-CI utility (blocking for any number
  that enters the proposal).

## 3. Methods primer — what each technique measures and how

*(§7 gives the reading for each; this is the working summary.)*

### 3.1 Domain-classifier two-sample test (C2ST) and proxy A-distance
Train a classifier (logistic regression and a small kNN — deliberately weak
and strong) on frozen embeddings to distinguish MC vs data; evaluate AUC on
held-out events. AUC 0.5 = indistinguishable domains; the **proxy
A-distance** `PAD = 2(2·acc − 1)` connects the same number to the
Ben-David domain-adaptation bound: the classifier-measurable divergence
between domains bounds how much a model trained on one domain can degrade
on the other. This is the single most interpretable headline number, and
it is the same quantity the DANN literature minimizes — measuring it is
step one of the method we would later propose to control it.
*Pitfalls:* any nuisance difference (composition!) inflates it — hence the
tiered samples of §4; always report the same statistic on null splits
(MC-vs-MC, data-vs-data) to calibrate the "indistinguishable" baseline.

### 3.2 Maximum Mean Discrepancy (MMD)
Kernel two-sample statistic: the distance between the mean embeddings of
the two samples in an RKHS (multi-scale RBF kernels over the frozen
features). Nonparametric, no classifier training, exact permutation
p-values, and differentiable (later usable as an alignment loss — same
"measure now, control later" story as PAD). Report the unbiased MMD² with
permutation-null p-value and bootstrap CI. *Pitfalls:* kernel bandwidth
choice (use the median heuristic + a bandwidth sweep); power decays in very
high dimension — also run on a PCA-reduced space and report both.

### 3.3 Prototype-occupancy divergence (Sonata-specific, cheap, novel-ish)
The Sonata head assigns every point to one of 4096 prototypes. Histogram
prototype usage separately for MC and data (the `PrototypeUsageLogger`
already computes usage); compare with Jensen–Shannon divergence and report
the count of domain-exclusive prototypes. This is a physics-legible
discretization of the embedding space: "N% of the learned vocabulary is
used only by data" is a proposal-friendly sentence, and per-prototype
inspection (what do the data-exclusive prototypes look like?) directly
surfaces *what* is mismodeled (noise clusters? track ends? high-charge
blobs?).

### 3.4 Centered Kernel Alignment (CKA)
Compares *representations of models*, not samples: e.g. does P1A.2 (MC-only
model) represent a common evaluation set the same way P1A.3 (data-only
model) does, and does the P5B mixture model sit between them? Layer-wise
CKA localizes where in the encoder the domains diverge. *Pitfalls:* CKA is
sensitive to dominant principal components and can mislead (Davari et al.);
use it as a comparative diagnostic across matched snapshots, never as a
standalone headline number.

### 3.5 Classifier-based density-ratio reweighting (DCTR-style) → systematics
The bridge from "gap number" to "systematic uncertainty," and the piece
that makes the proposal story concrete. The domain classifier of §3.1
yields per-event weights `w(x) = p(x)/(1−p(x))` that reweight MC to match
data *in the embedding space*. Procedure: (1) train the classifier on the
Tier-1 cosmic-controlled sample (§4); (2) reweight the MC; (3) re-derive a
downstream quantity (2-photon selection efficiency, probe-class IoU, any
analysis-level distribution) with and without weights; (4) the shift is a
data-driven estimate of the domain-shift systematic on that quantity, with
bootstrap CIs. This is the standard HEP reweighting idea (GBReweighter /
DCTR) applied to foundation-model embeddings instead of hand-picked
variables — which is exactly the pitch: the embedding automates the choice
of "which variables to compare data/MC in."
*Pitfalls:* weights are only valid where MC has support (clip and report
weight distributions); the classifier must not see composition confounds.

### 3.6 Alignment methods (proposed work, not needed for the proposal)
DANN (gradient-reversal domain classifier), Deep CORAL (covariance
matching), MMD-as-loss, and pivot-adversarial training are the *control*
counterparts of the *measurements* above: each minimizes one of the §3.1–3.2
statistics during training. The proposal narrative: Phase 1 measures the
gap (this plan); Phase 2 adds alignment terms to the joint pretraining and
verifies the gap and the derived systematics shrink without hurting probe
performance. No implementation needed now beyond citing the design.

### 3.7 Statistical rigor (applies to all of the above)
Every reported number carries: a permutation-test p-value where a null is
computable, a bootstrap CI over diagnostic events (the pending P0.2
utility), and a same-domain null (MC-vs-MC and data-vs-data splits of the
diag sets) plotted alongside. Given diag1k sizes (1k + 1k events,
~10⁸ points), event-level bootstrap is the correct resampling unit
(points within an event are strongly correlated).

## 4. Isolating the detector-domain shift: tiered samples

This answers the confounding question directly. **No new sample production
is required for the proposal-grade result (Tiers 0–2); a dedicated
cosmic-only MC production (Tier 3) is the right *proposed* work and can be
prepared in parallel if time allows.**

- **Tier 0 — raw event-level (confounded, still reported).** MC diag1k vs
  EXTBNB diag1k as-is. This measures the *total* shift the P5B model
  actually experienced during training — the honest upper bound. Reported
  as the anchor for the decomposition, never as "the" domain gap.

- **Tier 1 — cosmic-only MC view via truth masking (the workhorse).** Drop
  `origin == 1` (nu-matched) points from MC events before feature
  extraction, leaving simulated cosmics + ghosts vs real cosmics + ghosts.
  Two matched-preprocessing variants mirroring the P5B recipe axis:
  - *raw*: keep ghosts on both sides (pairs with P5B.1's training view);
  - *cleaned*: MC `hasmatch==1` points only vs EXTBNB LArMatch-filtered
    (pairs with P5B.2).
  The Tier-0 minus Tier-1 difference cleanly quantifies how much of the
  raw gap was just composition — itself a proposal figure. *Residual
  caveats to state honestly:* CORSIKA flux/multiplicity vs reality is a
  physics-model shift that remains (that is fine — it is part of what a
  simulation systematic should cover); nu-induced ghosts in MC are removed
  only insofar as they fail `hasmatch`, and LArMatch response itself
  differs between domains (which is why the preprocessing axis is kept
  explicit).

- **Tier 2 — object-conditioned comparison (composition-free by
  construction).** Select matched physics objects with the *same* procedure
  in both domains and compare embeddings conditioned on object type. The
  cleanest first object: long straight cosmic-muon tracks (through-going /
  stopping), selectable geometrically (PCA linearity + length + boundary
  conditions on the point cloud) without any learned model — avoiding
  circularity. Optional richer objects via the LArFormer reco (Michel
  candidates; stopping-muon Bragg ends). Per-object-class PAD/MMD then
  probes response mismodeling on a fixed physics population — the closest
  analogue to classic dE/dx calibration-sample comparisons, but in
  representation space.

- **Tier 3 — new samples (proposed work; start only if ahead of
  schedule).**
  1. *CORSIKA-only MC* (no nu overlay) through the same
     SimChTripletLabelMaker chain — removes every nu-induced correlation
     (including trigger/readout differences from the BNB window) rather
     than masking it; ~10–25k events is plenty for diag-scale two-sample
     work.
  2. *Beam-on BNB data* processed to h5 — enables the nu-side comparison
     using the existing 2-photon/pi0 selection: pi0 candidates in data vs
     MC through the frozen encoder is the flagship *physics* version of
     Tier 2 and connects directly to the proposal's 2-photon result.
  3. *(MicroBooNE-style hybrid)* overlay of data cosmics + MC nu, the
     collaboration-standard construction — worth citing as the natural
     factorization sample for the full program.

**Direct answer to "do I need a cosmic-muon / MC cosmic-muon sample?"**
Not for the first pass — Tier 1 (origin-masked MC) and Tier 2 (geometric
muon selection on both domains) get you a defensible decomposition from
data already on disk. A dedicated CORSIKA-only production (Tier 3.1) is the
cleanest version and worth queuing as CPU-side preparation, but should not
gate the proposal figures.

## 5. Measurements and proposal figures (priority order)

Each row is runnable on frozen snapshots (no pretraining resume required);
resumed checkpoints simply extend the x-axes later.

| # | figure | inputs | method |
|---|---|---|---|
| F1 | UMAP of the P5B.1 embedding, colored by domain; side panels colored by origin (MC) and lm_score (data) | latest matched snapshots, diag sets | qualitative anchor (adapt `visualize_sonata_umap.py`) |
| F2 | Domain-gap bar chart: PAD and MMD for Tier 0 vs Tier 1(raw/clean) vs Tier 2, with same-domain nulls and bootstrap CIs | P5B.1 (+P5B.2) latest snapshot | §3.1, §3.2, §3.7 — *the decomposition figure* |
| F3 | Gap vs images-seen: PAD/MMD per snapshot for P5B.1/.2/.3; overlay P1A.2-vs-P1A.3 cross-model CKA at matched anchors | full snapshot ladders | does joint pretraining align the domains over training, and does the recipe (raw/clean/symmetric) matter? |
| F4 | Prototype occupancy: JSD + domain-exclusive prototype count vs snapshot; gallery of top data-exclusive prototypes | P5B.1 snapshots | §3.3 — the physics-legible view |
| F5 | Systematics demo: 2-photon selection efficiency (or probe pion/proton IoU) before/after embedding-space reweighting, with CI | Tier-1 classifier + existing selection | §3.5 — *the "this becomes a systematic" figure* |
| F6 | Layer-wise CKA (P1A.2 vs P1A.3 vs P5B.1 on a common eval set) | matched ~45% snapshots | localizes where domains diverge in the encoder |

Minimum viable proposal set: **F1 + F2 + F5** (one qualitative, one
quantitative-decomposed, one systematics-connected). F3 is the strongest
"why joint pretraining" argument if time allows.

## 6. Work plan (≈4 weeks) and compute

**Week 1 — infrastructure (CPU/1 GPU).**
- Feature-extraction script: frozen snapshot → per-event pooled embedding
  (+ optional per-point features and prototype assignments) → .npz, for a
  filelist + tier-mask spec. Base on `visualize_sonata_umap.py`; run per
  diag set (1k events → minutes/snapshot on one GPU).
- Tier-1 masking flag in the extraction path (`origin != 1` drop; reuse
  `lartpc.py` mask machinery — check whether the inverse of `drop_cosmics`
  needs a ~10-line addition).
- P0.2 bootstrap-CI utility (event-level resampling; shared by all
  metrics). Null-calibration harness (MC-vs-MC, data-vs-data splits).
- Metrics module: PAD (logreg + kNN), multi-scale MMD + permutation test,
  prototype JSD, CKA. All plain numpy/sklearn — no new dependencies.

**Week 2 — Tier 0/1 measurements → F1, F2, F4 drafts.** Extract features
for P5B.1/.2/.3 and P1A.2/.3 latest matched snapshots; run the metric
battery; iterate on the decomposition figure. In parallel (background GPU):
bring up the Tufts pretraining sbatch (handoff P3) and resume **P5B.1
first**, then P5B.2/.3, then P1A.2/P1A.3.

**Week 3 — Tier 2 + snapshot ladders → F3, F6.** Geometric muon selection
on both diag sets; conditioned metrics. Ladder extraction over all
snapshots (still cheap). Start F5: train the Tier-1 domain classifier,
produce weights, wire into the 2-photon selection pipeline.

**Week 4 — F5 + write-up.** Reweighting propagation with CIs; assemble a
`RESULTS_*`-style memo mirroring `RESULTS_WAVE_A_DECISION.md`; freeze the
figures for the proposal. Buffer for the known unknowns (weight clipping,
UMAP hyperparams, null widths).

**Resume config note:** the archived configs reference Isambard filelist
paths (`/projects/u6jo/...`). Resume at Tufts with overrides, e.g.
`--options resume=True weight=<run>/model/model_last.pth
data.train.data_list_file=<repo>/lartpc/filelists/h5list_v3_mix1to1_train_tufts.txt
data.val.data_list_file=<repo>/lartpc/filelists/h5list_v3_combined_val_tufts.txt`
(gotcha 3 applies only if changing LR; we are not). Batch 48 total was
calibrated for 96 GB GH200 — calibrate once on the H100-80 before
committing (grad accumulation or 8-GPU split if needed).

## 7. Annotated reading list

**Core methods (read first, in this order):**
1. Ben-David et al., *A theory of learning from different domains*, Machine
   Learning 79 (2010) — the H-divergence/proxy-A-distance framework; why a
   domain classifier's accuracy bounds cross-domain degradation. The
   theoretical spine of the whole study.
2. Gretton et al., *A Kernel Two-Sample Test*, JMLR 13 (2012) — MMD:
   statistic, unbiased estimator, permutation testing.
3. Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests*, ICLR 2017
   (arXiv:1610.06545) — C2ST methodology and calibration; the modern
   justification for §3.1.
4. Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 17
   (2016, arXiv:1505.07818) — DANN; measurement→control counterpart.
5. Kornblith et al., *Similarity of Neural Network Representations
   Revisited*, ICML 2019 (arXiv:1905.00414) — CKA; plus Davari et al.,
   *Reliability of CKA as a Similarity Measure*, arXiv:2210.16156 for the
   failure modes.
6. Andreassen & Nachman, *Neural networks for full phase-space reweighting
   and parameter tuning* (DCTR), PRD 101, 091901 (2020, arXiv:1907.08209) —
   classifier-based density-ratio reweighting; §3.5's method. Companion:
   Rogozhnikov, *Reweighting with BDT* (arXiv:1608.05806) — the hep_ml
   GBReweighter standard.

**Particle-physics precedent (yes, there is precedent):**
7. Perdue et al. (MINERvA), *Reducing model bias in a deep learning
   classifier using domain adversarial neural networks in the MINERvA
   experiment*, JINST 13 P11020 (2018, arXiv:1808.08332) — **the direct
   neutrino-experiment precedent**: DANN with unlabeled data to control
   data/MC bias in vertex finding.
8. Louppe, Kagan & Cranmer, *Learning to Pivot with Adversarial Networks*,
   NeurIPS 2017 (arXiv:1611.01046) — training classifiers decorrelated
   from nuisance parameters; the systematics-aware-training frame.
9. D'Agnolo & Wulzer, *Learning New Physics from a Machine*, PRD 99, 015014
   (2019, arXiv:1806.02350) — ML two-sample testing with calibrated
   significance (NPLM); the statistically rigorous end of §3.1/3.2.
10. Nachman, *A guide for deploying deep learning in LHC searches*, SciPost
    Phys. 8, 090 (2020, arXiv:1909.03081) — how ML uncertainties should be
    framed for an analysis audience.
11. MicroBooNE, *Semantic segmentation with a sparse convolutional network
    for event reconstruction in MicroBooNE*, PRD 103, 052012 (2021,
    arXiv:2012.08513) — in-collaboration precedent for validating an
    MC-trained network on data with dedicated cosmic/Michel/pi0 samples;
    also the EM particle-ID CNN paper (JINST 14 P04012, arXiv:1808.07269).
    NOvA's muon-removed hybrid samples (see the CVN line of papers,
    arXiv:1604.01444) are the precedent for constructed data/MC hybrids.

**Foundation-model era (positions the proposal):**
12. Harris et al., *Re-simulation-based self-supervised learning for
    pretraining physics foundation models* (RS3L), PRD 111, 032010
    (arXiv:2403.07066) — SSL augmentations built from simulator variations
    to gain robustness to systematics; the closest philosophical neighbor
    to "pretrain across the sim/data boundary."
13. Young, Jwa & Terao, *Particle Trajectory Representation Learning with
    Masked Point Modeling* (PoLAr-MAE), arXiv:2502.02558 — SSL on LArTPC
    point clouds (PILArNet-M); the field context for our Sonata program.
14. Golling et al., *Masked particle modeling on sets*, arXiv:2401.13537;
    Mikuni & Nachman, *OmniLearn*, arXiv:2404.16091; and the recent
    sensor-level self-distillation line (e.g. *Panda*, arXiv:2512.01324) —
    the broader HEP foundation-model landscape for the related-work
    section.
15. DoReMi arXiv:2305.10429, "data-mixing-laws work"

**Where the proposal is novel:** the precedents either *control* the gap
during supervised training (7, 8), *test* distributions on hand-chosen
variables (9), or *build robustness* via simulator augmentations (12). The
piece with little direct precedent — and therefore the proposal's claim —
is using a *jointly pretrained* foundation model's embedding as the
measurement space in which the data/MC gap is decomposed (§4) and converted
into an analysis systematic via density-ratio reweighting (§3.5), with the
LArTPC point-cloud setting entirely unexplored for this.

## 8. Risks / honest caveats to carry into the proposal text

- Tier-1 masking removes nu *truth-matched* points, not nu-induced
  detector effects entangled with cosmics in the same event (small, state
  it).
- EXTBNB vs BNB-window trigger and readout differences remain in all tiers
  except 3.1; keep the claim scoped to "cosmic-region response."
- The mixture model has *seen* both domains — alignment in its embedding is
  partly trained-in. That is the point (one shared space), but always
  report the P1A single-domain models on the same tiers as the
  uncontaminated reference.
- diag1k × 2 is ~1k events/domain: fine for PAD/MMD power at the observed
  effect sizes, but quote CIs everywhere; expanding diagnostics to 5k
  (val lists) is cheap if power is marginal.
