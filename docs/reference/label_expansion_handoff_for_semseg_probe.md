# Label expansion ("A2 label completion"): handoff for the semseg feature-quality probe

Written 2026-10-03 from the `kpv2_pointcept` repository
(`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept`, branch
`nutufts_lartpc_keypointdev_v2`). All paths below are absolute so a session in
another pointcept checkout can read them directly. Everything described here is
**read-only reference** for that session. Do not modify files in this repository
or the shared data trees (see the gotchas in section 7).

---

## 1. Summary

- **What it is.** A post-processing pass over the LArTPC triplet HDF5 files.
  Unlabeled spacepoints (`trackid <= 0`) that sit in a thin shell around a
  particle's labeled spacepoints are *adopted* by the nearest labeled point (the
  donor). They copy its `trackid`, `pid`, and `origin`, and get `hasmatch = 1`.
  Original labels are kept in the same file as `<name>_precomplete`. A new
  per-point dataset, `label_completed`, records provenance.
- **Why.** In the all-simulation (LANTERN / corsika) files, simch-derived labels
  are thin. They cover about 55% of the local charge cloud, versus about 70% for
  overlay 2D-image labels. The missed points are the charge periphery: 95% lie
  within Δrow ≤ 2 and Δwire ≤ 2 of a labeled point. So the network was asked to
  call real charge periphery "ghost". This label mismatch is the overfitting
  and domain-transfer symptom that expansion fixes.
- **Operating point (locked 2026-08-18):** radius 0.5 cm in 3D, plus an
  image-shell guard of |Δrow| ≤ 2 (row = tick // 6) and |Δwire| ≤ 2 on **all
  three** planes.
- **Ready-made expanded all-sim data:**
  - **Train:** 219,791 single-event h5 files, expanded **in place** in the
    shared `v3_larmatch` tree. These are the nue, pi0filter, and chargedpiplus
    samples. The generic bnb_nu samples were **not** expanded.
  - **Val:** 1,500 expanded **copies** in a separate directory. The canonical
    10k val files themselves are untouched.
- **Semseg-probe impact.** With `label_mode="pid"` and a ghost class derived
  from `pid == 0` or `hasmatch == 0`, the expanded files work with **no code
  change**, because `pid` and `hasmatch` are rewritten. `ssnet_label` is
  **not** rewritten, so an ssnet-label probe gets no benefit (section 6).

---

## 2. The algorithm

Source: `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/complete_labels.py`
(127 lines, self-contained: numpy, h5py, scipy.cKDTree). Read the whole file;
it is short. Per file (one event, group `entry_0/triplet_data`):

```
donors     = points with trackid > 0          (simch-labeled, any particle, nu or cosmic)
candidates = points with trackid <= 0         (ghosts + unlabeled real periphery)
KD-tree on donor positions (cm); for each candidate, find the nearest donor within --radius
keep the pair iff, versus that nearest donor:
    |row_c - row_d| <= drow   (row = tick // 6)
    |uwire_c - uwire_d| <= dwire  AND  |vwire...| <= dwire  AND  |ywire...| <= dwire
adopted: trackid, pid, origin <- donor's; hasmatch <- 1
```

- Only one donor is tested per candidate: the nearest. If the nearest donor
  fails the image guard, the candidate stays unlabeled even when a farther
  donor would pass. Ambiguity is resolved by nearest donor.
- Adopted points never act as donors. The pass is single-shot, with an
  idempotency guard: a file that already has `label_completed` is skipped. A
  second pass would grow labels outward through adopted points.
- Files with zero donors still get `label_completed`, with all values 255, and
  the `label_completion` attribute shows `donors=0`.
- CLI defaults: `--radius 0.5 --drow 2 --dwire 2`. `--out-dir D` writes
  expanded copies to D. **Omitting `--out-dir` modifies the input in place.**
  The module docstring still says the default radius is 0.75; the argparse
  default of 0.5 is the one that counts.

### Validated performance (A2 QA, jobs 2513926/2514043)

| metric | before | after |
|---|---|---|
| val-sim per-particle completeness (labeled / local cloud) | 0.647 | 0.895 |
| overlay null test (2D labels already inclusive) | 0.759 | 0.968 |
| ghost fraction among adopted points (real-larmatch join) | – | 5.1% |
| net label noise | – | ~1.3% |

Typical adoption is +0.44 to +0.57 relative to the donor count. In the files
inspected for this note, about 50k points were adopted per event on top of
70–100k donors.

QA and display tool:
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/domain_shift/label_expansion_qa.py`

Design and decision log (search for "A2"):
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/pilot_matrix/SLICER_RETRAIN_PLAN.md`
Relevant sections:
- "What the measurements pin down" (row on thin simch labels)
- "A2 — Label completion" and "A2-integration"
- "A2 validation suite"
- the 2026-08-18 "A2 operating point LOCKED" log entry

---

## 3. Code to look at (absolute paths)

| file | role |
|---|---|
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/complete_labels.py` | **The expansion tool.** Copy it verbatim if you need to expand new files. |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/submit_complete_labels_array.sh` | SLURM array driver for **in-place** expansion over a file list (`FILELIST`, `NCHUNK`). Chunk lists are kept on shared storage. |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/make_completed_val.sh` | SLURM array that made the 1,500 expanded val **copies** (`--out-dir`). Use this as the template for out-of-place expansion. |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/training_data_ledger/LEDGER.md` | Provenance: LANTERN subsample rule, quota incident, repack, restore of 5 damaged files. |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/pointcept/datasets/lartpc.py` | `LArTPCDataset`, the loader the existing semseg probes use. It reads `pid`, `trackid`, `hasmatch`, and `origin` by name, so expanded labels are picked up automatically. It has **no** option to read `*_precomplete` or `label_completed`; that would be a new loader option (section 6). |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/pointcept/datasets/transform.py` (classes `HasmatchAsGhost`, `RemapGhostLabel`, around line 2035) | Ghost-target transforms. `HasmatchAsGhost` is the robust way to define ghost from `hasmatch`. |
| `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/configs/lartpc/sonata_pretrain/probes/linearprobe-sonata-v1m1-lartpc.py` | Existing linear-probe config in this repo (`label_mode="pid"`, `include_ghosts=True`, `exclude_other=True`, 6 classes). |

Container used for all of the above:
`/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif`
(`apptainer exec --bind /cluster:/cluster ...`)

---

## 4. On-disk schema of an expanded file

Group `entry_0/triplet_data`. The datasets added or changed by expansion are
marked †.

| dataset | dtype / meaning |
|---|---|
| `pos`, `tick`, `uwire`, `vwire`, `ywire`, `pixval`, `edep`, `lm_score`, `larmatch_feats`, `shower_score`, `truth_match`, `ssnet_label` | unchanged |
| `trackid` † | expanded (adopted points have the donor's track id) |
| `pid` † | expanded. PDG code; `0` = unlabeled/ghost |
| `origin` † | expanded (0 = none, 1 = nu, 2 = cosmic) |
| `hasmatch` † | expanded (adopted → 1); stays binary |
| `trackid_precomplete`, `pid_precomplete`, `origin_precomplete`, `hasmatch_precomplete` † | original labels |
| `label_completed` † (uint8) | 0 = originally labeled, 1 = adopted, 255 = still unlabeled |

Attribute `entry_0.attrs["label_completion"]`, for example
`"r=0.5cm drow=2 dwire=2 adopted=39293 donors=68257"`.

Invariants verified on sample files:
- Before expansion, `hasmatch == 1` is equivalent to `trackid > 0`.
- After expansion, every point with `label_completed == 255` has `pid == 0` and
  `hasmatch == 0`.
- Every adopted point had `hasmatch_precomplete == 0`.

**`ssnet_label` is not rewritten.** Adopted points keep their original
`ssnet_label`, which is 0 (bg/ghost) for these points.

To detect expansion: `"label_completed" in f["entry_0/triplet_data"]`.

---

## 5. Existing all-simulation data with expansion

All files are single-event h5 files from the LANTERN v3_larmatch production:
- corsika cosmics + BNB ν, MicroBooNE sim
- larmatch cut lm ≥ 0.15 applied at production
- root: `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_larmatch/`

### 5a. Train: expanded in place

**File list (219,791 files):**
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/training_data_ledger/lantern_train_subsample_enriched.txt`

| sample | files |
|---|---|
| bnb_nue_corsika | 96,833 |
| bnb_nu_chargedpiplus_corsika | 68,150 |
| bnb_nu_pi0filter_corsika | 54,808 |

- This list is a strict subset of the canonical LANTERN train list,
  `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_train.txt`
  (410,000 files). It shares no files with the val list.
- Random spot-check of 400 files: 400/400 carry `label_completed`.
- Five files are deliberately left out of this list:
  - list: `.../training_data_ledger/lantern_quota_damaged.txt`
  - They were damaged by a disk-quota incident, then restored and expanded on
    2026-08-20, and are now good. They can be re-added.

**Not expanded:** the generic BNB-ν samples, `bnb_nu_corsika_prod2` (94,409)
and `bnb_nu_corsika_set2_prod2` (95,795). List:
`.../training_data_ledger/lantern_train_removed_generic.txt` (190,204 files).
Spot-check: 0/100 carry `label_completed`.

> **Gotcha:** because expansion was in place, the canonical
> `h5list_mcall_lantern_train.txt` now resolves to a **mixture**. About 54% of
> its files (the enriched samples) have expanded labels and about 46% (generic)
> have original labels. Using that list unchanged gives mixed-definition labels.
> Use `lantern_train_subsample_enriched.txt`, or expand the generic files
> yourself (section 6c), or select per file on `label_completed`.

Composition caveat: the expanded training set has **no generic BNB-ν
events**, so it is enriched in νe, π0, and π± topologies. This was a
deliberate, storage-limited choice for the LArFormer mixture. For a semseg
probe it shifts the class balance (more electrons, gammas, and pions relative
to generic ν_μ CC).

### 5b. Val: expanded copies

**File list (1,500 files):**
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/data_prep/uboone_official/training_data_ledger/h5list_lantern_val_completed_1500.txt`

Files are in
`/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/_lantern_val_completed/`.
The directory name contains "overlay_train", but these are all-sim LANTERN files.

| sample | files |
|---|---|
| bnb_nu_corsika_set2_prod2 | 361 |
| bnb_nu_corsika_prod2 | 359 |
| bnb_nue_corsika | 349 |
| bnb_nu_chargedpiplus_corsika | 224 |
| bnb_nu_pi0filter_corsika | 207 |

- These are expanded copies of
  `.../training_data_ledger/lantern_val_subset_1500.txt`.
- That list is a subset of the canonical 10k val list,
  `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_val.txt`.
- The val subset **includes** generic BNB-ν, unlike the train set.
- The original val files (all 10k) were intentionally **not** modified, to keep
  earlier metric baselines comparable. Spot-check: 0/100 of the 10k carry
  `label_completed`. The `*_precomplete` arrays in the 1,500 copies give the
  original labels for exactly the same events, so a before/after comparison on
  one file set is possible.

### 5c. Overlay (for reference only; not requested)

The MC-overlay training files under
`/cluster/tufts/wongjiradlab/larbys/data/larformer/overlay_train/` were also
expanded at conversion time. They are not all-sim: the cosmics are real data and
carry no truth. Do not mix them into an all-sim probe without handling partial
truth.

---

## 6. How to use this in the semseg probe

### 6a. Using the existing expanded files (no new data processing)

- Train list: `lantern_train_subsample_enriched.txt`. Val list:
  `h5list_lantern_val_completed_1500.txt`.
- **`label_mode="pid"` works as is.** Adopted points change from pid 0 to the
  donor's PDG code.
  - **Ghost class from pid:** in this repo's `LArTPCDataset.PID_TO_CLASS`,
    `pid == 0` maps to index 8, not the ghost index 5 that `include_ghosts`
    and the 6-class probe config expect. Check how your repo maps pid 0 to the
    ghost class.
  - **Ghost class from hasmatch** (for example with `HasmatchAsGhost`, or
    `segment[hasmatch == 0] = ghost`): this is also consistent after
    expansion, because adopted points get `hasmatch = 1`.
- **`label_mode="ssnet"` gets no expansion**, because `ssnet_label` is
  untouched. Options:
  - (i) Derive the class from `pid` for points with `label_completed == 1`.
  - (ii) Extend `complete_labels.py` to also copy `ssnet_label` from the donor.
    This requires re-running (6c) on copies.
- **Per-point provenance** is available for:
  - loss down-weighting of adopted points (weight on `label_completed == 1`)
  - reporting metrics separately on original and adopted points
  - an ablation where the loader reads `*_precomplete` instead (labels before
    vs after, on identical files)

  None of these exist in `LArTPCDataset` in this repo. Each would be a small
  loader option: read the extra dataset, add it to `index_valid_keys`, and
  `Collect` it.
- **Baseline comparability:** probe metrics on expanded val are not directly
  comparable to earlier numbers on the unexpanded 10k val. The ghost/real
  boundary moves. In the four files sampled, 25–48% of former `pid == 0`
  points became particle points. Report both, or report on the 1,500-file set with both label versions.

### 6b. Applying expansion on the fly instead (alternative)

The algorithm needs only `pos`, `tick`, the three wire arrays, and the label
arrays, and it costs one KD-tree query per event. It can therefore run in the
dataset's `get_data()` on unexpanded files. Port `complete_file()` logic into a
transform or loader option, run before any GridSample or crop, on the full
event. This removes any dependence on the shared mutated files. It lets the
probe use all 410k train and 10k val files, including generic BNB-ν, with
consistent labels, and the parameters become config knobs. The cost is some
CPU per sample in the dataloader. Use the same parameters
(`radius=0.5, drow=2, dwire=2`) for parity with LArFormer training.

### 6c. Expanding more files offline

Use `complete_labels.py --out-dir <your dir> --h5 f1.h5 f2.h5 ...`, modeled on
`make_completed_val.sh` (SLURM array, `batch,preempt`, 2 CPU, 12 GB). See
section 7 before running anything.

---

## 7. Gotchas from the production campaign

1. **Never run in place on the shared `ub_on_tufts/hdf5/v3_larmatch` files.**
   Other projects read them. Always pass `--out-dir`. Note that the enriched
   train files are **already** expanded, and the guard skips them.
2. **Disk quota.** In-place h5 delete-and-recreate leaves dead extents (about 3x
   label-storage growth per file). In the 2026-08-19 campaign this filled the
   wongjiradlab volume and truncated 5 files. The current tool writes
   gzip-compressed datasets. Copies with `--out-dir` cost a full file each.
   Check free space before a large campaign, and `h5repack` afterwards if
   working in place.
3. **Node-local `/tmp` exhaustion.**
   - `make_completed_val.sh` stages its chunk list in `/tmp`.
   - In one campaign, 39/100 array chunks silently did nothing when `/tmp`
     filled, because `xargs` got empty input.
   - `submit_complete_labels_array.sh` was fixed to keep chunk lists on shared
     storage and to check the chunk length. Do the same.
4. **Idempotency.** Never expand a file twice. Check for `label_completed`
   first, as the tool does.
5. **Resolution-dependent parameters.** `radius` is in cm (raw `pos` units). The
   image guard uses raw `tick // 6` and integer wire numbers. Apply expansion
   **before** any coordinate scaling, voxelization, or point-count cap. If you
   port it into a loader, expand on the full event before `GridSample` or
   `SphereCrop`.
6. **Dirt / zero-label events** have no donors, so nothing is adopted. These
   events contribute only ghost labels.
