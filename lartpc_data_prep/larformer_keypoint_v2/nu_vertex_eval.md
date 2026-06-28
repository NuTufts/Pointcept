# Nu-Vertex Reconstruction Evaluation

Documentation for [`reco/eval_nu_vertex_reco.py`](reco/eval_nu_vertex_reco.py) —
an evaluation harness that measures how well the score-field keypoint
reconstruction (`reco/`, see [`keypoint_reco_spec.md`](keypoint_reco_spec.md))
recovers the **neutrino interaction vertex** from real LArFormer cascade
inference output.

It answers one practical question: **what nu-vertex score threshold should we
trust to select a neutrino vertex in an event slice?** — and quantifies the
resulting position resolution and vertex multiplicity.

---

## 1. What it consumes

The cascade inference tool
(`tools/run_larformer_keypoint2_cascade_inference.py --save-score-maps`) writes
one H5 per event slice. Example data lives in
[`reco_dev_data/keypoint2_out/`](reco_dev_data/keypoint2_out). These files are
**self-contained** — they carry both the network output and the truth, so the
evaluation needs nothing else (the `merged_sp/` slicer files are *not* required;
the inference output already carries the truth the network was trained against):

```
score_maps/nu_vertex/coords_cm   (N,3)   nu-slice spacepoints, detector cm
score_maps/nu_vertex/score       (N,)    per-spacepoint sigmoid nu-vertex score
score_maps/nu_vertex.attrs       level="spacepoint", kp_types=[0]
gt_keypoints/pos_cm,type         truth keypoints; the nu vertex is type == 0
gt_nu_vertex_cm                  (3,)    truth nu vertex (fallback if no type-0)
nu_vertex_cm                     (3,)    existing single-centroid decode (carried, unused)
```

The true nu vertex is taken from `gt_keypoints` where `type == 0`, falling back
to `gt_nu_vertex_cm`. A slice with **neither** (no finite truth) is treated as a
**background slice** — see §4.

## 2. What it does

For each slice it runs the greedy peel-and-fit reco
(`KeypointRecoTorch.reconstruct`) on the `nu_vertex` score field down to a low
score floor (`--min-score`, default `0.05`). The reco's Gaussian subtraction
provides non-maximum suppression, so a single pass yields a list of distinct
nu-vertex **candidates**, each with a position (cm) and a peak score, sorted
high → low. Everything downstream is derived from that per-slice candidate list:

- **max nu-vertex score** of a slice = the score of its top candidate.
- **best vertex** of a slice = the position of that top candidate.

## 3. What it measures (outputs)

All outputs go to `--output-dir` (default `nu_vertex_eval_out/`).

### 3.1 ROC: efficiency & purity vs max nu-vertex score

The core deliverable. Sweeping a score threshold `t` over `[min-score, 0.99]`
(`--n-roc-points` samples), each slice is classified by its **max** nu-vertex
score:

```
selected(t)   = slices whose max candidate score >= t            (signal OR bkg)
correct(t)    = SIGNAL slices that are selected AND whose best vertex
                is within --match-dist cm of the true nu vertex
efficiency(t) = correct(t) / (# signal slices)        # recall of the true vertex
purity(t)     = correct(t) / selected(t)              # 1 - false-selection rate
```

- **`nu_vertex_eff_purity_vs_threshold.png`** — efficiency and purity as
  functions of the threshold (the {0.1, 0.2, 0.5, 0.7, 0.9} marks are drawn as
  guide lines).
- **`nu_vertex_roc_purity_vs_efficiency.png`** — the parametric ROC, purity vs
  efficiency, coloured by threshold.
- **`nu_vertex_roc_table.csv`** — `threshold, n_selected, n_correct, n_truth,
  efficiency, purity` at every swept point.

As the threshold rises, efficiency falls (fewer slices clear the bar) and purity
generally rises (surviving high-score peaks are more often the real vertex) —
the trade-off the threshold choice has to balance.

### 3.2 True → reco vertex distance distribution

- **`nu_vertex_distance_distribution.png`** — histogram of the distance between
  the true nu vertex and the reco **max-score** vertex, **one entry per signal
  slice** (background slices have no truth and are excluded here). Median and
  mean are annotated.

The text summary additionally reports `p90` and the fraction of slices resolved
within {1, 3, 5, 10} cm.

### 3.3 Nu vertices per slice vs threshold

- **`nu_vertices_per_slice_vs_threshold.png`** — for each threshold in
  {0.1, 0.2, 0.5, 0.7, 0.9}, the distribution (over slices) of the **number of
  reco nu candidates** with score ≥ that threshold, drawn as grouped bars. This
  shows how vertex multiplicity collapses toward ~1 per slice as the threshold
  tightens. The per-threshold mean/std/max are in the text summary.

### 3.4 Tabular / text outputs

- **`nu_vertex_per_slice.csv`** — per-slice dump: `file, max_score,
  best_dist_cm, ncand_ge_{0.1,0.2,0.5,0.7,0.9}`.
- **`nu_vertex_eval_summary.txt`** — human-readable summary (also echoed to
  stdout): signal/background counts, the distance summary, the ROC at the five
  thresholds, and the multiplicity table.

## 4. Signal vs background slices (purity caveat)

The slicer does not only produce neutrino slices — some isolated slices are
cosmic-only and contain **no true nu vertex**. The script detects these (no
finite truth) and treats them as **background**:

- they are **not** counted in the efficiency denominator (efficiency is over
  signal slices only);
- but a background slice whose max nu score clears the threshold **is** counted
  as a *selection*, so it lowers **purity** — it is a genuine false positive.

This is what makes purity meaningful. (In the bundled `reco_dev_data` sample, 3
of 8 slices are background — events 00001, 00007, 00009 — so purity on that tiny
set sits around 0.5–0.6.) A signal slice that is selected but whose best vertex
is farther than `--match-dist` from truth is *also* an impurity (selected but
not correct).

## 5. How to run

Run inside the pointcept container (needs `numpy`, `h5py`, `torch`, `scipy`,
`matplotlib`). From the `Pointcept/` root, either as a module:

```bash
PYTHONPATH=. python3 -m lartpc_data_prep.larformer_keypoint_v2.reco.eval_nu_vertex_reco \
    lartpc_data_prep/larformer_keypoint_v2/reco_dev_data/keypoint2_out \
    --output-dir nu_vertex_eval_out
```

or by direct path (no `PYTHONPATH` needed):

```bash
python3 lartpc_data_prep/larformer_keypoint_v2/reco/eval_nu_vertex_reco.py \
    lartpc_data_prep/larformer_keypoint_v2/reco_dev_data/keypoint2_out \
    --output-dir nu_vertex_eval_out
```

The positional `input` may be a directory of `*.h5`, a glob, or a single file.

### 5.1 Options

| Flag | Default | Meaning |
|---|---|---|
| `input` (positional) | — | dir of keypoint2 inference H5s, a glob, or one file |
| `--output-dir` | `nu_vertex_eval_out` | where plots/CSVs/summary are written |
| `--match-dist` | `5.0` | max true→reco distance (cm) counted as a *correct* selection |
| `--min-score` | `0.05` | reco score floor; must be ≤ the smallest count threshold (0.1) |
| `--max-candidates` | `32` | cap on reco nu candidates per slice |
| `--n-roc-points` | `99` | number of threshold samples in the ROC sweep |
| `--radius-cm` | `10.0` | reco isolation radius (passed to `KeypointRecoParams`) |
| `--sigma-cm` | `3.0` | reco Gaussian width (= GT label σ) |
| `--fit-method` | `nls` | peak fitter: `nls` \| `loglinear` \| `centroid` |
| `--amplitude` | `peak` | subtraction amplitude: `peak` (C++) \| `fit` |
| `--device` | `cpu` | torch device for the reco |
| `--n-events` | `-1` | limit to the first N files (`-1` = all) |
| `--no-plots` | off | skip PNG generation (CSV/text only) |

### 5.2 Caveat: the candidate cap

If a slice produces more candidates above `--min-score` than `--max-candidates`,
the script prints a warning and the **low-threshold tail** of the
multiplicity distribution (§3.3) is truncated at the cap (the ROC and distance
results, which use only the top candidate, are unaffected). On a real,
larger-statistics sample, raise `--max-candidates` (and keep `--min-score` no
lower than needed) to capture the full multiplicity at score ≥ 0.1.

## 6. Interpreting the result

- Read **`nu_vertex_eff_purity_vs_threshold.png`** to pick the operating point:
  the threshold where efficiency and purity cross (or where purity reaches a
  target) is the score cut to use for nu-vertex selection.
- Cross-check the chosen cut against **`nu_vertices_per_slice_vs_threshold.png`**
  — a good cut should leave ≈1 vertex per signal slice.
- Use **`nu_vertex_distance_distribution.png`** to confirm the selected vertices
  are also *well-localized*, not merely present.

## 7. Related

- [`keypoint_reco_spec.md`](keypoint_reco_spec.md) — the reco algorithm spec
  (§7 "Evaluation & validation" motivates this script).
- [`reco/`](reco) — the reco package this evaluates (`keypoint_reco.py`,
  `gaussian_fit.py`, `io.py`, `run_keypoint_reco.py`).
- `reco/run_keypoint_reco.py` — the standalone reco runner (writes reco H5s and
  reports micro-averaged resolution); this eval script is the nu-vertex-focused,
  threshold-sweep companion to it.
