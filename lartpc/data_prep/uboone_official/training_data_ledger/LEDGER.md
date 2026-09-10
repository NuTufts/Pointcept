# Training-data ledger — overlay tier2 samples (created 2026-08-18)
# Rule (user): at most HALF of each sample may enter training; the
# remainder is RESERVED for inference/expectation vs beam data.
# Split: deterministic per-file (md5(path) even/odd on last hex digit
# parity of hash int) -> stable under list edits/reordering.

mcc9_v28_run1_bnboverlay: total=9538  trainpool=4740  reserved=4798
mcc9_v28_run1_nueintrinsics: total=4000  trainpool=1986  reserved=2014
mcc9_v29e_dl_run1_bnb_intrinsic_nue_LowE: total=617  trainpool=292  reserved=325
mcc9_v29e_dl_run3b_bnb_intrinsic_nue: total=3331  trainpool=1662  reserved=1669
mcc9_v29e_dl_run3b_bnb_intrinsic_nue_LowE: total=580  trainpool=287  reserved=293
mcc9_v29e_dl_run3b_bnb_nu: total=15566  trainpool=7826  reserved=7740
mcc9_v40a_run3_CCpi0: total=832  trainpool=392  reserved=440
mcc9_v40a_run3_NCpi0: total=1736  trainpool=840  reserved=896

EVAL LOCK: 767 run3b_bnb_nu trainpool files matched the satfix scale1500 eval production (basename match) -> moved to mcc9_v29e_dl_run3b_bnb_nu_EVAL_LOCKED.txt (never train). 733 reserved-side files also match (fine: reserved != training).

SMOKE VALIDATION (2026-08-18): one file per sample staged + tree
inventory — ALL 8 samples carry the full required product set
(image2d wire/instance/ancestor/segment/thrumu, chstatus, mcreco trees),
including the v40a CCpi0/NCpi0 dlana files (old-reco extras ignored;
truth + raw inputs intact, as user expected). stepA conversion smoke
(--adc wire -tb --mcc9) PASSED on mcc9_v40a_run3_CCpi0 and
mcc9_v28_run1_bnboverlay; truth content validated (pi0 in tree, labels
populated). Zero-label entries = genuine DIRT events (nu vertex outside
TPC, e.g. x=300cm) — normal for bnb overlay; the training list-builder
should require a minimum labeled-point count (dirt events give no
usable GT instance). Smoke h5s kept in
mcc9_scratch/tier2_staging_smoke/smoke_{ccpi0,v28bnb}; staged .root
copies removed.

LANTERN SUBSAMPLE (2026-08-19, user rule "remove from the bnb nu
files"): lantern_train_subsample_enriched.txt = all nue + pi0filter +
chargedpiplus files (219,796) — the non-generic samples alone exceed
the 205,341 overlay corpus, so ALL generic prod2/set2 files are removed
(lantern_train_removed_generic.txt, 190,204). Mixture = 48/52
overlay/sim, both sides enriched-composition. Generic topology returns
with the deferred max-stat overlay processing. NOTE: completion runs on
the TRAIN subsample ONLY — val/eval LANTERN files stay untouched to
preserve the campaign metric baselines.

MIX v1 (2026-08-19): h5list_mix_enriched_train_v1.txt = 186529 overlay (minlab200 dirt filter, from labeled_counts scan) + 219796 LANTERN enriched = 406325 events, seed-42 shuffle. Val list for training-time probe stays h5list_mcall_lantern_val.txt (untouched baselines); real gating = overlay+data battery (C1).

QUOTA INCIDENT (2026-08-19): in-place LANTERN completion filled the
wongjiradlab volume (h5 delete+create leaves DEAD EXTENTS — growth ~3x
label storage/file). Resolution: extbnb_dlreco scratch deleted after
tier2 verification at /cluster/tier2/wongjiradlab/wongjiradlabnu/...
(50,410/50,410 + spot-checks) -> 1.3T free. Damage: 0 schema-broken;
5 files quota-TRUNCATED (lantern_quota_damaged.txt) — excluded from
subsample + mix lists; restorable only from a v3_larmatch tier2 copy if
one exists. Remaining 24,461 PENDING resumed with gzip-compressed
writes + fault-tolerant tool (array 2623963). OPTIONAL follow-up:
h5repack campaign over the 195k DONE files to reclaim dead extents
(~1-2 TB) — proposed, not launched.

REPACK COMPLETE (2026-08-19, job 2624271): 200/200 chunks, 0 errors,
**2,178.7 GB reclaimed** over the 195k first-pass-completed LANTERN
files (source filters preserved, labels gzip'd, verified + atomic
replace). Volume: 3.4 TB free (89%) — better than pre-incident. The
24,461 resume-set files (compressed writer, ~10-20 GB residual leak)
left as-is.

RESTORE OF QUOTA-CORRUPTED FILES (2026-08-20, user request; source list
isambard .../filelists/h5_integrity_broken.txt == the 5 files ledgered
here as lantern_quota_damaged.txt). Needed for the scaling-law training
experiments. Procedure:
  1. Parent dlmerged = line <fileno> of the sample's prod2 inputlist
     (line number == fileno, verified for all 5). All 5 absent from
     tier1, present on tier2 (/cluster/tier2/wongjiradlab/larbys/data/
     ub_on_tufts/) -> staged to tier1 ub_on_tufts/dlmerged_scratch/
     (same staging area as the val-twin restore; these are ub_on_tufts
     corsika files, not mcc9), 2.8 GB, size-verified vs tier2.
  2. Remade with the ORIGINAL LANTERN pipeline (run_lantern_wconfig.sh,
     configs restore_broken/restore_<TAG>.conf: larmatch min-score 0.15,
     ADC wiremc, tick-forward, no mcc9) via RERUN_LINES_FILE mode ->
     SEPARATE verification tree hdf5/v3_larmatch_restore/<TAG>/merged_h5.
     Line-preserving staged inputlists keep lineno==fileno.
     Arrays 2668290 (nue: 2500,2799,3265) + 2668291 (cpiplus: 2314,5437).
  3. Verify (restore_broken/verify_restore.py): existence, readability,
     schema vs an intact sibling, lm_score min >= 0.15.
  4. THEN label-expand (complete_labels.py r=0.5 shell+-2) and place
     into the training tree / re-add to the affected file lists.
  RESULT (2026-08-20): ALL 5 RESTORED, LABEL-EXPANDED, AND PLACED.
  First pass 3/5; filenos 2314 + 3265 failed on NODE-LOCAL /tmp
  exhaustion (WORKDIR_BASE default) — 2314 died mid-Step-2 with
  errno 28, 3265 left a truncated larmatchme_larlite.root (Step 2 then
  emitted nothing). Retried with WORKDIR_BASE on shared scratch
  (dlmerged_scratch/_restore_workdir) -> both succeeded.
  Verified: lm_score min == 0.150 (cut applied), 21 datasets == intact
  siblings (16 base + 5 completion), point counts 84k-278k, adoption
  +0.44..+0.57 rel to donors (matches corpus). Placed over the corrupt
  live files via atomic temp+mv; restore-tree copies kept as backup at
  hdf5/v3_larmatch_restore/. NO list edits needed — all 5 are still
  referenced by the isambard scaling-law lists (h5list_v3_mc_only_train*,
  h5list_v3_mix1to1_train*, h5list_mcall_lantern_validated,
  h5list_tufts_source_mc_plus_extbnb), which now resolve to good files.
  NOTE: these 5 remain excluded from the S1 mix lists (excluded while
  broken; 5/406k is negligible — re-add at the next list rebuild).
  Byproduct entries from the same parents (~90 files) sit unused in the
  restore tree; delete when the backup is no longer wanted.
