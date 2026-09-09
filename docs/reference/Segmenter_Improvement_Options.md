# Segmenter improvement options (stage-3 particle)

Parked plan, recorded 2026-09-07. Context: the rebalance campaign took the
mu->e confusion 0.164 -> 0.087 (old-ep8 -> rebal-warm2-cew ep6, measured on
`larformer_reco/inputlists/merged_sp_confusion_test2000.txt`, nu primaries
KE>50) with pi->pi improved and a ~1-sigma/epoch monotone gamma slide to
watch (0.927 -> 0.912). Remaining issues: mu->e still ~9% (LArPID reaches
1%); photon selection is LOOSE — passes cosmic showers that burden the
single-photon search. Full diagnosis chain in
`lartpc/larformer_analysis/physics/pilot_matrix/SLICER_RETRAIN_PLAN.md`
(2026-09-03..06 entries).

## A. Decoupled classifier retrain (cRT) — cheapest, feasible
Species prediction is a single `nn.Linear(dim, num_classes)` on decoder
query embeddings (`pointcept/models/LArFormer/decoder.py:222`). Freeze
backbone+decoder+mask head, reset the head (optionally + last decoder
block), retrain with CLASS-BALANCED INSTANCE sampling (1:1 per class,
LArPID-style — possible because the head trains per matched query, not per
event). Matching stays stable with frozen masks (or reuse frozen-model
assignments). Fast-iteration variant: precompute query embeddings for the
whole cache once, sweep weights/sampling/tau-normalization offline.

## B. LoRA adapters in the backbone — supported machinery
Precedent: the production v6-lantern deghoster is Sonata+LoRA
(`configs/lartpc/lora_finetune/`). Attach LoRA to the PTv3 backbone
attention, train LoRA + head, everything else frozen. Representation
flexibility beyond A at a few % of full-training cost; frozen base
protects photon lanes.

## C. Capacity increases
- Decoder-side (warm-startable): more queries (now 32), deeper decoder,
  wider class MLP replacing the linear head.
- Backbone widening/deepening: breaks Sonata-pretrain compatibility ->
  requires re-pretrain; reserve for a planned Sonata refresh.

## D. Cosmic supervision — targets the loose-photon mechanism directly
`masked_no_object=True` EXCLUDES unmatched queries concentrated on
unlabeled (cosmic) points from the no-object CE — i.e., training never
punishes claiming a cosmic shower as a particle. Options:
  (i) explicit cosmic class supervised from hasmatch==0 charge in overlay
      events (label is free: unlabeled overlay charge IS cosmic);
 (ii) EXT-BNB events as hard negatives (pure cosmic by construction; no
      truth needed; addable to the cache);
(iii) partial/relaxed no-object penalty on cosmic-charge queries.

## E. Full-event context for the encoder (user insight, 2026-09-07)
The cache trains the embedder on NU-SLICE-ONLY points: the slicer sees all
deghosted points, but the cache saves only the nu slice, so the encoder
cannot form contextual features (surrounding cosmic activity) that would
help tag a photon as cosmic-like. Proposal: run the encoder on ALL
deghosted points (or a neighborhood around the slice) while restricting
the DECODER's queries/masks to nu-slice points. Requires a cache-format
change (store full-event points + slice mask) and retraining; composes
with A-D.

## F. Data levers
LANTERN generic sim: 190,204 muon-rich events already converted (needs
label completion + cache build only). Further generic-numu overlay
tranches: ~275k usable events, disk now available (5.3T free after the
2026-09-05 repack + cache deletions). Ledger:
`lartpc/data_prep/uboone_official/training_data_ledger/`.

## G. Post-hoc (no retrain) — STATUS: largely applied
- Per-shower cosmic BDT (`showerCosmicScore`) — APPLIED in the
  single-photon search; a second, VERTEX-FREE BDT was also developed to
  provide a no-vertex acceptance stream for photon events.
- Logit adjustment (prior-corrected argmax): measured tau=1 gives modest
  mu gains at pi->mu cost; calibration knob only.

## Suggested sequence (when resumed)
D(i or ii) + A together on top of the cew-ep6 checkpoint; escalate to B if
the frozen-embedding ceiling shows; E as the structural change alongside
the next cache rebuild; C-backbone only with a Sonata refresh.
Acceptance harness: the 2000-event confusion A/B + photon-purity metric
(cosmic-shower pass rate at fixed photon eff) to be added for D/E.
