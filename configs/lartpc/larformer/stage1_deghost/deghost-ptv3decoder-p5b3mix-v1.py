"""Deghoster PTv3-decoder on the P5B.3 MIXED sim+data Sonata encoder — the
encoder-swap arm of the domain-robustness A/B (2026-08-14).

Motivation: the pilot-matrix photon study showed the current ft deghoster
(frozen v7 encoder = EXTBNB-DATA-ONLY pretrain, + from-scratch PTv3 decoder
supervised on corsika) collapses on the run3b data-overlay domain (photon
charge keep 0.531 @ tau=0.2 vs 0.854 for the LoRA head on the same feed),
while inference plumbing was validated (run_deghost_eval: cascade feed
reproduces val within ~1%). Hypothesis: a mixed-domain SSL encoder
(P5B.3 = symmetric 1:1 sim+data mixture, stochastic LArMatch filter on both
domains) transfers better under the big supervised head.

EVERYTHING except the encoder checkpoint is IDENTICAL to
deghost-ptv3decoder-v1-frozenenc-extbnb.py (crop training, frozen encoder,
same recipe) so the A/B isolates the encoder. Architecture verified drop-in:
same PT-v3m2 trunk (3,3,3,9,3)/(48..512), grid 0.25 cm, LogTransform(0.01,
1000) strength; its swap_strength_columns is z-flip augmentation (p=0.5,
symmetric), not a channel-order change.

After this crop stage: full-event ft via deghost-ptv3decoder-p5b3mix-v2-
fullevent-ft.py, then judge on BOTH domains (run_deghost_eval corsika val +
overlay photon keep-curve) before any production decision.
"""

_base_ = ["./deghost-ptv3decoder-v1-frozenenc-extbnb.py"]

# P5B.3 mixed sim+data Sonata pretrain (epoch_18), consumed by
# SonataFinetuneCheckpointLoader via this top-level key.
weight = (
    "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept/"
    "sonata/p5b/P5B.3-mix_larmatch-s0/model/epoch_18.pth"
)

save_path = "exp/deghost_ptv3decoder_p5b3mix_v1"
