"""ptv3deghost cascade at deghost_threshold_val=0.35 — slicer-response test.

Tau-sweep (2026-08-06) showed the new deghoster reaches near-ceiling shower
completeness at low tau (gamma 0.805 @ tau=0.20 vs ceiling 0.815) at the
cost of kept-set purity (0.657 @ 0.20 vs 0.731 @ 0.50). This config reruns
the cascade at tau=0.35 to measure the SLICER's response to the dirtier
keep-set (slicer trained at tau ~ U(0.4, 0.6)): does the slicer gap stay
~0.013 or grow?
"""

_base_ = ["./larformer-slicer-m2frecipe-v2-ptv3deghost.py"]

model = dict(deghost_threshold_val=0.35)
save_path = "exp/larformer_slicer_m2frecipe_v2_ptv3deghost_tau035_eval"
