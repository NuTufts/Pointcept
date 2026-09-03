"""Battery cascade: v6-lantern deghoster + a CANDIDATE slicer checkpoint
taken from $LARFORMER_BATTERY_SLICER_CKPT (falls back to the deployed
m2f-v2 ep4). Used by the slicer-retrain gate battery
(run_slicer_battery.sh); everything else inherited from the v6lantern
cascade.

NOTE: no module-level `import os` — this file is pulled in via _base_,
and Config._substitute_base_vars deepcopies the namespace, which cannot
copy a module object ("cannot pickle 'module' object").
"""

_base_ = ["./larformer-fullcascade-v6lantern-tau020.py"]

_ckpt = __import__("os").environ.get(
    "LARFORMER_BATTERY_SLICER_CKPT", "").strip()
if _ckpt:
    model = dict(cascaded_slicer_weight=_ckpt)
del _ckpt
