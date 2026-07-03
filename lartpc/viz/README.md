# Shared Visualization Utilities

Common code for LArTPC event displays. Import with the repo root on `PYTHONPATH`
(or a script-local bootstrap): `from lartpc.viz.detector import DetectorOutline`.

- `detector.py` — MicroBooNE detector outline / wire-plane geometry for 3D
  displays (the formerly duplicated `detectoroutline.py`; the copy in
  `tools/viz_archive/detectoroutline.py` is a shim importing this one).
- `larformer_inference.py` — plotly helpers for decoding/displaying LArFormer
  inference outputs (formerly `pointcept/models/LArFormer/viz_inference.py`).

Event-display *scripts* live in `tools/viz/` (active) and `tools/viz_archive/`
(historical one-offs, kept runnable via the shim). As displays get touched,
extract their copy-pasted helpers (palettes, 3D scatter factories, wire-plane
rendering, plotly layouts) into this package instead of duplicating them.
