"""Compatibility shim: canonical detector geometry lives in lartpc/viz/detector.py."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
from lartpc.viz.detector import *  # noqa: F401,F403
from lartpc.viz.detector import DetectorOutline  # noqa: F401
