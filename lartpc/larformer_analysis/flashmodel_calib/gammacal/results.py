"""Provenance-stamped JSON results."""
import datetime
import json
import os
import socket
import subprocess
import sys

import numpy as np

from . import REPO_ROOT, VERSION


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_result(path, payload):
    payload = dict(payload)
    payload.update(date=datetime.datetime.now().isoformat(timespec="seconds"),
                   git_sha=git_sha(), gammacal_version=VERSION,
                   argv=sys.argv, hostname=socket.gethostname())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(payload), f, indent=1)
    return path
