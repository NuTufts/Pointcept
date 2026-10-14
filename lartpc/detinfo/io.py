"""HDF5 helpers and cache-file resolution."""
import glob
import os

import numpy as np

DEFAULT_DETINFO_ROOT = "/cluster/tufts/wongjiradlab/larbys/detinfo"
_REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _decode(a):
    if a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)
    return a


def read_table(group):
    """Table group -> dict column -> numpy array (strings decoded)."""
    return {k: _decode(group[k][()]) for k in group.keys()}


def read_attrs(group):
    out = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def read_tree(group):
    """Group -> nested dict: subgroups -> dicts, datasets -> arrays (strings decoded),
    attrs -> values; `<key>__json` attrs are decoded into `key`."""
    import json
    out = {}
    for k, v in read_attrs(group).items():
        if k.endswith("__json"):
            out[k[:-6]] = json.loads(v)
        else:
            out[k] = v
    for k in group.keys():
        obj = group[k]
        if hasattr(obj, "keys"):
            out[k] = read_tree(obj)
        else:
            a = obj[()]
            out[k] = a.decode() if isinstance(a, bytes) else (_decode(a) if hasattr(a, "dtype") else a)
    return out


def search_roots():
    """Where cache files are looked for, in order."""
    roots = []
    if os.environ.get("DETINFO_ROOT"):
        roots.append(os.environ["DETINFO_ROOT"])
    roots += [_REPO_DATA, DEFAULT_DETINFO_ROOT]
    return roots


def find_detinfo_file(detector, version=None):
    """Return the path of detinfo_<detector>_<version>.h5.

    Searches $DETINFO_ROOT, then lartpc/detinfo/data, then the shared cluster
    area, for <root>/<detector>/<product>_<version>/detinfo_<detector>_<version>.h5.
    With version=None the lexically latest version of the first root that has
    any is returned."""
    for root in search_roots():
        pat = os.path.join(root, detector, "*", f"detinfo_{detector}_{version or '*'}.h5")
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        f"no detinfo file for detector={detector!r} version={version!r} under {search_roots()}; "
        f"run lartpc/detinfo/extract/run_extract.sh")
