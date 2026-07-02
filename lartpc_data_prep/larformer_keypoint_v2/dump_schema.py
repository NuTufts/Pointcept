"""Dump the data schema of the keypoint-2 cascade inference H5 files.

Walks one (or more) of the per-event H5 files written by the cascade
inference and prints the full hierarchy: groups, datasets (shape + dtype +
a small value sample), and the attributes attached to each node. Groups whose
members repeat (e.g. `particle/0`, `particle/1`, ...) are collapsed so you see
the schema of one representative child instead of N copies.

Usage:
    # inside the pointcept container (h5py lives there):
    ./run_in_tufts_pointcept_container.sh \
        python lartpc_data_prep/larformer_keypoint_v2/dump_schema.py \
        lartpc_data_prep/larformer_keypoint_v2/output/test_500events/

    # a specific file, full (uncollapsed) listing, with value previews:
    python dump_schema.py output/test_500events/keypoint2_event00000_0.h5 \
        --no-collapse --values
"""
import argparse
import glob
import os
import re

import numpy as np
import h5py


def _fmt_attr(v):
    """Compact, readable repr of an attribute value."""
    if isinstance(v, bytes):
        return repr(v.decode("utf-8", "replace"))
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape} dtype={v.dtype} {np.array2string(v, threshold=8)}"
    return repr(v)


def _sample(dset):
    """Return a short preview of a dataset's contents."""
    try:
        if dset.size == 0:
            return "(empty)"
        flat = np.asarray(dset[()]).ravel()
        head = flat[:6]
        s = np.array2string(head, threshold=6, precision=4)
        return s + (" ..." if flat.size > head.size else "")
    except Exception as e:  # noqa: BLE001
        return f"(unreadable: {e})"


def _print_attrs(node, indent):
    for k in node.attrs:
        print(f"{indent}  @{k} = {_fmt_attr(node.attrs[k])}")


# strip a trailing integer index so particle/0, particle/12 collapse together
_IDX = re.compile(r"^\d+$")


def walk(node, indent="", collapse=True, show_values=False):
    if isinstance(node, h5py.Dataset):
        line = f"{indent}DSET {node.name.split('/')[-1]}  " \
               f"shape={node.shape} dtype={node.dtype}"
        if show_values:
            line += f"  sample={_sample(node)}"
        print(line)
        _print_attrs(node, indent)
        return

    # Group (or File)
    name = node.name if node.name == "/" else node.name.split("/")[-1]
    print(f"{indent}GRP  {name}/")
    _print_attrs(node, indent)

    keys = list(node.keys())
    if collapse and keys and all(_IDX.match(k) for k in keys):
        # repeated indexed children -> show one representative
        rep = sorted(keys, key=int)[0]
        print(f"{indent}  [{len(keys)} indexed children "
              f"'0..{len(keys) - 1}', showing '{rep}']")
        walk(node[rep], indent + "    ", collapse, show_values)
        return

    for k in keys:
        walk(node[k], indent + "  ", collapse, show_values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an H5 file, or a directory of H5 files")
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    help="do not collapse repeated indexed groups")
    ap.add_argument("--values", dest="values", action="store_true",
                    help="print a small sample of each dataset's contents")
    ap.add_argument("--max-files", type=int, default=1,
                    help="when given a directory, how many files to dump "
                         "(default 1; schema is shared across events)")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.h5")))
        if not files:
            raise SystemExit(f"no .h5 files found in {args.path}")
        print(f"# {len(files)} H5 files in {args.path}")
        files = files[: args.max_files]
    else:
        files = [args.path]

    for p in files:
        print("\n" + "=" * 78)
        print(f"FILE: {p}")
        print("=" * 78)
        with h5py.File(p, "r") as f:
            walk(f, collapse=args.collapse, show_values=args.values)


if __name__ == "__main__":
    main()
