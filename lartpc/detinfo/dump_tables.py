#!/usr/bin/env python3
"""
dump_tables.py -- inspect a detinfo HDF5 file (see SCHEMA.md). numpy + h5py only.

    python dump_tables.py FILE.h5                         # overview: groups, tables, columns, attrs
    python dump_tables.py FILE.h5 -g geometry/planes      # print a table
    python dump_tables.py FILE.h5 -g geometry/wires --where "(tpc==1)&(plane==2)" --head 5
    python dump_tables.py FILE.h5 -g geometry/opdets --cols opdet,center,shape
    python dump_tables.py FILE.h5 -g geometry/wires --csv wires.csv
    python dump_tables.py FILE.h5 -a clocks               # attributes of a group (scalars)
    python dump_tables.py FILE.h5 -t provenance/services_fhicl   # print a text dataset

A "table" is a group whose datasets all share the same first dimension
(n_rows attr). (N,k) columns are printed/exported as k sub-columns
(e.g. center -> center_x, center_y, center_z; bbox -> bbox_0..bbox_5).
"""
import argparse
import csv
import json
import sys

import numpy as np
import h5py


# (N,3) columns whose name contains one of these are xyz vectors
POSITIONAL = ("point", "center", "start", "end", "dir", "normal", "axis", "pos")
# (N,nplanes) columns, labelled _p0, _p1, ...
PER_PLANE = ("wire_coordinate", "nearest_wire", "tick")


def _fmt(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    if isinstance(v, np.ndarray):
        if v.size > 8:
            return f"array{v.shape} [{_fmt(v.flat[0])} .. {_fmt(v.flat[-1])}]"
        return "[" + ", ".join(_fmt(x) for x in v.ravel()) + "]"
    s = str(v)
    return s if len(s) <= 100 else s[:97] + "..."


def is_table(g):
    return isinstance(g, h5py.Group) and "n_rows" in g.attrs and all(
        isinstance(g[k], h5py.Dataset) for k in g)


def overview(f):
    print(f"file: {f.filename}")
    for k, v in f.attrs.items():
        print(f"  @{k} = {_fmt(v)}")

    def visit(name, obj):
        depth = name.count("/")
        ind = "  " * (depth + 1)
        if isinstance(obj, h5py.Group):
            if is_table(obj):
                cols = ", ".join(f"{c}{'' if obj[c].ndim == 1 else list(obj[c].shape[1:])}"
                                 for c in obj)
                print(f"{ind}[table] {name}  rows={obj.attrs['n_rows']}  cols: {cols}")
            else:
                print(f"{ind}[group] {name}")
            for k, v in obj.attrs.items():
                if k == "n_rows":
                    continue
                print(f"{ind}    @{k} = {_fmt(v)}")
        elif not is_table(obj.parent):
            print(f"{ind}[data]  {name} {obj.shape} {obj.dtype}")
    f.visititems(visit)


def table_columns(g, cols=None):
    names, arrays = [], []
    for c in (cols or list(g.keys())):
        a = g[c][()]
        if a.dtype.kind == "O" or a.dtype.kind == "S":
            a = np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)
        if a.ndim == 1:
            names.append(c); arrays.append(a)
        else:
            a2 = a.reshape(len(a), -1)
            if a2.shape[1] == 3 and any(t in c for t in POSITIONAL):
                sub = "xyz"
            elif c in PER_PLANE:
                sub = [f"p{i}" for i in range(a2.shape[1])]
            else:
                sub = [str(i) for i in range(a2.shape[1])]
            for i in range(a2.shape[1]):
                names.append(f"{c}_{sub[i]}"); arrays.append(a2[:, i])
    return names, arrays


def select_rows(g, where):
    n = int(g.attrs["n_rows"])
    if not where:
        return np.arange(n)
    env = {c: g[c][()] for c in g if g[c].ndim == 1 and g[c].dtype.kind in "iufb"}
    env["np"] = np
    mask = eval(where, {"__builtins__": {}}, env)  # noqa: S307 - local CLI tool
    return np.nonzero(np.asarray(mask))[0]


def print_table(g, cols, where, head):
    names, arrays = table_columns(g, cols)
    rows = select_rows(g, where)
    shown = rows if head is None else rows[:head]
    cells = [[_fmt(a[i]) for a in arrays] for i in shown]
    widths = [max([len(n)] + [len(r[j]) for r in cells]) for j, n in enumerate(names)]
    print("  ".join(n.rjust(w) for n, w in zip(names, widths)))
    for r in cells:
        print("  ".join(c.rjust(w) for c, w in zip(r, widths)))
    print(f"-- {len(shown)} of {len(rows)} selected rows ({g.attrs['n_rows']} total)")
    for k, v in g.attrs.items():
        if k != "n_rows":
            print(f"@{k} = {v}")


def export_csv(g, cols, where, path):
    names, arrays = table_columns(g, cols)
    rows = select_rows(g, where)
    with open(path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(names)
        for i in rows:
            w.writerow([a[i] for a in arrays])
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-g", "--table", help="table group to print")
    ap.add_argument("-a", "--attrs", help="group whose attributes to print")
    ap.add_argument("-t", "--text", help="string dataset to print")
    ap.add_argument("--cols", help="comma-separated column subset")
    ap.add_argument("--where", help="numpy boolean expression on 1-D numeric columns")
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--csv", help="export the selected table rows to CSV")
    ap.add_argument("--json", action="store_true", help="with -a: print as JSON")
    a = ap.parse_args()

    with h5py.File(a.file, "r") as f:
        if a.table:
            g = f[a.table]
            if not is_table(g):
                sys.exit(f"{a.table} is not a table")
            cols = a.cols.split(",") if a.cols else None
            if a.csv:
                export_csv(g, cols, a.where, a.csv)
            else:
                print_table(g, cols, a.where, a.head)
        elif a.attrs is not None:
            g = f[a.attrs] if a.attrs else f
            d = {k: (v.tolist() if isinstance(v, np.ndarray) else
                     v.item() if isinstance(v, np.generic) else v) for k, v in g.attrs.items()}
            if a.json:
                print(json.dumps(d, indent=1, default=str))
            else:
                for k, v in d.items():
                    print(f"{k:40s} {v}")
        elif a.text:
            v = f[a.text][()]
            print(v.decode() if isinstance(v, bytes) else v)
        else:
            overview(f)


if __name__ == "__main__":
    main()
