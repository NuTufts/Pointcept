#!/usr/bin/env python3
"""Assert the flash-calibration attrs of keypoint2 cascade files (needs h5py:
run inside the container, e.g. `pol_exec python3 lartpc/polaris/check_kp2_attrs.py ...`).

  check_kp2_attrs.py <file.h5> [...]            # explicit files
  check_kp2_attrs.py --tree <dir> --sample 20   # 20 random nu-stream files of a tree

Expected values (bnb5e19, calibrated table gamma) are the defaults; override
with --expect-* for other samples. Exit 1 on any mismatch or unreadable file.
"""
import argparse
import os
import random
import sys

import h5py


def find_files(tree, fm=False):
    suffix = "_fm_0.h5" if fm else "_0.h5"
    out = []
    for root, _d, files in os.walk(tree):
        for f in files:
            if f.startswith("keypoint2_event") and f.endswith(suffix) and (fm or not f.endswith("_fm_0.h5")):
                out.append(os.path.join(root, f))
    return sorted(out)


def check(path, exp):
    bad = []
    with h5py.File(path, "r") as h:
        fa = dict(h["flash"].attrs) if "flash" in h else {}
        stream = h.attrs.get("stream", b"")
        stream = stream.decode() if isinstance(stream, bytes) else str(stream)
        got = {
            "gamma_eff": float(fa.get("gamma_eff", float("nan"))),
            "gamma_spec": str(fa.get("gamma_spec", "")),
            "sample_kind": str(fa.get("sample_kind", "")),
            "flash_window": str(fa.get("flash_window", "")),
            "stream": stream,
        }
    for k, v in exp.items():
        if v is None:
            continue
        g = got[k]
        if k == "gamma_eff":
            okk = abs(g - float(v)) < 1e-3
        elif k == "flash_window":
            # stored as "%g,%g" ("2.8,5"), so compare numerically
            try:
                okk = ([float(x) for x in g.split(",")] == [float(x) for x in str(v).split(",")])
            except ValueError:
                okk = (g == v)
        elif k == "stream":
            # the nu file of an event whose nu slice is also the flash-matched
            # slice carries stream="nu,flashmatch"
            okk = (g.split(",")[0] == v)
        else:
            okk = (g == v)
        if not okk:
            bad.append(f"{k}: got {g!r} expected {v!r}")
    return got, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--tree")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--expect-gamma-eff", default="2.86073")
    ap.add_argument("--expect-spec", default="table")
    ap.add_argument("--expect-kind", default="data")
    ap.add_argument("--expect-window", default="2.8,5.0")
    ap.add_argument("--expect-stream", default="nu")
    args = ap.parse_args()
    files = list(args.files)
    if args.tree:
        allf = find_files(args.tree)
        random.seed(0)
        files += random.sample(allf, min(args.sample, len(allf)))
        print(f">>> {len(allf)} nu-stream files under {args.tree}; checking {min(args.sample, len(allf))}")
    if not files:
        sys.exit("no files")
    exp = {"gamma_eff": args.expect_gamma_eff, "gamma_spec": args.expect_spec,
           "sample_kind": args.expect_kind, "flash_window": args.expect_window,
           "stream": args.expect_stream}
    nbad = 0
    for p in files:
        try:
            got, bad = check(p, exp)
        except Exception as ex:  # noqa: BLE001
            print(f"FAIL {p}: unreadable ({type(ex).__name__}: {ex})"); nbad += 1; continue
        if bad:
            nbad += 1
            print(f"FAIL {os.path.basename(p)}: " + "; ".join(bad))
        else:
            print(f"ok   {os.path.basename(p)}: gamma_eff={got['gamma_eff']:.5f} spec={got['gamma_spec']} "
                  f"kind={got['sample_kind']} window={got['flash_window']} stream={got['stream']}")
    print(f">>> check_kp2_attrs: {len(files) - nbad}/{len(files)} ok -> {'PASS' if nbad == 0 else 'FAIL'}")
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
