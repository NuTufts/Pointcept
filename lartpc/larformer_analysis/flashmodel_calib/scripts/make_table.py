"""Turn closed calibration results into GAMMA_SCALE_TABLE entries.

    PYTHONPATH=$K python3 scripts/make_table.py results/<chain>/<tag>__muon.json [...]
        [--closure results/<chain>/<tag>__closure.json ...] [--write]

Prints the python dict entries for lartpc/flashmatch/flash_calib.py (and a
markdown row for CALIBRATION_LOG.md). With --write it inserts/replaces the
entries in flash_calib.py in place. A result is accepted only if its validity
gates passed and, unless --no-closure-check, a closure JSON for the same
(kind, period) with median 1.00 +- 0.05 is given.
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
from lartpc.larformer_analysis.flashmodel_calib.gammacal import REPO_ROOT  # noqa: E402

FC = os.path.join(REPO_ROOT, "lartpc", "flashmatch", "flash_calib.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--closure", nargs="*", default=[])
    ap.add_argument("--no-closure-check", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    closed = {}
    for p in args.closure:
        j = json.load(open(p))
        closed[(j["kind"], int(j["period"]))] = j
    entries = {}
    for p in args.results:
        j = json.load(open(p))
        key = (j["kind"], int(j["period"]))
        st = j["stats"]
        if not st.get("gate_ok"):
            print(f"SKIP {p}: validity gates failed (core {st.get('core_frac')}, "
                  f"peak {st.get('peak')} vs median {st.get('median')})")
            continue
        if not args.no_closure_check:
            c = closed.get(key)
            if c is None or abs(c["scale_abs"] - 1.0) > 0.05:
                print(f"SKIP {p}: no passing closure for {key} "
                      f"(closure median {None if c is None else c['scale_abs']})")
                continue
        rel = os.path.relpath(p, REPO_ROOT)
        entries[key] = dict(scale=round(float(j["scale_abs"]), 4),
                            err=round(float(j["scale_err"]), 4), N=int(j["N"]),
                            result=rel, chain=j["chain"], date=j["date"][:10],
                            sample=j["sample"], arm=j["arm"])
        print(f"| {j['sample']} | {j['kind']} | {j['period']} | {j['N']} | "
              f"{j['scale_abs']:.4f} +- {j['scale_err']:.4f} | `{rel}` |")
    if not entries:
        return
    body = "\n".join(
        f"        {k!r}: {v!r}," for k, v in sorted(entries.items()))
    print("\nGAMMA_SCALE_TABLE entries:\n" + body)
    if args.write:
        s = open(FC).read()
        pat = re.compile(r'(    "entries": \{\n)(.*?)(    \},\n\})', re.S)
        assert pat.search(s), "could not locate the entries block"
        new = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(3), s, count=1)
        open(FC, "w").write(new)
        print(f">>> wrote {len(entries)} entries into {FC}")


if __name__ == "__main__":
    main()
