#!/usr/bin/env python3
"""
Generate the Phase 0.5 file lists from the combined EXTBNB+MC shuffled list.

See lartpc/pretraining_studies/phase0_phase05_implementation_plan.md (WP1).

The source list is the one used by the v8 pretraining run
(h5list_v3_combined_extbnb_mc_shuffled.txt, 954,325 paths under the /data
squashfs mount). It is already shuffled; splits are taken deterministically
from the tail of each category so the output is fully reproducible with no RNG.

Categories (every line matches exactly one; verified 2026-07-13):
  - real data (EXTBNB):  path contains 'extbnb'   (532,645 files)
  - MC:                  path contains 'corsika'  (421,680 files;
                         bnb_nu_corsika_set2_prod2 / bnb_nue_corsika /
                         bnb_nu_chargedpiplus_corsika)

Outputs (into --outdir, default: this directory):
  h5list_v3_mc_only_train.txt      MC minus (diag1k + mc val)
  h5list_v3_mc_only_val.txt        5,000 MC files (never trained on)
  h5list_v3_mc_diag1k.txt          1,000 MC files, frozen diagnostic set
                                   (disjoint from BOTH train and val)
  h5list_v3_extbnb_only_train.txt  EXTBNB minus extbnb val
  h5list_v3_extbnb_only_val.txt    5,000 EXTBNB files
  h5list_v3_combined_train.txt     original order, minus every val/diag file
  h5list_v3_combined_val.txt       mc val + extbnb val (10,000 files)
  filelist_stats.txt               counts + sha256 of each output

Split layout (per category, taken from the END of the shuffled list):
  MC:      [ train ....................... | diag1k (1,000) | val (5,000) ]
  EXTBNB:  [ train ................................        | val (5,000) ]

Usage:
  python3 lartpc/filelists/make_filelists.py \
      --source /home/u6jo/twongj01.u6jo/ubpointcept/pointcept/h5list_v3_combined_extbnb_mc_shuffled.txt
"""
import argparse
import hashlib
import os
import sys

MC_TAG = "corsika"
DATA_TAG = "extbnb"
N_VAL = 5000
N_DIAG = 1000


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_list(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="combined shuffled list (h5list_v3_combined_extbnb_mc_shuffled.txt)")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-diag", type=int, default=N_DIAG)
    args = ap.parse_args()

    with open(args.source) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    mc = [ln for ln in lines if MC_TAG in ln]
    data = [ln for ln in lines if DATA_TAG in ln]
    unclassified = [ln for ln in lines if (MC_TAG in ln) == (DATA_TAG in ln)]
    if unclassified:
        sys.exit(f"ERROR: {len(unclassified)} lines match neither/both tags, "
                 f"e.g. {unclassified[0]}")
    assert len(mc) + len(data) == len(lines)

    n_val, n_diag = args.n_val, args.n_diag
    mc_val = mc[-n_val:]
    mc_diag = mc[-(n_val + n_diag):-n_val]
    mc_train = mc[:-(n_val + n_diag)]
    data_val = data[-n_val:]
    data_train = data[:-n_val]

    held_out = set(mc_val) | set(mc_diag) | set(data_val)
    combined_train = [ln for ln in lines if ln not in held_out]
    combined_val = mc_val + data_val

    os.makedirs(args.outdir, exist_ok=True)
    outputs = {
        "h5list_v3_mc_only_train.txt": mc_train,
        "h5list_v3_mc_only_val.txt": mc_val,
        "h5list_v3_mc_diag1k.txt": mc_diag,
        "h5list_v3_extbnb_only_train.txt": data_train,
        "h5list_v3_extbnb_only_val.txt": data_val,
        "h5list_v3_combined_train.txt": combined_train,
        "h5list_v3_combined_val.txt": combined_val,
    }

    stats = [f"source: {args.source}",
             f"source_sha256: {sha256_of(args.source)}",
             f"total={len(lines)} mc={len(mc)} extbnb={len(data)}"]
    for name, content in outputs.items():
        path = os.path.join(args.outdir, name)
        n = write_list(path, content)
        stats.append(f"{name}: n={n} sha256={sha256_of(path)}")

    # invariants
    assert not (set(mc_train) & held_out)
    assert not (set(data_train) & held_out)
    assert not (set(mc_diag) & set(mc_val))
    assert len(combined_train) == len(lines) - len(held_out)

    stats_path = os.path.join(args.outdir, "filelist_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats) + "\n")
    print("\n".join(stats))


if __name__ == "__main__":
    main()
