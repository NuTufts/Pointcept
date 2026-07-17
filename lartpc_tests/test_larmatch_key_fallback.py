"""
Gate tests for the larmatch_score_keys dataset parameter.

Background: the MC production wrote the LArMatch score as 'lm_score' while
the EXTBNB pipeline wrote 'larmatch_score'; LArTPCDataset only looked for
the latter, so filter_larmatch silently no-oped on MC. The new parameter
must (1) preserve that legacy behavior EXACTLY by default — live configs
(v8, P1A.4, P5B.2) rely on it — and (2) filter MC when opted in with
("larmatch_score", "lm_score").

Run inside the container with the squashfs bound at /data (Isambard) —
uses real MC (diag1k) and EXTBNB (extbnb_diag1k) files.
"""
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointcept.datasets.lartpc import LArTPCDataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MC_LIST = os.path.join(HERE, "..", "lartpc", "filelists", "h5list_v3_mc_diag1k.txt")
DATA_LIST = os.path.join(HERE, "..", "lartpc", "filelists", "h5list_v3_extbnb_diag1k.txt")
N_FILES = 4
LO, HI = 0.15, 0.75


def make_list(src, n):
    with open(src) as f:
        paths = [ln.strip() for ln in f if ln.strip()][:n]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tmp.write("\n".join(paths) + "\n")
    tmp.close()
    return tmp.name, paths


def make_ds(list_file, data_only, **kw):
    return LArTPCDataset(
        split="train", data_root="/tmp", data_list_file=list_file,
        transform=None, label_mode="ssnet", include_ghosts=True,
        exclude_other=False, data_only=data_only,
        filter_larmatch=True, larmatch_threshold_range=(LO, HI), **kw)


def main():
    mc_list, _ = make_list(MC_LIST, N_FILES)
    data_list, _ = make_list(DATA_LIST, N_FILES)
    try:
        # ---- 1. DEFAULT keys: MC filtering must remain a silent no-op ----
        ds_legacy = make_ds(mc_list, data_only=False)
        ds_nofilter = LArTPCDataset(
            split="train", data_root="/tmp", data_list_file=mc_list,
            transform=None, label_mode="ssnet", include_ghosts=True,
            exclude_other=False, data_only=False, filter_larmatch=False)
        for i in range(N_FILES):
            np.random.seed(1234 + i)
            a = ds_legacy.get_data(i)
            np.random.seed(1234 + i)
            b = ds_nofilter.get_data(i)
            assert a["coord"].shape == b["coord"].shape and \
                np.array_equal(a["coord"], b["coord"]), \
                "default larmatch_score_keys changed MC behavior"
        print("PASS  1. default keys: MC filter still no-ops (legacy preserved)")

        # ---- 2. Opt-in keys: MC IS filtered, with the exact expected mask ----
        ds_fb = make_ds(mc_list, data_only=False,
                        larmatch_score_keys=("larmatch_score", "lm_score"))
        n_checked = 0
        for i in range(N_FILES):
            path = ds_fb.data_list[i]
            with h5py.File(path, "r") as f:
                lm = np.asarray(f["/entry_0/triplet_data/lm_score"],
                                dtype=np.float32)
            np.random.seed(777 + i)
            item = ds_fb.get_data(i)
            # replay the RNG: threshold is the first uniform draw
            np.random.seed(777 + i)
            thr = np.random.uniform(LO, HI)
            expect = int((lm > thr).sum())
            if expect < 100:  # min_points retry path would kick in
                continue
            assert item["coord"].shape[0] == expect, \
                (item["coord"].shape[0], expect, thr)
            assert item["coord"].shape[0] < lm.shape[0], "nothing was filtered"
            n_checked += 1
        assert n_checked >= 2
        print(f"PASS  2. opt-in keys: MC filtered with exact expected mask "
              f"({n_checked} events)")

        # ---- 3. EXTBNB unaffected by the fallback (first key wins) ----
        for i in range(2):
            np.random.seed(4321 + i)
            a = make_ds(data_list, data_only=True).get_data(i)
            np.random.seed(4321 + i)
            b = make_ds(data_list, data_only=True,
                        larmatch_score_keys=("larmatch_score", "lm_score")).get_data(i)
            assert np.array_equal(a["coord"], b["coord"]), \
                "fallback changed EXTBNB behavior"
        print("PASS  3. EXTBNB identical with and without the fallback key")

        print("\nALL TESTS PASSED")
    finally:
        os.unlink(mc_list)
        os.unlink(data_list)


if __name__ == "__main__":
    main()
