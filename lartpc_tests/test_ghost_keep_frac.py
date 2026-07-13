"""
Test the ghost_keep_frac knob of LArTPCDataset (WP4 / P0.4 of
lartpc/pretraining_studies/phase0_phase05_implementation_plan.md).

Uses real MC files from the diagnostic list, so run inside the container with
the dataset squashfs bound at /data:

  apptainer exec --bind $SQSH:/data:image-src=/,ro --bind /projects/u6jo:/projects/u6jo \
      /projects/u6jo/containers/pointcept-sandbox \
      python3 lartpc_tests/test_ghost_keep_frac.py
"""
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointcept.datasets.lartpc import LArTPCDataset  # noqa: E402

DIAG_LIST = os.path.join(
    os.path.dirname(__file__), "..", "lartpc", "filelists", "h5list_v3_mc_diag1k.txt"
)
N_FILES = 5


def make_dataset(list_file, **kwargs):
    return LArTPCDataset(
        split="train",
        data_root="/tmp",
        data_list_file=list_file,
        transform=None,
        label_mode="ssnet",
        include_ghosts=True,
        exclude_other=False,
        data_only=False,
        **kwargs,
    )


def main():
    with open(DIAG_LIST) as f:
        paths = [ln.strip() for ln in f if ln.strip()][:N_FILES]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(paths) + "\n")
        list_file = tmp.name

    try:
        # ground truth counts straight from the files (dataset sorts its list)
        truth = {}
        for p in sorted(paths):
            with h5py.File(p, "r") as f:
                hm = np.asarray(f["/entry_0/triplet_data/hasmatch"])
            truth[os.path.splitext(os.path.basename(p))[0]] = (
                int((hm == 1).sum()), int((hm == 0).sum())
            )

        # mutual exclusion is enforced
        try:
            make_dataset(list_file, ghost_keep_frac=0.5, true_points_only=True)
            raise SystemExit("FAIL: exclusivity assert did not fire")
        except AssertionError:
            print("PASS  ghost_keep_frac + true_points_only rejected")

        for frac in (0.0, 0.5, 1.0):
            ds = make_dataset(list_file, ghost_keep_frac=frac)
            for i in range(len(ds.data_list)):
                d = ds.get_data(i)
                n_real_t, n_ghost_t = truth[d["name"]]
                n_real = int((d["hasmatch"] == 1).sum())
                n_ghost = int((d["hasmatch"] == 0).sum())
                assert n_real == n_real_t, (
                    f"{d['name']}: real points not preserved "
                    f"({n_real} != {n_real_t})"
                )
                kept = n_ghost / max(n_ghost_t, 1)
                # binomial 5-sigma tolerance around frac
                tol = 5.0 * np.sqrt(frac * (1 - frac) / max(n_ghost_t, 1))
                assert abs(kept - frac) <= tol, (
                    f"{d['name']}: ghost keep fraction {kept:.4f} "
                    f"outside {frac}±{tol:.4f}"
                )
            print(f"PASS  ghost_keep_frac={frac}: real preserved, "
                  f"ghost fraction within tolerance ({len(ds.data_list)} events)")

        # equivalence: frac=0.0 matches true_points_only
        ds_frac0 = make_dataset(list_file, ghost_keep_frac=0.0)
        ds_tpo = make_dataset(list_file, true_points_only=True)
        for i in range(len(ds_frac0.data_list)):
            a, b = ds_frac0.get_data(i), ds_tpo.get_data(i)
            assert a["coord"].shape == b["coord"].shape
            assert np.allclose(a["coord"], b["coord"])
        print("PASS  ghost_keep_frac=0.0 == true_points_only")

        print("\nALL TESTS PASSED")
    finally:
        os.unlink(list_file)


if __name__ == "__main__":
    main()
