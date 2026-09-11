"""List the stream='calib' cascade files of a --flash-calib-mode output dir in
event-index order (the kp2 list a calib sample entry points at).

    python3 scripts/make_calib_list.py <output_dir> <list.txt>
"""
import glob
import os
import re
import sys


def main():
    out_dir, dst = sys.argv[1], sys.argv[2]
    files = {}
    for p in glob.glob(os.path.join(out_dir, "**", "keypoint2_event*_calib_0.h5"), recursive=True):
        m = re.search(r"event(\d+)_calib_0\.h5$", os.path.basename(p))
        if m:
            files[int(m.group(1))] = os.path.abspath(p)
    with open(dst, "w") as f:
        for k in sorted(files):
            f.write(files[k] + "\n")
    print(f">>> {len(files)} calib files -> {dst}")


if __name__ == "__main__":
    main()
