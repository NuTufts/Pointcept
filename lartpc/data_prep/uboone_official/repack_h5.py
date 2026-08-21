"""Reclaim dead HDF5 extents left by in-place label completion (delete+
create leaks the old arrays inside the file). Python repack: copy every
group/dataset/attr to a temp file in the same directory — LABEL datasets
gzip-compressed, everything else contiguous (training read path
untouched) — verify, then atomically os.replace() over the original.
Original is never modified until the verified replace; failures leave it
intact and are reported per file.

    python3 repack_h5.py --h5 f1.h5 [f2.h5 ...]
"""
import argparse
import os

import h5py

COMPRESS_SUFFIXES = ("trackid", "pid", "origin", "hasmatch",
                     "trackid_precomplete", "pid_precomplete",
                     "origin_precomplete", "hasmatch_precomplete",
                     "label_completed")


def _copy(src_grp, dst_grp):
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v
    for name, obj in src_grp.items():
        if isinstance(obj, h5py.Group):
            _copy(obj, dst_grp.create_group(name))
        else:
            kw = {}
            if name in COMPRESS_SUFFIXES and obj.ndim >= 1 \
                    and obj.shape[0] > 0:
                kw = dict(compression="gzip", compression_opts=4)
            elif obj.compression is not None:
                # preserve the production file's own filters (everything
                # is gzip'd at source — decompressing would GROW files)
                kw = dict(compression=obj.compression,
                          compression_opts=obj.compression_opts,
                          chunks=obj.chunks, shuffle=obj.shuffle)
            d = dst_grp.create_dataset(name, data=obj[()], **kw)
            for k, v in obj.attrs.items():
                d.attrs[k] = v


def repack(path):
    tmp = path + ".repack_tmp"
    try:
        with h5py.File(path, "r") as fin, h5py.File(tmp, "w") as fout:
            _copy(fin, fout)
        # verify: same dataset tree, same shapes, label invariant
        with h5py.File(path, "r") as fin, h5py.File(tmp, "r") as fout:
            names_in, names_out = [], []
            fin.visit(names_in.append)
            fout.visit(names_out.append)
            assert names_in == names_out, "tree mismatch"
            td_i = fin["entry_0/triplet_data"]
            td_o = fout["entry_0/triplet_data"]
            assert td_i["trackid"].shape == td_o["trackid"].shape
            assert int((td_i["trackid"][()]
                        != td_o["trackid"][()]).sum()) == 0
        old = os.path.getsize(path)
        new = os.path.getsize(tmp)
        os.replace(tmp, path)
        return old, new
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", nargs="+", required=True)
    args = ap.parse_args()
    saved = 0
    n_err = 0
    for p in args.h5:
        try:
            old, new = repack(p)
            saved += old - new
        except Exception as e:
            n_err += 1
            print(f"ERROR {os.path.basename(p)}: {e!r}")
    print(f"[repack] {len(args.h5) - n_err}/{len(args.h5)} ok, "
          f"reclaimed {saved / 1e9:.2f} GB")
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
