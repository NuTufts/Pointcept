"""
CPU smoke validation of the generated Phase 0.5 configs (WP5): parse each
config, build its train/val datasets against the real squashfs data, pull one
sample through the full transform pipeline, and check the tensors the model
would receive.

Run inside the container with the squashfs bound at /data:
  apptainer exec --bind $SQSH:/data:image-src=/,ro --bind /projects/u6jo:/projects/u6jo \
      /projects/u6jo/containers/pointcept-sandbox \
      python3 lartpc_tests/validate_p05_configs.py
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointcept.utils.config import Config  # noqa: E402
from pointcept.datasets import build_dataset  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "lartpc", "p05")


def check_ssl(cfg, item, name):
    in_ch = cfg.model["backbone"]["in_channels"]
    assert "global_feat" in item and "local_feat" in item, sorted(item.keys())
    assert item["global_feat"].shape[1] == in_ch, (
        f"global_feat channels {item['global_feat'].shape[1]} != in_channels {in_ch}")
    assert item["local_feat"].shape[1] == in_ch
    assert "global_segment" in item, "M5 needs global_segment in the batch"
    assert item["global_segment"].shape[0] == item["global_coord"].shape[0]
    labels = item["global_segment"]
    n_labeled = int((labels >= 0).sum())
    assert n_labeled > 0, "no truth labels in views — data_only path suspected"
    assert len(cfg.snapshot_at_iters) == 11
    # snapshot schedule must hit the same images-seen anchors for every batch size
    assert cfg.snapshot_at_iters[0] * cfg.batch_size == 24_000
    assert cfg.snapshot_at_iters[-1] * cfg.batch_size == 4_608_000
    return (f"views ok, feat_ch={in_ch}, "
            f"labeled_frac={n_labeled / labels.shape[0]:.2f}")


def check_supervised(cfg, item, name):
    assert "feat" in item and "segment" in item, sorted(item.keys())
    assert item["feat"].shape[1] == 6
    assert item["segment"].shape[0] == item["coord"].shape[0]
    assert int((item["segment"] >= 0).sum()) > 0
    extra = ""
    if "nocharge" in name:
        strength_part = item["feat"][:, 3:]
        assert torch.all(strength_part == strength_part[0, 0]), \
            "ZeroKey did not produce constant charge features"
        extra = ", charge zeroed"
    return f"feat_ch={item['feat'].shape[1]}{extra}"


def main():
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.py")))
    assert paths, f"no configs found in {CONFIG_DIR}"
    n_fail = 0
    for path in paths:
        name = os.path.basename(path)
        try:
            cfg = Config.fromfile(path)
            if name.startswith("linearprobe"):
                # Tufts config: data lists don't exist on this cluster —
                # parse-only here; the model path is covered by the
                # supervised configs (same architecture).
                assert cfg.model["freeze_backbone"] is True
                assert cfg.model["backbone"]["backbone"]["in_channels"] == 6
                print(f"PASS  {name}: parse-only (Tufts data paths)")
                continue
            train_ds = build_dataset(cfg.data.train)
            item = train_ds[0]
            if name.startswith("pretrain-sonata"):
                info = check_ssl(cfg, item, name)
            else:
                info = check_supervised(cfg, item, name)
            val_ds = build_dataset(cfg.data.val)
            val_item = val_ds[0]
            assert "feat" in val_item and val_item["feat"].shape[1] in (4, 6)
            print(f"PASS  {name}: {info}")
        except Exception as e:
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(paths) - n_fail}/{len(paths)} configs validated")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
