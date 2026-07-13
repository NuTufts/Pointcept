"""
Test the weights-only snapshot feature of IterCheckpointSaver (WP2 of
lartpc/pretraining_studies/phase0_phase05_implementation_plan.md).

Checks, using a fake trainer (no GPU needed):
  1. snapshots are written exactly at the scheduled global steps
  2. filenames encode global_step and images_seen (= step * cfg.batch_size)
  3. snapshot content is weights-only (state_dict + metadata, no optimizer)
  4. the state_dict round-trips through SonataCheckpointLoader's key-remap
     convention (sonata-style 'student./teacher.' keys get 'backbone.'
     prepended; 'module.' prefixes stripped)

Run inside the pointcept container:
  python3 lartpc_tests/test_iter_snapshot_saver.py
"""
import glob
import logging
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pointcept.engines.hooks.misc import IterCheckpointSaver  # noqa: E402


class SonataLikeModel(nn.Module):
    """Minimal module whose state_dict keys mimic a Sonata pretrain model."""

    def __init__(self):
        super().__init__()
        self.student = nn.ModuleDict({"backbone": nn.Linear(4, 4)})
        self.teacher = nn.ModuleDict({"backbone": nn.Linear(4, 4)})


def make_fake_trainer(save_path, iters_per_epoch=10, batch_size=48):
    logging.basicConfig(level=logging.INFO)
    return SimpleNamespace(
        model=SonataLikeModel(),
        cfg=SimpleNamespace(save_path=save_path, batch_size=batch_size),
        logger=logging.getLogger("test"),
        epoch=0,
        comm_info={},
        train_loader=[None] * iters_per_epoch,
        _gradient_accumulation_counter=0,
        optimizer=None,
        scheduler=None,
        scaler=None,
        best_metric_value=0.0,
    )


def main():
    tmpdir = tempfile.mkdtemp(prefix="snapshot_test_")
    try:
        trainer = make_fake_trainer(tmpdir, iters_per_epoch=10, batch_size=48)
        hook = IterCheckpointSaver(
            save_iter_freq=10_000,  # resumable saves out of the way
            snapshot_at_iters=[2, 4, 15],
            snapshot_freq=None,
        )
        hook.trainer = trainer

        # epoch 0: iters 0..9 -> global steps 1..10
        for it in range(10):
            trainer.comm_info["iter"] = it
            hook.after_step()
        # epoch 1: iters 0..9 -> global steps 11..20 (crosses an epoch boundary)
        trainer.epoch = 1
        for it in range(10):
            trainer.comm_info["iter"] = it
            hook.after_step()

        snaps = sorted(glob.glob(os.path.join(tmpdir, "snapshot", "*.pth")))
        names = [os.path.basename(s) for s in snaps]
        expected = [
            "snapshot_iter0000002_img96.pth",
            "snapshot_iter0000004_img192.pth",
            "snapshot_iter0000015_img720.pth",
        ]
        assert names == expected, f"unexpected snapshots: {names} != {expected}"
        print(f"PASS  scheduled snapshots written: {names}")

        ckpt = torch.load(snaps[-1], map_location="cpu", weights_only=False)
        assert set(ckpt.keys()) == {
            "epoch", "iter_in_epoch", "global_step", "images_seen",
            "batch_size", "state_dict",
        }, f"unexpected keys: {sorted(ckpt.keys())}"
        assert "optimizer" not in ckpt and "rng_state" not in ckpt
        assert ckpt["global_step"] == 15
        assert ckpt["images_seen"] == 15 * 48
        assert ckpt["epoch"] == 1 and ckpt["iter_in_epoch"] == 5
        print(f"PASS  weights-only content: global_step={ckpt['global_step']} "
              f"images_seen={ckpt['images_seen']}")

        # SonataCheckpointLoader compatibility: it reads ckpt['state_dict'],
        # strips 'module.', and prepends 'backbone.' unless keys already start
        # with it (misc.py SonataCheckpointLoader.before_train).
        sample_key = next(iter(ckpt["state_dict"].keys()))
        stripped = sample_key[7:] if sample_key.startswith("module.") else sample_key
        assert not stripped.startswith("backbone."), (
            "snapshot keys must look like sonata pretrain keys "
            f"(student./teacher.*), got {sample_key}"
        )
        remapped = OrderedDict(
            ("backbone." + (k[7:] if k.startswith("module.") else k), v)
            for k, v in ckpt["state_dict"].items()
        )
        assert any(k.startswith("backbone.student.backbone.") for k in remapped)
        assert any(k.startswith("backbone.teacher.backbone.") for k in remapped)
        print("PASS  SonataCheckpointLoader remap convention holds")

        # snapshot_freq mode
        shutil.rmtree(os.path.join(tmpdir, "snapshot"))
        trainer.epoch = 0
        hook2 = IterCheckpointSaver(save_iter_freq=10_000, snapshot_freq=5)
        hook2.trainer = trainer
        for it in range(10):
            trainer.comm_info["iter"] = it
            hook2.after_step()
        names = sorted(
            os.path.basename(s)
            for s in glob.glob(os.path.join(tmpdir, "snapshot", "*.pth"))
        )
        assert names == [
            "snapshot_iter0000005_img240.pth",
            "snapshot_iter0000010_img480.pth",
        ], f"unexpected freq snapshots: {names}"
        print(f"PASS  snapshot_freq mode: {names}")

        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
