import os
import torch
from collections import OrderedDict
from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase
import pointcept.utils.comm as comm

# NOTE: copy_teacher_weights_to_student is imported lazily inside before_train()
# to avoid a circular import at module load time:
#   models/__init__.py → models/modules.py → engines/hooks/__init__.py
#   → lora_checkpoint_hook.py → models/sonata/__init__.py → models/modules.py  ← CYCLE


@HOOKS.register_module()
class LoRASonataCheckpointLoader(HookBase):
    """
    Checkpoint loader for SonataLoRASegmentorST.

    Identical key remapping to SonataCheckpointLoader:
    prepends 'backbone.' to all keys from the SONATA pretraining checkpoint.
    LoRA A/B and seg_head keys will be missing (expected) and reported.

    Teacher → Student weight initialisation.
    """

    def __init__(self, pretrained_path=None, strict=False):
        self.pretrained_path = pretrained_path
        self.strict = strict

    def before_train(self):
        # Deferred import — must not sit at module level (circular import).
        from pointcept.models.sonata.lora_sonata_student_teacher import (
            copy_teacher_weights_to_student,
        )
        self.trainer.logger.info("=> Loading SONATA checkpoint for LoRA fine-tuning ...")
        path = self.pretrained_path or self.trainer.cfg.weight
        if path and os.path.isfile(path):
            self.trainer.logger.info(f"Loading weight at: {path}")
            checkpoint = torch.load(
                path,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                "Remapping SONATA checkpoint keys (prepending 'backbone.')"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                # Strip DDP prefix if present
                if key.startswith("module."):
                    key = key[7:]
                # Prepend backbone. for SonataLoRASegmentorST
                new_key = "backbone." + key
                # Re-add module. prefix if running DDP
                if comm.get_world_size() > 1:
                    new_key = "module." + new_key
                weight[new_key] = value

            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            missing    = load_state_info[0]
            unexpected = load_state_info[1]
            lora_missing  = [k for k in missing if "lora_" in k]
            head_missing  = [k for k in missing if "seg_head" in k]
            other_missing = [k for k in missing
                             if k not in lora_missing and k not in head_missing]

            self.trainer.logger.info(
                f"LoRA adapter keys (new, expected missing): {len(lora_missing)}"
            )
            self.trainer.logger.info(
                f"seg_head keys (new, expected missing): {len(head_missing)}"
            )
            if other_missing:
                self.trainer.logger.warning(
                    f"Other missing keys: {other_missing[:10]}"
                )
            self.trainer.logger.info(
                f"Unexpected keys: {unexpected[:10] if unexpected else 'none'}"
            )

            # ----------------------------------------------------------------
            # Teacher → Student weight copy (Option 2 experiment)
            # ----------------------------------------------------------------
            # Unwrap DDP to reach the real model
            raw_model = (
                self.trainer.model.module
                if hasattr(self.trainer.model, "module")
                else self.trainer.model
            )
            if getattr(raw_model, "use_teacher_weights_init", False):
                self.trainer.logger.info(
                    "use_teacher_weights_init=True detected — copying teacher "
                    "weights into student encoder (LoRA adapters preserved)..."
                )
                n_copied = copy_teacher_weights_to_student(
                    backbone=raw_model.backbone,
                    logger=self.trainer.logger,
                )
                self.trainer.logger.info(
                    f"Teacher→Student copy complete: {n_copied} tensors."
                )
            else:
                self.trainer.logger.info(
                    "use_teacher_weights_init=False — student weights unchanged."
                )
            # ----------------------------------------------------------------

            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {path}")
