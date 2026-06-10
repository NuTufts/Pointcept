"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import itertools
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import (
    build_dataset,
    point_collate_fn,
    collate_fn,
    FastForwardSampler,
)
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry


TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        # Mid-epoch resume offset: number of dataloader iters to skip at the
        # start of the first resumed epoch. Set by CheckpointLoader from the
        # checkpoint's ``iter_in_epoch`` field; consumed once by Trainer.train.
        self.start_iter = 0
        # Set by SignalCheckpointHook on SIGUSR1 to request a clean stop at
        # the next safe boundary. The trainer breaks out of both loops without
        # running after_epoch so the partial-epoch checkpoint isn't clobbered.
        self._stop_requested = False
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0
        self._stop_requested = False
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                # Mid-epoch resume path. Two strategies:
                #   - Fast-skip (cfg.skip_dataloader_on_resume=True):
                #       Activate FastForwardSampler on the train_loader so the
                #       underlying sampler yields only kept indices. Workers
                #       never run the skipped batches' transforms. Augmentations
                #       on kept batches differ from the un-resumed counterfactual
                #       (acceptable for SSL); see configs/.../v8 config docs.
                #   - islice (default):
                #       Drive the dataloader through start_iter batches and drop
                #       them. Slow (~hour at ~5k skipped batches with heavy
                #       SONATA transforms) but preserves the original sequence's
                #       augmentation state on the kept batches.
                if self.start_iter > 0 and self.start_iter >= len(self.train_loader):
                    self.logger.warning(
                        f"Mid-epoch resume start_iter={self.start_iter} >= "
                        f"len(train_loader)={len(self.train_loader)}; "
                        f"running resumed epoch {self.epoch} with no new iters "
                        f"(after_epoch hooks like eval will still fire)."
                    )
                    self.start_iter = 0
                    data_iter = iter([])
                elif self.start_iter > 0 and getattr(
                    self.cfg, "skip_dataloader_on_resume", False
                ):
                    sampler = self.train_loader.sampler
                    # Hard-fail loudly if the loader's sampler isn't wrapped.
                    # Without this check, setting `skip_indices` on a plain
                    # DistributedSampler silently creates a stray attribute,
                    # the full epoch runs, AND enumerate(start=N) inflates
                    # comm_info["iter"] past len(train_loader) — corrupting
                    # all iter-keyed accounting (we hit this with
                    # LArFormerTrainer / ShowerClusteringTrainer subclasses
                    # that overrode build_train_loader without wrapping).
                    if not isinstance(sampler, FastForwardSampler):
                        raise RuntimeError(
                            f"skip_dataloader_on_resume=True but train_loader.sampler "
                            f"is {type(sampler).__name__}, not FastForwardSampler. "
                            f"Custom build_train_loader overrides must wrap the "
                            f"underlying sampler via self._build_train_sampler(...). "
                            f"Set skip_dataloader_on_resume=False to fall back to islice."
                        )
                    skip_samples = self.start_iter * self.cfg.batch_size_per_gpu
                    sampler.skip_indices = skip_samples
                    sampler._consumed = False
                    self.logger.info(
                        f"Mid-epoch resume (fast-skip): jumping to iter "
                        f"{self.start_iter}/{len(self.train_loader)} of epoch "
                        f"{self.epoch} via FastForwardSampler "
                        f"(skip {skip_samples} samples per rank)"
                    )
                    data_iter = enumerate(self.train_loader, start=self.start_iter)
                    self.start_iter = 0
                elif self.start_iter > 0:
                    self.logger.info(
                        f"Mid-epoch resume (islice): skipping first "
                        f"{self.start_iter} iters of epoch {self.epoch} "
                        f"({len(self.train_loader) - self.start_iter} iters remain)"
                    )
                    data_iter = itertools.islice(
                        enumerate(self.train_loader), self.start_iter, None
                    )
                    self.start_iter = 0
                else:
                    data_iter = enumerate(self.train_loader)
                self.data_iterator = data_iter
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                    # SignalCheckpointHook sets this on SIGUSR1 after writing
                    # a mid-epoch checkpoint; break out of both loops without
                    # running after_epoch (CheckpointSaver.after_epoch would
                    # otherwise overwrite the partial-epoch state with iter 0).
                    if self._stop_requested:
                        break
                if self._stop_requested:
                    self.logger.info(
                        "Stop requested; exiting training loop before after_epoch."
                    )
                    break
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = (
                output_dict["loss"] / self.cfg.gradient_accumulation_steps
            )  # scale loss

        # NaN/Inf detection: log batch info and skip batch when loss goes bad
        if not torch.isfinite(loss):
            self.logger.error(
                f"NaN/Inf loss detected! "
                f"Epoch {self.comm_info.get('epoch', '?')}, "
                f"Iter {self.comm_info.get('iter', '?')}. "
                f"Skipping this batch to prevent crash."
            )
            # Log sample names if available
            if "name" in input_dict:
                self.logger.error(f"  Batch sample names: {input_dict['name']}")
            # Log which input tensors contain NaN/Inf
            for key, val in input_dict.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    has_nan = torch.isnan(val).any().item()
                    has_inf = torch.isinf(val).any().item()
                    if has_nan or has_inf:
                        self.logger.error(
                            f"  Input '{key}': shape={val.shape}, "
                            f"has_nan={has_nan}, has_inf={has_inf}, "
                            f"min={val[torch.isfinite(val)].min().item() if torch.isfinite(val).any() else 'N/A'}, "
                            f"max={val[torch.isfinite(val)].max().item() if torch.isfinite(val).any() else 'N/A'}"
                        )
            # Log which output tensors contain NaN/Inf
            for key, val in output_dict.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    has_nan = torch.isnan(val).any().item()
                    has_inf = torch.isinf(val).any().item()
                    if has_nan or has_inf:
                        self.logger.error(
                            f"  Output '{key}': shape={val.shape}, "
                            f"has_nan={has_nan}, has_inf={has_inf}"
                        )
            # Skip this batch entirely: clear accumulated grads and reset counter
            self.optimizer.zero_grad()
            self._gradient_accumulation_counter = 0
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()
            self.comm_info["model_output_dict"] = output_dict
            return

        # Backward pass
        if self.cfg.enable_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp and self.scaler is not None:
                # Unscale gradients FIRST so inf/nan checks are on true gradient magnitudes
                self.scaler.unscale_(self.optimizer)

                # Check for NaN/Inf gradients on unscaled gradients
                nan_grad_params = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            grad = param.grad
                            nan_count = torch.isnan(grad).sum().item()
                            inf_count = torch.isinf(grad).sum().item()
                            grad_finite = grad[torch.isfinite(grad)]
                            nan_grad_params.append(name)
                            self.logger.error(
                                f"NaN/Inf gradient in '{name}': "
                                f"shape={list(grad.shape)}, "
                                f"nan={nan_count}, inf={inf_count}, "
                                f"finite_abs_max={grad_finite.abs().max().item() if grad_finite.numel() > 0 else 'N/A'}"
                            )
                if nan_grad_params:
                    self.logger.error(
                        f"Found NaN/Inf gradients in {len(nan_grad_params)} params at "
                        f"Epoch {self.comm_info.get('epoch', '?')}, "
                        f"Iter {self.comm_info.get('iter', '?')}. "
                        f"GradScaler will skip optimizer step and adjust scale."
                    )
                    if "name" in input_dict:
                        self.logger.error(f"  Batch sample names: {input_dict['name']}")

                if self.cfg.clip_grad is not None:
                    # clip_grad_norm_ returns the PRE-clip total norm; expose
                    # it as a logged scalar (InformationWriter picks up every
                    # 0-d tensor in model_output_dict -> train_batch/grad_norm).
                    output_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    ).detach()
                # scaler.step automatically skips the optimizer step if inf/nan gradients are present
                self.scaler.step(self.optimizer)

                # scaler.update adjusts the scale factor (reduces it on overflow, grows it on success)
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                # AMP without GradScaler (bfloat16) or non-AMP path:
                # check for NaN/Inf gradients manually
                nan_grad_params = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            grad = param.grad
                            nan_count = torch.isnan(grad).sum().item()
                            inf_count = torch.isinf(grad).sum().item()
                            grad_finite = grad[torch.isfinite(grad)]
                            nan_grad_params.append(name)
                            self.logger.error(
                                f"NaN/Inf gradient in '{name}': "
                                f"shape={list(grad.shape)}, "
                                f"nan={nan_count}, inf={inf_count}, "
                                f"finite_abs_max={grad_finite.abs().max().item() if grad_finite.numel() > 0 else 'N/A'}"
                            )
                if nan_grad_params:
                    self.logger.error(
                        f"Found NaN/Inf gradients in {len(nan_grad_params)} params at "
                        f"Epoch {self.comm_info.get('epoch', '?')}, "
                        f"Iter {self.comm_info.get('iter', '?')}. "
                        f"Skipping optimizer step to prevent weight corruption."
                    )
                    if "name" in input_dict:
                        self.logger.error(f"  Batch sample names: {input_dict['name']}")
                    self.optimizer.zero_grad()
                else:
                    if self.cfg.clip_grad is not None:
                        # Pre-clip total norm, exposed as a logged scalar
                        # (see the AMP branch above).
                        output_dict["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.clip_grad
                        ).detach()
                    self.optimizer.step()
                    self.scheduler.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def _build_train_sampler(self, train_data):
        """Build the train sampler, always wrapped in FastForwardSampler.

        Subclasses with custom build_train_loader (LArFormerTrainer,
        ShowerClusteringTrainer) should call this so mid-epoch fast-resume
        works for them too. With skip_indices=0 the wrapper is transparent.
        """
        if comm.get_world_size() > 1:
            base_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            base_sampler = torch.utils.data.RandomSampler(train_data)
        return FastForwardSampler(base_sampler, skip_indices=0)

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        train_sampler = self._build_train_sampler(train_data)

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            # Sampler controls order, so DataLoader's own shuffle is off.
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_val_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.eval_epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        # GradScaler is only needed for float16 AMP. bfloat16 has the same
        # exponent range as float32, so loss scaling is unnecessary and can
        # cause a "death spiral" where repeated overflow events erode the
        # scale factor until gradients underflow and training produces NaN.
        amp_dtype = getattr(self.cfg, "amp_dtype", "float16")
        if self.cfg.enable_amp and amp_dtype == "float16":
            if version.parse(torch.__version__) >= version.parse("2.4"):
                grad_scaler = partial(torch.amp.GradScaler, device="cuda")
            else:
                grad_scaler = torch.cuda.amp.GradScaler
            return grad_scaler()
        return None


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
