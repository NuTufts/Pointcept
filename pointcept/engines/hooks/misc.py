"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import gc
import wandb
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)
        # Account for mid-epoch resume: skipped iters won't run.
        self._remain_iter -= getattr(self.trainer, "start_iter", 0)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        # Include mid-epoch start_iter so the global iter counter (used for
        # tensorboard/wandb x-axis) lines up across resumes.
        self.curr_iter = (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + getattr(self.trainer, "start_iter", 0)
        )
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="Iter")
            wandb.define_metric("train_batch/*", step_metric="Iter")
            wandb.define_metric("train/*", step_metric="Epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:

                wandb.log(
                    {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
                )
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Iter": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:

                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )


@HOOKS.register_module()
class LREpochScheduler(HookBase):
    """
    Drives an epoch-level LR scheduler (e.g. FlatWithDecayLR) once per
    epoch from inside ``after_epoch``. The trainer's iteration-level
    ``self.scheduler.step()`` still runs every iter; for FlatWithDecayLR
    that's a no-op.

    Reads ``val_loss`` out of ``trainer.comm_info`` (set by the
    evaluator). Quietly does nothing if the configured scheduler doesn't
    expose a ``step_epoch`` method, so it's safe to leave in place even
    when the scheduler config is swapped back to OneCycleLR.

    Register this hook AFTER the evaluator (so val_loss is available)
    and BEFORE CheckpointSaver (so the saved scheduler state reflects
    this epoch's decay).
    """

    def __init__(self, log_to_writer=True):
        self.log_to_writer = log_to_writer

    def before_train(self):
        # Log the actual starting LR (post-resume override, if any) so
        # there's a single line in the train log confirming what the
        # optimizer is using before iter 0.
        scheduler = getattr(self.trainer, "scheduler", None)
        if scheduler is None:
            return
        if not is_main_process():
            return
        lrs = [g["lr"] for g in self.trainer.optimizer.param_groups]
        reset_lr = getattr(scheduler, "_reset_lr", None)
        if reset_lr is not None:
            self.trainer.logger.info(
                f"LREpochScheduler: reset_lr={reset_lr:.3e} active — "
                f"optimizer param_groups now at lr={lrs}"
            )
        else:
            self.trainer.logger.info(
                f"LREpochScheduler: starting lr={lrs}"
            )

    def after_epoch(self):
        scheduler = getattr(self.trainer, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "step_epoch"):
            return
        val_loss = self.trainer.comm_info.get("val_loss", None)
        info = scheduler.step_epoch(val_loss)
        if not is_main_process():
            return

        lr0 = info["lrs"][0] if info["lrs"] else float("nan")
        if info.get("decayed"):
            self.trainer.logger.info(
                f"LREpochScheduler: decay triggered "
                f"(reason={info.get('reason')}); "
                f"new lr={lr0:.3e} num_decays={info.get('num_decays')}"
            )
        if self.log_to_writer:
            current_epoch = self.trainer.epoch + 1
            writer = getattr(self.trainer, "writer", None)
            if writer is not None:
                writer.add_scalar("train/lr_epoch", lr0, current_epoch)
                writer.add_scalar(
                    "train/epochs_since_improvement",
                    info.get("epochs_since_improvement", 0),
                    current_epoch,
                )
                writer.add_scalar(
                    "train/epochs_since_decay",
                    info.get("epochs_since_decay", 0),
                    current_epoch,
                )
            if getattr(self.trainer.cfg, "enable_wandb", False):
                try:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "train/lr_epoch": lr0,
                            "train/epochs_since_improvement":
                                info.get("epochs_since_improvement", 0),
                            "train/epochs_since_decay":
                                info.get("epochs_since_decay", 0),
                        },
                        step=wandb.run.step,
                    )
                except Exception:
                    pass


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.scaler is not None
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


def _save_resumable_checkpoint(trainer, iter_in_epoch, save_rng_state=True,
                               keep_history=False, log_prefix="[IterCheckpointSaver]"):
    """Atomically write a mid-epoch resumable checkpoint on the main process.

    Shared by :class:`IterCheckpointSaver` and :class:`SignalCheckpointHook`.
    Caller is responsible for the is_main_process() / gradient-accumulation
    guards — this helper just writes the file.
    """
    ckpt = {
        "epoch": trainer.epoch,  # current epoch (NOT +1)
        "iter_in_epoch": iter_in_epoch,
        "state_dict": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": (
            trainer.scaler.state_dict() if trainer.scaler is not None else None
        ),
        "best_metric_value": trainer.best_metric_value,
    }
    if save_rng_state:
        import random
        import numpy as np

        ckpt["rng_state"] = {
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    filename = os.path.join(trainer.cfg.save_path, "model", "model_last.pth")
    trainer.logger.info(
        f"{log_prefix} Saving mid-epoch checkpoint: "
        f"epoch={trainer.epoch} iter={iter_in_epoch}/"
        f"{len(trainer.train_loader)} -> {filename}"
    )
    torch.save(ckpt, filename + ".tmp")
    os.replace(filename + ".tmp", filename)
    if keep_history:
        global_step = trainer.epoch * len(trainer.train_loader) + iter_in_epoch
        shutil.copyfile(
            filename,
            os.path.join(
                trainer.cfg.save_path, "model", f"iter_{global_step}.pth"
            ),
        )


@HOOKS.register_module()
class IterCheckpointSaver(HookBase):
    """
    Save a resumable checkpoint every ``save_iter_freq`` dataloader iterations.

    Lets SLURM jobs that get killed mid-epoch resume from the last save instead
    of restarting the partial epoch. Writes to the same ``model_last.pth`` that
    :class:`CheckpointSaver` writes at epoch boundaries, with two extra fields
    so :class:`CheckpointLoader` can pick up mid-epoch:

      - ``iter_in_epoch``: the next dataloader iter to run on resume
      - ``rng_state``: (optional) torch / cuda / numpy / python RNG snapshots

    The ``epoch`` field is the **current** (in-progress) epoch, NOT ``epoch+1``,
    because the epoch hasn't finished yet.

    Args:
        save_iter_freq: save every N dataloader iterations.
        keep_history: if True, also copy each save to
            ``model/iter_{global_step}.pth``.
        save_rng_state: if True, snapshot torch / cuda / numpy / python RNG
            (rank-0 state only; restored on every rank at resume).
    """

    def __init__(self, save_iter_freq=500, keep_history=False, save_rng_state=True):
        self.save_iter_freq = save_iter_freq
        self.keep_history = keep_history
        self.save_rng_state = save_rng_state

    def after_step(self):
        if not is_main_process():
            return
        # Only save at optimizer-step boundaries so we don't lose partial
        # gradients on resume. With gradient_accumulation_steps=1 this is
        # always true after run_step completes.
        if getattr(self.trainer, "_gradient_accumulation_counter", 0) != 0:
            return
        iter_in_epoch = self.trainer.comm_info["iter"] + 1
        if iter_in_epoch % self.save_iter_freq != 0:
            return
        # Skip the very last iter of the epoch — CheckpointSaver.after_epoch
        # will write a cleaner record (epoch+1, iter 0) moments later.
        if iter_in_epoch >= len(self.trainer.train_loader):
            return

        _save_resumable_checkpoint(
            self.trainer,
            iter_in_epoch,
            save_rng_state=self.save_rng_state,
            keep_history=self.keep_history,
            log_prefix="[IterCheckpointSaver]",
        )


@HOOKS.register_module()
class SignalCheckpointHook(HookBase):
    """
    Catch SIGUSR1 (or configured signal) and save a resumable checkpoint at the
    next safe boundary, then request a clean stop.

    Designed for SLURM ``--signal=USR1@N``: every rank installs its own handler
    in ``before_train``; the handler just sets a flag (signal handlers can't
    safely do CUDA work). ``after_step`` does an all-reduce of the flag every
    ``check_every_n_iter`` iters so all ranks agree, saves on rank 0, writes a
    RESUBMIT marker (so the SLURM script can chain the next job), and sets
    ``trainer._stop_requested`` so the trainer loop exits without running
    ``after_epoch`` (which would overwrite the partial checkpoint).

    Args:
        signum: signal number to catch (default SIGUSR1).
        check_every_n_iter: how often to do the cross-rank flag check.
        save_rng_state: snapshot RNG state in the saved checkpoint.
        resubmit_marker_name: filename written to ``cfg.save_path`` on signal.
    """

    def __init__(self, signum=None, check_every_n_iter=10, save_rng_state=True,
                 resubmit_marker_name="RESUBMIT"):
        import signal as _sig

        self.signum = signum if signum is not None else _sig.SIGUSR1
        self.check_every_n_iter = max(int(check_every_n_iter), 1)
        self.save_rng_state = save_rng_state
        self.resubmit_marker_name = resubmit_marker_name
        self._flag = False

    def before_train(self):
        import signal as _sig

        _sig.signal(self.signum, self._handler)
        self.trainer.logger.info(
            f"[SignalCheckpointHook] Installed handler for signal {self.signum} "
            f"(rank {comm.get_rank()}); check_every_n_iter={self.check_every_n_iter}"
        )

    def _handler(self, signum, frame):
        # Minimal: just flip the flag. The actual save happens at after_step.
        self._flag = True

    def after_step(self):
        # Fast path: skip the cross-rank check most iters when no rank has
        # received a signal yet. Each rank checks its own flag without any
        # collective; only on the periodic boundary (or when this rank's flag
        # is set) do we synchronize.
        iter_in_epoch = self.trainer.comm_info["iter"] + 1
        if iter_in_epoch % self.check_every_n_iter != 0 and not self._flag:
            return

        # Cross-rank agreement: any rank's flag triggers the stop on all ranks.
        local_flag = 1 if self._flag else 0
        if comm.get_world_size() > 1:
            t = torch.tensor(local_flag, device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            global_flag = int(t.item())
        else:
            global_flag = local_flag
        if global_flag == 0:
            return

        # Defer the save when mid-accumulation so we don't drop partial grads.
        # On the next call the flag is still set, the all-reduce repeats, and
        # we try again. With gradient_accumulation_steps=1 this never trips.
        if getattr(self.trainer, "_gradient_accumulation_counter", 0) != 0:
            return

        if is_main_process():
            self.trainer.logger.info(
                f"[SignalCheckpointHook] Signal received; saving and stopping "
                f"at epoch={self.trainer.epoch} iter={iter_in_epoch}"
            )
            _save_resumable_checkpoint(
                self.trainer,
                iter_in_epoch,
                save_rng_state=self.save_rng_state,
                keep_history=False,
                log_prefix="[SignalCheckpointHook]",
            )
            marker_path = os.path.join(
                self.trainer.cfg.save_path, self.resubmit_marker_name
            )
            with open(marker_path, "w") as f:
                f.write(
                    f"epoch={self.trainer.epoch} iter_in_epoch={iter_in_epoch}\n"
                )
            self.trainer.logger.info(
                f"[SignalCheckpointHook] Wrote resubmit marker: {marker_path}"
            )
        comm.synchronize()
        self.trainer._stop_requested = True


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False,
                 extend_scheduler=False, reset_optimizer=False):
        """
        Args:
            keywords: Keywords to match in layer names for weight loading
            replacement: Replacement string for matched keywords
            strict: Whether to require strict weight matching
            extend_scheduler: If True, when resuming, do NOT load the scheduler state.
                Instead, use the new scheduler (with potentially extended total_steps)
                and advance it to the current step. This allows extending training
                beyond the original epoch count while preserving optimizer state.
            reset_optimizer: If True, when resuming, do NOT load the optimizer state
                (Adam first/second moments, step count). Model weights, scheduler,
                epoch counter, best metric, and RNG all still restore. Use this to
                recover when train loss starts rising mid-run and the suspect is
                that Adam's `v_t` is mis-scaled (e.g. after a mid-run LR cut).
                Expect a brief transient (~500-1000 iters) where per-param steps
                are larger than usual due to v_t=0 + bias correction restart.
                REMEMBER TO REMOVE THIS FLAG ONCE THE NEXT CHECKPOINT SAVES,
                otherwise every subsequent resume keeps wiping Adam moments.
        """
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict
        self.extend_scheduler = extend_scheduler
        self.reset_optimizer = reset_optimizer

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                # Mid-epoch resume: IterCheckpointSaver writes iter_in_epoch
                # (the next dataloader iter to run) while CheckpointSaver does
                # not set it, so it defaults to 0 = clean epoch boundary.
                self.trainer.start_iter = checkpoint.get("iter_in_epoch", 0)
                if self.trainer.start_iter > 0:
                    self.trainer.logger.info(
                        f"Mid-epoch resume: continuing at iter "
                        f"{self.trainer.start_iter} of epoch {self.trainer.start_epoch}"
                    )
                    _maybe_reseed_workers_for_resume(self.trainer, checkpoint)
                self.trainer.best_metric_value = checkpoint["best_metric_value"]

                if self.extend_scheduler:
                    # CRITICAL: Save scheduler-related values from param_groups BEFORE loading
                    # optimizer state, because load_state_dict will overwrite them with old values.
                    scheduler = self.trainer.scheduler
                    saved_scheduler_params = []
                    for idx, group in enumerate(self.trainer.optimizer.param_groups):
                        saved_scheduler_params.append({
                            'initial_lr': group.get('initial_lr'),
                            'max_lr': group.get('max_lr'),
                        })

                    # Now load optimizer state (this overwrites initial_lr and max_lr in param_groups)
                    if not self.reset_optimizer:
                        self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    else:
                        self.trainer.logger.info(
                            "=> reset_optimizer=True: skipping optimizer state load "
                            "(Adam moments start at zero; expect ~500-1000 iter transient)"
                        )

                    # Do NOT load old scheduler state - use new scheduler with extended total_steps
                    # Set the scheduler's step counter to where we left off.
                    # Include iter_in_epoch so mid-epoch checkpoints written by
                    # IterCheckpointSaver fast-forward to the TRUE optimizer-step
                    # count (epoch-boundary checkpoints have iter_in_epoch=0, so
                    # the original behavior is unchanged for them).
                    steps_completed = (
                        checkpoint["epoch"] * len(self.trainer.train_loader)
                        + checkpoint.get("iter_in_epoch", 0)
                    ) // self.trainer.cfg.gradient_accumulation_steps
                    self.trainer.logger.info(
                        f"Extending scheduler: setting step counter to {steps_completed} "
                        f"(new total_steps: {self.trainer.cfg.scheduler.total_steps})"
                    )

                    # Restore the NEW scheduler's initial_lr and max_lr values that were saved
                    for idx, group in enumerate(self.trainer.optimizer.param_groups):
                        if idx < len(saved_scheduler_params):
                            if saved_scheduler_params[idx]['initial_lr'] is not None:
                                group['initial_lr'] = saved_scheduler_params[idx]['initial_lr']
                            if saved_scheduler_params[idx]['max_lr'] is not None:
                                group['max_lr'] = saved_scheduler_params[idx]['max_lr']
                        self.trainer.logger.info(
                            f"  Param group {idx}: initial_lr={group.get('initial_lr', 'N/A'):.6f}, "
                            f"max_lr={group.get('max_lr', 'N/A'):.6f}"
                        )

                    # Directly modify scheduler state to skip ahead efficiently
                    # This is much faster than calling step() 300k+ times
                    state = scheduler.state_dict()
                    state['last_epoch'] = steps_completed
                    state['_step_count'] = steps_completed + 1
                    scheduler.load_state_dict(state)

                    # Log the LR we'll start with
                    current_lr = scheduler.get_last_lr()[0]
                    self.trainer.logger.info(f"  Extended scheduler LR at resume: {current_lr:.6f}")
                else:
                    if not self.reset_optimizer:
                        self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    else:
                        self.trainer.logger.info(
                            "=> reset_optimizer=True: skipping optimizer state load "
                            "(Adam moments start at zero; expect ~500-1000 iter transient)"
                        )
                    self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])

                if self.trainer.scaler is not None and checkpoint.get("scaler") is not None:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])

                # Restore RNG state if IterCheckpointSaver saved it.
                _restore_rng_state(checkpoint.get("rng_state"), self.trainer.logger)
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


def _maybe_reseed_workers_for_resume(trainer, checkpoint):
    """When mid-epoch resume uses the fast-skip path (FastForwardSampler), the
    DataLoader workers never advance through the skipped batches' RNG, so the
    kept batches get transforms drawn from the workers' INITIAL RNG state — the
    same draws they would have made on the first batches of a fresh epoch.

    When ``cfg.resume_seed_strategy == "per_resume"``, swap the train_loader's
    ``worker_init_fn`` to a fresh seed derived from the checkpoint's epoch and
    iter_in_epoch so multiple resumes from the same checkpoint don't keep
    reproducing the same augmentation patterns. With ``persistent_workers=True``
    workers spawn on the first ``iter(loader)`` call (which happens later in
    ``Trainer.train``), so swapping ``worker_init_fn`` here is in time.
    """
    cfg = trainer.cfg
    strategy = getattr(cfg, "resume_seed_strategy", "deterministic")
    if strategy != "per_resume":
        return
    if not getattr(cfg, "skip_dataloader_on_resume", False):
        trainer.logger.info(
            "resume_seed_strategy=per_resume has no effect when "
            "skip_dataloader_on_resume=False (islice already advances worker RNG)."
        )
        return
    if cfg.seed is None:
        trainer.logger.warning(
            "resume_seed_strategy=per_resume requested but cfg.seed is None; "
            "cannot derive a new seed. Leaving worker_init_fn unchanged."
        )
        return
    from functools import partial
    from pointcept.engines.defaults import worker_init_fn as _worker_init_fn

    new_seed = (
        cfg.seed
        + 1_000_003 * int(checkpoint.get("epoch", 0))
        + int(checkpoint.get("iter_in_epoch", 0))
    )
    trainer.train_loader.worker_init_fn = partial(
        _worker_init_fn,
        num_workers=cfg.num_worker_per_gpu,
        rank=comm.get_rank(),
        seed=new_seed,
    )
    trainer.logger.info(
        f"resume_seed_strategy=per_resume: swapped worker_init_fn to use "
        f"derived seed {new_seed} (base {cfg.seed} + epoch/iter offset)."
    )


def _restore_rng_state(rng_state, logger):
    """Restore torch / cuda / numpy / python RNG state saved by IterCheckpointSaver."""
    if rng_state is None:
        return
    import random
    import numpy as np

    try:
        torch.set_rng_state(rng_state["torch"])
        if (
            torch.cuda.is_available()
            and rng_state.get("torch_cuda") is not None
        ):
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])
        logger.info("Restored RNG state (torch / cuda / numpy / python) from checkpoint")
    except Exception as e:
        logger.warning(f"Failed to restore RNG state: {e}")


@HOOKS.register_module()
class SonataCheckpointLoader(HookBase):
    """
    Checkpoint loader for SonataSegmentor that properly remaps SONATA checkpoint keys.

    SONATA checkpoints have keys like:
        student.backbone.embedding...
        teacher.backbone.embedding...

    SonataSegmentor expects keys like:
        backbone.student.backbone.embedding...
        backbone.teacher.backbone.embedding...

    This hook prepends "backbone." to all checkpoint keys.
    """
    def __init__(self, strict=False):
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading SONATA checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            # Detect whether the checkpoint's keys already match the model
            # layout. SONATA *pretrain* checkpoints have top-level keys
            # like 'student.backbone.embedding...' and need 'backbone.'
            # prepended. Resume checkpoints saved by CheckpointSaver during
            # this training have already been remapped, so prepending again
            # would produce 'backbone.backbone.student....' (the bug we hit).
            sample_key = next(iter(checkpoint["state_dict"].keys()), "")
            sample_stripped = (
                sample_key[7:] if sample_key.startswith("module.") else sample_key
            )
            already_remapped = sample_stripped.startswith("backbone.")

            weight = OrderedDict()
            if already_remapped:
                self.trainer.logger.info(
                    "Checkpoint keys already prefixed with 'backbone.' "
                    "(resume checkpoint) — skipping SONATA prepend remap"
                )
                for key, value in checkpoint["state_dict"].items():
                    if key.startswith("module."):
                        key = key[7:]
                    new_key = "module." + key if comm.get_world_size() > 1 else key
                    weight[new_key] = value
            else:
                self.trainer.logger.info(
                    "Remapping SONATA checkpoint keys (prepending 'backbone.')"
                )
                for key, value in checkpoint["state_dict"].items():
                    if key.startswith("module."):
                        key = key[7:]
                    new_key = "backbone." + key
                    if comm.get_world_size() > 1:
                        new_key = "module." + new_key
                    weight[new_key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.start_iter = checkpoint.get("iter_in_epoch", 0)
                if self.trainer.start_iter > 0:
                    self.trainer.logger.info(
                        f"Mid-epoch resume: continuing at iter "
                        f"{self.trainer.start_iter} of epoch {self.trainer.start_epoch}"
                    )
                    _maybe_reseed_workers_for_resume(self.trainer, checkpoint)
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.scaler is not None and checkpoint.get("scaler") is not None:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
                _restore_rng_state(checkpoint.get("rng_state"), self.trainer.logger)
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class SonataFinetuneCheckpointLoader(HookBase):
    """
    Checkpoint loader for fine-tuning DefaultSegmentorV2 from SONATA pretrained weights.

    SONATA checkpoints have keys like:
        student.backbone.embedding.stem.linear.weight
        student.backbone.enc.enc0.block0.cpe.0.weight
        teacher.backbone.embedding...  (ignored)

    DefaultSegmentorV2 expects keys like:
        backbone.embedding.stem.linear.weight
        backbone.enc.enc0.block0.cpe.0.weight

    This hook extracts student.backbone.* keys and maps them to backbone.*
    The decoder and seg_head are randomly initialized (not in SONATA checkpoint).
    """
    def __init__(self, strict=False, use_teacher=False):
        self.strict = strict
        self.use_teacher = use_teacher  # If True, load from teacher instead of student

    def before_train(self):
        self.trainer.logger.info("=> Loading SONATA checkpoint for fine-tuning ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )

            source = "teacher" if self.use_teacher else "student"
            prefix = f"{source}.backbone."
            self.trainer.logger.info(
                f"Remapping SONATA checkpoint: '{prefix}*' -> 'backbone.*'"
            )

            weight = OrderedDict()
            loaded_count = 0
            skipped_count = 0

            for key, value in checkpoint["state_dict"].items():
                # Remove module. prefix if present (from DDP)
                if key.startswith("module."):
                    key = key[7:]

                # Only load student.backbone.* (or teacher.backbone.*)
                if key.startswith(prefix):
                    # Map student.backbone.* -> backbone.*
                    new_key = "backbone." + key[len(prefix):]

                    # Skip mask_token - not used in fine-tuning
                    if "mask_token" in new_key:
                        skipped_count += 1
                        continue

                    # Add module. back if using DDP
                    if comm.get_world_size() > 1:
                        new_key = "module." + new_key
                    weight[new_key] = value
                    loaded_count += 1
                else:
                    skipped_count += 1

            self.trainer.logger.info(
                f"Loaded {loaded_count} weights from {source}.backbone.*, "
                f"skipped {skipped_count} (teacher/head/mask_token)"
            )

            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys (expected for decoder/head): {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")

            # Don't resume optimizer/scheduler for fine-tuning (different model)
            if self.trainer.cfg.resume:
                self.trainer.logger.warning(
                    "resume=True but not loading optimizer/scheduler state "
                    "(fine-tuning uses different architecture)"
                )
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        from pointcept.engines.test import TESTERS

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            tester.model.load_state_dict(weight, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
        extend_scheduler=False,
        original_total_steps=None,
    ):
        """
        Weight decay scheduler with cosine annealing.

        Args:
            base_value: Starting weight decay value
            final_value: Final weight decay value
            extend_scheduler: If True, compute the WD value that the ORIGINAL schedule
                would have at the resume point, and start the extended schedule from there.
                Requires original_total_steps to be set.
            original_total_steps: Total steps of the original training schedule.
                Required when extend_scheduler=True. Can be computed as:
                original_epochs * steps_per_epoch // gradient_accumulation_steps
        """
        self.base_value = base_value
        self.final_value = final_value
        self.extend_scheduler = extend_scheduler
        self.original_total_steps = original_total_steps
        self.scheduler = None

    def before_train(self):
        # Include mid-epoch start_iter: the WD schedule advances once per
        # before_step, so the iter pointer equals total #before_step calls so
        # far = start_epoch * len(loader) + start_iter.
        curr_step = (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + getattr(self.trainer, "start_iter", 0)
        )
        new_total_steps = self.trainer.cfg.scheduler.total_steps

        if self.extend_scheduler and self.original_total_steps:
            # Compute what the WD was at curr_step under the ORIGINAL schedule
            import numpy as np
            original_progress = min(curr_step / self.original_total_steps, 1.0)
            original_wd = self.final_value + 0.5 * (self.base_value - self.final_value) * (
                1 + np.cos(np.pi * original_progress)
            )
            self.trainer.logger.info(
                f"WeightDecaySchedular: Extending from original schedule"
            )
            self.trainer.logger.info(
                f"  Original WD at step {curr_step}: {original_wd:.6f}"
            )
            self.trainer.logger.info(
                f"  (Original schedule: {self.original_total_steps} steps, "
                f"base={self.base_value}, final={self.final_value})"
            )

            # Create extended schedule that starts at original_wd and goes to final_value
            remaining_steps = new_total_steps - curr_step
            self.scheduler = CosineScheduler(
                base_value=original_wd,
                final_value=self.final_value,
                total_iters=remaining_steps,
            )
            self.scheduler.iter = 0  # Start fresh from 0 for remaining steps
            self.trainer.logger.info(
                f"  Extended WD schedule: {original_wd:.6f} -> {self.final_value} "
                f"over {remaining_steps} remaining steps"
            )
        else:
            # Standard behavior: cosine from base to final over total steps
            self.scheduler = CosineScheduler(
                base_value=self.base_value,
                final_value=self.final_value,
                total_iters=new_total_steps,
            )
            self.scheduler.iter = curr_step
            current_wd = self.scheduler.get(curr_step)
            self.trainer.logger.info(
                f"WeightDecaySchedular: WD at step {curr_step}: {current_wd:.6f}"
            )

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()


@HOOKS.register_module()
class LossWeightScheduler(HookBase):
    """
    Schedules loss weights during training with linear interpolation.

    This hook modifies the loss_weight attribute of loss functions in the
    model's criteria list during training. Weights transition linearly from
    initial to final values over a configurable fraction of training.

    Args:
        loss_weights: List of dicts specifying weight schedules, each containing:
            - loss_index (int): Index in model.criteria.criteria list
            - initial_weight (float): Starting weight value
            - final_weight (float): Ending weight value
        transition_fraction (float): Fraction of training at which transition
            completes (0.0 to 1.0). After this point, weights stay at final values.
            Default: 1.0 (transition over entire training)
        schedule_type (str): Granularity of weight updates:
            - "epoch": Update at epoch boundaries (default, more stable)
            - "step": Update every step (smoother transitions)

    Example config:
        dict(
            type="LossWeightScheduler",
            loss_weights=[
                dict(loss_index=0, initial_weight=1.0, final_weight=0.0),  # FocalLoss
                dict(loss_index=1, initial_weight=0.0, final_weight=1.0),  # LovaszLoss
            ],
            transition_fraction=0.5,  # Complete transition at 50% of training
        )
    """

    def __init__(self, loss_weights, transition_fraction=1.0, schedule_type="epoch"):
        self.loss_weights = loss_weights
        self.transition_fraction = transition_fraction
        self.schedule_type = schedule_type
        self.criteria = None
        self.total_epochs = None
        self.total_steps = None
        self.transition_epochs = None
        self.transition_steps = None
        self.current_epoch = 0
        self.current_step = 0
        self.step_log_interval = 100  # Log every N steps when schedule_type="step"

    def _get_model_criteria(self):
        """Get the criteria list from the model, handling DDP wrapping."""
        model = self.trainer.model
        if comm.get_world_size() > 1:
            model = model.module
        return model.criteria.criteria

    def _compute_weights(self, progress):
        """Compute interpolated weights given progress in [0, 1]."""
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))
        weights = {}
        for spec in self.loss_weights:
            idx = spec["loss_index"]
            initial = spec["initial_weight"]
            final = spec["final_weight"]
            # Linear interpolation
            weight = initial + (final - initial) * progress
            weights[idx] = weight
        return weights

    def _apply_weights(self, weights):
        """Apply weights to the criteria list."""
        for idx, weight in weights.items():
            if idx < len(self.criteria):
                self.criteria[idx].loss_weight = weight

    def _log_weights(self, weights, step_for_logging):
        """Log current weights to tensorboard/wandb."""
        if self.trainer.writer is not None:
            for idx, weight in weights.items():
                self.trainer.writer.add_scalar(
                    f"params/loss_weight_{idx}", weight, step_for_logging
                )
            if self.trainer.cfg.enable_wandb:
                log_dict = {"Iter": step_for_logging}
                for idx, weight in weights.items():
                    log_dict[f"params/loss_weight_{idx}"] = weight
                wandb.log(log_dict, step=wandb.run.step)

    def before_train(self):
        """Initialize scheduler parameters and get reference to criteria."""
        self.criteria = self._get_model_criteria()
        self.total_epochs = self.trainer.max_epoch
        steps_per_epoch = len(self.trainer.train_loader)
        self.total_steps = self.total_epochs * steps_per_epoch

        # Compute transition points
        self.transition_epochs = int(self.total_epochs * self.transition_fraction)
        self.transition_steps = int(self.total_steps * self.transition_fraction)

        # Handle resume: set current epoch/step from trainer
        self.current_epoch = self.trainer.start_epoch
        self.current_step = self.current_epoch * steps_per_epoch

        self.trainer.logger.info(
            f"LossWeightScheduler: schedule_type={self.schedule_type}, "
            f"transition_fraction={self.transition_fraction}"
        )
        self.trainer.logger.info(
            f"  Total epochs: {self.total_epochs}, transition at epoch {self.transition_epochs}"
        )
        self.trainer.logger.info(
            f"  Total steps: {self.total_steps}, transition at step {self.transition_steps}"
        )

        # Log initial weight configuration
        for spec in self.loss_weights:
            idx = spec["loss_index"]
            self.trainer.logger.info(
                f"  Loss[{idx}]: {spec['initial_weight']:.3f} -> {spec['final_weight']:.3f}"
            )

        # Apply initial weights
        if self.schedule_type == "epoch":
            progress = self.current_epoch / self.transition_epochs if self.transition_epochs > 0 else 1.0
        else:
            progress = self.current_step / self.transition_steps if self.transition_steps > 0 else 1.0
        weights = self._compute_weights(progress)
        self._apply_weights(weights)

        # Log current weights after resume
        self.trainer.logger.info(
            f"  Initial weights at epoch {self.current_epoch}: {weights}"
        )

    def before_epoch(self):
        """Update weights at epoch boundary if using epoch-based scheduling."""
        if self.schedule_type == "epoch":
            progress = self.current_epoch / self.transition_epochs if self.transition_epochs > 0 else 1.0
            weights = self._compute_weights(progress)
            self._apply_weights(weights)

            # Log to tensorboard/wandb (use step count for x-axis consistency)
            step_for_logging = self.current_step
            self._log_weights(weights, step_for_logging)

            self.trainer.logger.info(
                f"LossWeightScheduler: epoch {self.current_epoch}, weights={weights}"
            )

    def before_step(self):
        """Update weights at step boundary if using step-based scheduling."""
        if self.schedule_type == "step":
            progress = self.current_step / self.transition_steps if self.transition_steps > 0 else 1.0
            weights = self._compute_weights(progress)
            self._apply_weights(weights)

            # Log periodically to avoid log spam
            if self.current_step % self.step_log_interval == 0:
                self._log_weights(weights, self.current_step)

    def after_step(self):
        """Increment step counter."""
        self.current_step += 1

    def after_epoch(self):
        """Increment epoch counter."""
        self.current_epoch += 1
