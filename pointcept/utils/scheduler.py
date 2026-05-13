"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[int(rate * total_steps) for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        power=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        gamma=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        eta_min=0,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class FlatWithDecayLR(lr_scheduler._LRScheduler):
    """
    Flat learning rate with explicit, epoch-driven decays.

    Per-iteration ``step()`` is a no-op — the LR stays at whatever
    ``param_groups[i]['lr']`` was last set to. Decays happen only when
    ``step_epoch(val_loss)`` is called once per epoch (by the
    ``LREpochScheduler`` hook).

    Two decay triggers, selected by ``mode``:
      - ``"epoch"``   : multiply LR by ``gamma`` every ``step_period_epochs``.
      - ``"plateau"`` : multiply LR by ``gamma`` after ``patience_epochs``
                        epochs with no val-loss improvement greater than
                        ``min_delta``. After each plateau decay, a
                        ``cooldown_epochs`` window suppresses further
                        plateau triggers.
      - ``"both"``    : either trigger fires (whichever comes first); each
                        keeps its own counter.

    LR floor is ``min_lr``; further decays are skipped once the floor is
    hit (also resets the relevant counters so we don't keep flapping).

    Constructor accepts (and ignores) ``total_steps`` because Pointcept's
    ``Trainer.build_scheduler`` injects it for the OneCycle/Cosine families.

    Resume note: the OneCycleLR-aware ``extend_scheduler=True`` path in
    ``CheckpointLoader`` rewrites ``initial_lr`` / ``max_lr`` and is NOT
    appropriate here. When resuming a FlatWithDecayLR run, leave
    ``extend_scheduler=False`` so ``scheduler.load_state_dict`` restores
    counters and ``best_val_loss`` exactly.
    """

    def __init__(
        self,
        optimizer,
        mode="both",
        gamma=0.5,
        min_lr=1e-7,
        step_period_epochs=20,
        patience_epochs=8,
        min_delta=1e-4,
        cooldown_epochs=2,
        reset_lr=None,
        reset_counters=False,
        total_steps=None,  # accepted for trainer compatibility, unused
        last_epoch=-1,
    ):
        if mode not in ("epoch", "plateau", "both"):
            raise ValueError(
                f"FlatWithDecayLR: mode must be one of "
                f"'epoch'|'plateau'|'both', got {mode!r}"
            )
        self.mode = mode
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)
        self.step_period_epochs = int(step_period_epochs)
        self.patience_epochs = int(patience_epochs)
        self.min_delta = float(min_delta)
        self.cooldown_epochs = int(cooldown_epochs)
        # On resume, `load_state_dict` will override every param_group's
        # `lr` with this value (if not None). Use this to manually drop
        # the LR below whatever was saved when continuing a run. The
        # override is applied every time `load_state_dict` is called, so
        # remove it from the config (or set to None) after the resume run
        # has produced its first checkpoint, otherwise subsequent resumes
        # will keep stomping the LR back to this value.
        self._reset_lr = None if reset_lr is None else float(reset_lr)
        # If True, also wipe plateau/period counters and best_val_loss on
        # load — useful when reset_lr changes the loss landscape enough
        # that the old "best" is no longer comparable.
        self._reset_counters = bool(reset_counters)

        self.best_val_loss = float("inf")
        self.epochs_since_decay = 0
        self.epochs_since_improvement = 0
        self.cooldown_remaining = 0
        self.num_decays = 0

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # No iteration-driven schedule: hold whatever's currently set on
        # each param group. Returning the live values makes
        # `get_last_lr()` consistent with what the optimizer is using.
        return [group["lr"] for group in self.optimizer.param_groups]

    def _apply_decay(self, reason):
        new_lrs = []
        any_above_floor = False
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.gamma, self.min_lr)
            if new_lr < group["lr"]:
                any_above_floor = True
            group["lr"] = new_lr
            new_lrs.append(new_lr)
        if any_above_floor:
            self.num_decays += 1
        return new_lrs, any_above_floor, reason

    def step_epoch(self, val_loss=None):
        """
        Call once at the end of each epoch. Returns a dict describing
        whether a decay happened and the new LRs (handy for logging).
        """
        decayed = False
        reason = None

        # plateau bookkeeping (only meaningful when val_loss provided)
        if val_loss is not None and val_loss == val_loss:  # not NaN
            improved = val_loss < (self.best_val_loss - self.min_delta)
            if improved:
                self.best_val_loss = float(val_loss)
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
        # epoch-period bookkeeping always advances
        self.epochs_since_decay += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # epoch-period trigger
        if (
            self.mode in ("epoch", "both")
            and self.step_period_epochs > 0
            and self.epochs_since_decay >= self.step_period_epochs
        ):
            _, did, _ = self._apply_decay("epoch_period")
            self.epochs_since_decay = 0
            if did:
                decayed = True
                reason = "epoch_period"

        # plateau trigger (independent counter; can fire same epoch as
        # the period trigger but we only count it as one decay event)
        if (
            not decayed
            and self.mode in ("plateau", "both")
            and self.cooldown_remaining == 0
            and val_loss is not None
            and val_loss == val_loss
            and self.patience_epochs > 0
            and self.epochs_since_improvement >= self.patience_epochs
        ):
            _, did, _ = self._apply_decay("plateau")
            self.epochs_since_improvement = 0
            self.cooldown_remaining = self.cooldown_epochs
            if did:
                decayed = True
                reason = "plateau"

        return dict(
            decayed=decayed,
            reason=reason,
            lrs=[g["lr"] for g in self.optimizer.param_groups],
            best_val_loss=self.best_val_loss,
            epochs_since_decay=self.epochs_since_decay,
            epochs_since_improvement=self.epochs_since_improvement,
            cooldown_remaining=self.cooldown_remaining,
            num_decays=self.num_decays,
        )

    # Two sets of attributes:
    #   _CONFIG_ONLY_KEYS  — tuning knobs that always come from the
    #     CURRENT config. They're constructor args, not runtime state.
    #     If we persisted these in the checkpoint, _LRScheduler's
    #     `self.__dict__.update(state_dict)` would clobber the freshly-
    #     constructed value on resume — making it impossible to change
    #     e.g. step_period_epochs from one resume to the next.
    #   _STATEFUL_KEYS — true runtime state that MUST survive resumes
    #     (counters, best-so-far, etc).
    _CONFIG_ONLY_KEYS = (
        "mode",
        "gamma",
        "min_lr",
        "step_period_epochs",
        "patience_epochs",
        "min_delta",
        "cooldown_epochs",
        "_reset_lr",
        "_reset_counters",
    )
    _STATEFUL_KEYS = (
        "best_val_loss",
        "epochs_since_decay",
        "epochs_since_improvement",
        "cooldown_remaining",
        "num_decays",
    )

    def state_dict(self):
        state = super().state_dict()
        # Strip config-only keys so they don't ride in the checkpoint
        # and override what the user puts in the config on a later resume.
        for key in self._CONFIG_ONLY_KEYS:
            state.pop(key, None)
        for key in self._STATEFUL_KEYS:
            state[key] = getattr(self, key)
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        # Drop config-only keys from incoming state_dict before super
        # restores. Critical for OLDER checkpoints written before this
        # filter existed — they still carry baked-in values for
        # step_period_epochs / patience_epochs / etc. that would
        # otherwise overwrite the current config.
        for key in self._CONFIG_ONLY_KEYS:
            state_dict.pop(key, None)
        for key in self._STATEFUL_KEYS:
            if key in state_dict:
                setattr(self, key, state_dict.pop(key))
        super().load_state_dict(state_dict)

        # Apply manual reset_lr override from current config. Runs AFTER
        # the parent restores last_epoch/_step_count, and AFTER the
        # caller has already done optimizer.load_state_dict(), so this is
        # the final word on what the optimizer's param_groups carry.
        if self._reset_lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = self._reset_lr
            # Keep get_last_lr() / future state_dicts consistent.
            self._last_lr = [self._reset_lr for _ in self.optimizer.param_groups]
        if self._reset_counters:
            self.best_val_loss = float("inf")
            self.epochs_since_decay = 0
            self.epochs_since_improvement = 0
            self.cooldown_remaining = 0
            self.num_decays = 0


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))
        self.iter = 0

    def get(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):
        value = self.get(self.iter)
        self.iter += 1
        return value

    def reset(self):
        self.iter = 0

    def __getitem__(self, it):
        return self.get(it)


def build_scheduler(cfg, optimizer):
    cfg = copy.deepcopy(cfg)
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
