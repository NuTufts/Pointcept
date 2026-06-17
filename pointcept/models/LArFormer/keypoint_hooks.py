"""Training hooks for the LArFormer keypoint heads.

`KeypointRegWarmupHook` — before each epoch, broadcasts the current epoch to
every `LArFormerLoss` in the model so the keypoint start/end regression can
warm up with a robust loss (smooth_l1) for the first `kp_reg_warmup_epochs`
before switching to β-NLL (see `LArFormerLoss._eff_reg_kind`). No-op when
`kp_reg_warmup_epochs == 0`, so it is safe to always register.

Registers on import; configs trigger it via the import side effect of
`pointcept.models.LArFormer` plus a `dict(type="KeypointRegWarmupHook")` entry.
"""
from pointcept.engines.hooks.builder import HOOKS
from pointcept.engines.hooks.default import HookBase

from .losses import LArFormerLoss


@HOOKS.register_module()
class KeypointRegWarmupHook(HookBase):
    """Push `trainer.epoch` into every LArFormerLoss each epoch."""

    def _set(self) -> None:
        model = getattr(self.trainer, "model", None)
        if model is None:
            return
        ep = int(getattr(self.trainer, "epoch", 0))
        for m in model.modules():
            if isinstance(m, LArFormerLoss):
                m.set_epoch(ep)

    def before_train(self) -> None:
        # Cover the resume case so the very first resumed epoch starts with the
        # correct warm-up state before any step runs.
        self._set()

    def before_epoch(self) -> None:
        self._set()
