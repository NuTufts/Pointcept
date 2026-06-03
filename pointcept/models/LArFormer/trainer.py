"""
LArFormerTrainer — subclass of Pointcept's DefaultTrainer that uses
`larformer_collate` for the train/val DataLoaders.

The default trainer hard-codes `point_collate_fn` / `collate_fn`, which try
to batch every dict field with torch.default_collate. That breaks for
LArFormerDataset because it returns:
    - per-event tensors with variable N (per-spacepoint fields)
    - per-event lists of dicts (gt_instances)
    - per-event lists of variable-length LongTensors (fragment_indices, when
      emit_fragments=True)

`larformer_collate` handles flat-concat + per-event-list packing.

Registration is side-effect-only: this module isn't imported by
`pointcept/models/LArFormer/__init__.py` to avoid the circular import that
also tripped ShowerClusteringTrainer (engines.train imports pointcept.models
before TRAINERS is defined). Configs must import this module explicitly:

    import pointcept.models.LArFormer.trainer as _trainer_module
    del _trainer_module

then set `train = dict(type="LArFormerTrainer")`.
"""

from functools import partial

import torch
import torch.utils.data

from pointcept.datasets import build_dataset
from pointcept.datasets.larformer import larformer_collate
from pointcept.engines.defaults import worker_init_fn
from pointcept.engines.train import TRAINERS, Trainer, AMP_DTYPE
from pointcept.utils import comm


@TRAINERS.register_module()
class LArFormerTrainer(Trainer):
    """DefaultTrainer + larformer_collate."""

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        # Use the base Trainer's wrapped-sampler helper so mid-epoch fast-resume
        # (FastForwardSampler) works for this trainer too. Sampler controls
        # ordering, so DataLoader's own shuffle is off.
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

        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=larformer_collate,
            # Custom collate yields nested Python lists (gt_instances,
            # fragment_indices_per_event); pin_memory only handles flat
            # tensors and would warn or fail.
            pin_memory=False,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
        )

    def build_val_loader(self):
        if not self.cfg.evaluate:
            return None
        val_data = build_dataset(self.cfg.data.val)
        if comm.get_world_size() > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_data)
        else:
            val_sampler = None
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.cfg.batch_size_val_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_val_per_gpu,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=larformer_collate,
        )

    # ------------------------------------------------------------------
    # Optional second train loader — for the mask-denoising path.
    # ------------------------------------------------------------------

    def build_train_dn_loader(self):
        """Optional sibling of `build_train_loader` for the denoising path.

        Activated when `cfg.data.train_dn` is set. The convention is:

            data = dict(
                train     = dict(..., source_set_filter="stage2_pass"),
                train_dn  = dict(..., source_set_filter="union"),
                val       = dict(..., source_set_filter="stage2_pass"),
            )

        So the matcher sees the inference-realistic SP set while the
        mask-denoising path sees the full GT-anchored SP set. The two
        datasets typically read the same files — only the per-event
        source-set filter differs. Returns None when `train_dn` isn't
        configured, which falls back to single-loader training.
        """
        train_dn_cfg = getattr(self.cfg.data, "train_dn", None)
        if train_dn_cfg is None:
            return None
        dn_data = build_dataset(train_dn_cfg)
        if comm.get_world_size() > 1:
            dn_sampler = torch.utils.data.distributed.DistributedSampler(
                dn_data)
        else:
            dn_sampler = None
        return torch.utils.data.DataLoader(
            dn_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(dn_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=dn_sampler,
            collate_fn=larformer_collate,
            pin_memory=False,
            drop_last=True,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
        )

    def _ensure_dn_iter(self):
        """Lazy-init the DN iterator and the underlying DataLoader. Returns
        None when no DN loader is configured."""
        if not hasattr(self, "_train_dn_loader"):
            self._train_dn_loader = self.build_train_dn_loader()
        if self._train_dn_loader is None:
            return None
        if (not hasattr(self, "_train_dn_iter")
                or self._train_dn_iter is None):
            self._train_dn_iter = iter(self._train_dn_loader)
        return self._train_dn_iter

    def _next_dn_batch(self):
        """Pull one batch from the DN loader; wrap-around at end of epoch."""
        dn_iter = self._ensure_dn_iter()
        if dn_iter is None:
            return None
        try:
            return next(dn_iter)
        except StopIteration:
            self._train_dn_iter = iter(self._train_dn_loader)
            return next(self._train_dn_iter)

    # ------------------------------------------------------------------
    # Dual-forward train step
    # ------------------------------------------------------------------

    def run_step(self):
        """Single- or dual-forward step.

        With `cfg.data.train_dn` unset, this is identical to the base
        class — the matcher dataset is the only input and one forward
        produces all losses.

        With `cfg.data.train_dn` set, runs TWO forwards per iter:
          - matcher batch (`stage2_pass`)  → keep all NON-`loss_dn_*`
                                             components from this forward
          - denoising batch (`union`)      → keep ONLY `loss_dn_*`
                                             components from this forward
        Total loss = sum of the two splits. The model's `mask_denoising`
        config still triggers DN-query construction in BOTH forwards, but
        the DN losses from the matcher forward are discarded (they were
        anchored against the same SPs the matcher saw, so they don't
        add the truth-anchor signal mask denoising is supposed to give).
        """
        if self._ensure_dn_iter() is None:
            return super().run_step()
        return self._dual_run_step()

    def _dual_run_step(self):
        """Two-forward variant. See `run_step` docstring."""
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            auto_cast = torch.cuda.amp.autocast

        matcher_input = self.comm_info["input_dict"]
        dn_input = self._next_dn_batch()

        # Move both to GPU (in-place mutation — same convention as the
        # base run_step for flat-tensor fields; nested lists / dicts in
        # gt_instances are moved inside the model's forward).
        for d in (matcher_input, dn_input):
            for key in d.keys():
                if isinstance(d[key], torch.Tensor):
                    d[key] = d[key].cuda(non_blocking=True)

        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        with auto_cast(
            enabled=self.cfg.enable_amp,
            dtype=AMP_DTYPE[self.cfg.amp_dtype],
        ):
            out_matcher = self.model(matcher_input)
            out_dn = self.model(dn_input)

            # Matcher contribution: everything in `out_matcher` EXCEPT
            # `loss_dn_*`. Compute by subtracting DN bits from total, so
            # we don't have to know every regular-loss key explicitly.
            dn_in_matcher = sum(
                (v for k, v in out_matcher.items()
                 if k.startswith("loss_dn_")),
                start=torch.zeros((), device=out_matcher["loss"].device),
            )
            matcher_loss = out_matcher["loss"] - dn_in_matcher

            # DN contribution: ONLY `loss_dn_*` from the DN forward. If
            # `weight_dn_loss` already scales these inside the model,
            # we don't re-scale.
            dn_loss = sum(
                (v for k, v in out_dn.items()
                 if k.startswith("loss_dn_")),
                start=torch.zeros((), device=out_matcher["loss"].device),
            )
            loss = (matcher_loss + dn_loss) / self.cfg.gradient_accumulation_steps

        # NaN/Inf guard — same skip-batch behaviour as the base class but
        # without the verbose per-tensor diagnostic (the same guard runs
        # in the base run_step path for the non-DN cases).
        if not torch.isfinite(loss):
            self.logger.error(
                f"NaN/Inf loss in DN dual-forward step "
                f"(matcher={float(matcher_loss):.4f}, "
                f"dn={float(dn_loss):.4f}); skipping batch."
            )
            self.optimizer.zero_grad()
            self._gradient_accumulation_counter = 0
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()
            # Publish the matcher output so downstream hooks (logging
            # writers) still see a "model_output_dict".
            self.comm_info["model_output_dict"] = out_matcher
            return

        if self.cfg.enable_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.clip_grad is not None:
                if self.cfg.enable_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad,
                )
            if self.cfg.enable_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self._gradient_accumulation_counter = 0
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()

        # Compose a merged output_dict for downstream hooks. Keep the
        # matcher dict as the base (the InformationWriter prints all
        # `loss_*` keys) and stamp in the DN forward's dn_* losses
        # under a `loss_dn_<...>_dn` suffix so they don't collide with
        # the matcher's discarded DN keys.
        merged = dict(out_matcher)
        merged["loss"] = matcher_loss + dn_loss
        for k, v in out_dn.items():
            if k.startswith("loss_dn_"):
                merged[f"{k}_dnforward"] = v
        self.comm_info["model_output_dict"] = merged
