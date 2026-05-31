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
from pointcept.engines.train import TRAINERS, Trainer
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
