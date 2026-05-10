"""
ShowerClusteringTrainer — subclass of Pointcept's DefaultTrainer that uses
shower_clustering_collate for the train/val DataLoaders. The default trainer
hard-codes point_collate_fn / collate_fn, which don't know how to batch our
per-event lists of variable-size fragment indices and GT instance dicts.

Phase 8 of the design (see pointcept/docs/shower_clustering_design.md).
"""

from functools import partial

import torch
import torch.utils.data

from pointcept.utils import comm
from pointcept.engines.train import TRAINERS, Trainer
from pointcept.engines.defaults import worker_init_fn
from pointcept.datasets import build_dataset
from pointcept.datasets.shower_clustering import shower_clustering_collate


@TRAINERS.register_module()
class ShowerClusteringTrainer(Trainer):
    """DefaultTrainer + custom collate for shower-clustering."""

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data)
        else:
            train_sampler = None

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
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=shower_clustering_collate,
            pin_memory=False,  # custom collate yields nested lists; pin_memory
                               # only handles flat tensors and would warn or fail
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_val_per_gpu,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=shower_clustering_collate,
            )
        return val_loader
