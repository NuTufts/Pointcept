"""
FastForwardSampler

Wraps any PyTorch Sampler to support fast mid-epoch resume: on the FIRST
``__iter__`` after ``skip_indices`` is set, skips that many items from the
wrapped sampler without driving the DataLoader's workers through their
transforms. Subsequent epochs yield from index 0 normally.

Used by Trainer when ``cfg.skip_dataloader_on_resume=True`` so resuming
mid-epoch doesn't pay the islice cost (~1h with heavy SONATA transforms).
The tradeoff: workers don't advance through the skipped batches' RNG, so
the kept batches see different augmentations than the un-resumed
counterfactual — fine for SSL pretraining with huge IID datasets.
"""

from torch.utils.data import Sampler


_SENTINEL = object()


class FastForwardSampler(Sampler):
    """Wraps ``base_sampler``; on the first ``__iter__``, skips the first
    ``skip_indices`` items it yields. After that the wrapper is "consumed"
    and behaves like ``base_sampler`` for all subsequent epochs.

    Args:
        base_sampler: the underlying sampler (e.g. DistributedSampler,
            RandomSampler). ``set_epoch`` is forwarded to it.
        skip_indices: number of items to skip on the first iteration.
            Set this BEFORE the first ``iter(loader)`` call of the resumed
            epoch and reset ``_consumed = False`` if reactivating.
    """

    def __init__(self, base_sampler, skip_indices=0):
        self.base_sampler = base_sampler
        self.skip_indices = int(skip_indices)
        self._consumed = False

    def __iter__(self):
        it = iter(self.base_sampler)
        if not self._consumed and self.skip_indices > 0:
            for _ in range(self.skip_indices):
                if next(it, _SENTINEL) is _SENTINEL:
                    break
        self._consumed = True
        yield from it

    def __len__(self):
        n = len(self.base_sampler)
        if not self._consumed and self.skip_indices > 0:
            return max(n - self.skip_indices, 0)
        return n

    def set_epoch(self, epoch):
        # Forward to DistributedSampler (or any other epoch-aware sampler)
        # so per-epoch shuffling still works.
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)
