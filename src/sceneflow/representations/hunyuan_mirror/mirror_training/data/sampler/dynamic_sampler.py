from typing import Callable, Optional

import numpy as np
from torch.utils.data import Sampler, DistributedSampler


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self._aspect_ratio = None
        self._view_idxs = None
        self._source_view_idxs = None
        self._target_pixels = None

    def __iter__(self):
        """
        Yields a sequence of (sample_idxs, _aspect_ratio, (_view_idxs, _source_view_idxs), _target_pixels).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for sample_idxs in indices_iter:
            yield (sample_idxs, self._aspect_ratio, (self._view_idxs, self._source_view_idxs), self._target_pixels)

    def update_parameters(self, aspect_ratio, view_idxs, source_view_idxs, target_pixels):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            view_idxs: The number of images to set.
            source_view_idxs: The number of source images to set.
            target_pixels: The number of pixels per image to set.
        """
        self._aspect_ratio = aspect_ratio
        self._view_idxs = view_idxs
        self._source_view_idxs = source_view_idxs
        self._target_pixels = target_pixels


class DynamicBatchSampler(Sampler):
    def __init__(self, sampler, min_view_size, max_view_size, epoch=0, seed=42,
                 max_img_per_gpu=24, aspect_ratio_range=None, num_pixels_range=[100000, 250000], decay=0.5, allview_p=0.2):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            min_view_size: min_images numbers per sample.
            max_view_size: max_images numbers per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            num_pixels_range: List containing [min_pixels, max_pixels] for target resolution sampling.
            decay: Decay parameter for source view sampling weight distribution. Higher values favor more source views.
            allview_p: Probability of using all views as source views (no novel views).
        """
        self.sampler = sampler
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.rng = np.random.default_rng(seed=seed)
        self.aspect_ratio_range = aspect_ratio_range
        self.num_pixels_range = num_pixels_range

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(min_view_size, max_view_size+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if min_view_size <= n <= max_view_size])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        self.decay = decay
        self.allview_p = allview_p

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)
    
    def _sample_source_view_idxs(self, _view_idxs):
        """
        Sample the number of source views using a weighted distribution.
        Higher number of source views have relatively higher probability.
        """
        if _view_idxs <= 2 or self.rng.random() < self.allview_p:
            return _view_idxs
        
        # Valid range for source views count
        min_source = max(self.min_view_size, int(_view_idxs//2 + 0.5))
        max_source = min(self.max_view_size - 1, _view_idxs - 1)  # Reserve space for at least 1 novel view
        
        if min_source > max_source:
            return _view_idxs
        
        # Create weights: higher counts have higher weights, but not linear growth
        counts = list(range(min_source, max_source + 1))
        weights = [(count - min_source + 1)**self.decay for count in counts]

        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        return self.rng.choice(counts, p=weights)

    def _sample_view_idxs_and_ar_and_tp(self):
        """Sample view_idxs and aspect_ratio according to the specified rules."""
        _view_idxs = int(self.rng.choice(self.possible_nums, p=self.normalized_weights))
        _source_view_idxs = self._sample_source_view_idxs(_view_idxs)
        if self.aspect_ratio_range is not None:
            _aspect_ratio = float(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1], size=1))
        else:
            _aspect_ratio = 1.0
        min_pixels = self.num_pixels_range[0]
        max_pixels = self.num_pixels_range[1]
        _target_pixels = int(self.rng.integers(min_pixels, max_pixels + 1))

        return _view_idxs, _source_view_idxs, _aspect_ratio, _target_pixels

    def _batch_size_for(self, view_idxs: int) -> int:
        """Calculate batch_size based on max_img_per_gpu and view count (floor division, minimum 1)."""
        bs = int(np.floor(self.max_img_per_gpu / max(1, view_idxs)))
        return max(1, bs)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng = np.random.default_rng(seed=epoch + 777)

    def __iter__(self):
        """
        Dynamically sample and consume the underlying sampler.
        All samples in each batch share the same aspect_ratio / view_idxs.
        """
        sampler_iterator = iter(self.sampler)
        remaining = len(self.sampler)
        
        while remaining > 0:
            v, sv, ar, tp = self._sample_view_idxs_and_ar_and_tp()
            bs = self._batch_size_for(v)
            
            # Synchronize dynamic parameters to the underlying sampler (for dataset usage)
            self.sampler.update_parameters(
                aspect_ratio=ar,
                view_idxs=v,
                source_view_idxs=sv,
                target_pixels=tp,
            )
            
            current_batch = []
            for _ in range(bs):
                try:
                    item = next(sampler_iterator)
                    current_batch.append(item)
                    remaining -= 1
                except StopIteration:
                    break
            
            if current_batch:
                yield current_batch

    def __len__(self):
        # Return the length of the underlying sampler as an estimate of sample count
        return len(self.sampler)