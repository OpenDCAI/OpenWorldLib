import numpy as np
from typing import List, Tuple, Union, Any


class BaseDataset:
    """Base composable dataset with operator overloading.

    Operators:
        ds1 + ds2    -> Concatenate datasets
        N @ dataset  -> Random sample to size N (with replacement if needed)
        K * dataset  -> Repeat each element K times
    """

    def __add__(self, other: "BaseDataset") -> "AddedDataset":
        return AddedDataset([self, other])

    def __rmul__(self, repeat_factor: int) -> "RepeatedDataset":
        return RepeatedDataset(repeat_factor, self)

    def __rmatmul__(self, sample_size: int) -> "SampledDataset":
        return SampledDataset(sample_size, self)

    def set_epoch(self, epoch: int) -> None:
        """Called before each epoch for datasets requiring reproducible sampling."""
        pass


class AddedDataset(BaseDataset):
    """Concatenates multiple datasets sequentially."""

    def __init__(self, datasets: List[BaseDataset]):
        assert len(datasets) > 0 and all(isinstance(d, BaseDataset) for d in datasets)
        self.datasets = datasets
        self._cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return int(self._cumulative_sizes[-1])

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, index: Union[int, Tuple[int, ...]]) -> Any:
        # Extract primary index and optional secondary indices
        primary_idx, *secondary = (
            (index, ) if not isinstance(index, tuple) else index
        )

        if not (0 <= primary_idx < len(self)):
            raise IndexError(
                f"Index {primary_idx} out of range for dataset of size {len(self)}"
            )

        # Binary search to find which dataset contains this index
        dataset_idx = int(np.searchsorted(self._cumulative_sizes, primary_idx, "right"))
        dataset = self.datasets[dataset_idx]

        # Calculate offset within the selected dataset
        offset = primary_idx - (
            int(self._cumulative_sizes[dataset_idx - 1]) if dataset_idx > 0 else 0
        )

        return dataset[(offset, *secondary)] if secondary else dataset[offset]

    def __repr__(self) -> str:
        return " + \n".join(f"{i+1}. {repr(d)}" for i, d in enumerate(self.datasets))

    @property
    def _resolutions(self):
        # Verify all datasets have the same resolutions
        resolutions = self.datasets[0]._resolutions
        assert all(
            tuple(d._resolutions) == tuple(resolutions) for d in self.datasets[1:]
        )
        return resolutions

    @property
    def num_views(self):
        # Verify all datasets have the same number of views
        views = self.datasets[0].num_views
        assert all(d.num_views == views for d in self.datasets[1:])
        return views


class SampledDataset(BaseDataset):
    """Randomly samples a fixed number of elements with epoch-based shuffling."""

    def __init__(self, sample_size: int, base_dataset: BaseDataset):
        assert isinstance(sample_size, int) and sample_size > 0
        self.sample_size = sample_size
        self.base_dataset = base_dataset
        self._indices = None

    def set_epoch(self, epoch: int) -> None:
        """Generate shuffled indices with tiling if sample_size > base dataset size."""
        rng = np.random.RandomState(epoch + 777)
        base_size = len(self.base_dataset)

        # Shuffle and tile to reach sample_size
        shuffled = rng.permutation(base_size)
        tiles_needed = (self.sample_size + base_size - 1) // base_size
        self._indices = np.tile(shuffled, tiles_needed)[: self.sample_size]

    def __len__(self) -> int:
        return self.sample_size

    def __getitem__(self, index: Union[int, Tuple[int, ...]]) -> Any:
        assert (
            self._indices is not None
        ), "Call set_epoch() before accessing SampledDataset"

        # Extract primary index and remap through shuffled indices
        primary_idx, *secondary = (
            (index, ) if not isinstance(index, tuple) else index
        )
        mapped_idx = int(self._indices[primary_idx])

        return (
            self.base_dataset[(mapped_idx, *secondary)]
            if secondary
            else self.base_dataset[mapped_idx]
        )

    def __repr__(self) -> str:
        return f"{self.sample_size:_} @ {repr(self.base_dataset)}"

    @property
    def _resolutions(self):
        return self.base_dataset._resolutions

    @property
    def num_views(self):
        return self.base_dataset.num_views


class RepeatedDataset(BaseDataset):
    """Repeats each element k times consecutively."""

    def __init__(self, repeat_factor: int, base_dataset: BaseDataset):
        assert isinstance(repeat_factor, int) and repeat_factor > 0
        self.repeat_factor = repeat_factor
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return self.repeat_factor * len(self.base_dataset)

    def __getitem__(self, index: Union[int, Tuple[int, ...]]) -> Any:
        # Map repeated index back to base dataset index
        primary_idx, *secondary = (
            (index, ) if not isinstance(index, tuple) else index
        )
        base_idx = primary_idx // self.repeat_factor

        return (
            self.base_dataset[(base_idx, *secondary)]
            if secondary
            else self.base_dataset[base_idx]
        )

    def __repr__(self) -> str:
        return f"{self.repeat_factor} * {repr(self.base_dataset)}"

    @property
    def _resolutions(self):
        return self.base_dataset._resolutions

    @property
    def num_views(self):
        return self.base_dataset.num_views
