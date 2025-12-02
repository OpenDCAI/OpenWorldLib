from typing import Any, Optional

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from mirror_training.utils.image import ImageAugmentation
from mirror_training.data.train import *
from mirror_training.data.eval import *
from mirror_training.data.sampler.dynamic_sampler import (
    DynamicBatchSampler,
    DynamicDistributedSampler,
)

from mirror_training.utils.misc import get_rank

class WorldMirrorDataModule(LightningDataModule):
    def __init__(
        self,
        train_datasets: list[str],
        validation_datasets: list[str],
        max_images_per_gpu: int = 24,
        min_num_of_views: int = 2,
        num_workers: int = 6,
        num_workers_val: int = 2,
        pin_memory: bool = True,
        aspect_ratios: list[float] | None = None,
        num_pixels_range: list[int] | None = None,
        decay: float = 0.5,
        allview_p: float = 0.2,
        transform=ImageAugmentation(apply_aug=False),
    ) -> None:
        """Initialize a WorldMirrorDataModule.

        :param train_datasets: List of training dataset path strings.
        :param validation_datasets: List of validation dataset path strings.
        :param max_images_per_gpu: Maximum number of images to process per GPU in a batch.
        :param min_num_of_views: Minimum number of views required for multi-view learning.
        :param num_workers: Number of worker processes for training data loading.
        :param num_workers_val: Number of worker processes for validation data loading.
        :param pin_memory: Whether to pin memory in GPU for faster data transfer.
        :param aspect_ratios: List of allowed aspect ratios for dynamic batching.
        :param num_pixels_range: Range of pixel counts [min, max] for dynamic resolution.
        :param decay: Decay factor for dynamic batch sampling strategy.
        :param allview_p: Probability of using all available views in a sample.
        :param transform: Transform to apply to the data.
        """
        super().__init__()

        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.max_images_per_gpu = max_images_per_gpu
        self.min_num_of_views = min_num_of_views
        self.num_workers = num_workers
        self.num_workers_val = num_workers_val
        self.pin_memory = pin_memory
        self.aspect_ratios = aspect_ratios
        self.num_pixels_range = num_pixels_range
        self.decay = decay
        self.allview_p = allview_p
        self.transform = transform
        
        # Placeholder for training dataloader, initialized in train_dataloader()
        self.train_loader: Optional[DataLoader] = None
        # Placeholder for validation dataloader, initialized in val_dataloader()
        self.val_loader: Optional[DataLoader] = None

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader with dynamic batching.
        """
        # Validate that all dataset entries are strings
        assert all(
            isinstance(dataset, str) for dataset in self.train_datasets
        ), "All datasets must be strings"

        # Concatenate all training dataset strings into a single expression with "+" separator
        train_datasets_concat = " + ".join(self.train_datasets)
        
        # Evaluate the concatenated string to create the actual dataset object
        if isinstance(train_datasets_concat, str):
            train_datasets_concat = eval(train_datasets_concat)

        # Build distributed sampler for multi-GPU training
        distributed_sampler = DynamicDistributedSampler(
            dataset=train_datasets_concat, 
            rank=get_rank(), 
            shuffle=True
        )

        # Build dynamic batch sampler that adjusts batch size based on image resolution and aspect ratio
        dynamicbatch_sampler = DynamicBatchSampler(
            distributed_sampler,
            self.min_num_of_views,
            self.max_images_per_gpu,
            max_img_per_gpu=self.max_images_per_gpu,
            aspect_ratio_range=self.aspect_ratios,
            num_pixels_range=self.num_pixels_range,
            decay=self.decay,
            allview_p=self.allview_p,
        )

        # Create the training dataloader with dynamic batching
        self.train_loader = DataLoader(
            train_datasets_concat,
            batch_sampler=dynamicbatch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Keep workers alive between epochs if num_workers > 0
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # Initialize dataloader epoch counter
        self.train_loader = self.set_dataloader_epoch(self.train_loader)

        return self.train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: A CombinedLoader containing all validation dataloaders in sequential mode.
        """        
        # Validate that all validation dataset entries are strings
        assert all(
            isinstance(dataset, str) for dataset in self.validation_datasets
        ), "All datasets must be strings"

        # Evaluate each dataset string to instantiate the actual dataset objects
        val_datasets = [eval(dataset) for dataset in self.validation_datasets]

        # Create individual validation dataloaders for each dataset
        val_loaders = []
        for dataset in val_datasets:
            # Create distributed sampler without shuffling for validation
            distributed_sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
            
            # Create validation dataloader with batch size of 1
            val_dataloader = DataLoader(
                dataset,
                batch_size=1,
                sampler=distributed_sampler,
                num_workers=self.num_workers_val,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers_val > 0 else False,
            )

            # Initialize epoch counter for the validation dataloader
            val_dataloader = self.set_dataloader_epoch(val_dataloader)

            val_loaders.append(val_dataloader)

        # Combine all validation loaders to run sequentially
        self.val_loader = CombinedLoader(val_loaders, mode="sequential")
        return self.val_loader

    def set_dataloader_epoch(self, dataloader):
        """Set the epoch counter to 0 for the dataloader and its samplers.

        This ensures proper data shuffling and synchronization across epochs.

        :param dataloader: The dataloader to initialize.
        :return: The dataloader with epoch set to 0.
        """
        # Set epoch for the dataset
        dataloader.dataset.set_epoch(0)

        # Set epoch for the sampler if it exists and supports epoch tracking
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(0)

        # Set epoch for the batch sampler if it exists and supports epoch tracking
        if hasattr(dataloader, "batch_sampler") and hasattr(
            dataloader.batch_sampler, "set_epoch"
        ):
            dataloader.batch_sampler.set_epoch(0)

        return dataloader
