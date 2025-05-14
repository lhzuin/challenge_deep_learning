from torch.utils.data import DataLoader, random_split
import torch

from data.dataset import Dataset


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata

        self.val_ratio = val_ratio
        self.seed = seed

        # build the two subsets exactly once
        self._create_split()

    def _create_split(self):
        full = Dataset(
            self.dataset_path,
            "train_val_gpt",
            transforms=self.test_transform,   # no augmentations for the split
            metadata=self.metadata,
        )

        val_len = int(self.val_ratio * len(full))
        train_len = len(full) - val_len

        self.train_set, self.val_set = random_split(
            full,
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test_gpt",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )