from torch.utils.data import DataLoader, random_split, Subset,ConcatDataset
import torch
import numpy as np
import random

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
        self.aug = 4
        # build the two subsets exactly once
        #self._create_split_newest()
        self._create_champion()
        #self._create_split()

    def _create_split(self):
        full = Dataset(
            self.dataset_path,
            "train_val_gpt_aug_3",
            transforms=self.test_transform,   # no augmentations for the split
            metadata=self.metadata,
        )
        
        val_len = int(self.val_ratio * len(full)/self.aug)
        val_idx,train_idx= self.random_split_range(len(full), val_len)
        self.train_set=ConcatDataset([Subset(full, train_idx*self.aug+i) for i in range(self.aug)])
        self.val_set=Subset(full, val_idx*self.aug+3)
        
    def random_split_range(self, n, p):
        indices = list(range(n))
        rng = random.Random(self.seed)  # Use self.seed for reproducibility
        rng.shuffle(indices)
        return indices[:p], indices[p:]
    def _create_split_newest(self):
        full = Dataset(
            self.dataset_path,
            "train_val_gpt_aug3",
            transforms=self.test_transform,   # no augmentations for the split
            metadata=self.metadata,
        )

        years = full.info["year"].values
        aug   = full.info["aug"].values
        newest_idx = np.where((years >= 2022) + (1- aug))[0].tolist()
        old_idx    = np.where(years <  2022)[0].tolist()

        # 3) wrap with Subset
        self.train_set = Subset(full, old_idx)
        self.val_set   = Subset(full, newest_idx)
    def _create_champion(self):
        full = Dataset(
            self.dataset_path,
            "train_val_gpt_aug3",
            transforms=self.test_transform,   # no augmentations for the split
            metadata=self.metadata,
        )
        n=len(full)
        num=np.random.randint(0, n-1)
        while full.info["aug"][num] :
            num-=1
        self.val_set=Subset(full, [num])
        self.train_set=Subset(full, [i for i in range(n) if full.info["Unnamed: 0"][i] != full.info["Unnamed: 0"][num]])

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