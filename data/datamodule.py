from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
from data.random_per_id import RandomPerIdDataset
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
        metadata=None,
        val_ratio: float = 0.1,
        seed: int = 42,
        train_on_log=False,
        augmentation=True
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = list(metadata) if metadata is not None else ["title"]

        self.val_ratio = val_ratio
        self.seed = seed
        self.aug = 4
        self.train_on_log = train_on_log
        self.augmentation = augmentation
        # build the two subsets exactly once
        self._create_split_newest()
        #self._create_champion()
        #self._create_split()

    def _create_split(self):
        base = Dataset(
            self.dataset_path,
            "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms=self.test_transform,   # no augmentations for the split
            metadata=self.metadata,
            train_on_log=self.train_on_log
        )

        if self.augmentation:
            # 1 — wrapper that produces ONE sample per base-id
            full = RandomPerIdDataset(base, mix_fields=("title", "summary"))

            # 2 — operate only on ORIGINAL rows (aug==0)
            originals = base.info.query("aug == 0")
            all_ids   = originals["base_id"].unique().tolist()

            rng = random.Random(self.seed)
            rng.shuffle(all_ids)

            val_len  = int(self.val_ratio * len(all_ids))
            val_ids  = set(all_ids[:val_len])
            train_ids= set(all_ids[val_len:])

            # 3 — translate base-ids → indices in full.base_ids
            train_idx, val_idx = [], []
            for i, bid in enumerate(full.base_ids):
                (train_idx if bid in train_ids else val_idx).append(i)

            # 4 — final splits
            self.train_set = Subset(full, train_idx)                   # random image/epoch
            self.val_set   = Subset(base,                              # fixed originals
                                    originals.index[originals.base_id.isin(val_ids)].tolist())
        else:
            val_len = int(self.val_ratio * len(base))
            train_len = len(base) - val_len

            self.train_set, self.val_set = random_split(
                base,
                lengths=[train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed),
            )

        
    def random_split_range(self, n, p):
        indices = list(range(n))
        rng = random.Random(self.seed)  # Use self.seed for reproducibility
        rng.shuffle(indices)
        return indices[:p], indices[p:]
    
    def _create_split_newest(self):
        # 1) base Dataset (all rows, incl. aug)
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log=self.train_on_log
        )
        if self.augmentation:
            # 2) wrapper that yields 1 sample per base_id
            full = RandomPerIdDataset(
                base_dataset = base,
                mix_fields   = ("title", "summary"),
            )

            # 3) which base_id belong to the val set?  (≥ 2022 & not augmented)
            val_row_mask = (base.info["year"] >= 2022) & (base.info["aug"] == 0)
            val_ids      = set(base.info.loc[val_row_mask, "base_id"])

            # 4) map base_id → index in full.base_ids
            train_idx, val_idx = [], []
            for i, bid in enumerate(full.base_ids):
                (val_idx if bid in val_ids else train_idx).append(i)

            self.train_set = Subset(full, train_idx)      # indices guaranteed < len(full)
            # validation stays a “stable” plain Dataset (no random mixing)
            self.val_set   = Subset(base, val_row_mask.to_numpy().nonzero()[0])
        else:
            years = base.info["year"].values
            aug   = base.info["aug"].values
            newest_idx = np.where((years >= 2022) & (aug == 0))[0].tolist()
            old_idx    = np.where(years <  2022)[0].tolist()
            self.train_set = Subset(base, old_idx)
            self.val_set   = Subset(base, newest_idx)
    def _create_champion(self):
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log=self.train_on_log
        )
        if self.augmentation:
            full = RandomPerIdDataset(base, mix_fields=("title", "summary"))
            # choose ONE original row whose aug==0
            originals = base.info.query("aug == 0").reset_index()      # keep csv index
            champ_row = originals.sample(n=1,
                                random_state=self.seed).iloc[0]
            champ_id  = champ_row["base_id"]

            # indices in the wrapper
            champ_csv_idx =  [int(champ_row["index"])]
            train_idx = [i for i, bid in enumerate(full.base_ids) if bid != champ_id]

            # build the two sets
            self.train_set = Subset(full, train_idx)                   # all other videos
            self.val_set   = Subset(base, champ_csv_idx)   # the one original
        else:
            rng = random.Random(self.seed)
            # list of actual DataFrame row‐indices where aug==0
            orig_idxs = base.info.index[base.info["aug"] == 0].tolist()

            # pick one of those
            champ_idx = rng.choice(orig_idxs)

            # val is that one row
            self.val_set = Subset(base, [champ_idx])

            # train is everything else (in base row‐index space)
            train_idxs = [i for i in base.info.index.tolist() if i != champ_idx]
            self.train_set = Subset(base, train_idxs)

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
            train_on_log=self.train_on_log
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )