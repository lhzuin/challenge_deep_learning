from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
import torch
import os
from data.random_per_id import RandomPerIdDataset
from torch.utils.data import WeightedRandomSampler

from data.dataset import Dataset
import pickle


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
        augmentation=True,
        build="newest",
        epoch=0
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
        self.epoch = epoch
        self.build(build)
        

        
    def build(self, build):
        if build == "newest":
            self._create_split_newest()
        elif build == "champion":
            self._create_champion()
        elif build == "split":
            self._create_split()
        elif build == "epoque_validation":
            self._create_epoque(self.epoch, validation=True)
        elif build == "epoque_champion":
            self._create_epoque(self.epoch, validation=False)
        elif build == "outliers":
            self._create_outlier(self.epoch, validation=True)
        elif build == "outliers_champ":
            self._create_outlier(self.epoch, validation=False)
        else:
            raise ValueError(f"Unknown build type: {build}")

    def _create_outlier(self, epoch, validation=True):
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log = self.train_on_log
        )
        if epoch == 0:
            base = base.subset((base.info["year"] >= 2020+2*(1-validation)).to_numpy().nonzero()[0])
        elif epoch <= 3:
            if self.train_on_log:
                mask=(base.info["views"] -10 <-3) |( base.info["views"] -10 >3)
            else:
                mask=(np.log1p(base.info["views"]) - 10  <-3) |( np.log1p(base.info["views"]) - 10 >2)
            base = base.subset(((base.info["year"] >= 2020+2*(1-validation)) | mask).to_numpy().nonzero()[0])
        
        val_row_mask = (base.info["year"] >= 2022+5*(1-validation)) & (base.info["aug"] == 0)
        val_ids = set(base.info.loc[val_row_mask, "id"])
        full = RandomPerIdDataset(
            base_dataset = base,
            mix_fields   = ("title", "summary"),
        )
        train_idx, val_idx = [], []
        for i, bid in enumerate(full.base_ids):
            (val_idx if bid in val_ids else train_idx).append(i)

        self.train_set = Subset(full, train_idx)
        self.val_set = Subset(base, val_row_mask.to_numpy().nonzero()[0]) if validation else Subset(base, train_idx[:5])
        
    def _create_epoque(self, epoch, validation=True):
        start_year = 2022 - 5*epoch - 2*validation 
        end_year = 2023 - 2*validation 
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log = self.train_on_log
        )
        base = base.subset((base.info["year"] >= start_year).to_numpy().nonzero()[0])
        full = RandomPerIdDataset(
            base_dataset = base,
            mix_fields   = ("title", "summary"),
        )
        val_row_mask = (base.info["year"] > end_year) & (base.info["aug"] == 0)
        val_ids = set(base.info.loc[val_row_mask, "id"])
        train_idx, val_idx = [], []
        for i, bid in enumerate(full.base_ids):
            (val_idx if bid in val_ids else train_idx).append(i)

        self.train_set = Subset(full, train_idx)
        self.val_set = Subset(base, val_row_mask.to_numpy().nonzero()[0]) if validation else Subset(base, train_idx[:5])

    def __len__(self):
        return len(self.train_set)

    def _create_split(self):
        base = Dataset(
            self.dataset_path,
            "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms=self.test_transform,
            metadata=self.metadata,
            train_on_log=self.train_on_log
        )
        if self.augmentation:
            full = RandomPerIdDataset(base, mix_fields=("title", "summary"))
            originals = base.info.query("aug == 0")
            all_ids = originals["id"].unique().tolist()
            rng = random.Random(self.seed)
            rng.shuffle(all_ids)
            val_len = int(self.val_ratio * len(all_ids))
            val_ids = set(all_ids[:val_len])
            train_ids = set(all_ids[val_len:])
            train_idx, val_idx = [], []
            for i, bid in enumerate(full.base_ids):
                (train_idx if bid in train_ids else val_idx).append(i)
            self.train_set = Subset(full, train_idx)
            self.val_set = Subset(base, originals.index[originals.base_id.isin(val_ids)].tolist())
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
        rng = random.Random(self.seed)
        rng.shuffle(indices)
        return indices[:p], indices[p:]

    def _create_split_newest(self):
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log = self.train_on_log
        )
        if self.augmentation:
            full = RandomPerIdDataset(
                base_dataset = base,
                mix_fields   = ("title", "summary"),
            )
            val_row_mask = (base.info["year"] >= 2022) & (base.info["aug"] == 0)
            val_ids = set(base.info.loc[val_row_mask, "id"])
            train_idx, val_idx = [], []
            for i, bid in enumerate(full.base_ids):
                (val_idx if bid in val_ids else train_idx).append(i)
            self.train_set = Subset(full, train_idx)
            self.val_set = Subset(base, val_row_mask.to_numpy().nonzero()[0])
        else:
            years = base.info["year"].values
            aug = base.info["aug"].values
            newest_idx = np.where((years >= 2022) & (aug == 0))[0].tolist()
            old_idx = np.where(years < 2022)[0].tolist()
            self.train_set = Subset(base, old_idx)
            self.val_set = Subset(base, newest_idx)

    def _create_champion(self):
        base = Dataset(
            dataset_path = self.dataset_path,
            split        = "train_val_gpt_aug3" if self.augmentation else "train_val_gpt",
            transforms   = self.train_transform,
            metadata     = self.metadata.copy(),
            train_on_log = self.train_on_log
        )
        if self.augmentation:
            full = RandomPerIdDataset(base, mix_fields=("title", "summary"))
            originals = base.info.query("aug == 0").reset_index()
            champ_row = originals.sample(n=1, random_state=self.seed).iloc[0]
            champ_id = champ_row["id"]
            champ_csv_idx = [int(champ_row["index"])]
            train_idx = [i for i, bid in enumerate(full.base_ids) if bid != champ_id]
            self.train_set = Subset(full, train_idx)
            self.val_set = Subset(base, champ_csv_idx)
        else:
            rng = random.Random(self.seed)
            orig_idxs = base.info.index[base.info["aug"] == 0].tolist()
            champ_idx = rng.choice(orig_idxs)
            self.val_set = Subset(base, [champ_idx])
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
    def train_class_dataloader(self):
        # 1) build the flat list of true labels for every index in self.train_set
        all_labels = []
        # self.train_set is a Subset, so .indices gives the underlying Dataset indices
        base_ds = self.train_set.dataset
        for idx in self.train_set.indices:
            # your Dataset.__getitem__ already attaches "label" as torch.long
            lab = base_ds[idx]["label"].item()
            all_labels.append(lab)
        all_labels = np.array(all_labels, dtype=np.int64)

        # 2) count and desired proportions
        counts    = np.bincount(all_labels, minlength=3).astype(np.float32)
        desired_p = np.array([0.25, 0.50, 0.25], dtype=np.float32)
        # if you wanted "normal" exactly twice low/high in absolute terms:
        # desired_p = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        # desired_p /= desired_p.sum()

        # 3) per-class weight = desired_p[c] / counts[c]
        class_weights = desired_p / counts
        # 4) per-sample weights
        sample_weights = class_weights[all_labels]

        # 5) sampler
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True
        )

        # 6) DataLoader with sampler (drop shuffle!)
        return DataLoader(
            self.train_set,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers,
            pin_memory  = True,
        )
    
    def train_dataloader_by_class(self, class_filter):
        """
        If class_filter is None, returns the vanilla shuffle DataLoader.
        Otherwise filters to only those base‐IDs whose 3‐way label == class_filter,
        and returns a rebalanced sampler DataLoader.
        """
        from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
        import math

        class_filter = int(class_filter)

        # 1) get the RandomPerIdDataset wrapper and its base DataFrame
        full_wrapper: RandomPerIdDataset = self.train_set.dataset
        base_df = full_wrapper.base.info

        # 2) build base_id → label from the original (aug==0) rows
        orig = base_df.query("aug == 0")[["base_id", "views"]]
        mu, sigma = full_wrapper.base.mu, full_wrapper.base.sigma
        def make_label(v):
            l = math.log(v + 1)
            if   l < mu - sigma:     return 0
            elif l > mu + 0.5*sigma: return 2
            else:                    return 1
        label_map = {row.base_id: make_label(row.views) for row in orig.itertuples()}

        # 4) otherwise pick only those indices in full_wrapper.base_ids
        idxs = [i for i, bid in enumerate(full_wrapper.base_ids)
                if label_map.get(bid, -1) == class_filter]
        if len(idxs)==0:
            raise ValueError(f"No samples for class_filter={class_filter}")

        # 5) build a Subset and sampler to rebalance if you like
        sub = Subset(full_wrapper, idxs)
        

        return DataLoader(sub,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader_by_class(self, class_filter):
        """
        Like val_dataloader(), but only keeps those val examples whose 3-way label == class_filter.
        """
        from torch.utils.data import Subset, DataLoader

        class_filter = int(class_filter)
        base_ds  = self.val_set.dataset   # this is your Dataset instance
        val_idxs = self.val_set.indices   # the row‐indices into base_ds

        # n.b. we pull the precomputed label out of each sample
        keep = []
        for idx in val_idxs:
            lbl = base_ds[idx]["label"].item()
            if lbl == class_filter:
                keep.append(idx)

        if not keep:
            raise ValueError(f"No validation samples for class_filter={class_filter}")

        subset = Subset(base_ds, keep)
        return DataLoader(
            subset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
        )

class ConcatDataModule():
    def __init__(self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title"],
        val_ratio: float = 0.1,
        seed: int = 42,
        num_epochs: int = 1,
        planification="same_newest",
        train_on_log=False,
        augmentation=True):

        self.dataset_path = dataset_path
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_ratio = val_ratio
        self.seed = seed
        self.aug = 4
        self.num_epochs = num_epochs
        self.name = planification
        self.val = None
        self.planification = []
        self.train_on_log = train_on_log
        self.augmentation = augmentation
        self.planify()

    def save_planification(self):
        filepath = f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_{self.train_on_log}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.planification, f)
        with open(f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_{self.train_on_log}_val.pkl", "wb") as f:
            pickle.dump(self.val, f)

    def planify(self):
        planification = self.name
        if os.path.exists(f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_{self.train_on_log}.pkl") and os.path.exists(f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_val.pkl"):
            with open(f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_{self.train_on_log}.pkl", "rb") as f:
                self.planification = pickle.load(f)
            with open(f"{self.dataset_path}/{self.name}_{self.num_epochs}_{self.batch_size}_{self.train_on_log}_val.pkl", "rb") as f:
                self.val = pickle.load(f)
            return
        else:
            if planification == "same_newest":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="newest",
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log  
                ) ]*self.num_epochs
            elif planification == "same_champion":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="champion",
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                )] * self.num_epochs
            elif planification == "same_split":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="split",
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                )] * self.num_epochs
            elif planification == "epoque_validation":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="epoque_validation",
                    epoch=i,
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                ) for i in range(self.num_epochs)]
            elif planification == "epoque_champion":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="epoque_champion",
                    epoch=i,
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                ) for i in range(self.num_epochs)]
            elif planification == "outliers":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="outliers",
                    epoch=i,
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                ) for i in range(self.num_epochs)]
            elif planification == "outliers_champ":
                self.planification = [DataModule(
                    self.dataset_path,
                    self.train_transform,
                    self.test_transform,
                    self.batch_size,
                    self.num_workers,
                    self.metadata,
                    self.val_ratio,
                    self.seed,
                    build="outliers_champ",
                    epoch=i,
                    augmentation=self.augmentation,
                    train_on_log=self.train_on_log
                ) for i in range(self.num_epochs)]
            else:
                raise ValueError(f"Unknown planification: {planification}")
        self.val = self.planification[0].val_dataloader()
        self.planification = [p.train_dataloader() for p in self.planification if p is not None]
        self.save_planification()



