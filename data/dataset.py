import torch
import json
import pandas as pd
from PIL import Image
import numpy as np
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, train_on_log=False):
        self.train_on_log = train_on_log
        self.min_year = 2011
        self.max_year = 2025
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        #print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        if self.train_on_log:
                info["views"] = info["views"].fillna(0).astype(float)
                info["views"] = np.log1p(info["views"]).astype(float)
        self.info = info
        logs = np.log1p(self.info["views"].values.astype(np.float64))
        self.mu, self.sigma = logs.mean(), logs.std()
        
        if "description" in info.columns:
            info["description"] = info["description"].fillna("")
        
        if "summary" in info.columns:
            info["summary"] = info["summary"].fillna("")

        
        self.has_year_z = False
        self.has_year_norm = False
        self.has_year_idx = False
        self.has_date_sin = False
        self.has_channel_idx = False
        self.has_title = False
        self.has_summary = False

        if "summary" in metadata:
            self.has_summary = True
            metadata.remove("summary")
            self.summary = info["summary"].astype("string")

        if "title" in metadata:
            self.has_title = True
            metadata.remove("title")
            self.title = info["title"].astype("string")

        if "year_z" in metadata:
            self.has_year_z = True
            metadata.remove("year_z")
            self.years = info["year"].astype(int).values
        
        if "year_norm" in metadata:
            self.has_year_norm = True
            metadata.remove("year_norm")
            self.years = info["year"].astype(int).values
            
        
        if "year_idx" in metadata:
            self.has_year_idx = True
            metadata.remove("year_idx")
            self.years = info["year"].astype(int).values

        if "date_sin" in metadata:
            self.has_date_sin = True
            metadata.remove("date_sin")
            ts = pd.to_datetime(
                info["date"],
                format="ISO8601",  # all ISO-8601 variants
                utc=True
            )
            self.month = ts.dt.month.values  # 1–12
            self.weekday = ts.dt.weekday.values # 0–6
            self.hour = ts.dt.hour.values  # 0–23

        if "channel_idx" in metadata:
            self.has_channel_idx = True
            metadata.remove("channel_idx")
            # load the **same** mapping for train _and_ test
            with open("channel_to_idx.json") as f:
                self.ch_to_idx = json.load(f)
            self.unk_idx = self.ch_to_idx["__unk__"]

            channels = info["channel"].astype(str)
            self.ch_idx = channels.map(lambda c: self.ch_to_idx.get(c, self.unk_idx)).values

        info["meta"] = (
            info[metadata]
                .fillna("")
                .astype("string")
                .agg(lambda r: " + ".join(v for v in r if v), axis=1)
        )
        if "views" in info.columns:
            
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values

        # - transforms
        self.transforms = transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],
        }

        if self.has_title:
            value["title"]   = str(self.title.iloc[idx])
        
        if self.has_summary:
            value["summary"] = str(self.summary.iloc[idx])

        # 0) year_z 
        if self.has_year_z:
            z = (self.years[idx] - self.mu) / self.sigma
            z_clipped = np.clip(z, -3.0, 3.0)
            value["year_z"] = torch.tensor([z_clipped], dtype=torch.float32)

        # 1) year_norm ∈ [0,1]
        if self.has_year_norm:
            y = self.years[idx]
            #yr_norm = (y - self.min_year) / (self.max_year - self.min_year)
            yr_norm = 2*((y - self.min_year)/(self.max_year-self.min_year)) - 1.0
            value["year_norm"] = torch.tensor([yr_norm], dtype=torch.float32)

        # 2) year bucket for embedding (0..max_year-min_year,  else last idx)
        if self.has_year_idx:
            bucket = min(y, self.max_year) - self.min_year
            value["year_idx"] = torch.tensor(bucket, dtype=torch.long)

        # 3) cyclical month/day/hour
        if self.has_date_sin:
            m, w, h = self.month[idx], self.weekday[idx], self.hour[idx]
            value["m_sin"], value["m_cos"] = (
                torch.tensor([np.sin(2*np.pi*m/12)], dtype=torch.float32),
                torch.tensor([np.cos(2*np.pi*m/12)], dtype=torch.float32),
            )
            value["d_sin"], value["d_cos"] = (
                torch.tensor([np.sin(2*np.pi*w/7 )], dtype=torch.float32),
                torch.tensor([np.cos(2*np.pi*w/7 )], dtype=torch.float32),
            )
            value["h_sin"], value["h_cos"] = (
                torch.tensor([np.sin(2*np.pi*h/24)], dtype=torch.float32),
                torch.tensor([np.cos(2*np.pi*h/24)], dtype=torch.float32),
            )

        # 4) channel idx
        if self.has_channel_idx:
            value["channel_idx"] = torch.tensor(self.ch_idx[idx], dtype=torch.long)
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            if self.train_on_log:
                y = torch.tensor(self.targets[idx], dtype=torch.float32)
                y = torch.clamp(y, min=0.0)
                value["target"] = torch.log1p(y)
            else:
                value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        # build a 3-way label via ±2σ on the log-view
        raw_views = self.targets[idx]
        l = math.log(raw_views + 1)
        if   l < self.mu - self.sigma:
            label = 0   # low
        elif l > self.mu + 0.5*self.sigma:
            label = 2   # high
        else:
            label = 1   # normal
        value["label"] = torch.tensor(label, dtype=torch.long)
        return value
    
    def subset(dataset, indices):
        """Return a subset of the dataset."""
        new_dataset = self.__class__(self.dataset_path, self.split, self.transforms, [], train_on_log=self.train_on_log)
        new_dataset.info = self.info.iloc[indices].reset_index(drop=True)
        new_dataset.ids = self.ids[indices]
        new_dataset.text = self.text[indices]
        if self.has_title:
            new_dataset.title = self.title.iloc[indices].reset_index(drop=True)
        if self.has_summary:
            new_dataset.summary = self.summary.iloc[indices].reset_index(drop=True)
        if self.has_year_z:
            new_dataset.years = self.years[indices]
        if self.has_year_norm:
            new_dataset.years = self.years[indices]
        if self.has_year_idx:
            new_dataset.years = self.years[indices]
        if self.has_date_sin:
            new_dataset.month = self.month[indices]
            new_dataset.weekday = self.weekday[indices]
            new_dataset.hour = self.hour[indices]
        if self.has_channel_idx:
            new_dataset.ch_idx = self.ch_idx[indices]
            new_dataset.ch_to_idx = self.ch_to_idx
        new_dataset.transforms = self.transforms
        new_dataset.has_year_z = self.has_year_z
        new_dataset.has_year_norm = self.has_year_norm
        new_dataset.has_year_idx = self.has_year_idx
        new_dataset.has_date_sin = self.has_date_sin
        new_dataset.has_channel_idx = self.has_channel_idx
        new_dataset.has_title = self.has_title
        new_dataset.has_summary = self.has_summary
        return new_dataset
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class CustomConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum(self.datasets)

    def _cumsum(self, datasets):
        r, s = [], 0
        for d in datasets:
            l = len(d)
            r.append(l + s)
            s += l
        return r

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumulative_sizes[-1]
