import torch
import pandas as pd
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        if "description" in info.columns:
            info["description"] = info["description"].fillna("")
        if "summary" in info.columns:
            info["summary"] = info["summary"].fillna("")
        #info["meta"] = info[metadata].astype(str, copy=True).agg(" + ".join, axis=1)
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
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
