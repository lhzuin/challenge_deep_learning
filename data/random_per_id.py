import numpy as np, torch

class RandomPerIdDataset(torch.utils.data.Dataset):
    """
    Wrap an existing `Dataset` so that:
      • __len__ = # unique original videos          (base_id)
      • each __getitem__ returns ONE image/row chosen at random
        +   independent random rows for chosen text fields (title, summary…).
    Nothing else changes - all numeric tensors come from the inner Dataset.
    """
    def __init__(self, base_dataset, mix_fields=("title", "summary")):
        self.base = base_dataset
        self.mix_fields = mix_fields

        # -- group dataframe rows by base_id once ---------------------------
        df = self.base.info  
        self.groups = (
            df.groupby("Unnamed: 0", sort=False)
              .apply(lambda g: g.index.to_numpy()) 
              .to_dict()
        )
        self.base_ids = list(self.groups.keys())

        self.rng = np.random.default_rng() 

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.base_ids)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        rows = self.groups[self.base_ids[idx]]

        # --- one random row for the *image* and all numeric tensors --------
        idx_img = int(self.rng.choice(rows))
        sample  = self.base[idx_img] 

        # --- independently mix specified text fields ----------------------
        for field in self.mix_fields:
            flag = getattr(self.base, f"has_{field}", False)
            if flag:
                idx_f = int(self.rng.choice(rows))
                sample[field] = str(self.base.info.at[idx_f, field])
        return sample

    # ----------------------------------------------------------------------
    def set_epoch(self, epoch:int):
        "Deterministic but epoch-specific reshuffle (call once per epoch)."
        self.rng = np.random.default_rng(seed=epoch)