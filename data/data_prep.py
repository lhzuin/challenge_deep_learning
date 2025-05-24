# utils/make_base_id.py  (run once during data-prep)
import pandas as pd, re, sys, pathlib

csv_in  = pathlib.Path("dataset/train_val_gpt_aug3.csv")
csv_out = csv_in.with_stem(csv_in.stem)

df = pd.read_csv(csv_in)

# strip the “_aug?” suffix → --DnfroyKQ8, --DnfroyKQ8_aug2, … → --DnfroyKQ8
df["base_id"] = df["id"].str.replace(r"_aug\d+$", "", regex=True)

df.to_csv(csv_out, index=False)
print("✨ wrote", csv_out)