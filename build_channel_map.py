import pandas as pd
import json

df = pd.read_csv("dataset/train_val_gpt.csv")
channels = sorted(df["channel"].unique().tolist())

# Reserve the last index for any “unknown” channels
mapping = { ch:i for i,ch in enumerate(channels) }
mapping["__unk__"] = len(channels)

with open("channel_to_idx.json","w") as f:
    json.dump(mapping, f, indent=2)