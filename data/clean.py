#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np

# 1) CONFIGURATION
CSV_IN  = "dataset/train_val_gpt_aug3.csv"
JSON_IN = "channel_to_idx.json"

CSV_OUT = "dataset/train_val_gpt_aug3_cleaned.csv"
JSON_OUT= "channel_to_idx_cleaned.json"

# a) channels to drop entirely
DROP_CHANNELS = {
    "UCN9-Y2E0_fLxtcWAgr89CsQ",
    "UC-vrN89jIox3XqAKbEMeFgQ",
    "UCWnTC8o8AycL6a2eRrKSdBg",
    "UCyp-EzJrdTOd9uNPvmdst-w",
    "UCpJ6Dn01AqjvFVN2EiK72Ag",
    "UC8P49XCWzrcRVF1uD0ZdEIA",
    "UCFhLnzPQFjTtlUuf39emMfg",
    "UCpa2QppVwCuivaWe5qfVY_Q",
    "UCCykQNoqyABmoa3ZoaU93XA",
}

# b) channels for 30% outlier removal
OUTLIER_CHANNELS = {
    "UC-1rx8j9Ggp8mp4uD0ZdEIA",
    "UC6P24bhhCmMPOcujA9PKPTA",
}

# c) groups mapping: map every channel in a group to the group's first channel
GROUPS = [
    [
      "UCZtZ5HkDba6_zM_wpaQn60w",
      "UCk6Znl7ryh2dKCX5opiu7ng",
      "UC6qhhbnMeoOPFBf-cjUgn3g",
      "UCcRbWavyHkNsXdfot0yo8zw",
      "UCFfovcqnybrMbIrNjyND-nA",
    ],
    [
      "UCpUhcwq3oB7HKvrNeQmxJsg",
      "UCDp2VWtNgiaTjkhfCvdZmjA",
      "UCHUnLXalwe66kBndfmH5cXA",
    ],
    [
      "UCN5mdjo8Odux4PvR297-K4Q",
      "UCUrJu1fEq7kH9CBXaPH0PGw",
    ],
    [
      "UC9pJs4nFKM9xuW_enR4hHig",
      "UCTCQA8vgOHhK9hKWsPncy7w",
      "UCNvn1EEO-ZgN2IHkCLnA1Ng",
      "UC2zrmdd_rWlMc2rAufX9bSg",
    ],
    [
      "UC-1rx8j9Ggp8mp4uD0ZdEIA",
      "UCrUPTKJg_Oe-Bc4aC5CzrEQ",
      "UCamfFGyiS8aGLBK4sQ661ew",
    ],
    [
      "UC71x8bBOwIEscxAltguHXMQ",
      "UCO7o35HtN0bRnCP1y5NNCng",
    ],
    [
      "UCKMV8axq3zTUmiyXav2Ytew",
      "UCIAegrInVcw19QXmblowt2g",
    ],
    [
      "UCVNNriW_Mc2prgngYZ69T0A",
      "UCX894EiI3iOKez6ULmZFlOg",
      "UCl6p6SZMiI966k1ZzdzY9sA",
    ],
]

# build a flat mapping from channel → rep
GROUP_MAP = {}
for group in GROUPS:
    rep = group[0]
    for ch in group:
        GROUP_MAP[ch] = rep

# ------------------------------------------------------------------------
# 2) LOAD
df = pd.read_csv(CSV_IN)

# 3) DROP unwanted channels
df = df[~df["channel"].isin(DROP_CHANNELS)].copy()

# 4) OUTLIER removal for specified channels
def drop_outliers(sub: pd.DataFrame, q=0.30):
    low  = sub["views"].quantile(q/2)
    high = sub["views"].quantile(1 - q/2)
    return sub[(sub["views"] >= low) & (sub["views"] <= high)]

keep_frames = []
for ch, sub in df.groupby("channel"):
    if ch in OUTLIER_CHANNELS:
        sub = drop_outliers(sub, q=0.30)
    keep_frames.append(sub)
df = pd.concat(keep_frames, ignore_index=True)

# 5) GROUP remapping
df["channel"] = df["channel"].map(lambda ch: GROUP_MAP.get(ch, ch))

# 6) REBUILD channel_to_idx.json
with open(JSON_IN, "r") as f:
    old_map = json.load(f)

# drop removed channels
new_keys = sorted({old_map[ch] for ch in df["channel"].unique() if ch in old_map})
# reassign clean indices 0..N-1
new_map = { ch: idx for idx, ch in enumerate(sorted(df["channel"].unique())) }

# 7) SAVE
df.to_csv(CSV_OUT, index=False)
with open(JSON_OUT, "w") as f:
    json.dump(new_map, f, indent=2)

print("Saved cleaned CSV →", CSV_OUT)
print("Saved cleaned JSON →", JSON_OUT)