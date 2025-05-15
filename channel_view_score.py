import pandas as pd

df = pd.read_csv("dataset/train_val_gpt.csv")
# compute average views per channel
channel_means = df.groupby("channel")["views"].mean()
max_mean      = channel_means.max()
# normalize to [0,1]
channel_score = (channel_means / max_mean).to_dict()

# save to disk
import json
with open("channel_score.json", "w") as f:
    json.dump(channel_score, f)