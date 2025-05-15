import pandas as pd, numpy as np, json

# 1. load your training CSV
df = pd.read_csv("dataset/train_val_gpt.csv")

# 2. compute log1p of views
df["log_views"] = np.log1p(df["views"])

# 3. channel → mean(log_views)
means = df.groupby("channel")["log_views"].mean()

# 4. normalize to [0,1] by dividing by the max
norm = means.max()
means = (means / norm).to_dict()

# 5. write out JSON
with open("channel_mean_log.json","w") as f:
    json.dump(means, f)