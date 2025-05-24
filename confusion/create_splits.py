import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import hydra
from tqdm import tqdm

# Suppose datamodule is your DataModule instance

@hydra.main(config_path="../configs", config_name="confusion", version_base="1.1")
def build(cfg):
    datamodule= hydra.utils.instantiate(cfg.datamodule)
    

        
    with open("confusion/datamodule.pkl", "wb") as f:
        pickle.dump(datamodule, f)
if __name__ == "__main__":
    build()
    