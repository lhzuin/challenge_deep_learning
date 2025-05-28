import hydra
from confusion_matrix import Confusion_matrix
import torch
import numpy as np
import	pickle as pkl
import sys, os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

@hydra.main(config_path="../configs", config_name="submission")
def mec(cfg_model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure the model is instantiated correctly from the config
    model = hydra.utils.instantiate(cfg_model.model.instance).to(device)
    checkpoint = torch.load(cfg_model.checkpoint_path, map_location=device)
    #print(checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    try:
        #datamodule=pkl.load(open("confusion/datamodule.pkl", "rb"))
        datamodule = hydra.utils.instantiate(cfg_model.datamodule)
    except:
        datamodule = hydra.utils.instantiate(cfg_model.datamodule)
    val_set = datamodule.val_dataloader()
    pbar = tqdm(val_set, desc=f"heh {0}", leave=False)
    confusion_mat = Confusion_matrix(num_classes=20)
    for i, batch in enumerate(pbar):
        batch["image"] = batch["image"].to(device)
        with torch.no_grad():
            preds = model(batch).squeeze().cpu().numpy()
            if cfg_model.train_on_log:
                preds = np.expm1(preds)       # exp(pred) – 1
                preds = np.clip(preds, 0, None)
            # Ensure batch["label"] is on CPU and numpy
            true_labels = np.int16(np.log(batch["target"].squeeze().cpu().numpy()))
            pred_labels = np.int16(preds-1)
            confusion_mat.update(true_labels, pred_labels)

    confusion_mat.print_matrix(save_path=f'confusion/confusion_matrix/{cfg_model.model.name}{cfg_model.time_stamp}confusion_matrix.png')

if __name__ == "__main__":
    mec()