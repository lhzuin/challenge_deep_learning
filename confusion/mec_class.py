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
    val_loader = datamodule.val_dataloader()
    pbar = tqdm(val_loader, desc="Confusion pass", leave=False)
    # our classifier has 3 classes: low, normal, high 
    confusion_mat = Confusion_matrix(num_classes=3)
    for i, batch in enumerate(pbar):
        batch["image"] = batch["image"].to(device)
        with torch.no_grad():
            # get raw logits [B,3]
            logits = model(batch)               # shape (B,3)
            # predicted class = argmax over dim=1
            pred_labels = logits.argmax(dim=1).cpu().numpy()
        # true labels must have been stored as integer 0/1/2 in `batch["label"]`
        true_labels = batch["label"].cpu().numpy()
        confusion_mat.update(true_labels, pred_labels)

    # dump out a heatmap as PNG
    confusion_mat.print_matrix(
        save_path=f'confusion/{cfg_model.model.name}_confusion_matrix.png',
        class_names=["low","normal","high"]
    )

if __name__ == "__main__":
    mec()