import os, sys, math
import torch
import hydra
from omegaconf import OmegaConf
from torch.nn import Softmax
from torch import nn

@hydra.main(config_path="configs", config_name="validate", version_base="1.1")
def validate(cfg):
    # 1) setup device and dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = hydra.utils.instantiate(cfg.datamodule)
    val_loader = dm.val_dataloader()

    # 2) load the classifier
    clf: nn.Module = hydra.utils.instantiate(cfg.model.instance).to(device)
    clf.load_state_dict(torch.load(cfg.ckpt_class, map_location=device))
    clf.eval()
    softmax = Softmax(dim=1)

    # 3) build three regressors (one per class)
    regressors = []
    for c in [0, 1, 2]:
        m = hydra.utils.instantiate(cfg.model.instance).to(device)
        # swap out the head for a 1-D output
        m.head = nn.Sequential(
            nn.LayerNorm(m.head[0].normalized_shape),
            nn.Dropout(cfg.model.head_dropout),
            nn.Linear(cfg.model.head_hidden_dim, 1),
        ).to(device)
        ckpt = os.path.join(cfg.ckpt_dir, f"{cfg.model.name}_{c}.pt")
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        regressors.append(m)

    # 4) loss function
    loss_fn = hydra.utils.instantiate(cfg.loss_fn).to(device)

    # 5) loop over validation set
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in val_loader:
            # move tensors
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            # classifier → class probabilities
            logits = clf(batch)                 # [B,3]
            probs  = softmax(logits)            # [B,3]

            # regressors → one prediction each
            preds = []
            for m in regressors:
                p = m(batch).view(-1,1)         # [B,1]
                preds.append(p)
            preds = torch.cat(preds, dim=1)    # [B,3]

            # combine
            if dm.train_on_log:
                # weighted average in log‐space
                final = (probs * preds).sum(dim=1)      # [B]
            else:
                # geometric average in raw space
                logp = torch.log(preds.clamp(min=1e-6))
                lg   = (probs * logp).sum(dim=1)
                final = torch.exp(lg)                   # [B]

            # compute loss against the true target
            target = batch["target"].view(-1)
            loss = loss_fn(final, target)
            total_loss += loss.item() * final.size(0)
            total_n    += final.size(0)

    print(f"\n>>> Validation Loss: {total_loss/total_n:.4f}")

if __name__ == "__main__":
    validate()