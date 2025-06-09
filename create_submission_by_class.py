# create_submission.py
import os
import torch
import hydra
import pandas as pd
from torch.nn import Softmax
from omegaconf import OmegaConf
from torch import nn

@hydra.main(config_path="configs", config_name="submission_by_class", version_base="1.1")
def create_submission(cfg):
    OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
    # 1) device & dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = hydra.utils.instantiate(cfg.datamodule)
    test_loader = dm.test_dataloader()

    # 2) load 3-way classifier
    clf: nn.Module = hydra.utils.instantiate(cfg.classif).to(device)
    clf.load_state_dict(torch.load(cfg.ckpt_class, map_location=device))
    clf.eval()
    softmax = Softmax(dim=1)

    # 3) load the three regressors (original heads)
    regressors = []
    for c in [0, 1, 2]:
        m: nn.Module = hydra.utils.instantiate(cfg.model.instance).to(device)
        ckpt = os.path.join(cfg.ckpt_dir, f"{cfg.model.name}_{c}.pt")
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        regressors.append(m)

    # 4) inference loop
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            # move tensors
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # class logits → probabilities
            probs = softmax(clf(batch))   # [B,3]

            # each regressor → [B,1]
            preds = torch.cat([ r(batch).view(-1,1) for r in regressors ], dim=1)  # [B,3]

            # combine:
            if dm.train_on_log:
                # preds are log1p(y); final is expectation(log1p y)
                final = (probs * preds).sum(dim=1)
                # invert log1p → raw y
                view_preds = torch.expm1(final).clamp(min=0.0)
            else:
                # preds are raw y; geometric mean
                logp = torch.log(preds.clamp(min=1e-6))
                lg   = (probs * logp).sum(dim=1)
                view_preds = torch.exp(lg)

            # collect
            ids = batch["id"]
            for i, vid in enumerate(ids):
                rows.append((int(vid), float(view_preds[i].cpu().item())))

    # 5) write submission.csv
    submission = pd.DataFrame(rows, columns=["ID","views"])
    out_path = os.path.join(cfg.root_dir, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"📝 Submission written to {out_path}")

if __name__ == "__main__":
    create_submission()