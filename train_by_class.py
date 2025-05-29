import torch
import wandb
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler

from utils.sanity import show_images
import signal, sys
import os
from PIL import Image

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

def get_text_blocks(peft_model):
    """
    Return the list/ModuleList that contains the transformer blocks
    inside a PEFT-wrapped text backbone, for both BERT- and LLama/Gemma-like
    architectures. Returns None if nothing is trainable (all frozen).
    """
    # 1) DistilBERT / BERT / RoBERTa
    if hasattr(peft_model, "transformer"):
        return peft_model.transformer.layer

    # 2) Llama-family: PeftModel.model.model.layers
    if hasattr(peft_model, "model"):
        inner = peft_model.model                       # GemmaForCausalLM
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return inner.model.layers
        if hasattr(inner, "layers"):                   # some Falcon-style models
            return inner.layers

    # 3) Nothing found → return None
    return None

@hydra.main(config_path="configs", config_name="train_by_class", version_base="1.1")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"🏃‍♂️ Training process PID = {os.getpid()}")
    print(f"To early stop, do: kill -SIGUSR1 {os.getpid()}")
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    model   = hydra.utils.instantiate(cfg.model.instance).to(device)
    scaler = GradScaler(device="cuda")

    # Configuring Early Stop
    def save_and_exit(*_):
        torch.save(model.state_dict(), cfg.checkpoint_path)
        print(f"🔖 checkpoint written to {cfg.checkpoint_path}")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, save_and_exit)
    # -------------------------------------------------------------------

    opt_cfg = OmegaConf.to_container(cfg.optim, resolve=True, enum_to_str=True)
    
    lr_body = opt_cfg.pop("lr_body")

    lr_image_adapter =  opt_cfg.pop("lr_image_adapter")  # only image‐LoRA
    lr_text_adapter = opt_cfg.pop("lr_text_adapter")    # only text‐LoRA (title+summary)
    lr_head = opt_cfg.pop("lr_head")    # head MLP, year/date/channel embeds, etc.
    lr_fusion = opt_cfg.pop("lr_fusion")

    # ------------------------------------------------------------------ #
    # Build the two parameter groups                                  #
    # ------------------------------------------------------------------ #
    param_groups = []
    decay = cfg.layer_decay
    try :
        num_blocks = len(model.img_encoder.visual.trunk.blocks)
        # collect all transformer blocks, assign lr = lr_body * decay**(depth)
        for depth, module in enumerate(model.img_encoder.visual.trunk.blocks):
            
            param_groups.append({
            "params": module.parameters(),
            "lr": lr_body * (decay ** (num_blocks - depth - 1))
            })
    
    except AttributeError:
        num_blocks = len(model.img_encoder1.visual.trunk.blocks)
        # collect all transformer blocks, assign lr = lr_body * decay**(depth)
        for depth, module in enumerate(model.img_encoder1.visual.trunk.blocks):
            
            param_groups.append({
            "params": module.parameters(),
            "lr": lr_body * (decay ** (num_blocks - depth - 1))
            })

    # ───────── text blocks for layer-decay LR ─────────
    text_blocks = get_text_blocks(model.text_encoder)  # your helper
    if any(p.requires_grad for p in model.text_encoder.base_model.parameters()):
        num_text_blocks = len(text_blocks)
        for depth, module in enumerate(text_blocks):
            layer_lr = lr_body * (decay ** (num_text_blocks - depth - 1))
            param_groups.append({"params": module.parameters(), "lr": layer_lr})
    else:
        print("⚠  text backbone frozen → skipping layer-wise LR groups.")

    # --- collect LoRA adapter parameters at head_lr ---
    # --- collect image vs text LoRA adapters separately ---
    img_lora_params = []
    txt_lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad or "lora_" not in name:
            continue
        if name.startswith("img_encoder") or "img_encoder" in name:
            img_lora_params.append(param)
        else:
            txt_lora_params.append(param)

    if img_lora_params:
        param_groups.append({"params": img_lora_params, "lr": lr_image_adapter})
    if txt_lora_params:
        param_groups.append({"params": txt_lora_params, "lr": lr_text_adapter})


    
    head_params = list(model.head.parameters())
    # always optimize any of these if they exist:
    for attr in ("year_proj","ch_emb","cy_proj","year_emb","date_proj"):
        if hasattr(model, attr):
            head_params += list(getattr(model, attr).parameters())
    
    param_groups.append({"params": head_params,     "lr": lr_head})

    # after adapter_params and head_params...
    if hasattr(model, "fusion_transformer"):
        fusion_params = list(model.fusion_transformer.parameters())
        param_groups.append({
            "params": fusion_params,
            "lr": lr_fusion,   # or a small fraction of lr_body if you prefer
        })
    seen = set()
    for g in param_groups:
        #uniq = [p for p in g["params"] if id(p) not in seen]
        #g["params"] = uniq                      # drop dups in-place
        #seen.update(map(id, uniq))
        # split into decay vs no_decay
        decay, no_decay = [], []
        for p in g["params"]:
            if id(p) in seen: continue
            seen.add(id(p))
            # any 1D param is bias or LayerNorm weight → no decay
            if p.ndim == 1:
                no_decay.append(p)
            else:
                decay.append(p)
        # apply your global weight_decay only to decay set
        wd = cfg.optim.weight_decay
        g["params"]       = decay
        g["weight_decay"] = wd
        if no_decay:
            param_groups.append({
                "params": no_decay,
                "weight_decay": 0.0,
                "lr":        g["lr"]
            })
    optimizer = hydra.utils.instantiate(opt_cfg, params=param_groups,_convert_="all")
    # ── dataloaders ────────────────────────────────────────────
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # if training a regressor, only keep samples of that class
    if cfg.class_filter is not None:
        ds = datamodule.train_set  # this is a Subset
        # gather only indices whose label == class_filter
        idxs = [i for i in ds.indices
                if ds.dataset[i]["label"].item() == cfg.class_filter]
        datamodule.train_set = torch.utils.data.Subset(ds.dataset, idxs)
        train_loader = torch.utils.data.DataLoader(
            datamodule.train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    else:
        train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()
    train_transform = hydra.utils.instantiate(cfg.datamodule.train_transform)

    img_dir = "data/centroids"  # Path to the directory containing images
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    transfo=torch.zeros(len(img_files), 1,3, 224, 224)
    for i,fname in enumerate(img_files):
        #img = Image.open(os.path.join(img_dir, fname)).convert('L').resize((64, 64))  # grayscale, 64x64
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')  # full hd
        transfo[i]= train_transform(img).unsqueeze(0)
    torch.save(transfo,'data/transformed_centroids.pt')

    # ── cosine-with-warmup scheduler ───────────────────────────
    if cfg.use_warmup:
        num_epochs      = cfg.epochs
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )



    # -- sanity check
    train_sanity = show_images(train_loader, name="assets/sanity/train_images")
    (
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if logger is not None
        else None
    )
    if val_loader is not None:
        val_sanity = show_images(val_loader, name="assets/sanity/val_images")
        logger.log(
            {"sanity_checks/val_images": wandb.Image(val_sanity)}
        ) if logger is not None else None

    best_val_loss = float("inf")
    epochs_since_improve = 0
    patience = cfg.early_stopping.patience
    min_epochs = cfg.early_stopping.min_epochs
    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            with autocast(device_type="cuda"):
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()
            if cfg.use_warmup:
                scheduler.step()
                if logger is not None:
                    lrs = scheduler.get_last_lr()
                    logger.log({"lr_head": lrs[0], "lr_body": lrs[1]})
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        print(f"[Epoch {epoch:02d}] Training loss: {epoch_train_loss:.4f}")

        if logger is not None:
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                }
            )


        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                with torch.no_grad():
                    preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            (
                logger.log(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )
            # ––– check for improvement –––
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improve = 0
                torch.save(model.state_dict(), cfg.checkpoint_path)
                print(f"[Epoch {epoch:02d}] New best val loss: {best_val_loss:.4f} (saved)")
            else:
                epochs_since_improve += 1
                print(f"[Epoch {epoch:02d}] No improvement: {epoch_val_loss:.4f} (best {best_val_loss:.4f}), patience {epochs_since_improve}/{patience}")
                if epochs_since_improve >= patience and epoch >= min_epochs:
                    print(f"Early stopping triggered. Stopping at epoch {epoch}.")
                    break
        
        if hasattr(model, "epoch_scheduler_hook"):
            model.epoch_scheduler_hook() 

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
