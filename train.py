import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← AJOUTE CETTE LIGNE

from PIL import Image
import wandb
import hydra
from torch import cuda, device as torch_device, save as torch_save, no_grad, zeros
from torch import Tensor,save
from torch.cuda import empty_cache, ipc_collect
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from torch.amp import autocast, GradScaler

from utils.sanity import show_images
import signal
import sys
import os

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

@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    if cuda.is_available():
        dev = torch_device("cuda")
    elif hasattr(cuda, "backends") and hasattr(cuda.backends, "mps") and cuda.backends.mps.is_available():
        dev = torch_device("mps")
    else:
        dev = torch_device("cpu")
    
    print(f"🏃‍♂️ Training process PID = {os.getpid()}")
    print(f"To early stop, do: kill -SIGUSR1 {os.getpid()}")

    # Instantiate model and loss
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    model = hydra.utils.instantiate(cfg.model.instance).to(dev)
    continu = False
    scaler = GradScaler(device="cuda")
    continu=False
    if continu:# Load the saved state dict
        checkpoint_path = 'checkpoints/SIGLIP_DISTILBERT_ATTENTION_LORA_2025-05-22_23-15-08.pt'
        empty_cache()
        ipc_collect()
        model.load_state_dict(torch_save(checkpoint_path, map_location=dev))

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
    try:
        num_blocks = len(model.img_encoder.visual.trunk.blocks)
        # collect all transformer blocks, assign lr = lr_body * decay**(depth)
        for depth, module in enumerate(model.img_encoder.visual.trunk.blocks):
            param_groups.append({
            "params": module.parameters(),
            "lr": lr_body * (decay ** (num_blocks - depth - 1))
            })
    
    except AttributeError:
        try:
            num_blocks = len(model.img_encoder1.visual.trunk.blocks)
            # collect all transformer blocks, assign lr = lr_body * decay**(depth)
            for depth, module in enumerate(model.img_encoder1.visual.trunk.blocks):
                
                param_groups.append({
                "params": module.parameters(),
                "lr": lr_body * (decay ** (num_blocks - depth - 1))
                })
        except AttributeError:
            resnet_layers = [
                model.img_encoder.layer1,
                model.img_encoder.layer2,
                model.img_encoder.layer3,
                model.img_encoder.layer4,
            ]
            num_blocks = len(resnet_layers)
            for depth, module in enumerate(resnet_layers):
                param_groups.append({
                    "params": module.parameters(),
                    "lr": body_lr * (decay ** (num_blocks - depth - 1))
                })

    # ───────── text blocks for layer-decay LR ─────────
    text_blocks = get_text_blocks(model.text_encoder)
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
    for attr in ("year_proj", "ch_emb", "cy_proj", "year_emb", "date_proj"):
        if hasattr(model, attr):
            head_params += list(getattr(model, attr).parameters())
    
    param_groups.append({"params": head_params,     "lr": lr_head})

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
    datamodule_planification = datamodule.planification
    
    train_transform = hydra.utils.instantiate(cfg.datamodule.train_transform)

    img_dir = "data/centroids"
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    transfo = zeros(len(img_files), 1, 3, 224, 224)
    for i, fname in enumerate(img_files):
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        transfo[i] = train_transform(img).unsqueeze(0)
    torch_save(transfo, 'data/transformed_centroids.pt')

    # ── cosine-with-warmup scheduler ───────────────────────────
    if cfg.use_warmup:
       
        total_steps = sum([
                len(loader) for loader in datamodule_planification]
            ) 
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

    # -- sanity check
    if cfg.sanity_check.enabled:
        train_loader = datamodule_planification[0]
        val_loader = datamodule_planification[0] if len(datamodule_planification) > 1 else None
    
        train_sanity = show_images(train_loader, name="assets/sanity/train_images")
        if logger is not None:
            logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if val_loader is not None and logger is not None:
            val_sanity = show_images(val_loader, name="assets/sanity/val_images")
            logger.log({"sanity_checks/val_images": wandb.Image(val_sanity)})

    best_val_loss = float("inf")
    epochs_since_improve = 0
    patience = cfg.early_stopping.patience
    min_epochs = cfg.early_stopping.min_epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        train_loader = datamodule_planification[epoch]
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(dev)
            batch["target"] = batch["target"].to(dev).squeeze()
            with autocast(device_type="cuda"):
                preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
            if logger is not None:
                logger.log({"loss": loss.detach().cpu().numpy()})
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

        if logger is not None:
            logger.log({
                "epoch": epoch,
                "train/loss_epoch": epoch_train_loss,
            })

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        val_loader = datamodule.val
        if val_loader is not None:
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(dev)
                batch["target"] = batch["target"].to(dev).squeeze()
                with no_grad():
                    preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
            if logger is not None:
                logger.log({
                    "epoch": epoch,
                    **val_metrics,
                })
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_since_improve = 0
                torch_save(model.state_dict(), cfg.checkpoint_path)
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

    torch_save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
