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
import math

#torch.autograd.set_detect_anomaly(True)

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

def run_epochs(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
               device, scaler, logger, start_epoch, num_epochs, checkpoint_path, patience=2, min_epochs=2, accum_steps = 2):
    
    best_val, no_imp = float("inf"), 0
    last_epoch = start_epoch
    global_step  = 0
    for e in tqdm(range(start_epoch, start_epoch + num_epochs)):
        last_epoch = e+1
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(e)
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {e}", leave=False)
        for i, batch in enumerate(pbar):
            # ----- forward  ----------------------------------------------------
            batch["image"]  = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()

            with autocast(device_type="cuda"):
                preds = model(batch).squeeze()
                loss  = loss_fn(preds, batch["target"]) / accum_steps   # ① scale down

            # ----- logging  ----------------------------------------------------
            if logger is not None:
                logger.log({"loss": loss.detach().cpu().item() * accum_steps})                            # ② log up-scaled

            # ----- backward / optim -------------------------------------------
            scaler.scale(loss).backward()

            # update only every `accum_steps`
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            epoch_train_loss += (loss.detach().cpu().numpy() * accum_steps* len(batch["image"]))
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()* accum_steps})
            
        # after the loop over pbar
        if (i + 1) % accum_steps != 0:     # leftovers
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
        epoch_train_loss /= num_samples_train
        print(f"[Epoch {e:02d}] Training loss: {epoch_train_loss:.4f}")
        if logger is not None:
            logger.log(
                {
                    "epoch": e,
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
                        "epoch": e,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )
            # ––– check for improvement –––
            if epoch_val_loss < best_val:
                best_val = epoch_val_loss
                epochs_since_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[Epoch {e:02d}] New best val loss: {best_val:.4f} (saved)")
            else:
                epochs_since_improve += 1
                print(f"[Epoch {e:02d}] No improvement: {epoch_val_loss:.4f} (best {best_val:.4f}), patience {epochs_since_improve}/{patience}")
                if epochs_since_improve >= patience and e >= min_epochs:
                    print(f"Early stopping triggered. Stopping at epoch {e}.")
                    break

    print(
        f"""Epoch {last_epoch-1}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )
    return last_epoch, best_val, no_imp

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

@hydra.main(config_path="configs", config_name="train_thaw", version_base="1.1")
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


    # Instantiate model and loss
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    

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
    
    # ── dataloaders ────────────────────────────────────────────
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    accum_steps = cfg.accum_steps

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

    # Phase 1: Head and Fusion only
    num_epochs = cfg.num_epochs_phase1
    for p in model.parameters(): 
        p.requires_grad = False
    for name,p in model.named_parameters():
        if name.startswith("fusion_transformer") or name.startswith("head") or any(name.startswith(attr) for attr in ("year_proj","ch_emb","cy_proj","date_proj")):
            p.requires_grad = True
    
    meta_params   = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(("year_proj",
                                                        "ch_emb",
                                                        "cy_proj",
                                                        "date_proj"))]

    head_params = list(model.head.parameters())
    param_groups_phase1 = [
        {"params": head_params,   "lr": cfg.model.lr_head_phase1},
        {"params": fusion_params, "lr": cfg.model.lr_fusion_phase1, "weight_decay": cfg.model.weight_decay_fusion_phase1},
        {"params": meta_params,   "lr": cfg.model.lr_head_phase1}
    ]

    # instantiate a *new* Hydra sub‐config on the fly
    head_fusion_optim_cfg = {
        "_target_": opt_cfg["_target_"],        # torch.optim.AdamW
        "weight_decay": opt_cfg["weight_decay"],
        "betas": opt_cfg["betas"],
    }

    optimizer = hydra.utils.instantiate(
        head_fusion_optim_cfg,
        params=param_groups_phase1,
        _convert_="all",
    )
    scheduler = None
    if cfg.use_warmup:
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        """
        steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = num_epochs*steps_per_epoch,
                eta_min = cfg.model.lr_fusion_phase1_final
        )
        """

    scaler = GradScaler(device="cuda")
    last_e, best_val, patience_cnt = run_epochs(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, scaler, logger, start_epoch=0, num_epochs=num_epochs, 
        checkpoint_path=cfg.checkpoint_path, patience=cfg.early_stopping.patience, min_epochs=cfg.early_stopping.min_epochs, accum_steps=accum_steps
    )

    # Phase 2: Image-LoRA only
    num_epochs = cfg.num_epochs_phase2
    for p in model.parameters(): p.requires_grad = False
    # Unfreeze only the image-adapter weights:
    for name, p in model.named_parameters():
        if "img_encoder" in name and "lora_" in name:
            p.requires_grad = True


    param_groups_phase2 = [
        {"params": img_lora_params, "lr": cfg.model.lr_image_adapter_phase2, "weight_decay": cfg.model.weight_decay_image_lora},
    ]

    optimizer = hydra.utils.instantiate(
        head_fusion_optim_cfg,            # same minimal config
        params=param_groups_phase2,
        _convert_="all",
    )

    
    if cfg.use_warmup:
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    scaler = GradScaler(device="cuda")
    last_e, best_val, patience_cnt = run_epochs(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, scaler, logger, start_epoch=last_e, num_epochs=num_epochs,
        checkpoint_path=cfg.checkpoint_path, patience=cfg.early_stopping.patience, min_epochs=cfg.early_stopping.min_epochs, accum_steps=accum_steps
    )


    # Phase 3: Text-LoRA only
    num_epochs = cfg.num_epochs_phase3
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name and "img_encoder" not in name:
            p.requires_grad = True

    # ❶  Re-compute the list *now*, not earlier
    txt_lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n and "img_encoder" not in n
    ]
    assert txt_lora_params, "Phase-3 optimizer received no parameters!"

    param_groups_phase3 = [
        {"params": txt_lora_params,
        "lr":     cfg.model.lr_text_adapter_phase3,
        "weight_decay": cfg.model.weight_decay_text_lora},
    ]

    optimizer = hydra.utils.instantiate(
        head_fusion_optim_cfg,
        params=param_groups_phase3,
        _convert_="all",
    )

    if cfg.use_warmup:
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

    scaler = GradScaler(device="cuda")

    last_e, best_val, patience_cnt = run_epochs(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, scaler, logger, start_epoch=last_e, num_epochs=num_epochs,
        checkpoint_path=cfg.checkpoint_path, patience=cfg.early_stopping.patience, min_epochs=cfg.early_stopping.min_epochs, accum_steps=accum_steps
    )

    # Phase 4: Joint fine-tuning
    num_epochs = cfg.num_epochs_phase4

    # unfreeze adapters + fusion + head

    for name, p in model.named_parameters():
        p.requires_grad = False
        if ("lora_" in name or name.startswith("fusion_transformer") or name.startswith("head") or any(name.startswith(attr) for attr in ("year_proj","ch_emb","cy_proj","date_proj"))):
            p.requires_grad = True
        else:
            p.requires_grad = False
    img_lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n and "img_encoder" in n
    ]
    txt_lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n and "img_encoder" not in n
    ]
    head_params   = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("head")]
    fusion_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("fusion_transformer")]
    meta_params   = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(("year_proj",
                                                        "ch_emb",
                                                        "cy_proj",
                                                        "date_proj"))]
    param_groups_phase4 = [
    {"params": head_params,   "lr": cfg.model.lr_head_phase4, "weight_decay": cfg.model.weight_decay_head_phase4},
    {"params": fusion_params, "lr": cfg.model.lr_fusion_phase4, "weight_decay": cfg.model.weight_decay_fusion_phase4},
    {"params": meta_params,   "lr": cfg.model.lr_meta_phase4, "weight_decay": cfg.model.weight_decay_meta_phase4},
    {"params": img_lora_params, "lr": cfg.model.lr_image_adapter_phase4, "weight_decay": cfg.model.weight_decay_image_lora},
    {"params": txt_lora_params, "lr": cfg.model.lr_text_adapter_phase4, "weight_decay": cfg.model.weight_decay_text_lora},
]


    # Rebuild optimizer with your full param_groups (image/text/adapters + head + fusion)
    optimizer = hydra.utils.instantiate(
        head_fusion_optim_cfg,
        params=param_groups_phase4,
        _convert_="all",
    )
    
    if cfg.use_warmup:
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    scaler = GradScaler(device="cuda")
    last_e, best_val, patience_cnt = run_epochs(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, scaler, logger, start_epoch=last_e, num_epochs=num_epochs,#cfg.epochs - last_e,
        checkpoint_path=cfg.checkpoint_path, patience=cfg.early_stopping.patience, min_epochs=cfg.early_stopping.min_epochs, accum_steps=accum_steps
    )

    # Phase 5: Full unfreeze
    num_epochs = cfg.num_epochs_phase5
    for p in model.parameters(): p.requires_grad=True
    param_groups_phase5 = [
        {
            "params": model.parameters(),
            "lr": cfg.model.lr_full_finetune,
            "weight_decay": cfg.model.weight_decay_phase5,
        }
    ]

    # 3) fresh optimizer
    optimizer = hydra.utils.instantiate(
        head_fusion_optim_cfg,      # your AdamW config
        params=param_groups_phase5,
        _convert_="all",
    )
    if cfg.use_warmup:
        num_batches     = len(train_loader)
        total_steps     = num_epochs * num_batches
        num_warmup_steps = int(total_steps * cfg.warmup_fraction)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*cfg.model.num_epochs_phase5,
            eta_min=cfg.model.lr_full_finetune_final)

    last_e, best_val, patience_cnt = run_epochs(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        device, scaler, logger, start_epoch=last_e, num_epochs=num_epochs,
        checkpoint_path=cfg.checkpoint_path, patience=cfg.early_stopping.patience, min_epochs=cfg.early_stopping.min_epochs, accum_steps=accum_steps
    )


    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
