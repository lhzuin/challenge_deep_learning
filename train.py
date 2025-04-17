import torch
import wandb
import hydra
from tqdm import tqdm
import os


from utils.sanity import show_images

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="train")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
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

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- training loop
        model.train()
        epoch_train_loss = 0
        epoch_train_mae = 0
        epoch_train_mse = 0
        epoch_train_msle = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()

            preds = model(batch).squeeze()

            loss = loss_fn(preds, batch["target"])
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])

            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
            epoch_train_mae += (
                torch.nn.L1Loss(reduction="sum")(preds, batch["target"])
                .detach()
                .cpu()
                .numpy()
            )
            epoch_train_mse += (
                torch.nn.MSELoss(reduction="sum")(preds, batch["target"])
                .detach()
                .cpu()
                .numpy()
            )
            epoch_train_msle += (
                torch.nn.MSELoss(reduction="sum")(
                    torch.log1p(preds),
                    torch.log1p(batch["target"]),
                )
                .detach()
                .cpu()
                .numpy()
            )
            num_samples_train += len(batch["image"])

        epoch_train_loss /= num_samples_train
        epoch_train_mae /= num_samples_train
        epoch_train_mse /= num_samples_train
        epoch_train_msle /= num_samples_train

        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                    "train/mae_epoch": epoch_train_mae,
                    "train/mse_epoch": epoch_train_mse,
                    "train/msle_epoch": epoch_train_msle,
                }
            )
            if logger is not None
            else None
        )

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        epoch_val_mae = 0
        epoch_val_mse = 0
        epoch_val_msle = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None: 
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                with torch.no_grad():
                    preds = model(batch).squeeze()

                # loss = loss_fn(preds, targets)
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                epoch_val_mae += (
                    torch.nn.L1Loss(reduction="sum")(preds, batch["target"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                epoch_val_mse += (
                    torch.nn.MSELoss(reduction="sum")(preds, batch["target"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                epoch_val_msle += (
                    torch.nn.MSELoss(reduction="sum")(
                        torch.log1p(preds),
                        torch.log1p(batch["target"]),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                num_samples_val += len(batch["image"])

            epoch_val_loss /= num_samples_val
            epoch_val_mse /= num_samples_val
            epoch_val_mae /= num_samples_val
            epoch_val_msle /= num_samples_val

            val_metrics["val/loss_epoch"] = epoch_val_loss
            val_metrics["val/mae_epoch"] = epoch_val_mae
            val_metrics["val/mse_epoch"] = epoch_val_mse
            val_metrics["val/msle_epoch"] = epoch_val_msle
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

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        - Train MAE: {epoch_train_mae:.4f}
        - Train MSE: {epoch_train_mse:.4f}
        - Train MSLE: {epoch_train_msle:.4f}
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}
        - Val MAE: {epoch_val_mae:.4f} 
        - Val MSE: {epoch_val_mse:.4f}
        - Val MSLE: {epoch_val_msle:.4f}"""
    )

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
