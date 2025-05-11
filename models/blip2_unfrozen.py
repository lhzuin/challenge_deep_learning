import torch, torch.nn as nn
from transformers import Blip2Model, Blip2Processor


class BLIP2Regressor(nn.Module):
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        frozen: bool = True,
        unfreeze_enable: bool = False,
        unfreeze_after_epochs: int | None = None,   # e.g. 3
        unfreeze_top_blocks:   int = 4,
    ):
        super().__init__()

        self.backbone = Blip2Model.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        )
        self.processor = Blip2Processor.from_pretrained(model_name)

        # freeze everything at first
        if frozen:
            self.backbone.vision_model.requires_grad_(False)
            self.backbone.language_model.requires_grad_(False)
            self.backbone.qformer.requires_grad_(False)

        hidden = self.backbone.config.qformer_config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        # ▸ progressive-unfreeze bookkeeping
        self._unfreeze_cfg = {
            "enable": unfreeze_enable,
            "after_epochs": unfreeze_after_epochs,
            "top_layers":   unfreeze_top_blocks,
        }
        self._global_step = 0
        self._current_epoch = 0
        self._did_unfreeze = False

    # ------------------------------------------------------------------ #
    # public helpers (called *implicitly* from train.py — no edit needed) #
    # ------------------------------------------------------------------ #

    def epoch_scheduler_hook(self):
        """Call this once per epoch."""
        self._current_epoch += 1
        self._maybe_unfreeze()

    # ------------------------------------------------------------------ #
    # internal                                                            #
    # ------------------------------------------------------------------ #
    def _maybe_unfreeze(self):
        if self._did_unfreeze or not self._unfreeze_cfg["enable"]:
            return

        epoch_cond = (
            self._unfreeze_cfg["after_epochs"] is not None
            and self._current_epoch >= self._unfreeze_cfg["after_epochs"]
        )

        if epoch_cond:
            self._unfreeze_top_qformer_layers(self._unfreeze_cfg["top_layers"])
            self._did_unfreeze = True
            print(f"[BLIP-2] ✔ Unfroze top {self._unfreeze_cfg['top_layers']} Q-Former layers")

    def _unfreeze_top_qformer_layers(self, n):
        layers = list(self.backbone.qformer.encoder.layer)
        for layer in layers[-n:]:
            layer.requires_grad_(True)

    # ------------------------------------------------------------------ #
    # forward (unchanged except hook call)                               #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        device = next(self.parameters()).device

        pixel_values = batch["image"].to(device)
        txt = self.processor.tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        out = self.backbone(
            pixel_values=pixel_values,
            input_ids=txt.input_ids,
            attention_mask=txt.attention_mask,
            output_hidden_states=True,
        )

        # ---- robust pooling for every transformers version ----
        if getattr(out, "qformer_outputs", None) is not None:     # ≥ 4.38
            q_last = out.qformer_outputs.last_hidden_state        # (B, Nq, Hd)
        elif getattr(out, "q_hidden_states", None) is not None:   # ≤ 4.36
            q_last = out.q_hidden_states[-1]                      # list → tensor
        else:                                                     # future proof
            raise RuntimeError("Cannot locate Q-Former hidden states in output")

        pooled_q = q_last.mean(dim=1)                             # (B, Hd)

        return self.head(pooled_q).squeeze(1)