import torch, torch.nn as nn
from transformers import Blip2Model, Blip2Processor

class BLIP2Regressor(nn.Module):
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", frozen=True):
        super().__init__()

        # backbone + processor give you transforms & tokenizer in one call
        self.backbone = Blip2Model.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.processor = Blip2Processor.from_pretrained(model_name)

        if frozen:
            self.backbone.vision_model.requires_grad_(False)
            self.backbone.language_model.requires_grad_(False)

        hidden = self.backbone.config.qformer_config.hidden_size  # = 768
        self.reg_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, batch):
        device = next(self.parameters()).device

        pixel_values = batch["image"].to(device)

        txt = self.processor.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,          
            max_length=128,    
            return_tensors="pt"
        ).to(device)

        out = self.backbone(
            pixel_values=pixel_values,
            input_ids=txt.input_ids,
            attention_mask=txt.attention_mask,
            return_dict=True      
        )

        # ---- robust pooling for every transformers version ----
        if getattr(out, "qformer_outputs", None) is not None:     # ≥ 4.38
            q_last = out.qformer_outputs.last_hidden_state        # (B, Nq, Hd)
        elif getattr(out, "q_hidden_states", None) is not None:   # ≤ 4.36
            q_last = out.q_hidden_states[-1]                      # list → tensor
        else:                                                     # future proof
            raise RuntimeError("Cannot locate Q-Former hidden states in output")

        pooled_q = q_last.mean(dim=1)                             # (B, Hd)

        return self.reg_head(pooled_q).squeeze(1)