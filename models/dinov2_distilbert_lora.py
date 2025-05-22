import torch, torch.nn as nn
import transformers
from peft import get_peft_model, LoraConfig
# Cross-modal fusion layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import timm
from .lora import LoRA
import types



class DINOv2DistilBert(nn.Module):
    def __init__(
        self,
        num_channels: int = 47,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.15,
        ch_emb_dim: int = 16,
        year_proj_dim: int = 8,
        date_proj_dim: int = 16,
        cy_hidden: int = 16,
        ):

        super().__init__()

        lora_rank = 4

        distilbert_name = "distilbert-base-uncased"
        
        # image tower -----------------------------------------------------------
        # ── 1) Vision tower: DINOv2 ViT-L/14 via timm ─────────────
        self.img_encoder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,              # remove head
        )
        self.img_encoder.visual = types.SimpleNamespace(
            trunk = types.SimpleNamespace(
                blocks = self.img_encoder.blocks   # original list of blocks
            )
        )
        # freeze all weights
        for p in self.img_encoder.parameters():
            p.requires_grad = False

        # inject LoRA into each block.attn.qkv
        for blk in self.img_encoder.blocks:
            # blk.attn.qkv is the fused linear of shape [in, 3*out]
            blk.attn.qkv = LoRA(blk.attn.qkv, r=lora_rank)

        # infer vision embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.img_encoder.default_cfg["input_size"][1:])
            feat  = self.img_encoder(dummy)
            clip_dim = feat.shape[-1]

        # text tower
        self.text_encoder = transformers.DistilBertModel.from_pretrained(distilbert_name)
        self.tokenizer    = transformers.DistilBertTokenizerFast.from_pretrained(distilbert_name)
        text_lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],  # key and value proj
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.text_encoder = get_peft_model(self.text_encoder, text_lora_cfg)
        for p in self.text_encoder.base_model.parameters():
            p.requires_grad = False
        bert_dim = self.text_encoder.config.hidden_size

        D = min(clip_dim, bert_dim)

        self.title_proj   = nn.Linear(bert_dim, D)
        self.sum_proj     = nn.Linear(bert_dim, D)   
        
        self.img_proj     = nn.Linear(clip_dim, D)

        # ── continuous year MLP ─────────────────────────────────
        self.year_proj = nn.Sequential(
            nn.Linear(1, year_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(year_proj_dim),
        )
    

        # ── channel×year interaction ────────────────────────────
        
        self.cy_proj = nn.Sequential(
            nn.Linear(ch_emb_dim + year_proj_dim, cy_hidden),
            nn.ReLU(),
            nn.LayerNorm(cy_hidden),
        )
        
        # ── channel embedding ────────────────────────────────────
        self.ch_emb = nn.Embedding(num_channels, ch_emb_dim)
        

        # ── date features MLP ───────────────────────────────────
        self.date_proj = nn.Sequential(
            nn.Linear(6, date_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(date_proj_dim),
        )


        # ── Cross-modal fusion Transformer ──────────────────────
        # We have three “tokens” (img, summary, title). We'll reshape them into a sequence.
        fusion_dim = D  # same dimension for all modality embeddings
        
        encoder_layer = TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=4,           # 4 attention heads
            dim_feedforward= fusion_dim * 2,
            dropout=0.4,
            activation="relu",
            batch_first=True 
        )
        self.fusion_transformer = TransformerEncoder(
            encoder_layer,
            num_layers=2       # two-layer fusion
        )
    
        # ── regression head (unchanged dims except now head sees one token instead of concat)
        # We'll pool (mean) the transformer outputs over the 3 tokens, then send through head.
        meta_dim = year_proj_dim + ch_emb_dim + cy_hidden + date_proj_dim
        joint_dim = fusion_dim + meta_dim
        self.head = nn.Sequential(
            nn.LayerNorm(joint_dim),               # normalize across all joint features
            nn.Dropout(head_dropout),
            nn.Linear(joint_dim, head_hidden_dim), # accept the full concatenated vector
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        device = batch["image"].device
        if isinstance(batch["image"], dict):
            img_out = self.img_encoder(**{k: v.to(device) for k, v in batch["image"].items()})
        else:
            img_out = self.img_encoder(batch["image"].to(device))
        cls_feat = img_out 
        img_f    = self.img_proj(cls_feat)      

        tok_title = self.tokenizer(batch["title"], padding=True, truncation=True, max_length=64, return_tensors="pt").to(img_f.device)
        tok_title = {k:v.to(img_f.device) for k,v in tok_title.items()}
        tok_sum = self.tokenizer(batch["summary"], padding=True, truncation=True, max_length=64, return_tensors="pt").to(img_f.device)
        tok_sum = {k:v.to(img_f.device) for k,v in tok_sum.items()}


        out_title = self.text_encoder(**tok_title)
        out_sum = self.text_encoder(**tok_sum)

        t_f = self.title_proj(out_title.last_hidden_state[:,0])
        s_f = self.sum_proj(out_sum.last_hidden_state[:,0])

        joint_f = [img_f, t_f, s_f]
        yr_norm = batch["year_norm"].to(img_f.device)       # [B,1]
        yr_f = self.year_proj(yr_norm) 
        joint_f.append(yr_f)

        # channel embed
        ch_f = self.ch_emb(batch["channel_idx"].to(img_f.device))
        joint_f.append(ch_f)
        
        # channel×year interaction
        cy_in = torch.cat([ch_f, yr_f], dim=1)
        cy_f  = self.cy_proj(cy_in)
        joint_f.append(cy_f)
        
        # 3) date flags
        date = torch.cat([batch[k].to(img_f.device) for k in
            ["m_sin","m_cos","d_sin","d_cos","h_sin","h_cos"]], dim=1)
        date_f = self.date_proj(date)
        joint_f.append(date_f)
        joint = torch.cat(joint_f, dim=-1)

        # Combine different latent spaces
        modalities = torch.stack([img_f, t_f, s_f], dim=1)     # [B, 3, D]
        fused      = self.fusion_transformer(modalities)      # [B, 3, D]
        pooled     = fused.mean(dim=1)  # [B, D]

        # 4) append metadata embeddings as before
        meta = torch.cat([yr_f, ch_f, cy_f, date_f], dim=-1)  # [B, meta_dim]

        # 5) final joint
        joint = pooled + 0  # copy pooled into new tensor
        joint = torch.cat([joint, meta], dim=-1)  # [B, D + meta_dim]

        # 6) head
        return self.head(joint).squeeze(1)