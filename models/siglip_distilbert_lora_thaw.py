import torch, torch.nn as nn, open_clip
import transformers
from peft import get_peft_model, LoraConfig


class SigLIPDistilBertLoraThaw(nn.Module):
    def __init__(
        self,
        num_channels: int = 47,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.25,
        metadata_dropout: float = 0.40,
        proj_dropout: float = 0.10,
        ch_emb_dim: int = 16,
        year_proj_dim: int = 8,
        date_proj_dim: int = 16,
        cy_hidden: int = 16,
        ):

        super().__init__()

        siglip_name="ViT-B-16-SigLIP"
        pretrained="webli"

        distilbert_name = "distilbert-base-uncased"

        self.fusion_transformer = nn.Identity()
       
        # image tower -----------------------------------------------------------
        self.img_encoder, self.pre_tf, self.val_tf = open_clip.create_model_and_transforms(
            siglip_name, pretrained=pretrained
        )

        vision_lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,          
            target_modules=["qkv"],  # this covers the fused qkv projection
            lora_dropout=0.05,       # renamed from dropout
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.img_encoder = get_peft_model(self.img_encoder, vision_lora_cfg)

        for name, p in self.img_encoder.named_parameters():
            # keep only the tiny LoRA matrices trainable
            if "lora_" not in name:
                p.requires_grad = False 

        # compute the embedding dimension by doing a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = self.img_encoder.encode_image(dummy).shape[-1]

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
        for name, p in self.text_encoder.base_model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        
        bert_dim = self.text_encoder.config.hidden_size

        D = min(clip_dim, bert_dim)

        self.title_proj = nn.Sequential(
            nn.Linear(bert_dim, D),
            nn.ReLU(),
            nn.Dropout(proj_dropout),   # new
            )
        self.sum_proj = nn.Sequential(
            nn.Linear(bert_dim, D),
            nn.ReLU(),
            nn.Dropout(proj_dropout),   # new
            )
       
        self.img_proj     = nn.Linear(clip_dim, D)

        # ── continuous year MLP ─────────────────────────────────
        self.year_proj = nn.Sequential(
            nn.Linear(1, year_proj_dim),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
            nn.LayerNorm(year_proj_dim),
        )
   

        # ── channel×year interaction ────────────────────────────
       
        self.cy_proj = nn.Sequential(
            nn.Linear(ch_emb_dim + year_proj_dim, cy_hidden),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
            nn.LayerNorm(cy_hidden),
        )
       
        # ── channel embedding ────────────────────────────────────
        self.ch_emb = nn.Embedding(num_channels, ch_emb_dim)
       
        # ── bucketed year embedding ─────────────────────────────
        #self.year_emb  = nn.Embedding(num_year_buckets, year_emb_dim)

        # ── date features MLP ───────────────────────────────────
        self.date_proj = nn.Sequential(
            nn.Linear(6, date_proj_dim),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
            nn.LayerNorm(date_proj_dim),
        )


        # ── regression head ──────────────────────────────────────
        joint_dim = (
            3*D             +  # image + text
            ch_emb_dim      +
            year_proj_dim   +
            date_proj_dim   +
            cy_hidden
        )
        self.head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Dropout(head_dropout),
            nn.Linear(joint_dim,head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),  
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        img_f = self.img_encoder.encode_image(batch["image"])
        img_f = self.img_proj(img_f)

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

        # Year bucket
        #yr_emb_f = self.year_emb(batch["year_idx"].to(img_f.device))   # [B, year_emb_dim]
        #joint_f.append(yr_emb_f)
       
        # 3) date flags
        date = torch.cat([batch[k].to(img_f.device) for k in
            ["m_sin","m_cos","d_sin","d_cos","h_sin","h_cos"]], dim=1)
        date_f = self.date_proj(date)
        joint_f.append(date_f)
        joint = torch.cat(joint_f, dim=-1)
        return self.head(joint).squeeze(1)