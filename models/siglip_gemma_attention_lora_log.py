import torch, torch.nn as nn, open_clip
#from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM as AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
# Cross-modal fusion layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os
token = os.environ.get("HF_TOKEN")


class SigLIPGemmaAttentionLoraLog(nn.Module):
    def __init__(
        self,
        num_channels: int = 47,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
        metadata_dropout: float = 0.3,
        ch_emb_dim: int = 16,
        year_proj_dim: int = 8,
        date_proj_dim: int = 16,
        cy_hidden: int = 16,
        ):

        super().__init__()

        siglip_name="ViT-B-16-SigLIP"
        pretrained="webli"

        gemma_name = "google/gemma-3-1b-it"
        
        
        # image tower -----------------------------------------------------------
        self.img_encoder, self.pre_tf, self.val_tf = open_clip.create_model_and_transforms(
            siglip_name, pretrained=pretrained
        )

        vision_lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,          
            target_modules=["qkv"],  # this covers the fused qkv projection
            lora_dropout=0.1,    
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.img_encoder = get_peft_model(self.img_encoder, vision_lora_cfg)

        for name, p in self.img_encoder.named_parameters(): 
            if "lora_" not in name:               
                p.requires_grad = False

        # compute the embedding dimension by doing a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = self.img_encoder.encode_image(dummy).shape[-1]

        # text tower
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_threshold=6.0,        # HF default, keeps accuracy
        )
        self.text_encoder = AutoModel.from_pretrained(
                gemma_name,
                device_map="auto",
                quantization_config=bnb_cfg,
                attn_implementation="eager",  
                trust_remote_code=True,
                token=token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                gemma_name,
                trust_remote_code=True,
                token=token
        )

        # Gemma has no pad-token in the original vocab → add one once:
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # ── build the *first* adapter → wraps into a PeftModel ─────────
        title_cfg = LoraConfig(r=8, lora_alpha=16,
                               target_modules=["q_proj","k_proj","v_proj"],
                               lora_dropout=0.3, bias="none",
                               task_type="FEATURE_EXTRACTION")
        self.text_encoder = get_peft_model(self.text_encoder, title_cfg, "title_adapter")
        
        # ── add the second adapter on the same wrapper ─────────────────
        sum_cfg = LoraConfig(r=8, lora_alpha=16,
                             target_modules=["q_proj","k_proj","v_proj"],
                             lora_dropout=0.3, bias="none",
                             task_type="FEATURE_EXTRACTION")
        self.text_encoder.add_adapter("summary_adapter", sum_cfg)
        #self.text_encoder = get_peft_model(self.text_encoder, text_lora_cfg)
        for name, p in self.text_encoder.base_model.named_parameters():
            if "lora_" not in name:                          # …except LoRA
                p.requires_grad = False
        txt_dim = self.text_encoder.config.hidden_size

        D = min(clip_dim, txt_dim)

        self.title_proj   = nn.Linear(txt_dim, D)
        self.sum_proj     = nn.Linear(txt_dim, D)   
        
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
        

        # ── date features MLP ───────────────────────────────────
        self.date_proj = nn.Sequential(
            nn.Linear(6, date_proj_dim),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
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
            num_layers=1       # two-layer fusion
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
        img_f = self.img_encoder.encode_image(batch["image"])
        img_f = self.img_proj(img_f)

        tok_title = self.tokenizer(batch["title"], padding=True, truncation=True, max_length=96, return_tensors="pt").to(img_f.device)
        tok_title = {k:v.to(img_f.device) for k,v in tok_title.items()}
        tok_sum = self.tokenizer(batch["summary"], padding=True, truncation=True, max_length=96, return_tensors="pt").to(img_f.device)
        tok_sum = {k:v.to(img_f.device) for k,v in tok_sum.items()}

        # ---- encode title ----
        self.text_encoder.set_adapter("title_adapter")
        out_title = self.text_encoder(**tok_title, output_hidden_states=True)
        h_title   = out_title.hidden_states[-1][:, 0].to(self.title_proj.weight.dtype)
        t_f       = self.title_proj(h_title)
        #out_title = self.text_encoder(**tok_title)
        #t_f = self.title_proj(out_title.last_hidden_state[:,0])

        # ---- encode summary ----
        self.text_encoder.set_adapter("summary_adapter")
        out_sum   = self.text_encoder(**tok_sum, output_hidden_states=True)
        h_sum     = out_sum.hidden_states[-1][:, 0].to(self.sum_proj.weight.dtype)
        s_f       = self.sum_proj(h_sum) 

        yr_norm = batch["year_norm"].to(img_f.device)       # [B,1]
        yr_f = self.year_proj(yr_norm) 


        # channel embed
        ch_f = self.ch_emb(batch["channel_idx"].to(img_f.device))
        
        # channel×year interaction
        cy_in = torch.cat([ch_f, yr_f], dim=1)
        cy_f  = self.cy_proj(cy_in)
        
        # 3) date flags
        date = torch.cat([batch[k].to(img_f.device) for k in
            ["m_sin","m_cos","d_sin","d_cos","h_sin","h_cos"]], dim=1)
        date_f = self.date_proj(date)

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