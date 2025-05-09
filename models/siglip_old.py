import torch, torch.nn as nn, open_clip


class SigLIPRegressor2(nn.Module):
    def __init__(self, frozen: bool = True):
        super().__init__()

        # backbone + CLIP transforms
        self.backbone, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP", pretrained="webli"
            )
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

        if frozen:
            self.backbone.visual.requires_grad_(False)
            if hasattr(self.backbone, "text"):
                self.backbone.text.requires_grad_(False)

        # ---------- universal, always-works dim finder ----------
        embed_dim = self._infer_joint_dim()
        # --------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 1),
        )

    def _infer_joint_dim(self) -> int:
        """Return the shared vision-text embedding size for *any* CLIP family."""
        # 1️⃣ Fast paths for most models
        if hasattr(self.backbone.visual, "output_dim"):
            return self.backbone.visual.output_dim
        if hasattr(self.backbone.visual, "embed_dim"):
            return self.backbone.visual.embed_dim
        if hasattr(self.backbone, "text_projection"):
            return self.backbone.text_projection.shape[1]

        # 2️⃣ Fallback that *always* works (even SigLIP)
        with torch.no_grad():
            # build a dummy image with the resolution expected by the preprocess
            # (all ViT/SigLIP backbones accept arbitrary sizes, so 224 is safe)
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone.encode_image(dummy)
        return feat.shape[-1]

    def forward(self, batch):
        img_feat = self.backbone.encode_image(batch["image"])
        txt_feat = self.backbone.encode_text(
            self.tokenizer(batch["text"]).to(img_feat.device)
        )
        return self.head(torch.cat([img_feat, txt_feat], dim=-1)).squeeze(1)