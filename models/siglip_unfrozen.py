import torch, torch.nn as nn, open_clip


class SigLIPRegressor(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        frozen=True,
        # ───── progressive-unfreeze knobs (YAML-controllable) ─────
        unfreeze_enable=False,
        unfreeze_epoch_fraction=None,      # e.g. 3
        total_epochs=None,
        unfreeze_top_blocks=2,           # how many ViT / text blocks
        unfreeze_proj=True,              # also turn on .proj/.text_projection

        add_year_proj = True,
        year_proj_dim = 8
        ):

        super().__init__()
        self.add_year_proj = add_year_proj
        self.unfreeze_unable = unfreeze_enable
        if unfreeze_enable:
            unfreeze_after_epochs = int(total_epochs*unfreeze_epoch_fraction)
        else:
            unfreeze_after_epochs = None

        # backbone -----------------------------------------------------------
        self.backbone, self.pre_tf, self.val_tf = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Add year projection MLP
        if self.add_year_proj:
            self.year_proj = nn.Sequential(
                nn.Linear(1, year_proj_dim),
                nn.ReLU(),
                nn.LayerNorm(year_proj_dim),
            )

        # freeze all ----------------------------------------------------------
        if frozen:
            self.backbone.visual.requires_grad_(False)
            if hasattr(self.backbone, "text"):
                self.backbone.text.requires_grad_(False)

        # reg-head -----------------------------------------------------------
        D = self._joint_dim()
        joint_dim = 2 * D + year_proj_dim if self.add_year_proj else 2*D
        self.head = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1))

        # scheduler bookkeeping ---------------------------------------------
        self._cfg = dict(enable=unfreeze_enable, T=unfreeze_top_blocks,
                         after_ep=unfreeze_after_epochs,
                         proj=unfreeze_proj)
        self._epoch = 0
        self._step  = 0
        self._done  = False

    # ------------------------------------------------------------------ #
    # hooks (call these from train.py, exactly like for BLIP-2)          #
    # ------------------------------------------------------------------ #

    def epoch_scheduler_hook(self):
        self._epoch += 1
        if self.unfreeze_unable:
            self._maybe_unfreeze()

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        img_f = self.backbone.encode_image(batch["image"])
        txt_f = self.backbone.encode_text(
            self.tokenizer(batch["text"]).to(img_f.device)
        )
        if self.add_year_proj:
            yr_z = batch["year_z"].to(img_f.device)       # [B,1]
            yr_f = self.year_proj(yr_z) 
            joint = torch.cat([img_f, txt_f, yr_f], dim=-1)
        else:
            joint = torch.cat([img_f, txt_f], dim=-1)
        return self.head(joint).squeeze(1)

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    def _maybe_unfreeze(self):
        if self._done or not self._cfg["enable"]:
            return
        if (self._cfg["after_ep"] is not None and self._epoch >= self._cfg["after_ep"]):
            self._unfreeze()
            self._done = True
            print(f"[SigLIP] ✔ Unfroze proj {'and' if self._cfg['proj'] else ''}"
                  f" top {self._cfg['T']} blocks")

    def _unfreeze(self):
        # 1. projection layers
        if self._cfg["proj"]:
            if hasattr(self.backbone.visual, "proj"):
                self.backbone.visual.proj.requires_grad_(True)
            if hasattr(self.backbone, "text_projection"):
                self.backbone.text_projection.requires_grad_(True)

        # 2. last N ViT blocks
        if hasattr(self.backbone.visual, "trunk"):
            blocks = self.backbone.visual.trunk.blocks   # timm style
        else:
            blocks = self.backbone.visual.transformer.resblocks  # openai style
        for blk in list(blocks)[-self._cfg["T"]:]:
            blk.requires_grad_(True)

        # 3. last N text blocks
        txt = getattr(self.backbone, "text", None)
        if txt is not None:
            for blk in list(txt.transformer.resblocks)[-self._cfg["T"]:]:
                blk.requires_grad_(True)

    def _joint_dim(self):
        if hasattr(self.backbone.visual, "output_dim"):
            return self.backbone.visual.output_dim
        if hasattr(self.backbone, "text_projection"):
            return self.backbone.text_projection.shape[1]
        return self.backbone.encode_image(torch.zeros(1,3,224,224)).shape[-1]