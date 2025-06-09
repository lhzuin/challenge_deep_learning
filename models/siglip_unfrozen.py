import torch, torch.nn as nn, open_clip


class SigLIPRegressor(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        frozen=True,
        # ───── progressive-unfreeze knobs (YAML-controllable) ─────
        unfreeze_enable=False,
        unfreeze_epoch_fraction=None, 
        total_epochs=None,
        unfreeze_top_blocks=2,
        unfreeze_proj=True,      
        ):
        super().__init__()
        if unfreeze_enable:
            unfreeze_after_epochs = int(total_epochs*unfreeze_epoch_fraction)
        else:
            unfreeze_after_epochs = None

        # backbone -----------------------------------------------------------
        self.backbone, self.pre_tf, self.val_tf = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # ── regression head ──────────────────────────────────────
        D = self._joint_dim()
        joint_dim = 2*D

        self.head = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim,1))

        # freeze all ----------------------------------------------------------
        if frozen:
            self.backbone.visual.requires_grad_(False)
            if hasattr(self.backbone, "text"):
                self.backbone.text.requires_grad_(False)    

        # scheduler bookkeeping ---------------------------------------------
        self._cfg = dict(enable=unfreeze_enable, T=unfreeze_top_blocks,
                         after_ep=unfreeze_after_epochs,
                         proj=unfreeze_proj)
        self._epoch = 0
        self._step  = 0
        self._done  = False

    def epoch_scheduler_hook(self):
        self._epoch += 1
        self._maybe_unfreeze()

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        img_f = self.backbone.encode_image(batch["image"])
        txt_f = self.backbone.encode_text(
            self.tokenizer(batch["text"]).to(img_f.device)
        )

        joint_f = [img_f, txt_f]
        joint = torch.cat(joint_f, dim=-1)
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