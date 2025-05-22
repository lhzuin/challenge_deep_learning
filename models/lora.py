import torch.nn as nn
import math
import torch

# --- Inspiration: https://github.com/RobvanGastel/dinov2-finetune/blob/main/dino_finetune/model/lora.py ---
class LoRA(nn.Module):
    """Low-Rank Adaptation for Q and V in a fused qkv linear."""
    def __init__(self, 
                 qkv: nn.Linear, 
                 r: int):
        super().__init__()
        self.qkv = qkv
        dim    = qkv.in_features
        # manually register four low-rank weight matrices on the same device/dtype
        dev, dtype = qkv.weight.device, qkv.weight.dtype
        self.a_q_weight = nn.Parameter(torch.empty((r,  dim), device=dev, dtype=dtype))
        self.b_q_weight = nn.Parameter(torch.empty((dim, r), device=dev, dtype=dtype))
        self.a_v_weight = nn.Parameter(torch.empty((r,  dim), device=dev, dtype=dtype))
        self.b_v_weight = nn.Parameter(torch.empty((dim, r), device=dev, dtype=dtype))
        # init
        nn.init.kaiming_uniform_(self.a_q_weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q_weight)
        nn.init.kaiming_uniform_(self.a_v_weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v_weight)

    def forward(self, x):
        """
        x: (B, N, in_dim)
        returns fused qkv with LoRA deltas: (B, N, 3*out_dim)
        """
        B, N, in_dim = x.shape

        # original fused projection
        qkv = self.qkv(x)                      # (B, N, 3*out_dim)
        out_dim = qkv.shape[-1] // 3

        # flatten batch+tokens → (B·N, in_dim) for matmul
        x_flat = x.reshape(-1, in_dim)         # (B·N, in_dim)

        # low-rank updates:  x Aᵀ Bᵀ  →  (B·N, out_dim)
        new_q = (x_flat @ self.a_q_weight.t()) @ self.b_q_weight.t()
        new_v = (x_flat @ self.a_v_weight.t()) @ self.b_v_weight.t()

        # reshape back to (B, N, out_dim)
        new_q = new_q.view(B, N, out_dim)
        new_v = new_v.view(B, N, out_dim)

        # add into q and v slices of the fused tensor
        qkv[:, :, :out_dim] += new_q
        qkv[:, :, -out_dim:] += new_v

        return qkv