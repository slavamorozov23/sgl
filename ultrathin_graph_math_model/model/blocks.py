import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


class SpatialCubeLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        from .attention import MultiLatentAttention

        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiLatentAttention(d_model, nhead=nhead, d_c=config.MLA_D_C, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False) # Gate projection
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False) # Value projection
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False) # Output projection
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor], print_debug: bool = False) -> torch.Tensor:
        # 1. FFN блок
        residual_1 = x
        normed_x_ffn = self.norm1(x)
        gate = F.silu(self.w1(normed_x_ffn))
        value = self.w3(normed_x_ffn)
        ff_hidden = gate * value
        ff_output = self.linear2(ff_hidden)
        x = residual_1 + self.dropout2(ff_output) # Используем dropout2 для FFN

        # 2. MLA блок
        residual_2 = x
        normed_x_attn = self.norm2(x)
        attn_output = self.self_attn(normed_x_attn, is_causal=True, freqs_cis=freqs_cis, print_debug=print_debug)
        x = residual_2 + self.dropout1(attn_output) # Используем dropout1 для Attention

        return x


class Gate(nn.Module):
    def __init__(self, d_model: int, num_cubes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_cubes + 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate mean representation across the sequence length dimension
        sequence_representation = x.mean(dim=1)  # Shape: [batch_size, d_model]
        return self.linear(sequence_representation)