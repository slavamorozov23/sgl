import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config
import torch.nn.init as init


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the complex frequency tensor for Rotary Positional Embedding (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Return complex numbers: exp(i * freqs * t) = cos(freqs * t) + i * sin(freqs * t)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to a single tensor (Q or K) typically of shape [..., seq_len, dim]"""
    dim_rope = x.shape[-1]
    assert dim_rope % 2 == 0, "RoPE dimension must be even"
    # seq_len dimension is expected to be -2
    seq_len = x.shape[-2]

    if freqs_cis.shape[0] < seq_len:
         raise ValueError(f"freqs_cis has length {freqs_cis.shape[0]} but input needs {seq_len}")

    # Slice freqs_cis to match the sequence length and move to input device
    freqs = freqs_cis[:seq_len].to(x.device) # Shape: [seq_len, dim_rope // 2]

    # Reshape x for complex number representation
    x_r = x.float().reshape(*x.shape[:-1], dim_rope // 2, 2)
    x_c = torch.view_as_complex(x_r) # Shape: [..., seq_len, dim_rope // 2]

    # Reshape freqs for broadcasting to match x_c dimensions before seq_len
    freqs_b = freqs.view(1, *([1] * (x_c.ndim - 3)), seq_len, freqs.shape[-1])

    # Apply rotary embedding via complex multiplication
    x_out_c = x_c * freqs_b
    x_out = torch.view_as_real(x_out_c).flatten(start_dim=-2) # Flatten the last two dims

    return x_out.type_as(x)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention layer where queries have multiple heads,
    but keys and values share a single head projected to head_dim.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.head_dim, bias=False) # Shared K projection
        self.Wv = nn.Linear(d_model, self.head_dim, bias=False) # Shared V projection
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.Wq(x).view(bsz, seqlen, self.nhead, self.head_dim)
        k = self.Wk(x) # Shape: [bsz, seqlen, head_dim]
        v = self.Wv(x) # Shape: [bsz, seqlen, head_dim]

        # Apply RoPE if provided
        if freqs_cis is not None and freqs_cis.numel() > 0:
             # Apply standard RoPE to Queries
             q = q.transpose(1, 2) # [bsz, nhead, seqlen, head_dim]
             q = apply_rotary_emb_single(q, freqs_cis) # Apply RoPE directly

             # Reshape K for RoPE, apply, and add back dummy head dim
             k_flat = k.reshape(bsz, seqlen, self.head_dim)
             k_flat = apply_rotary_emb_single(k_flat, freqs_cis)
             k = k_flat.view(bsz, seqlen, self.head_dim).unsqueeze(1) # [bsz, 1, seqlen, head_dim]

             v = v.unsqueeze(1) # Add dummy head dim: [bsz, 1, seqlen, head_dim]

        else:
             # Prepare for attention: q[b,nh,s,hd], k[b,1,s,hd], v[b,1,s,hd]
             q = q.transpose(1, 2)
             k = k.unsqueeze(1)
             v = v.unsqueeze(1)

        # Scaled Dot-Product Attention (handles broadcasting K/V head dim)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal and seqlen > 1
        )

        # Combine heads and project output
        out = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
        out = self.Wo(out)
        return self.resid_dropout(out)


class MultiLatentAttention(nn.Module):
    """
    Attention mechanism with latent compression for Keys and Values.
    Optionally compresses Queries as well. Implements KV caching.
    Uses standard RoPE: applies rotary embedding to both Q and K (no positional bias addition).
    """
    def __init__(self, d_model: int, nhead: int, d_c: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_c = d_c # Dimension of compressed latent space

        # Query projection/compression layers
        self.compress_query = config.MLA_COMPRESS_QUERY
        if self.compress_query:
            self.W_dq = nn.Linear(d_model, d_c, bias=False)
            self.W_uq = nn.Linear(d_c, d_model, bias=False)
            init.kaiming_uniform_(self.W_dq.weight, a=0, mode='fan_in')
            init.kaiming_uniform_(self.W_uq.weight, a=0, mode='fan_in')
        else:
            self.Wq = nn.Linear(d_model, d_model, bias=False)
            init.kaiming_uniform_(self.Wq.weight, a=0, mode='fan_in') # Add initialization for Wq

        # Latent KV projection and reconstruction layers
        self.W_dkv = nn.Linear(d_model, d_c, bias=False) # Compress K/V
        self.W_uk = nn.Linear(d_c, d_model, bias=False)  # Reconstruct K
        self.W_uv = nn.Linear(d_c, d_model, bias=False)  # Reconstruct V
        init.kaiming_uniform_(self.W_dkv.weight, a=0, mode='fan_in') # Changed initialization
        init.kaiming_uniform_(self.W_uk.weight, a=0, mode='fan_in')  # Changed initialization
        init.kaiming_uniform_(self.W_uv.weight, a=0, mode='fan_in')  # Changed initialization

        # Output projection and dropout
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        init.kaiming_uniform_(self.Wo.weight, a=0, mode='fan_in') # Add initialization for Wo
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)






    def forward(self, x: torch.Tensor, is_causal: bool = False, freqs_cis: Optional[torch.Tensor] = None, print_debug: bool = False) -> torch.Tensor:
        bsz, input_seqlen, d_model = x.shape


        # Query projection
        if self.compress_query:
            c_q = self.W_dq(x)
            q = self.W_uq(c_q).view(bsz, input_seqlen, self.nhead, self.head_dim)
        else:
            q = self.Wq(x).view(bsz, input_seqlen, self.nhead, self.head_dim)

        # Compress K/V input tokens
        c_kv_new = self.W_dkv(x)

        # Use current sequence (no KV cache)
        c_kv = c_kv_new

        # Determine sequence length for K, V based on cache/input
        kv_seqlen = c_kv.shape[1]

        # Reconstruct K, V from (potentially cached) compressed representation
        k = self.W_uk(c_kv).view(bsz, kv_seqlen, self.nhead, self.head_dim)
        v = self.W_uv(c_kv).view(bsz, kv_seqlen, self.nhead, self.head_dim)

        # Apply standard RoPE if provided
        if freqs_cis is not None and freqs_cis.numel() > 0:
            # Add check for freqs_cis length
            if freqs_cis.shape[0] < kv_seqlen:
                raise ValueError(f"freqs_cis length {freqs_cis.shape[0]} insufficient for kv_seqlen {kv_seqlen}")

            # --- RoPE for Q ---
            q_rope = q.transpose(1, 2) # [bsz, nhead, input_seqlen, head_dim]
            q_flat = q_rope.reshape(bsz * self.nhead, input_seqlen, self.head_dim)
            q_flat = apply_rotary_emb_single(q_flat, freqs_cis)
            q = q_flat.view(bsz, self.nhead, input_seqlen, self.head_dim)

            # --- RoPE for K ---
            k_rope = k.transpose(1, 2) # [bsz, nhead, kv_seqlen, head_dim]
            k_flat = k_rope.reshape(bsz * self.nhead, kv_seqlen, self.head_dim)
            k_flat = apply_rotary_emb_single(k_flat, freqs_cis)
            k = k_flat.view(bsz, self.nhead, kv_seqlen, self.head_dim)

            # Transpose V for attention
            v = v.transpose(1, 2) # [bsz, nhead, kv_seqlen, head_dim]
        else: # No RoPE
            # Transpose q, k, v to [bsz, nhead, seqlen, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        # Q: [bsz, nhead, input_seqlen, head_dim]
        # K: [bsz, nhead, kv_seqlen, head_dim]
        # V: [bsz, nhead, kv_seqlen, head_dim]
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal # Let SDPA handle masking based on Q/K/V lengths
        )
        # attn shape: [bsz, nhead, input_seqlen, head_dim]

        # Add optional logging for attention norm
        if print_debug:
            attn_norm = attn.norm().item()
            console.print(f"MLA Attention Norm: {attn_norm:.4f}", style="cyan")

        # Combine heads and project output
        out = attn.transpose(1, 2).reshape(bsz, input_seqlen, -1)
        out = self.Wo(out)
        return self.resid_dropout(out)