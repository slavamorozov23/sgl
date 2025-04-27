import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple, List

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dim_rope = x.shape[-1]
    rope_dim_half = dim_rope // 2
    assert dim_rope % 2 == 0, "RoPE dimension must be even"

    if freqs_cis.shape[0] == 0:
         if x.shape[1] == 0:
              return x
         else:
              raise ValueError(f"freqs_cis has length 0 but input sequence (dim 1) has length {x.shape[1]}")

    assert freqs_cis.shape[0] == x.shape[1], \
        f"Sequence length mismatch for RoPE: freqs_cis has {freqs_cis.shape[0]} but input has {x.shape[1]} at dim 1"
    assert freqs_cis.shape[1] == rope_dim_half, \
        f"RoPE dimension mismatch: freqs_cis has {freqs_cis.shape[1]} but expected {rope_dim_half}"

    x_r = x.float().reshape(*x.shape[:-1], rope_dim_half, 2)
    x_c = torch.view_as_complex(x_r)

    freqs_cis_b = freqs_cis.view(1, *freqs_cis.shape)
    while freqs_cis_b.ndim < x_c.ndim:
         freqs_cis_b = freqs_cis_b.unsqueeze(0)

    x_out_c = x_c * freqs_cis_b.to(x_c.device)

    x_out_r = torch.view_as_real(x_out_c)
    x_out = x_out_r.flatten(start_dim=-2)

    return x_out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        rms_inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms_inv

    def forward(self, x):
        normalized_x = self._norm(x.float()).type_as(x)
        return self.weight * normalized_x

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, self.head_dim, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, freqs_cis: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len_in, _ = x.shape

        q = self.Wq(x)
        k_new = self.Wk(x)
        v_new = self.Wv(x)
        q = q.view(batch_size, seq_len_in, self.nhead, self.head_dim)

        if freqs_cis is not None:
            if freqs_cis.numel() == 0 and seq_len_in > 0:
                 raise ValueError("Received empty freqs_cis for non-empty sequence.")
            elif freqs_cis.numel() > 0:
                k_new_rotated = apply_rotary_emb_single(k_new, freqs_cis)
                q_reshaped = q.transpose(1, 2).reshape(batch_size * self.nhead, seq_len_in, self.head_dim)
                q_rotated_reshaped = apply_rotary_emb_single(q_reshaped, freqs_cis)
                q_rotated = q_rotated_reshaped.view(batch_size, self.nhead, seq_len_in, self.head_dim).transpose(1, 2)
            else: # seq_len_in == 0
                 q_rotated = q
                 k_new_rotated = k_new
        else: # freqs_cis is None
            q_rotated = q
            k_new_rotated = k_new

        q_final = q_rotated.transpose(1, 2)
        k_new_final = k_new_rotated.unsqueeze(1)
        v_new_final = v_new.unsqueeze(1)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_final = torch.cat((past_k, k_new_final), dim=2)
            v_final = torch.cat((past_v, v_new_final), dim=2)
        else:
            k_final = k_new_final
            v_final = v_new_final

        current_kv = (k_final, v_final)

        attn_output = F.scaled_dot_product_attention(
            q_final, k_final, v_final,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_in, self.d_model)
        output = self.Wo(output)
        output = self.resid_dropout(output)

        return output, current_kv


class MQAEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiQueryAttention(d_model, nhead, dropout=dropout)
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, is_causal: bool = False, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, freqs_cis: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = src
        attn_output, current_kv = self.self_attn(self.norm1(x), is_causal=is_causal, past_kv=past_kv, freqs_cis=freqs_cis)
        x = x + self.dropout1(attn_output)

        norm_x = self.norm2(x)
        swish_gate = F.silu(self.w1(norm_x))
        value = self.w3(norm_x)
        gated_output = swish_gate * value
        ff_output = self.linear2(gated_output)
        x = x + self.dropout2(ff_output)

        return x, current_kv


class SimpleMQA_Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_seq_len: int, dropout: float = 0.1, vocab_size: int = None, rope_theta: float = 10000.0):
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size must be provided")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MQAEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        head_dim = d_model // nhead
        freqs_cis = precompute_freqs_cis(head_dim, self.max_seq_len * 2, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.fc_out.bias is not None:
            self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

        for layer in self.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'Wq') and isinstance(attn.Wq, nn.Linear):
                    attn.Wq.weight.data.uniform_(-initrange, initrange)
                if hasattr(attn, 'Wk') and isinstance(attn.Wk, nn.Linear):
                    attn.Wk.weight.data.uniform_(-initrange, initrange)
                if hasattr(attn, 'Wv') and isinstance(attn.Wv, nn.Linear):
                    attn.Wv.weight.data.uniform_(-initrange, initrange)
                if hasattr(attn, 'Wo') and isinstance(attn.Wo, nn.Linear):
                    attn.Wo.weight.data.uniform_(-initrange, initrange)

            if hasattr(layer, 'w1') and isinstance(layer.w1, nn.Linear):
                layer.w1.weight.data.uniform_(-initrange, initrange)
            if hasattr(layer, 'w3') and isinstance(layer.w3, nn.Linear):
                layer.w3.weight.data.uniform_(-initrange, initrange)
            if hasattr(layer, 'linear2') and isinstance(layer.linear2, nn.Linear):
                layer.linear2.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_ids: torch.Tensor, past_kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len_in = input_ids.shape

        start_pos = 0
        if past_kv_cache and past_kv_cache[0] is not None and past_kv_cache[0][0] is not None:
            start_pos = past_kv_cache[0][0].shape[2]

        end_pos = start_pos + seq_len_in
        if end_pos > self.freqs_cis.shape[0]:
             raise ValueError(
                f"Requested RoPE frequencies up to position {end_pos} exceed precomputed buffer length {self.freqs_cis.shape[0]}. "
                f"Input seq len: {seq_len_in}, Cache len: {start_pos}"
            )
        freqs_cis = self.freqs_cis[start_pos : end_pos]

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        present_kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        if past_kv_cache is None:
            past_kv_cache = [None] * self.num_layers

        is_causal = seq_len_in > 1

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv_cache[i]
            x, current_kv = layer(x, is_causal=is_causal, past_kv=layer_past_kv, freqs_cis=freqs_cis)
            present_kv_cache.append(current_kv)

        x = self.norm(x)
        # Check if the output embedding layer size matches the model's embedding size
        if self.fc_out.in_features != x.size(-1):
             # This can happen after resize_token_embeddings if fc_out is not tied or updated
             # A simple fix might be to re-initialize fc_out if vocab size changed,
             # but often embeddings are tied, or output layer is handled differently.
             # For now, raise an error if sizes mismatch.
             raise RuntimeError(f"Output layer input dimension ({self.fc_out.in_features}) "
                                f"does not match transformer output dimension ({x.size(-1)})")

        output = self.fc_out(x)

        return output, present_kv_cache