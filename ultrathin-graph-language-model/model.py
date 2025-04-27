import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple, List, Dict, Set
import numpy as np
import warnings
import random

import config # Assuming RMSNorm is defined here or imported appropriately
# Placeholder for RMSNorm if not defined elsewhere
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return freqs_cis

# --- Function apply_rotary_emb_single (Modified RoPE handling) ---
def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dim_rope = x.shape[-1]
    rope_dim_half = dim_rope // 2
    assert dim_rope % 2 == 0, "RoPE dimension must be even"

    if freqs_cis.shape[0] == 0:
         if x.shape[1] == 0: return x
         else: raise ValueError(f"freqs_cis has length 0 but input sequence (dim 1) has length {x.shape[1]}")

    if freqs_cis.shape[0] < x.shape[1]:
        raise ValueError(f"Sequence length mismatch for RoPE: freqs_cis has {freqs_cis.shape[0]} but input needs at least {x.shape[1]} at dim 1")
    freqs_cis_used = freqs_cis[:x.shape[1]]

    assert freqs_cis_used.shape[1] == rope_dim_half, \
        f"RoPE dimension mismatch: freqs_cis_used has {freqs_cis_used.shape[1]} but expected {rope_dim_half}"

    x_r = x.float().reshape(*x.shape[:-1], rope_dim_half, 2)
    x_c = torch.view_as_complex(x_r)

    freqs_cis_b = freqs_cis_used.unsqueeze(0)
    if x_c.ndim == 4:
         freqs_cis_b = freqs_cis_b.unsqueeze(1)

    x_out_c = x_c * freqs_cis_b.to(x_c.device)
    x_out_r = torch.view_as_real(x_out_c)
    x_out = x_out_r.flatten(start_dim=-2)

    return x_out.type_as(x)

# --- Class MultiQueryAttention (Modified forward for RoPE) ---
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

    def forward(self, x: torch.Tensor, is_causal: bool = False, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len_in, _ = x.shape

        q = self.Wq(x)
        k_new = self.Wk(x)
        v_new = self.Wv(x)

        q = q.view(batch_size, seq_len_in, self.nhead, self.head_dim)

        if freqs_cis is not None:
            if freqs_cis.numel() > 0 and seq_len_in > 0:
                 current_freqs_cis = freqs_cis[:seq_len_in]
                 k_new_rotated = apply_rotary_emb_single(k_new, current_freqs_cis)
                 q_transposed = q.transpose(1, 2)
                 q_reshaped = q_transposed.reshape(batch_size * self.nhead, seq_len_in, self.head_dim)
                 q_rotated_reshaped = apply_rotary_emb_single(q_reshaped, current_freqs_cis)
                 q_rotated_transposed = q_rotated_reshaped.view(batch_size, self.nhead, seq_len_in, self.head_dim)
                 q_rotated = q_rotated_transposed.transpose(1, 2)
            else:
                 q_rotated = q
                 k_new_rotated = k_new
        else:
            q_rotated = q
            k_new_rotated = k_new

        k_final = k_new_rotated.unsqueeze(1)
        v_final = v_new.unsqueeze(1)

        q_final = q_rotated.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q_final, k_final, v_final, attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal and seq_len_in > 1
        )

        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_in, self.d_model)
        output = self.Wo(output)
        output = self.resid_dropout(output)
        return output

# --- Class SpatialCubeLayer (Modified forward for RoPE passing) ---
class SpatialCubeLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiQueryAttention(d_model, nhead=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        attn_output = self.self_attn(self.norm1(x), is_causal=True, freqs_cis=freqs_cis)
        x = residual + self.dropout1(attn_output)

        residual = x
        norm_x = self.norm2(x)
        swish_gate = F.silu(self.w1(norm_x))
        value = self.w3(norm_x)
        ff_output = self.linear2(swish_gate * value)
        x = residual + self.dropout2(ff_output)
        return x

# --- Class SpatialGraphTransformer (Modified __init__, _compute_neighbors, forward return signature) ---
class SpatialGraphTransformer(nn.Module):
    def __init__(self, num_cubes: int, d_model: int, dim_feedforward: int,
                 max_seq_len: int, dropout: float, vocab_size: int,
                 max_distance: float,
                 rope_theta: float = 10000.0,
                 entry_exit_ratio: float = 0.1):
        super().__init__()
        if vocab_size is None: raise ValueError("vocab_size must be provided")
        if not (0 < entry_exit_ratio <= 1.0):
             raise ValueError("entry_exit_ratio must be between 0 (exclusive) and 1.0")

        self.num_cubes = num_cubes
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout_emb = nn.Dropout(dropout)

        self.cubes = nn.ModuleList([
            SpatialCubeLayer(d_model, nhead=config.NHEAD, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_cubes)
        ])

        num_entry_exit = max(1, int(self.num_cubes * entry_exit_ratio))
        all_cube_indices = list(range(self.num_cubes))
        self.entry_exit_indices = set(random.sample(all_cube_indices, num_entry_exit))
        self.entry_exit_list = sorted(list(self.entry_exit_indices))
        # Store start candidates for visualization, accessible after init
        self.start_candidates = self.entry_exit_list # Keep for viz
        print(f"Initialized with {len(self.entry_exit_indices)} entry/exit cubes: {self.entry_exit_list}")

        self.cube_positions = nn.Parameter(torch.randn(num_cubes, 3), requires_grad=False)
        self.neighbor_map = self._compute_neighbors()
        self.gating_linear = nn.Linear(d_model, num_cubes + 1, bias=False)
        self.final_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        head_dim = d_model // config.NHEAD
        freqs_cis = precompute_freqs_cis(head_dim, self.max_seq_len * 2, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.init_weights()

        self.register_buffer(
            'neighbor_biases',
            torch.zeros((self.num_cubes, self.num_cubes + 1), dtype=torch.float)
        )
        self.register_buffer(
            'visit_counts_epoch',
            torch.zeros((self.num_cubes, self.num_cubes + 1), dtype=torch.float)
        )

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        if hasattr(self.gating_linear, 'weight'):
            nn.init.kaiming_uniform_(self.gating_linear.weight, a=math.sqrt(5))
        for cube in self.cubes:
             for name, param in cube.named_parameters():
                  if param.dim() > 1 and 'weight' in name:
                      param.data.uniform_(-initrange, initrange)

    def _compute_neighbors(self) -> Dict[int, List[int]]:
        neighbor_map = {}
        num_cubes = self.num_cubes
        target_cube_neighbors = max(1, int(num_cubes * 0.1))
        target_cube_neighbors = min(target_cube_neighbors, num_cubes - 1)
        all_cube_indices = list(range(num_cubes))

        for i in range(num_cubes):
            potential_cube_neighbors = [j for j in all_cube_indices if j != i]
            if not potential_cube_neighbors:
                sampled_neighbors = []
            elif len(potential_cube_neighbors) <= target_cube_neighbors:
                sampled_neighbors = potential_cube_neighbors
            else:
                sampled_neighbors = random.sample(potential_cube_neighbors, target_cube_neighbors)

            neighbors = list(sampled_neighbors)

            if i in self.entry_exit_indices:
                neighbors.append(config.EXIT_TOKEN_INDEX)

            neighbor_map[i] = neighbors
        return neighbor_map

    def bias_for_pair(self, from_idx: int, to_idx: int):
        to_col = self.num_cubes if to_idx == config.EXIT_TOKEN_INDEX else to_idx
        # Add boundary checks for safety
        if 0 <= from_idx < self.num_cubes and 0 <= to_col <= self.num_cubes:
            return self.neighbor_biases[from_idx, to_col]
        else:
            warnings.warn(f"Invalid indices requested for bias_for_pair: from {from_idx}, to {to_idx} (col {to_col})", RuntimeWarning)
            return torch.tensor(0.0, device=self.neighbor_biases.device) # Return neutral bias

    # Updated return signature: removed aux_loss
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int, List[int]]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            warnings.warn(f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len}). Truncating.", RuntimeWarning)
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        elif seq_len == 0:
             warnings.warn("Input sequence length is 0. Returning zeros.", RuntimeWarning)
             dummy_logits = torch.zeros(batch_size, 0, self.fc_out.out_features, device=input_ids.device, dtype=self.fc_out.weight.dtype)
             # Return dummy values consistent with the new signature
             return dummy_logits, 0, []

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout_emb(x)

        current_cube_index = random.choice(self.entry_exit_list)

        visited_path_indices: Set[int] = {current_cube_index}
        # Removed: total_aux_loss initialization
        current_freqs_cis = self.freqs_cis
        step_count = 0
        cubes_visited_count = 1
        route_bigrams = []
        cubes_visited_history = [current_cube_index]
        batch_transition_pairs = []

        while True:
            if step_count >= config.ROUTING_SAFETY_LIMIT:
                warnings.warn(f"Routing safety limit ({config.ROUTING_SAFETY_LIMIT}) reached. Exiting loop.", RuntimeWarning)
                if current_cube_index in self.entry_exit_indices:
                    next_choice_actual_index = config.EXIT_TOKEN_INDEX
                    batch_transition_pairs.append((current_cube_index, next_choice_actual_index)) # Record forced transition
                    break
                else:
                     warnings.warn(f"Safety limit reached in non-exit cube {current_cube_index}. Cannot force exit. Stopping path.", RuntimeWarning)
                     break

            min_path_length = max(1, int(self.num_cubes * config.MIN_PATH_LENGTH_RATIO))

            if cubes_visited_count > config.MAX_CUBES_PER_PATH:
                 if current_cube_index in self.entry_exit_indices:
                     next_choice_actual_index = config.EXIT_TOKEN_INDEX
                     batch_transition_pairs.append((current_cube_index, next_choice_actual_index))
                     break
                 else:
                     warnings.warn(f"Max cubes per path ({config.MAX_CUBES_PER_PATH}) reached in non-exit cube {current_cube_index}. Forcing move to neighbor.", RuntimeWarning)
                     gating_input = x.mean(dim=1)
                     all_scores = self.gating_linear(gating_input)
                     potential_neighbors = self.neighbor_map.get(current_cube_index, []) # Use .get for safety
                     non_exit_neighbors = [n for n in potential_neighbors if n != config.EXIT_TOKEN_INDEX]

                     if not non_exit_neighbors:
                         warnings.warn(f"Max path reached in non-exit cube {current_cube_index} with no non-exit neighbors. Stopping path.", RuntimeWarning)
                         break

                     score_indices = [idx for idx in non_exit_neighbors]
                     if not score_indices: # Should be covered by `if not non_exit_neighbors` but double check
                          warnings.warn(f"Empty score_indices in max_path forced move for cube {current_cube_index}. Stopping path.", RuntimeWarning)
                          break
                     score_indices_tensor = torch.tensor(score_indices, device=x.device, dtype=torch.long)

                     # Check bounds before indexing all_scores
                     if score_indices_tensor.max() >= all_scores.shape[1] or score_indices_tensor.min() < 0:
                          warnings.warn(f"Invalid score indices in forced move: {score_indices}. Scores shape: {all_scores.shape}. Stopping path.", RuntimeWarning)
                          break

                     relevant_scores = all_scores[:, score_indices_tensor]

                     neighbor_biases_values = torch.tensor(
                         [self.bias_for_pair(current_cube_index, idx).item() for idx in non_exit_neighbors],
                         device=x.device,
                         dtype=relevant_scores.dtype
                     ).unsqueeze(0)
                     relevant_scores_with_bias = relevant_scores + neighbor_biases_values

                     routing_probs = F.softmax(relevant_scores_with_bias, dim=-1)
                     choice_idx_in_filtered_list = torch.argmax(routing_probs[0], dim=-1).item()
                     next_choice_actual_index = non_exit_neighbors[choice_idx_in_filtered_list]
                     batch_transition_pairs.append((current_cube_index, next_choice_actual_index))
                     # Now set current_cube_index and continue loop to process this *forced* neighbor
                     current_cube_index = next_choice_actual_index
                     visited_path_indices.add(current_cube_index)
                     # Do NOT increment cubes_visited_count here as we're already over limit
                     step_count += 1
                     cubes_visited_history.append(current_cube_index)
                     # Skip the normal processing for this iteration as the choice was forced
                     continue


            # --- Normal Processing ---
            current_cube = self.cubes[current_cube_index]
            x = current_cube(x, freqs_cis=current_freqs_cis)

            gating_input = x.mean(dim=1)
            all_scores = self.gating_linear(gating_input)
            potential_neighbors = self.neighbor_map.get(current_cube_index, []) # Use .get for safety

            can_exit_legally = (cubes_visited_count >= min_path_length) and (current_cube_index in self.entry_exit_indices)

            if can_exit_legally:
                filtered_neighbors = potential_neighbors
            else:
                filtered_neighbors = [n for n in potential_neighbors if n != config.EXIT_TOKEN_INDEX]

            if not filtered_neighbors:
                 warnings.warn(f"No valid neighbors found for cube {current_cube_index} (can_exit_legally={can_exit_legally}). Path terminated.", RuntimeWarning)
                 break

            eps = config.ROUTING_EPSILON_GREEDY
            exploration = False
            explorable_neighbors = [n for n in filtered_neighbors if n != config.EXIT_TOKEN_INDEX]
            can_explore = len(explorable_neighbors) > 0

            if can_explore and random.random() < eps:
                 next_choice_actual_index = random.choice(explorable_neighbors)
                 exploration = True
            else:
                score_indices = [idx if idx != config.EXIT_TOKEN_INDEX else self.num_cubes for idx in filtered_neighbors]
                # Check if score_indices is empty (should not happen if filtered_neighbors is not empty)
                if not score_indices:
                     warnings.warn(f"Empty score_indices for cube {current_cube_index} despite non-empty filtered_neighbors. Path terminated.", RuntimeWarning)
                     break
                score_indices_tensor = torch.tensor(score_indices, device=x.device, dtype=torch.long)

                if score_indices_tensor.max() >= all_scores.shape[1] or score_indices_tensor.min() < 0:
                     # This indicates a potential issue with indexing or neighbor definition
                     warnings.warn(f"Score indices out of bounds: Indices={score_indices}, Max={score_indices_tensor.max()}, Min={score_indices_tensor.min()}, ScoreShape={all_scores.shape}. Path terminated.", RuntimeWarning)
                     break

                relevant_scores = all_scores[:, score_indices_tensor]

                neighbor_biases_values = torch.tensor(
                    [self.bias_for_pair(current_cube_index, idx).item() for idx in filtered_neighbors],
                    device=x.device,
                    dtype=relevant_scores.dtype
                ).unsqueeze(0)
                relevant_scores_with_bias = relevant_scores + neighbor_biases_values

                routing_probs = F.softmax(relevant_scores_with_bias, dim=-1)
                # Ensure routing_probs is not empty or NaN before argmax
                if routing_probs.numel() == 0 or torch.isnan(routing_probs).any():
                     warnings.warn(f"Invalid routing_probs for cube {current_cube_index}. Path terminated.", RuntimeWarning)
                     break
                choice_idx_in_filtered_list = torch.argmax(routing_probs[0], dim=-1).item()
                # Ensure choice index is valid for filtered_neighbors list
                if choice_idx_in_filtered_list >= len(filtered_neighbors):
                     warnings.warn(f"Argmax index {choice_idx_in_filtered_list} out of bounds for filtered_neighbors (len {len(filtered_neighbors)}) for cube {current_cube_index}. Path terminated.", RuntimeWarning)
                     break
                next_choice_actual_index = filtered_neighbors[choice_idx_in_filtered_list]


            batch_transition_pairs.append((current_cube_index, next_choice_actual_index))

            if next_choice_actual_index != config.EXIT_TOKEN_INDEX:
                route_bigrams.append((current_cube_index, next_choice_actual_index))

            if next_choice_actual_index == config.EXIT_TOKEN_INDEX:
                break

            current_cube_index = next_choice_actual_index
            visited_path_indices.add(current_cube_index)
            step_count += 1
            cubes_visited_count += 1
            cubes_visited_history.append(current_cube_index)
            # --- End Normal Processing ---


        # --- Bias updates (Adaptive Mechanism) ---
        with torch.no_grad():
            for from_idx, to_idx in batch_transition_pairs:
                to_col = self.num_cubes if to_idx == config.EXIT_TOKEN_INDEX else to_idx
                if 0 <= from_idx < self.num_cubes and 0 <= to_col <= self.num_cubes:
                     self.visit_counts_epoch[from_idx, to_col] += 1.0
                else:
                     warnings.warn(f"Invalid indices for visit_counts_epoch: from {from_idx}, to {to_idx} (col {to_col})", RuntimeWarning)

            update_after_n = 1000
            total_visits = self.visit_counts_epoch.sum().item()

            if total_visits >= update_after_n:
                num_possible_targets = self.visit_counts_epoch.shape[1]
                total_visits_from_source = self.visit_counts_epoch.sum(dim=1, keepdim=True)
                # Avoid division by zero for sources with no visits
                valid_sources_mask = total_visits_from_source.squeeze() > 0
                # Calculate expected only for sources that were actually visited
                expected = torch.zeros_like(self.visit_counts_epoch)
                if valid_sources_mask.any():
                    expected[valid_sources_mask] = total_visits_from_source[valid_sources_mask] / (num_possible_targets + 1e-9)

                # Compare only where expected is calculated (i.e., where source was visited)
                overload = torch.zeros_like(self.neighbor_biases, dtype=torch.bool)
                underload = torch.zeros_like(self.neighbor_biases, dtype=torch.bool)
                if valid_sources_mask.any():
                     overload[valid_sources_mask] = self.visit_counts_epoch[valid_sources_mask] > expected[valid_sources_mask]
                     underload[valid_sources_mask] = (self.visit_counts_epoch[valid_sources_mask] > 0) & \
                                                     (self.visit_counts_epoch[valid_sources_mask] < expected[valid_sources_mask])

                gamma = config.ROUTING_BIAS_UPDATE_RATE
                # Update biases using masks - ensure mutable if needed or reassign
                new_biases = self.neighbor_biases.clone()
                new_biases[overload] -= gamma
                new_biases[underload] += gamma
                self.neighbor_biases = torch.clamp(new_biases, -6.0, 6.0)

                self.visit_counts_epoch.zero_()

        # --- Route Penalty applied via Bias Update (using ROUTE_PENALTY_LAMBDA) ---
        # This section implicitly penalizes frequently used non-exit transitions by adjusting biases
        with torch.no_grad():
             L = len(route_bigrams)
             if L > 0:
                 M = torch.zeros_like(self.neighbor_biases[:, :self.num_cubes]) # Only for non-exit transitions
                 for (from_idx, to_idx) in route_bigrams:
                     if to_idx != config.EXIT_TOKEN_INDEX: # Already filtered but double-check
                          if 0 <= from_idx < self.num_cubes and 0 <= to_idx < self.num_cubes:
                              M[from_idx, to_idx] += 1.0
                          else:
                              warnings.warn(f"Invalid indices for M update: from {from_idx}, to {to_idx}", RuntimeWarning)

                 M /= L # Normalize counts

                 eta = config.ROUTING_BIAS_UPDATE_RATE # Reusing rate, maybe should be different?
                 lam = config.ROUTE_PENALTY_LAMBDA
                 # Apply penalty only to non-exit transitions recorded in M
                 new_biases = self.neighbor_biases.clone()
                 new_biases[:, :self.num_cubes] -= eta * lam * M
                 self.neighbor_biases = torch.clamp(new_biases, -6.0, 6.0)
        # --- End Bias Updates ---

        x = self.final_norm(x)
        logits = self.fc_out(x)

        # Return logits, number of unique cubes visited, and the history of cube indices
        return logits, len(visited_path_indices), cubes_visited_history