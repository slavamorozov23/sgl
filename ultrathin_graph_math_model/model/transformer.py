import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import warnings
from typing import Optional, List, Set, Tuple, Dict

from .blocks import RMSNorm, SpatialCubeLayer, Gate
from .attention import precompute_freqs_cis
import config
from rich.console import Console
console = Console()
from collections import Counter, defaultdict
import json

class SpatialGraphTransformer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int,
                 max_seq_len: int, dropout: float, vocab_size: int,
                 rope_theta: float = 10000.0):
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size must be provided")

        # --- Load graph configuration from JSON ---
        json_path = getattr(config, 'POINTS_JSON_PATH', 'points.json')
        console.log(f"Loading graph configuration from: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Support new format where points are under 'points' key
            if isinstance(raw, dict) and 'points' in raw:
                points_data = raw['points']
            else:
                points_data = raw
            if not isinstance(points_data, list) or not points_data:
                raise ValueError("JSON data must be a non-empty list of points.")
        except FileNotFoundError:
            console.print(f"[bold red]Error: Graph configuration file not found at '{json_path}'. Exiting.[/bold red]")
            raise
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"[bold red]Error parsing graph configuration file '{json_path}': {e}. Exiting.[/bold red]")
            raise

        # --- Extract parameters from JSON ---
        num_cubes_loaded = len(points_data)
        loaded_uids = sorted([p['uid'] for p in points_data])
        if loaded_uids != list(range(num_cubes_loaded)):
            raise ValueError(f"Point uids in '{json_path}' must be sequential from 0 to {num_cubes_loaded-1}. Found: {loaded_uids}")
        self.num_cubes = num_cubes_loaded
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        start_candidates_list = []
        neighbor_map_loaded = defaultdict(list)
        cube_positions_loaded = {}

        expected_keys = {'uid', 'is_input', 'path_neighbors'}
        for point in points_data:
            if not expected_keys.issubset(point.keys()):
                raise ValueError(f"Point object in JSON missing required keys ({expected_keys}). Found: {point}")
            uid = point['uid']
            is_input = point['is_input']
            neighbors_data = point['path_neighbors']

            if is_input:
                start_candidates_list.append(uid)

            current_neighbors = []
            # neighbors_data is now a list of integers (neighbor UIDs)
            for neighbor_uid in neighbors_data:
                if not isinstance(neighbor_uid, int):
                    # Add a check for type, although the error suggests it's already int
                    raise ValueError(f"Invalid neighbor uid type for point {uid}: {neighbor_uid}. Expected int.")
                # Validate the neighbor UID
                if neighbor_uid != config.EXIT_TOKEN_INDEX and not (0 <= neighbor_uid < self.num_cubes):
                    raise ValueError(f"Neighbor uid {neighbor_uid} for point {uid} is out of bounds (0-{self.num_cubes-1}).")
                current_neighbors.append(neighbor_uid)
            neighbor_map_loaded[uid] = current_neighbors

        self.start_candidates = sorted(list(set(start_candidates_list)))
        if not self.start_candidates:
            console.print("[bold yellow]Warning: No input points (is_input=true) found in JSON. Using cube 0 as default start.[/bold yellow]")
            self.start_candidates = [0]

        self.neighbor_map = dict(neighbor_map_loaded)
        for i in range(self.num_cubes):
            if i not in self.neighbor_map:
                console.print(f"[bold yellow]Warning: Cube {i} has no neighbors defined in JSON.[/bold yellow]")
                self.neighbor_map[i] = []

        console.log(f"Graph loaded: {self.num_cubes} cubes, {len(self.start_candidates)} start candidates.")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout_emb = nn.Dropout(dropout)

        # Cube layers
        self.cubes = nn.ModuleList([
            SpatialCubeLayer(d_model, nhead=config.NHEAD, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(self.num_cubes)
        ])

        # Gating
        self.gate = Gate(d_model, self.num_cubes)
        self.final_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        # RoPE buffer
        head_dim = d_model // config.NHEAD
        freqs = precompute_freqs_cis(head_dim, self.max_seq_len * 2, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        nn.init.kaiming_uniform_(self.gate.linear.weight, a=math.sqrt(5))
        for cube in self.cubes:
            for name, param in cube.named_parameters():
                if param.dim() > 1 and 'weight' in name:
                    param.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int, List[int], torch.Tensor, torch.Tensor, bool]:
        """
        Forward pass: per-sample routing.
        Returns:
            logits: Tensor [bsz, seq_len, vocab_size]
            cubes_used_count: int
            cubes_visited_history: List[int]
            aux_variance_loss: Tensor
            aux_load_balancing_loss: Tensor
            max_cubes_limit_reached: bool
        """
        bsz, seq_len = input_ids.shape
        print_debug = False

        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        x_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x_emb = self.dropout_emb(x_emb)

        x_out = x_emb.clone()
        device = x_out.device

        current_cubes = torch.tensor([random.choice(self.start_candidates) for _ in range(bsz)], device=device)
        active_mask = torch.ones(bsz, dtype=torch.bool, device=device)
        steps = torch.zeros(bsz, dtype=torch.long, device=device)
        visit_counts = [Counter() for _ in range(bsz)]
        for i in range(bsz):
            visit_counts[i][current_cubes[i].item()] = 1

        history0 = [current_cubes[0].item()]
        max_limit_reached = False

        accumulated_probs_sum = torch.zeros(self.num_cubes + 1, device=device)
        accumulated_load_balancing_term = torch.tensor(0.0, device=device)
        total_active_routing_steps = 0

        while active_mask.any():
            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            processed_x = torch.zeros_like(x_out[active_indices])
            for cube_idx in range(self.num_cubes):
                mask_k = current_cubes[active_indices] == cube_idx
                if mask_k.any():
                    rel_idx = torch.nonzero(mask_k, as_tuple=False).squeeze(1)
                    inp_k = x_out[active_indices[rel_idx]]
                    out_k = self.cubes[cube_idx](inp_k, freqs_cis=self.freqs_cis, print_debug=print_debug)
                    processed_x[rel_idx] = out_k
            x_out[active_indices] = processed_x

            scores_active = self.gate(x_out[active_indices])

            temp = getattr(config, 'ROUTING_TEMPERATURE', 1.0)
            if temp > 0:
                probs_active = F.softmax(scores_active / temp, dim=-1)
                accumulated_probs_sum += probs_active.sum(dim=0)
                avg_probs = probs_active.mean(dim=0)
                lb = (self.num_cubes + 1) * torch.sum(avg_probs.pow(2))
                accumulated_load_balancing_term += lb
                total_active_routing_steps += probs_active.size(0)

            chosen_next = torch.full_like(current_cubes[active_indices], config.EXIT_TOKEN_INDEX)
            for idx, global_i in enumerate(active_indices.tolist()):
                cur = current_cubes[global_i].item()
                neigh = self.neighbor_map.get(cur, [])

                min_len = max(1, math.ceil(self.num_cubes * config.MIN_PATH_LENGTH_RATIO))
                orig_filtered = [n for n in neigh if not (len(visit_counts[global_i]) < min_len and n == config.EXIT_TOKEN_INDEX)]
                if not orig_filtered:
                    continue

                filtered = [n for n in orig_filtered if visit_counts[global_i].get(n, 0) < config.MAX_CUBES_PER_PATH]
                if not filtered:
                    max_limit_reached = True
                    continue

                map_idx = []
                inv_map = {}
                for j, n in enumerate(filtered):
                    idx_score = self.num_cubes if n == config.EXIT_TOKEN_INDEX else n
                    map_idx.append(idx_score)
                    inv_map[j] = n
                idx_t = torch.tensor(map_idx, device=device, dtype=torch.long)
                sample_scores = scores_active[idx, idx_t]

                if random.random() < config.ROUTING_EPSILON_GREEDY:
                    sel = random.randrange(len(filtered))
                else:
                    sel = torch.argmax(sample_scores).item()
                chosen_next[idx] = inv_map[sel]

            steps[active_indices] += 1

            if active_mask[0]:
                pos0 = (active_indices == 0).nonzero(as_tuple=False)
                if pos0.numel() > 0:
                    next0 = chosen_next[pos0.item()]
                    history0.append(next0.item())

            exit_mask = chosen_next == config.EXIT_TOKEN_INDEX
            limit_mask = steps[active_indices] >= config.ROUTING_SAFETY_LIMIT
            stop_mask = exit_mask | limit_mask
            cont_mask = ~stop_mask
            cont_idx = active_indices[cont_mask]
            current_cubes[cont_idx] = chosen_next[cont_mask]
            for idx, global_i in enumerate(active_indices.tolist()):
                if cont_mask[idx]:
                    visit_counts[global_i][chosen_next[idx].item()] += 1
            active_mask[active_indices[stop_mask]] = False

        if total_active_routing_steps > 0:
            avg_probs_per_target = accumulated_probs_sum / total_active_routing_steps
            aux_variance_loss = torch.var(avg_probs_per_target)
            aux_load_balancing_loss = accumulated_load_balancing_term / total_active_routing_steps
        else:
            aux_variance_loss = torch.tensor(0.0, device=device)
            aux_load_balancing_loss = torch.tensor(0.0, device=device)

        x_final = self.final_norm(x_out)
        logits = self.fc_out(x_final)
        cubes_used_count = len(history0)
        cubes_visited_history = history0
        return logits, cubes_used_count, cubes_visited_history, aux_variance_loss, aux_load_balancing_loss, max_limit_reached