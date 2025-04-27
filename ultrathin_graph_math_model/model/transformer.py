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
                points_data = json.load(f)
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

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int, List[int], torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        print_debug = False

        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout_emb(x)
        current = random.choice(self.start_candidates)
        visited: Set[int] = {current}
        history: List[int] = [current]
        visit_counts = Counter({current: 1})  # track uses per cube
        trans: List[Tuple[int,int]] = []
        steps = 0

        if print_debug: console.print(f"\n--- [DEBUG BATCH] Start Routing from: {current} ---", style="bold blue")

        # Initialize accumulators for auxiliary loss
        accumulated_probs_sum = torch.zeros(self.num_cubes + 1, device=x.device)
        total_steps_with_probs = 0
        accumulated_load_balancing_loss = torch.tensor(0.0, device=x.device)
        total_routing_steps = 0

        while True:
            if steps >= config.ROUTING_SAFETY_LIMIT:
                # ... (обработка лимита) ...
                break

            # --- ДИАГНОСТИКА: ШАГ РОУТИНГА ---
            if print_debug and steps < 10: # Ограничим вывод первыми 10 шагами
                console.print(f"  [Step {steps}] Current Cube: {current}", style="blue")

            cube = self.cubes[current]
            x = cube(x, freqs_cis=self.freqs_cis, print_debug=print_debug)
            if print_debug and steps < 10:
                if hasattr(cube.self_attn, 'kv_cache') and cube.self_attn.kv_cache is not None:
                    console.print(f"  [Step {steps}] MLA KV Cache Size: {cube.self_attn.kv_cache.shape}", style="cyan")
                else:
                    console.print(f"  [Step {steps}] MLA KV Cache: None", style="cyan")
            scores = self.gate(x)

            # Calculate and accumulate probabilities for auxiliary loss
            temperature = getattr(config, 'ROUTING_TEMPERATURE', 1.0)
            if temperature > 0:
                temp_scores = scores / temperature
                full_probs_this_step = F.softmax(temp_scores, dim=-1) # [bsz, num_cubes + 1]
                avg_probs_per_target_step = full_probs_this_step.mean(dim=0) # [num_cubes + 1]
                num_choices = self.num_cubes + 1
                sum_squared_probs = torch.sum(avg_probs_per_target_step * avg_probs_per_target_step)
                step_load_balancing_loss = num_choices * sum_squared_probs
                accumulated_load_balancing_loss = accumulated_load_balancing_loss + step_load_balancing_loss
                total_routing_steps += 1
                accumulated_probs_sum += avg_probs_per_target_step
                total_steps_with_probs += 1

            # Получаем соседей
            neigh = self.neighbor_map.get(current, [])
            if print_debug and steps < 10:
                console.print(f"    Original Neighbors: {neigh}", style="cyan")

            # ... (логика фильтрации соседей) ...
            min_len = max(1, math.ceil(self.num_cubes * config.MIN_PATH_LENGTH_RATIO))
            filtered_neigh = list(neigh)
            removed_exit = False
            if len(history) < min_len:
                if config.EXIT_TOKEN_INDEX in filtered_neigh:
                    filtered_neigh.remove(config.EXIT_TOKEN_INDEX)
                    removed_exit = True

            if print_debug and steps < 10:
                console.print(f"    Filtered Neighbors (min_len={min_len}, hist={len(history)}, removed_exit={removed_exit}): {filtered_neigh}", style="cyan")

            if not filtered_neigh:
                 # ... (обработка отсутствия соседей) ...
                 break

            # Enforce max uses per cube in this path
            filtered_neigh = [n for n in filtered_neigh if visit_counts.get(n, 0) < config.MAX_CUBES_PER_PATH]
            if not filtered_neigh:
                console.print(f"[bold red]Max cubes per path ({config.MAX_CUBES_PER_PATH}) reached for all neighbors. Ending route.[/]", style="red")
                break

            neighbor_score_indices = []
            original_neighbor_indices_map = {}
            for i, n_idx in enumerate(filtered_neigh):
                # ... (логика получения score_idx) ...
                if n_idx == config.EXIT_TOKEN_INDEX: score_idx = self.num_cubes
                elif 0 <= n_idx < self.num_cubes: score_idx = n_idx
                else: continue
                neighbor_score_indices.append(score_idx)
                original_neighbor_indices_map[i] = n_idx

            if not neighbor_score_indices:
                 # ... (обработка отсутствия индексов) ...
                 break

            idx_tensor = torch.tensor(neighbor_score_indices, device=scores.device, dtype=torch.long)
            neighbor_scores = scores[:, idx_tensor].mean(dim=0)

            # Apply neighbor biases
            # Removed neighbor bias application as the mechanism is being removed.

            temperature = getattr(config, 'ROUTING_TEMPERATURE', 1.0)

            # --- ДИАГНОСТИКА: ОЦЕНКИ И ВЕРОЯТНОСТИ ---
            if print_debug and steps < 10:
                console.print(f"      Neighbor Scores: {neighbor_scores.detach().cpu().numpy()}", style="green")
                console.print(f"      Temperature: {temperature}", style="green")

            if temperature <= 0:
                if print_debug and steps < 10: console.print(f"      Mode: Argmax (T <= 0)", style="yellow")
                chosen_relative_idx = torch.argmax(neighbor_scores).item()
            else:
                probs = F.softmax(neighbor_scores / temperature, dim=-1)
                # --- ДИАГНОСТИКА: ВЕРОЯТНОСТИ ---
                if print_debug and steps < 10:
                     console.print(f"      Probabilities: {probs.detach().cpu().numpy()}", style="magenta")

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    if print_debug and steps < 10: console.print(f"      Mode: Fallback (NaN/Inf) -> Uniform", style="red")
                    chosen_relative_idx = random.randrange(len(filtered_neigh))
                else:
                    try:
                        if print_debug and steps < 10: console.print(f"      Mode: Multinomial Sampling", style="yellow")
                        chosen_relative_idx = torch.multinomial(probs, num_samples=1).item()
                    except RuntimeError as e:
                        if print_debug and steps < 10: console.print(f"      Mode: Fallback (Multinomial Error {e}) -> Argmax", style="red")
                        chosen_relative_idx = torch.argmax(neighbor_scores).item()

            # --- ДИАГНОСТИКА: ВЫБОР ---
            if chosen_relative_idx >= len(filtered_neigh):
                 console.print(f"[bold red]      ERROR: chosen_relative_idx {chosen_relative_idx} out of bounds for filtered_neigh {filtered_neigh}[/]")
                 # Handle error - perhaps break or choose randomly from what's available
                 if filtered_neigh:
                     chosen_relative_idx = random.randrange(len(filtered_neigh))
                 else: # Should not happen if checks above work
                      break
            next_cube = filtered_neigh[chosen_relative_idx]

            if print_debug and steps < 10:
                console.print(f"      Chosen Relative Idx: {chosen_relative_idx}", style="bold green")
                console.print(f"      --> Next Cube: {next_cube}", style="bold green")
            # ---- КОНЕЦ ИЗМЕНЕННОЙ ЛОГИКИ ----

            trans.append((current, next_cube))

            # Update neighbor biases based on visited path
            # Removed neighbor bias update as the mechanism is being removed.

            if next_cube == config.EXIT_TOKEN_INDEX:
                if print_debug: console.print(f"  [Step {steps+1}] EXIT token chosen. Route ended.", style="bold blue")
                break

            # Update neighbor biases based on visited path
            # Update neighbor biases based on visited path
            # Removed neighbor bias update as the mechanism is being removed.

            visited.add(next_cube)
            history.append(next_cube)
            visit_counts[next_cube] = visit_counts.get(next_cube, 0) + 1
            current = next_cube
            steps += 1

        # Calculate auxiliary routing loss after the loop
        if total_routing_steps > 0:
            final_load_balancing_loss = accumulated_load_balancing_loss / total_routing_steps
        else:
            final_load_balancing_loss = torch.tensor(0.0, device=x.device)

        if total_steps_with_probs > 0:
            avg_probs_per_target = accumulated_probs_sum / total_steps_with_probs # Avg over steps [num_cubes+1]
            # Loss: Variance of these average probabilities (encourage balance)
            # Smaller variance means more balanced probabilities.
            aux_routing_loss = torch.var(avg_probs_per_target)
        else:
            aux_routing_loss = torch.tensor(0.0, device=x.device) # No routing steps, no aux loss

        # Apply final_norm and fc_out to x
        x = self.final_norm(x)
        logits = self.fc_out(x)

        if print_debug: console.print(f"--- [DEBUG BATCH] Final Path: {history} (Length: {len(history)}) ---", style="bold blue")

        return logits, len(history), history, aux_routing_loss, final_load_balancing_loss