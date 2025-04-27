import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import json
import os

import config
from torch.cuda.amp import autocast
from torch.amp import GradScaler

console = Console()

def train(model, dataloader, optimizer, criterion, epoch, total_epochs, device):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_cubes_used = 0

    try:
        num_batches = len(dataloader)
        if num_batches == 0:
            console.print("[red]Warning: Dataloader returned length 0. Progress bar may not display correctly.[/red]")
    except TypeError:
        num_batches = -1
        console.print("[yellow]Warning: Dataloader does not support len(). Progress bar total will be indeterminate.[/yellow]")

    if epoch == 0:
        viz_pos_path = config.VIZ_POSITIONS_FILE
        if not os.path.exists(viz_pos_path):
            try:
                cube_positions_list = model.cube_positions.cpu().numpy().tolist()
                with open(viz_pos_path, 'w') as f:
                    json.dump(cube_positions_list, f)
                console.log(f"Saved cube positions for visualization to {viz_pos_path}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save cube positions for visualization: {e}[/yellow]")
        viz_start_path = config.VIZ_START_CANDIDATES_FILE
        try:
            if hasattr(model, 'start_candidates'):
                with open(viz_start_path, 'w') as f:
                    json.dump(model.start_candidates, f)
                console.log(f"Saved start candidates for visualization to {viz_start_path}")
            else:
                console.print(f"[yellow]Warning: Model does not have 'start_candidates' attribute. Skipping save to {viz_start_path}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save start candidates: {e}[/yellow]")

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        TextColumn("Cubes: {task.fields[cubes_used]}/{task.fields[num_total_cubes]}"),
        TextColumn("Path: {task.fields[path_str]}")
    ]

    epoch_progress = Progress(*progress_columns, console=console)
    task_id = epoch_progress.add_task(
        f"Epoch {epoch+1}/{total_epochs}",
        total=num_batches if num_batches > 0 else None,
        loss=float('inf'),
        cubes_used=0,
        num_total_cubes=config.NUM_CUBES,
        path_str=""
    )

    processed_batches = 0
    viz_paths_path = config.VIZ_PATHS_FILE
    mode = 'w' if epoch == 0 else 'a'
    try:
        paths_file = open(viz_paths_path, mode)
        action = 'Truncated' if epoch == 0 else 'Opened'
        console.log(f"{action} paths file for visualization ({mode}): {viz_paths_path}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not open paths file for visualization: {e}. Skipping path saving.[/yellow]")
        paths_file = None

    scaler = GradScaler()
    with epoch_progress:
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with autocast():
                logits, cubes_used_count, cubes_visited_history = model(input_ids)
                main_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = main_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            processed_batches += 1
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_cubes_used += cubes_used_count

            avg_loss = total_loss / processed_batches

            if len(cubes_visited_history) <= 4:
                parts = [f"[{idx}]" for idx in cubes_visited_history]
            else:
                first2 = cubes_visited_history[:2]
                last2 = cubes_visited_history[-2:]
                skipped = len(cubes_visited_history) - 4
                parts = [f"[{idx}]" for idx in first2] + ["..."] + [f"[{idx}]" for idx in last2]
            path_str = " -> ".join(parts)

            epoch_progress.update(
                task_id,
                loss=avg_loss,
                cubes_used=cubes_used_count,
                path_str=path_str
            )
            epoch_progress.update(task_id, advance=1)

            if paths_file:
                try:
                    transitions = []
                    if len(cubes_visited_history) > 1:
                        transitions = [[cubes_visited_history[j], cubes_visited_history[j+1]]
                                       for j in range(len(cubes_visited_history)-1)]
                    data_to_save = {"epoch": epoch, "batch": i, "transitions": transitions}
                    paths_file.write(json.dumps(data_to_save) + '\n')
                    paths_file.flush()
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not save path data for batch {i}: {e}[/yellow]")

        if num_batches <= 0:
            epoch_progress.update(task_id, total=processed_batches)

    if paths_file:
        paths_file.close()

    return total_loss / processed_batches if processed_batches > 0 else 0.0