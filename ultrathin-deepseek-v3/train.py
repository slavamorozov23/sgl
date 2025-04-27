import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

def train(model, dataloader, optimizer, criterion, epoch, total_epochs, device):
    """ Trains the custom model for one epoch. """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_columns = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(),
        TimeElapsedColumn(), TextColumn("Loss: {task.fields[loss]:.4f}"),
    ]
    epoch_progress = Progress(*progress_columns, console=console)
    task_id = epoch_progress.add_task(f"Epoch {epoch+1}/{total_epochs}", total=num_batches, loss=float('inf'))

    with epoch_progress:
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs, _ = model(input_ids, past_kv_cache=None) # Assuming no cache during training steps

            # Ensure outputs and labels are compatible with the criterion
            # Output shape: [Batch, SeqLen, VocabSize] -> [Batch*SeqLen, VocabSize]
            # Labels shape: [Batch, SeqLen] -> [Batch*SeqLen]
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            epoch_progress.update(task_id, advance=1, loss=avg_loss)

    return total_loss / num_batches