# --- START OF FILE train.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import redis
import json
import os

import config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.amp import autocast, GradScaler
from rich.live import Live
from rich.text import Text
from rich.console import Group
from torch.optim.lr_scheduler import SequentialLR  # Add scheduler import
from metrics import metrics # Import metrics collector
import tokenizer_utils # Импортируем наш модуль

console = Console()

# Настройки Redis (можно вынести в config.py)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PATH_LIST_KEY = 'spatial_graph:training_paths' # Ключ списка в Redis

import multiprocessing
redis_client = None # Инициализируем клиент как None по умолчанию

def init_redis():
    global redis_client
    try:
        # decode_responses=False для отправки байт JSON
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
        redis_client.ping() # Проверка соединения
        console.log(f"[Redis] Successfully connected to {REDIS_HOST}:{REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        console.print(f"[bold red][Redis] Connection Error: {e}. Path data will not be sent.[/bold red]")
        redis_client = None # Убедимся, что клиент None при ошибке

# Гарантируем инициализацию Redis при запуске как main
if __name__ == "__main__" or multiprocessing.current_process().name == 'MainProcess':
    init_redis()

def train(model, dataloader, optimizer, criterion, epoch, total_epochs, device, scheduler: SequentialLR):  # Accept scheduler
    model.train()
    total_loss = 0
    total_cubes_used = 0
    total_correct_tokens = 0
    total_target_tokens = 0

    try:
        num_batches = len(dataloader)
        if num_batches == 0:
             console.print("[red]Warning: Dataloader returned length 0.[/red]")
    except TypeError:
         num_batches = -1
         console.print("[yellow]Warning: Dataloader does not support len(). Progress is indeterminate.[/yellow]")

    # Setup progress bar
    progress_columns = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        TextColumn("Epoch Acc: {task.fields[acc]:.2f}%"),
        TextColumn("Batch Acc: {task.fields[batch_acc]:.2f}%")
    ]
    epoch_progress = Progress(*progress_columns, console=console)
    task_id = epoch_progress.add_task(
        f"Epoch {epoch+1}/{total_epochs}",
        total=num_batches if num_batches > 0 else None,
        loss=float('inf'),
        acc=0.0,
        batch_acc=0.0, # Initialize batch_acc
        path_str="",
        cubes_used=0, num_total_cubes=model.num_cubes
    )

    processed_batches = 0
    # sliding-window latency aggregators
    window_ffn_time = window_mla_time = window_gate_time = 0.0
    window_ffn_calls = window_mla_calls = window_gate_calls = 0
    window_batches = 0

    scaler = GradScaler()
    limit_messages = [] # List to collect messages about max cubes limit
    # Training loop with Progress only
    with epoch_progress:
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with autocast(device_type=device.type):
                # Model forward pass - now returns 12 values including metrics
                logits, cubes_used_count, cubes_visited_history, aux_variance_loss, aux_load_balancing_loss, max_cubes_limit_reached, ffn_time, ffn_calls, mla_time, mla_calls, gate_time, gate_calls = model(input_ids)

                # --- START: Conditional Label Masking ---
                if config.TRAIN_ON_LAST_ANSWER_TOKEN_ONLY:
                    # Get EOS token ID
                    eos_id = tokenizer_utils.get_eos_token_id()
                    if eos_id is None:
                        console.print("[red]Error: EOS token ID not found, cannot apply TRAIN_ON_LAST_ANSWER_TOKEN_ONLY logic.[/red]")
                        # Fallback to using original labels if EOS is missing
                        current_labels = labels
                    else:
                        # Create a mask filled with ignore_index
                        masked_labels = torch.full_like(labels, config.IGNORE_INDEX)
                        # Iterate over each sequence in the batch
                        for b_idx in range(labels.size(0)):
                            # Find indices of all EOS tokens in the current sequence
                            eos_indices = (labels[b_idx] == eos_id).nonzero(as_tuple=True)[0]
                            # If at least one EOS token is found
                            if eos_indices.numel() > 0:
                                # Get the index of the *first* EOS token
                                first_eos_idx = eos_indices[0].item()
                                # If the EOS token is not the very first token in the sequence
                                if first_eos_idx > 0:
                                    # The target index is the one right before the first EOS
                                    target_idx = first_eos_idx - 1
                                    # Copy the original label value to the masked labels at the target position
                                    masked_labels[b_idx, target_idx] = labels[b_idx, target_idx]
                        current_labels = masked_labels
                else:
                    # If the flag is False, use the original labels
                    current_labels = labels
                # --- END: Conditional Label Masking ---

                # Calculate main loss using potentially masked labels
                main_loss = criterion(logits.view(-1, logits.size(-1)), current_labels.view(-1))
                loss = main_loss \
                     + config.AUX_LOSS_WEIGHT * aux_variance_loss \
                     + config.LOAD_BALANCING_LOSS_WEIGHT * aux_load_balancing_loss

            # Accumulate latency metrics for the sliding window
            window_ffn_time += ffn_time
            window_ffn_calls += ffn_calls
            window_mla_time += mla_time
            window_mla_calls += mla_calls
            window_gate_time += gate_time
            window_gate_calls += gate_calls
            window_batches += 1

            # Check the flag returned by the model
            if max_cubes_limit_reached:
                limit_messages.append(f"Batch {i+1}: Max cubes per path ({config.MAX_CUBES_PER_PATH}) reached. Path truncated due to cube-use limit.")

            if torch.isnan(loss) or torch.isinf(loss):
                 console.print(f"[red]Warning: NaN/Inf loss at batch {i}. Skipping update.[/red]")
                 continue

            # --- ИЗМЕНЕНО: Условное логирование и сброс счетчиков задержки ---
            if window_batches >= 10:
                # Логируем только если флаг включен
                if config.LOG_AVG_LATENCY:
                    avg_ffn_latency = window_ffn_time / window_ffn_calls if window_ffn_calls > 0 else 0.0
                    avg_mla_latency = window_mla_time / window_mla_calls if window_mla_calls > 0 else 0.0
                    avg_gate_latency = window_gate_time / window_gate_calls if window_gate_calls > 0 else 0.0
                    console.log(f"[Batch {i+1}] Avg Latency (last 10 batches): FFN={avg_ffn_latency:.6f}s, MLA={avg_mla_latency:.6f}s, Gate={avg_gate_latency:.6f}s")

                # Сбрасываем счетчики всегда, когда условие window_batches >= 10 выполняется
                window_ffn_time = window_mla_time = window_gate_time = 0.0
                window_ffn_calls = window_mla_calls = window_gate_calls = 0
                window_batches = 0
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # Backward pass and optimization step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            nan_inf_found = False
            zero_grad_found = True  # Предположим, что все нулевые, пока не найдем ненулевой
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                # Проверка на NaN/Inf
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan_inf_found = True
                    break  # Прерываем проверку, если нашли NaN/Inf
                # Проверка на ненулевой градиент
                if p.grad.norm().item() > 1e-9:
                    zero_grad_found = False

            # --- ИЗМЕНЕНО: Условное логирование проверки градиентов ---
            if config.LOG_GRADIENT_CHECKS:
                if nan_inf_found:
                    console.log(f"[bold red]Batch {i}: NaN/Inf градиенты обнаружены! Шаг оптимизатора будет пропущен.[/bold red]")
                elif zero_grad_found:
                    console.log(f"[yellow]Batch {i}: Все градиенты нулевые или близкие к нулю. Нет сигнала обучения?[/yellow]")
                else:
                    # Keep original gradient log
                    if i % 10 == 0:
                        console.log(f"[green]Batch {i}: Градиенты в норме.[/green]")
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # --- ИЗМЕНЕНО: Условное логирование и сброс счетчиков задержки (второе место) ---
            # Этот блок кажется дублирующим, но применим ту же логику
            if window_batches >= 10:
                # Логируем только если флаг включен
                if config.LOG_AVG_LATENCY:
                    avg_ffn_latency = window_ffn_time / window_ffn_calls if window_ffn_calls > 0 else 0.0
                    avg_mla_latency = window_mla_time / window_mla_calls if window_mla_calls > 0 else 0.0
                    avg_gate_latency = window_gate_time / window_gate_calls if window_gate_calls > 0 else 0.0
                    console.log(f"[Batch {i+1}] Avg Latency (last 10 batches): FFN={avg_ffn_latency:.6f}s, MLA={avg_mla_latency:.6f}s, Gate={avg_gate_latency:.6f}s")

                # Сбрасываем счетчики всегда, когда условие window_batches >= 10 выполняется
                window_ffn_time = window_mla_time = window_gate_time = 0.0
                window_ffn_calls = window_mla_calls = window_gate_calls = 0
                window_batches = 0
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # Условное выполнение шага оптимизатора
            if not nan_inf_found:
                scaler.step(optimizer)
                scaler.update()
                # --- НОВОЕ: Шаг планировщика после оптимизатора ---
                scheduler.step()
                # --- Конец нового шага планировщика ---
                if i % 50 == 0:  # Optional LR logging
                    current_lr = optimizer.param_groups[0]['lr']
                    console.log(f"[LR Step {i}] Current LR: {current_lr:.8f}")
            else:
                # Если были NaN/Inf, пропускаем шаг и обязательно обнуляем градиенты
                optimizer.zero_grad()
                scaler.update()  # Важно: всегда обновлять scaler, даже если шаг пропущен

            processed_batches += 1
            total_loss += loss.item()
            total_cubes_used += cubes_used_count

            # Calculate batch accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Use the potentially masked labels (current_labels) for accuracy calculation mask
                mask = current_labels != config.IGNORE_INDEX
                # Compare predictions with the potentially masked labels
                batch_correct = ((preds == current_labels) & mask).sum().item()
                batch_total = mask.sum().item()
                batch_acc = (batch_correct / batch_total) * 100 if batch_total > 0 else 0.0

            total_correct_tokens += batch_correct
            total_target_tokens += batch_total

            # Update averages and progress bar
            avg_loss = total_loss / processed_batches
            avg_acc = total_correct_tokens / total_target_tokens if total_target_tokens > 0 else 0.0

            parts = [f"[{idx}]" for idx in cubes_visited_history]
            path_str = " -> ".join(parts) if parts else "[No Path]"

            epoch_progress.update(
                task_id,
                advance=1,
                loss=avg_loss,
                acc=avg_acc * 100.0,
                batch_acc=batch_acc,
                path_str=path_str,
                cubes_used=cubes_used_count
            )

            # --- ИЗМЕНЕНО: Условное выполнение всего блока отладки батча ---
            if config.LOG_BATCH_DEBUG_INFO:
                # --- НОВАЯ ЛОГИКА: Отладка согласно ТЗ ---
                target_tokens_str = "N/A"
                predicted_tokens_str = "N/A"
                input_sequence_str = "N/A"
                import numpy as np # Добавим импорт numpy здесь

                try: # Внешний try для всего блока новой логики
                    # 1. Декодируем входную последовательность
                    input_ids_np = input_ids[0].cpu().numpy()
                    labels_np = labels[0].cpu().numpy()
                    # Убедимся, что preds на CPU перед конвертацией в numpy
                    preds_np = preds[0].cpu().numpy()

                    pad_token_id = tokenizer_utils.get_pad_token_id()
                    # Убираем паддинг из input_ids для декодирования
                    actual_input_ids = [token_id for token_id in input_ids_np if token_id != pad_token_id]
                    if actual_input_ids:
                        input_sequence_str = tokenizer_utils.decode(actual_input_ids)
                    else:
                        input_sequence_str = "[Empty Sequence after Padding Removal]"

                    # 2. Находим позиции и ID целевых токенов
                    target_indices = np.where(labels_np != config.IGNORE_INDEX)[0]
                    if target_indices.size > 0:
                        target_token_ids = labels_np[target_indices]
                        target_tokens_str = tokenizer_utils.decode(target_token_ids.tolist())

                        # 3. Получаем предсказанные ID для целевых позиций
                        predicted_token_ids = preds_np[target_indices]
                        predicted_tokens_str = tokenizer_utils.decode(predicted_token_ids.tolist())
                    else:
                        target_tokens_str = "[No Target Tokens]"
                        predicted_tokens_str = "[No Target Tokens]"

                    # 4. Вывод отладочной информации (каждые 20 батчей или при аномалиях)
                    if i % 20 == 0 or nan_inf_found or zero_grad_found:
                        console.print(f"--- Batch {i+1} Debug Info ---")
                        for msg in limit_messages: console.print(f"  [yellow]{msg}[/yellow]")
                        limit_messages = [] # Очищаем сообщения после вывода
                        console.print(f"  Input Sequence (Q_A<eos>): {repr(input_sequence_str)}")
                        console.print(f"  Target Tokens: {repr(target_tokens_str)}")
                        console.print(f"  Predicted Tokens: {repr(predicted_tokens_str)}")
                        path_display = path_str[:100] + ('...' if len(path_str) > 100 else '')
                        console.print(f"  Path: {path_display}")
                        console.print(f"  Avg Loss: {avg_loss:.4f}, Acc@Target: {avg_acc*100:.2f}%")
                        console.print(f"  Aux Var Loss: {aux_variance_loss.item():.4f}")
                        console.print(f"  Aux LB Loss: {aux_load_balancing_loss.item():.4f}")
                        console.print("-" * 25)

                except Exception as e_debug_new:
                    console.print(f"[Warning] Error during new debug info generation: {e_debug_new}")
                    import traceback
                    console.print(traceback.format_exc())
            # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

            # --- Отправка данных о пути в Redis ---
            if redis_client: # Отправляем только если есть соединение
                try:
                    # Collect path(s) data: send full path for each sample in the batch
                    path_data = list(cubes_visited_history)
                    batch_size = input_ids.size(0)
                    paths = [path_data for _ in range(batch_size)]

                    # Формируем данные для отправки с метриками
                    message_data = {
                        "epoch": epoch,
                        "batch_index": i,
                        "paths": paths,
                        "avg_loss": avg_loss,
                        "acc_target": avg_acc,
                        "aux_var_loss": aux_variance_loss.item(),
                        "aux_lb_loss": aux_load_balancing_loss.item(),
                    }

                    # Сериализуем в JSON (байты)
                    message_json_bytes = json.dumps(message_data).encode('utf-8')

                    # Отправляем в Redis (используем RPUSH для добавления в конец списка)
                    redis_client.rpush(REDIS_PATH_LIST_KEY, message_json_bytes)

                except redis.exceptions.RedisError as e:
                    console.print(f"[bold red][Redis] Error sending path data: {e}[/bold red]")
                    # Возможно, стоит переподключиться или отключить отправку на время
                    # redis_client = None # Отключаем отправку при ошибке Redis
                except Exception as e:
                     console.print(f"[bold yellow][Redis] Error preparing path data: {e}[/bold yellow]")
            # --- Конец отправки данных в Redis ---


        if num_batches<=0 and processed_batches>0:
             epoch_progress.update(task_id,total=processed_batches)

    # --- Отправка Маркера Конца Эпохи в Redis ---
    if redis_client:
        try:
            end_epoch_message = {"epoch": epoch, "status": "epoch_end"}
            message_json_bytes = json.dumps(end_epoch_message).encode('utf-8')
            redis_client.rpush(REDIS_PATH_LIST_KEY, message_json_bytes)
            console.log(f"[Redis] Sent epoch {epoch} end marker.")
        except redis.exceptions.RedisError as e:
            console.print(f"[bold red][Redis] Error sending epoch end marker: {e}[/bold red]")
        except Exception as e:
             console.print(f"[bold yellow][Redis] Error preparing epoch end marker: {e}[/bold yellow]")
    # --- Конец отправки Маркера Конца Эпохи ---


    avg_epoch_loss = total_loss/processed_batches if processed_batches else 0.0
    avg_epoch_acc = total_correct_tokens/total_target_tokens if total_target_tokens else 0.0
    avg_epoch_cubes_used = total_cubes_used/processed_batches if processed_batches else 0.0
    console.print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Avg Acc@Target: {avg_epoch_acc*100:.2f}%, Avg Cubes Used: {avg_epoch_cubes_used:.2f}")
    return avg_epoch_loss

# --- END OF FILE train.py ---
