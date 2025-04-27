# --- START OF FILE train.py ---

import torch
import torch.nn as nn
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

def train(model, dataloader, optimizer, criterion, epoch, total_epochs, device):
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

    # Save visualization data only on the first epoch
    if epoch == 0:
        viz_pos_path = config.VIZ_POSITIONS_FILE
        if viz_pos_path:
            if not os.path.exists(viz_pos_path):
                try:
                    cube_positions_list = model.cube_positions.cpu().numpy().tolist()
                    os.makedirs(os.path.dirname(viz_pos_path), exist_ok=True)
                    with open(viz_pos_path, 'w') as f:
                        json.dump(cube_positions_list, f)
                    console.log(f"Saved cube positions to {viz_pos_path}")
                except Exception as e:
                     console.print(f"[yellow]Warning: Could not save cube positions: {e}[/yellow]")
        else:
            console.print("[yellow]Warning: config.VIZ_POSITIONS_FILE is empty.[/yellow]")

        viz_start_path = config.VIZ_START_CANDIDATES_FILE
        if viz_start_path:
            try:
                start_candidates_list = model.entry_exit_list
                os.makedirs(os.path.dirname(viz_start_path), exist_ok=True)
                with open(viz_start_path, 'w') as f:
                    json.dump(start_candidates_list, f)
                console.log(f"Saved start candidates to {viz_start_path}")
            except AttributeError:
                 console.print(f"[yellow]Warning: Model missing 'entry_exit_list'. Skipping start candidates save.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save start candidates: {e}[/yellow]")
        else:
            console.print("[yellow]Warning: config.VIZ_START_CANDIDATES_FILE is empty.[/yellow]")

    # Setup progress bar
    progress_columns = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(),
    ]
    epoch_progress = Progress(*progress_columns, console=console)
    task_id = epoch_progress.add_task(
        f"Epoch {epoch+1}/{total_epochs}",
        total=num_batches if num_batches > 0 else None,
        loss=float('inf'),
        acc=0.0,
        path_str="",
        cubes_used=0, num_total_cubes=model.num_cubes
    )

    processed_batches = 0
    viz_paths_path = config.VIZ_PATHS_FILE
    mode = 'w' if epoch == 0 else 'a'
    paths_file = None

    # Setup path visualization file
    if viz_paths_path:
        try:
            os.makedirs(os.path.dirname(viz_paths_path), exist_ok=True)
            paths_file = open(viz_paths_path, mode)
            action = 'Truncated/Created' if epoch == 0 else 'Opened'
            console.log(f"{action} paths file: {viz_paths_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not open paths file: {e}. Skipping path saving.[/yellow]")
    else:
        console.print("[yellow]Warning: config.VIZ_PATHS_FILE is empty. Skipping path saving.[/yellow]")

    scaler = GradScaler()
    additional_info_text = Text("")
    renderable = Group(epoch_progress, additional_info_text)

    # Training loop with Live display
    with Live(renderable, refresh_per_second=10):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with autocast(device_type=device.type):
                logits, cubes_used_count, cubes_visited_history, aux_variance_loss, aux_load_balancing_loss = model(input_ids)
                main_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = main_loss \
                     + config.AUX_LOSS_WEIGHT * aux_variance_loss \
                     + config.LOAD_BALANCING_LOSS_WEIGHT * aux_load_balancing_loss

            if torch.isnan(loss) or torch.isinf(loss):
                 console.print(f"[red]Warning: NaN/Inf loss at batch {i}. Skipping update.[/red]")
                 continue

            # Backward pass and optimization step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            processed_batches += 1
            total_loss += loss.item()
            total_cubes_used += cubes_used_count

            # Calculate accuracy at the single target token position
            batch_correct = 0
            batch_total = 0
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = labels != config.IGNORE_INDEX
                batch_correct = ((preds == labels) & mask).sum().item()
                batch_total = mask.sum().item()

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
                path_str=path_str,
                cubes_used=cubes_used_count
            )

            # Prepare dynamic logging text (for first item in batch)
            try:
                input_ids_list = input_ids[0].cpu().tolist()
                labels_list = labels[0].cpu().tolist()
                tokenizer = dataloader.dataset.tokenizer
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                ignore_index = config.IGNORE_INDEX

                # Decode display input
                disp_ids = [t for t in input_ids_list if t != pad_token_id]
                input_text_display = tokenizer.decode(disp_ids, skip_special_tokens=False).strip()

                # Decode prediction task
                valid_idxs = [j for j,v in enumerate(labels_list) if v!=ignore_index]
                if valid_idxs:
                    pred_pos = valid_idxs[0]
                    pre_ids = [t for t in input_ids_list[:pred_pos+1] if t!=pad_token_id]
                    task_context = tokenizer.decode(pre_ids, skip_special_tokens=False).strip()
                    target_id = labels_list[pred_pos]
                    target_str = tokenizer.decode([target_id], skip_special_tokens=False)
                else:
                    task_context = input_text_display
                    target_str = ""

                # Reconstruct full question before '='
                eq = tokenizer.encode("=", add_special_tokens=False)[0]
                if eq in input_ids_list:
                    q_tokens = input_ids_list[:input_ids_list.index(eq)]
                    question = tokenizer.decode(q_tokens, skip_special_tokens=False).strip()
                else:
                    question = input_text_display
            except:
                input_text_display, task_context, target_str, question = "", "", "", ""

            # --- НАЧАЛО: Код для извлечения Q/A из batch ---
            first_q = "N/A"
            first_a = "N/A"
            try:
                if isinstance(batch, dict) and \
                   'original_q' in batch and batch['original_q'] and \
                   'original_a' in batch and batch['original_a']:

                    q_data = batch['original_q'][0]
                    a_data = batch['original_a'][0]

                    # Декодируем, если это байты (из HDF5 кэша)
                    if isinstance(q_data, bytes):
                        first_q = q_data.decode('utf-8', errors='ignore')
                    elif isinstance(q_data, str):
                        first_q = q_data
                    else:
                        first_q = str(q_data)

                    if isinstance(a_data, bytes):
                        first_a = a_data.decode('utf-8', errors='ignore')
                    elif isinstance(a_data, str):
                        first_a = a_data
                    else:
                        first_a = str(a_data)

                    # Ограничим длину для вывода
                    first_q = first_q[:80] + "..." if len(first_q) > 80 else first_q
                    first_a = first_a[:80] + "..." if len(first_a) > 80 else first_a
            except Exception as e:
                pass # Оставляем N/A
            # --- КОНЕЦ: Код для извлечения Q/A из batch ---

            # --- НАЧАЛО: Формирование Predicted строки ---
            predicted_str_new = "N/A"
            try:
                input_ids_list = input_ids[0].cpu().tolist()
                labels_list = labels[0].cpu().tolist()
                # Extract question token IDs
                try:
                    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
                    newline_idx = input_ids_list.index(newline_token_id)
                    question_token_ids = input_ids_list[:newline_idx+1]
                except ValueError:
                    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                    if pad_token_id in input_ids_list:
                        question_token_ids = input_ids_list[:input_ids_list.index(pad_token_id)]
                    else:
                        question_token_ids = input_ids_list
                # True first answer token
                true_first_a_token_id = config.IGNORE_INDEX
                if 'newline_idx' in locals() and newline_idx + 1 < len(labels_list):
                    val = labels_list[newline_idx+1]
                    if val != config.IGNORE_INDEX:
                        true_first_a_token_id = val
                # Secondary forward pass for next token
                predicted_first_a_token_id = -1
                if question_token_ids:
                    with torch.no_grad():
                        q_tensor = torch.tensor([question_token_ids], device=device)
                        logits_q, *_ = model(q_tensor)
                        next_logits = logits_q[:, -1, :]
                        predicted_first_a_token_id = torch.argmax(next_logits, dim=-1).item()
                # Decode tokens
                pred_token_str = repr(tokenizer.decode([predicted_first_a_token_id])) if predicted_first_a_token_id != -1 else "N/A"
                true_token_str = repr(tokenizer.decode([true_first_a_token_id])) if true_first_a_token_id != config.IGNORE_INDEX else "N/A"
                correctness_mark = ""
                if predicted_first_a_token_id == true_first_a_token_id and true_first_a_token_id != config.IGNORE_INDEX:
                    correctness_mark = " [CORRECT]"
                elif predicted_first_a_token_id != true_first_a_token_id and true_first_a_token_id != config.IGNORE_INDEX and predicted_first_a_token_id != -1:
                    correctness_mark = f" [WRONG (True: {true_token_str})]"
                elif predicted_first_a_token_id == -1:
                    correctness_mark = " [PRED FAILED]"
                elif true_first_a_token_id == config.IGNORE_INDEX:
                    correctness_mark = " [TRUE N/A]"
                predicted_str_new = f"Predict#1: {first_q} -> {pred_token_str}{correctness_mark}"
            except Exception as e:
                predicted_str_new = f"Predict#1: N/A (error: {e})"
            # --- КОНЕЦ: Формирование Predicted строки ---

            # Update dynamic display
            additional_info_text.plain = "\n".join([
                f"Avg Loss: {avg_loss:.4f}, Acc@Target: {avg_acc*100:.2f}%",
                f"Aux Var Loss: {aux_variance_loss.item():.4f}",
                f"Aux LB Loss: {aux_load_balancing_loss.item():.4f}",
                f"Path: {path_str}",
                f"Q (Batch[0]): {first_q}",
                f"A (Batch[0]): {first_a}",
                predicted_str_new,
            ])

            # Save paths data (original file logic)
            # This logic is now replaced by sending to Redis
            # if paths_file:
            #     try:
            #         hist = list(cubes_visited_history)
            #         transitions = [[hist[j], hist[j+1]] for j in range(len(hist)-1)]
            #         paths_file.write(json.dumps({"epoch":epoch,"batch":i,"path":hist,"trans":transitions})+"\n")
            #         if i%100==0: paths_file.flush()
            #     except:
            #         paths_file=None

            # --- Отправка данных о пути в Redis ---
            if redis_client: # Отправляем только если есть соединение
                try:
                    # Collect path(s) data: send full path for each sample in the batch
                    path_data = list(cubes_visited_history)
                    batch_size = input_ids.size(0)
                    paths = [path_data for _ in range(batch_size)]

                    # Формируем данные для отправки
                    message_data = {
                        "epoch": epoch,
                        "batch_index": i,
                        "paths": paths,
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

    # Close the original paths file if it was opened (though we are now using Redis)
    if paths_file:
        try: paths_file.close()
        except: pass

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