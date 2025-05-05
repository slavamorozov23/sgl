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

                main_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
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
                mask = labels != config.IGNORE_INDEX
                batch_correct = ((preds == labels) & mask).sum().item()
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
                # Prepare dynamic logging text (for first item in batch)
                try:
                    input_ids_list = input_ids[0].cpu().tolist()
                    labels_list = labels[0].cpu().tolist()
                    # tokenizer = dataloader.dataset.tokenizer # Больше не получаем токенизатор так
                    # Используем tokenizer_utils напрямую
                    pad_token_id = tokenizer_utils.get_pad_token_id()
                    eos_token_id = tokenizer_utils.get_eos_token_id()
                    effective_pad_id = pad_token_id if pad_token_id is not None else eos_token_id # Используем EOS как запасной
                    ignore_index = config.IGNORE_INDEX

                    # Decode display input
                    disp_ids = [t for t in input_ids_list if t != effective_pad_id]
                    # input_text_display = tokenizer.decode(disp_ids, skip_special_tokens=False).strip()
                    input_text_display = tokenizer_utils.decode(disp_ids).strip() # skip_special_tokens=False по умолчанию

                    # Decode prediction task
                    valid_idxs = [j for j,v in enumerate(labels_list) if v!=ignore_index]
                    if valid_idxs:
                        pred_pos = valid_idxs[0]
                        pre_ids = [t for t in input_ids_list[:pred_pos+1] if t!=effective_pad_id]
                        # task_context = tokenizer.decode(pre_ids, skip_special_tokens=False).strip()
                        task_context = tokenizer_utils.decode(pre_ids).strip()
                        target_id = labels_list[pred_pos]
                        # target_str = tokenizer.decode([target_id], skip_special_tokens=False)
                        target_str = tokenizer_utils.decode([target_id])
                    else:
                        task_context = input_text_display
                        target_str = ""

                    # Reconstruct full question before '='
                    # eq = tokenizer.encode("=", add_special_tokens=False)[0]
                    eq = tokenizer_utils.encode("=")[0] # add_special_tokens=False по умолчанию
                    if eq in input_ids_list:
                        q_tokens = input_ids_list[:input_ids_list.index(eq)]
                        # question = tokenizer.decode(q_tokens, skip_special_tokens=False).strip()
                        question = tokenizer_utils.decode(q_tokens).strip()
                    else:
                        question = input_text_display
                except Exception as log_err:
                    input_text_display, task_context, target_str, question = f"Err:{log_err}", "", "", ""

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
                    # --- ИЗМЕНЕННАЯ ЛОГИКА ---
                    # Получаем оригинальный вопрос из батча (уже извлечен выше как first_q)
                    # Токенизируем оригинальный вопрос, чтобы получить правильные ID для предсказания
                    question_token_ids = tokenizer_utils.encode(first_q) # add_special_tokens=False по умолчанию

                    # Добавляем токен '\n' (или его ID), так как модель ожидает его в конце вопроса
                    # (согласно логике process_batch_for_math_mp, где s = f"{q_clean}\n{a_clean}")
                    # newline_token_id = tokenizer_utils.encode('\n')[0] # Получаем ID новой строки
                    # question_token_ids.append(newline_token_id) # Добавляем его

                    # Найти первый целевой токен (не IGNORE_INDEX) из labels
                    labels_list = labels[0].cpu().tolist()
                    true_first_a_token_id = config.IGNORE_INDEX
                    label_start_index = len(question_token_ids) # Индекс, с которого начинаются метки ответа
                    if label_start_index < len(labels_list):
                        for j in range(label_start_index, len(labels_list)):
                             if labels_list[j] != config.IGNORE_INDEX:
                                  true_first_a_token_id = labels_list[j]
                                  break

                    # Secondary forward pass for next token prediction
                    predicted_first_a_token_id = -1
                    if question_token_ids: # Убедимся, что список токенов не пуст
                        with torch.no_grad():
                            # Создаем тензор из ID вопроса
                            q_tensor = torch.tensor([question_token_ids], device=device)
                            # Получаем логиты от модели
                            logits_q, *_ = model(q_tensor)
                            # Берем логиты для последнего токена в последовательности вопроса
                            next_logits = logits_q[:, -1, :]
                            # --- НОВОЕ: Получаем вероятности ---
                            # Используем полное имя вместо псевдонима F
                            probabilities = torch.nn.functional.softmax(next_logits, dim=-1)
                            # Находим ID токена с максимальным логитом (предсказанный)
                            predicted_first_a_token_id = torch.argmax(probabilities, dim=-1).item()
                            # --- НОВОЕ: Получаем уверенность (вероятность) предсказанного токена ---
                            confidence = probabilities[0, predicted_first_a_token_id].item() * 100
                    else:
                         print(f"[DIAG Worker {current_process().pid}] Warning: question_token_ids is empty for debug prediction.")
                         confidence = 0.0 # Устанавливаем уверенность в 0, если не было предсказания


                    # Decode predicted token using tokenizer_utils
                    pred_token_str = repr(tokenizer_utils.decode([predicted_first_a_token_id])) if predicted_first_a_token_id != -1 else "N/A"
                    # true_token_str не нужен для нового формата вывода

                    # Determine correctness mark with confidence
                    correctness_mark = ""
                    # Используем :.2f для форматирования уверенности с двумя знаками после запятой
                    if predicted_first_a_token_id != -1 and true_first_a_token_id != config.IGNORE_INDEX:
                        if predicted_first_a_token_id == true_first_a_token_id:
                            correctness_mark = f" [Correct, conf {confidence:.2f}%]"
                        else:
                            correctness_mark = f" [Wrong, conf {confidence:.2f}%]"
                    elif predicted_first_a_token_id == -1:
                        correctness_mark = " [PRED FAILED]"
                    elif true_first_a_token_id == config.IGNORE_INDEX:
                         # Если истинный токен N/A (Stage 1), просто показываем предсказание и уверенность
                         correctness_mark = f" [True N/A, conf {confidence:.2f}%]"

                    # Format the final debug string
                    predicted_str_new = f"Predict#1: {first_q} -> {pred_token_str}{correctness_mark}"

                except Exception as e:
                    # Логируем ошибку с трассировкой для лучшей диагностики
                    import traceback
                    print(f"[ERROR] Failed to generate debug prediction string: {e}")
                    print(traceback.format_exc())
                    predicted_str_new = f"Predict#1: N/A (error: {e})"
                # --- КОНЕЦ: Формирование Predicted строки ---

                # Debug printout (every 20 batches or if anomaly)
                if i % 20 == 0 or nan_inf_found or zero_grad_found:
                    console.print(f"--- Batch {i+1} Debug Info ---")
                    # Print collected limit messages
                    for msg in limit_messages:
                        console.print(f"  [yellow]{msg}[/yellow]")
                    limit_messages = [] # Clear the list after printing
                    console.print(f"  Avg Loss: {avg_loss:.4f}, Acc@Target: {avg_acc*100:.2f}%")
                    console.print(f"  Aux Var Loss: {aux_variance_loss.item():.4f}")
                    console.print(f"  Aux LB Loss: {aux_load_balancing_loss.item():.4f}")
                    path_display = path_str[:100] + ('...' if len(path_str) > 100 else '')
                    console.print(f"  Path: {path_display}")
                    console.print(f"  Q (Batch[0]): {first_q}")
                    console.print(f"  A (Batch[0]): {first_a}")
                    console.print(f"  {predicted_str_new}")
                    # --- Новая строка: уверенность в правильном токене ---
                    try:
                        if 'pred_pos' in locals() and 'target_id' in locals():
                            # logits: [batch, seq, vocab], берем [0, pred_pos, :]
                            import torch.nn.functional as F
                            true_logits = logits[0, pred_pos, :]
                            probs = F.softmax(true_logits, dim=-1)
                            prob_true = probs[target_id].item()
                            console.print(f"  Model confidence in TRUE token: {prob_true*100:.2f}%")
                    except Exception as e:
                        console.print(f"  [yellow]Could not compute confidence: {e}[/yellow]")
                    console.print("-" * 25)
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
