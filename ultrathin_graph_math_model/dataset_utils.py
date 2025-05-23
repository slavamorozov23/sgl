# --- START OF FILE dataset_utils.py ---

import numpy as np
import traceback # Import traceback for detailed error logging
import random
from text_utils import clean_text
from multiprocessing import current_process
import tokenizer_utils # Импортируем наш модуль

# tokenizer_mp = None # Глобальный токенизатор больше не нужен
max_seq_len_mp = None
ignore_index_mp = None

# def init_math_worker(tok, seq_len, ignore_idx): # Убираем tok
def init_math_worker(unused_tok_arg, seq_len, ignore_idx): # Оставляем аргумент для совместимости вызова, но не используем
    """Initializes global variables for ProcessPoolExecutor workers."""
    # global tokenizer_mp, max_seq_len_mp, ignore_index_mp # Убираем tokenizer_mp
    global max_seq_len_mp, ignore_index_mp
    # tokenizer_mp = tok # Не устанавливаем глобальный токенизатор
    max_seq_len_mp = seq_len
    ignore_index_mp = ignore_idx
    # Можно добавить вызов get_tokenizer() здесь, чтобы убедиться, что он инициализирован в воркере
    pid = current_process().pid # Получаем PID один раз
    try:
        # --- ДОБАВИТЬ ЛОГ ПЕРЕД ИНИЦИАЛИЗАЦИЕЙ ---
        # print(f"[DIAG Worker {pid}] init_math_worker started. Initializing tokenizer...") # Используем print для надежности в воркерах
        tokenizer_utils.get_tokenizer()
        # --- ДОБАВИТЬ ЛОГ ПОСЛЕ ИНИЦИАЛИЗАЦИИ ---
        # print(f"[DIAG Worker {pid}] Tokenizer initialized successfully in init_math_worker.") # Используем print
    except Exception as e:
        print(f"[CRITICAL Worker {pid}] Failed to initialize tokenizer in worker: {e}")

def process_batch_for_math_mp(batch, show_samples=False): # Removed stage1_mode parameter
    """
    Processes a batch of questions and answers for the math task.
    Concatenates question and answer with '_', adds EOS, and masks question tokens in labels.
    """
    # global tokenizer_mp, max_seq_len_mp, ignore_index_mp # Убираем tokenizer_mp
    global max_seq_len_mp, ignore_index_mp # Оставляем нужные глобальные переменные
    pid = current_process().pid # Получаем PID один раз
    # --- УБРАН ЛОГ В НАЧАЛЕ ФУНКЦИИ ---
    # print(f"[DIAG Worker {pid}] process_batch_for_math_mp started.")
    out = {'input_ids': [], 'labels': [], 'original_q': [], 'original_a': []}
    debug_messages = [] # Keep this for warnings, add direct prints for errors


    # Получаем ID спец токенов один раз в начале
    # --- УБРАН ЛОГ ПЕРЕД ПОЛУЧЕНИЕМ ID ---
    # print(f"[DIAG Worker {pid}] Getting token IDs...")
    eos_id = tokenizer_utils.get_eos_token_id()
    pad_id = tokenizer_utils.get_pad_token_id()
    # --- УБРАН ЛОГ ПОСЛЕ ПОЛУЧЕНИЯ ID ---
    # print(f"[DIAG Worker {pid}] Token IDs received (EOS: {eos_id}, PAD: {pad_id}).")
    if pad_id is None: # Если pad_id все еще None после инициализации в tokenizer_utils
        pad_id = eos_id # Используем EOS как запасной вариант
        if pad_id is None:
             print(f"[CRITICAL process_batch Worker {pid}] Tokenizer has neither PAD nor EOS token ID. Padding will fail.")
             # Установим pad_id в 0 как запасной вариант, хотя это может быть не идеально
             pad_id = 0
    # newline_id больше не нужен, будем определять индекс по длине вопроса

    qs, ans = batch['question'], batch['answer']

    # Initial display of 3 random cleaned examples if requested
    if show_samples and len(qs) > 0:
        sample_indices = random.sample(range(len(qs)), min(3, len(qs)))
        print("Sample cleaned examples:")
        for idx in sample_indices:
            q_clean = clean_text(qs[idx])
            a_clean = clean_text(ans[idx])
            print(f"Q: {q_clean}")
            print(f"A: {a_clean}")
            print("---")

    for i, (q, a) in enumerate(zip(qs, ans)):
        try:
            # --- ДОБАВИТЬ ЛОГ ПЕРЕД ОЧИСТКОЙ ---
            # print(f"[DIAG Worker {pid} Item {i}] Cleaning text...")
            # print(f"[DEBUG process_batch {i}] Raw Q: {repr(q)}") # Optional: Log raw input
            # print(f"[DEBUG process_batch {i}] Raw A: {repr(a)}") # Optional: Log raw input

            # Clean question and answer using shared function
            q_clean = clean_text(q)
            a_clean = clean_text(a)
            # print(f"[DEBUG process_batch {i}] Clean Q: {repr(q_clean)}") # Optional: Log cleaned input
            # print(f"[DEBUG process_batch {i}] Clean A: {repr(a_clean)}") # Optional: Log cleaned input

            # Concatenate with newline and add EOS to answer
            # --- ИЗМЕНЕНО: Используем '_' как разделитель и добавляем EOS к ответу ---
            eos_token_str = tokenizer_utils.get_tokenizer().eos_token or "" # Получаем строку EOS
            s = f"{q_clean}_{a_clean}{eos_token_str}" # Добавляем EOS к ответу перед кодированием
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            # print(f"[DEBUG process_batch {i}] Combined S: {repr(s)}") # Optional: Log combined string

            # Tokenize using tokenizer_utils and get offset mapping
            prepared_s = tokenizer_utils.prepare_text_for_encoding(s) # Подготовка текста
            # Используем __call__ для получения offset_mapping
            encoding = tokenizer_utils.get_tokenizer()(
                prepared_s,
                return_offsets_mapping=True,
                add_special_tokens=False # Убедимся, что спец. токены не добавляются здесь
            )
            ids = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']
            # print(f"[DEBUG process_batch {i} Worker {current_process().pid}] Initial IDs: {ids}") # Optional: Log token IDs
            # print(f"[DEBUG process_batch {i} Worker {current_process().pid}] Offset Mapping: {offset_mapping}") # Optional: Log offsets

            # # Add EOS token if missing and tokenizer has one # --- УДАЛЕНО: EOS теперь добавляется к ответу выше ---
            # if eos_id is not None and (not ids or ids[-1] != eos_id):
            #     ids.append(eos_id)
            #     # print(f"[DEBUG process_batch {i}] IDs after EOS: {ids}") # Optional: Log token IDs after EOS

            # --- НАЧАЛО НОВОЙ ЛОГИКИ input_ids/labels ---
            original_len = len(ids)
            if original_len < 2:
                debug_messages.append(f"Warning: Skipping short sequence (len {original_len}): {s[:50]}...")
                print(f"[DEBUG process_batch {i} Worker {pid}] Skipping: original_len < 2 (len={original_len})") # ADDED DEBUG
                continue

            # --- НОВЫЙ МЕТОД: Определение индекса разделения с использованием offset_mapping ---
            separator_token = '_'
            answer_token_start_index_in_ids = -1 # Индекс первого токена ответа

            try:
                # Находим позицию разделителя в исходной строке 's'
                # Ищем ПОСЛЕДНИЙ разделитель, на случай если он есть в вопросе
                separator_char_index = s.rfind(separator_token)

                if separator_char_index == -1:
                    debug_messages.append(f"Warning: Separator '{separator_token}' not found in string '{s[:100]}...'. Skipping.")
                    print(f"[WARN process_batch {i} Worker {pid}] Skipping: Separator '{separator_token}' not found in string.")
                    continue

                # Ищем первый токен, который начинается ПОСЛЕ индекса разделителя
                found_split_token = False
                for token_idx, (start_offset, end_offset) in enumerate(offset_mapping):
                    # Ищем токен, который начинается на или после позиции СЛЕДУЮЩЕЙ за разделителем
                    if start_offset > separator_char_index:
                        answer_token_start_index_in_ids = token_idx
                        found_split_token = True
                        # print(f"[DEBUG process_batch {i} Worker {pid}] Found split point: char_idx={separator_char_index}, token_idx={token_idx}, offset=({start_offset},{end_offset})") # Optional Debug
                        break

                if not found_split_token:
                    # Это может произойти, если разделитель был последним символом или токены после него отсутствуют/обрезаны
                    debug_messages.append(f"Warning: Could not find token starting after separator index {separator_char_index} in '{s[:100]}...'. Skipping.")
                    print(f"[WARN process_batch {i} Worker {pid}] Skipping: Could not find token starting after separator index {separator_char_index}.")
                    continue

                # Дополнительная проверка: не выходит ли индекс за пределы? (маловероятно, но на всякий случай)
                if answer_token_start_index_in_ids >= len(ids):
                     debug_messages.append(f"Warning: Calculated answer_start_idx ({answer_token_start_index_in_ids}) >= len(ids) ({len(ids)}) using offset mapping. Skipping.")
                     print(f"[WARN process_batch {i} Worker {pid}] Skipping: Calculated answer_start_idx {answer_token_start_index_in_ids} >= len(ids) {len(ids)} using offset mapping.")
                     continue

            except Exception as e_offset:
                 debug_messages.append(f"Error during offset mapping processing: {e_offset}")
                 print(f"[ERROR process_batch {i} Worker {pid}] Skipping: Error during offset mapping processing: {e_offset}.")
                 continue
            # --- Конец нового метода определения индекса ---

            # Truncate or pad ids
            ids_truncated = ids[:max_seq_len_mp] if original_len > max_seq_len_mp else ids

            # Build input_ids
            input_tokens = ids_truncated[:-1] # All tokens except the last one (usually EOS)
            input_ids_final = np.full(max_seq_len_mp, pad_id, dtype=np.int64)
            len_input = min(len(input_tokens), max_seq_len_mp)
            input_ids_final[:len_input] = input_tokens[:len_input]

            # --- Logic for labels (Only Stage 2 remains) ---
            labels_final = np.full(max_seq_len_mp, ignore_index_mp, dtype=np.int64)
            # answer_token_start_index_in_ids уже рассчитан выше
            # Индекс в labels, куда записываем первый токен ответа = answer_token_start_index_in_ids - 1 (т.к. labels сдвинуты на 1)
            label_start_pos_for_first_answer_token = answer_token_start_index_in_ids - 1

            # Итерируемся по токенам ПОСЛЕ рассчитанного индекса вопроса и разделителя в ids_truncated
            for j, tok_id in enumerate(ids_truncated[answer_token_start_index_in_ids:]):
                 # Новая логика: позиция метки = (индекс начала ответа в ids - 1) + смещение j
                 label_pos = label_start_pos_for_first_answer_token + j
                 if label_pos < max_seq_len_mp:
                     # Устанавливаем ID текущего токена ответа (tok_id) как цель для предыдущей позиции (label_pos)
                     labels_final[label_pos] = tok_id
                 else:
                     # Прекращаем, если вышли за пределы max_seq_len_mp
                     if len(debug_messages) < 10:
                          debug_messages.append(
                               f"Warning: Answer tokens truncated for labels (max_seq_len: {max_seq_len_mp})."
                          )
                     break

            # Validate final lengths
            if len(input_ids_final) != max_seq_len_mp or len(labels_final) != max_seq_len_mp:
                if len(debug_messages) < 10:
                    debug_messages.append(f"Warning: Length mismatch after processing. Skipping example. Input: {len(input_ids_final)}, Label: {len(labels_final)}")
                print(f"[WARN process_batch {i} Worker {pid}] Skipping: Length mismatch. Input: {len(input_ids_final)}, Label: {len(labels_final)} vs {max_seq_len_mp}.") # DEBUG Already added
                continue

            # Append to output
            out['input_ids'].append(input_ids_final)
            out['labels'].append(labels_final)
            out['original_q'].append(q_clean) # Store cleaned versions
            out['original_a'].append(a_clean)
            # --- КОНЕЦ НОВОЙ ЛОГИКИ input_ids/labels ---

        except Exception as e:
            print(f"[ERROR process_batch {i}] Failed processing item. Error: {e}")
            print(f"  Raw Q: {repr(q)}")
            print(f"  Raw A: {repr(a)}")
            print(traceback.format_exc()) # Print full traceback for the error
            debug_messages.append(f"ERROR processing item {i}: {e}")
            print(f"[ERROR process_batch {i} Worker {pid}] Skipping: Exception during processing: {e}") # DEBUG Already added
            continue # Skip this item and continue with the next

    return out, debug_messages

# --- END OF FILE dataset_utils.py ---
