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
        print(f"[DIAG Worker {pid}] init_math_worker started. Initializing tokenizer...") # Используем print для надежности в воркерах
        tokenizer_utils.get_tokenizer()
        # --- ДОБАВИТЬ ЛОГ ПОСЛЕ ИНИЦИАЛИЗАЦИИ ---
        print(f"[DIAG Worker {pid}] Tokenizer initialized successfully in init_math_worker.") # Используем print
    except Exception as e:
        print(f"[CRITICAL Worker {pid}] Failed to initialize tokenizer in worker: {e}")

def process_batch_for_math_mp(batch, show_samples=False, stage1_mode=False):
    """
    Processes a batch of questions and answers for the math task.
    Now concatenates question and answer with '\\n', logs inputs, and masks the last token.
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
            # print(f"[DEBUG process_batch {i}] Raw Q: {repr(q)}") # Optional: Log raw input
            # print(f"[DEBUG process_batch {i}] Raw A: {repr(a)}") # Optional: Log raw input

            # Clean question and answer using shared function
            q_clean = clean_text(q)
            a_clean = clean_text(a)
            # print(f"[DEBUG process_batch {i}] Clean Q: {repr(q_clean)}") # Optional: Log cleaned input
            # print(f"[DEBUG process_batch {i}] Clean A: {repr(a_clean)}") # Optional: Log cleaned input

            # Concatenate with newline
            s = f"{q_clean}\n{a_clean}"
            # print(f"[DEBUG process_batch {i}] Combined S: {repr(s)}") # Optional: Log combined string

            # Tokenize using tokenizer_utils
            # ids = tokenizer_mp.encode(s, add_special_tokens=False)
            prepared_s = tokenizer_utils.prepare_text_for_encoding(s) # Подготовка текста
            ids = tokenizer_utils.encode(prepared_s) # add_special_tokens=False по умолчанию
            # print(f"[DEBUG process_batch {i} Worker {current_process().pid}] Initial IDs: {ids}") # Optional: Log token IDs

            # Add EOS token if missing and tokenizer has one
            if eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)
                # print(f"[DEBUG process_batch {i}] IDs after EOS: {ids}") # Optional: Log token IDs after EOS

            # --- НАЧАЛО НОВОЙ ЛОГИКИ input_ids/labels ---
            original_len = len(ids)
            if original_len < 2:
                debug_messages.append(f"Warning: Skipping short sequence (len {original_len}): {s[:50]}...")
                continue

            # --- Определение индекса разделения вопроса и ответа ---
            # Кодируем только вопрос, чтобы узнать его длину в токенах
            prepared_q_clean = tokenizer_utils.prepare_text_for_encoding(q_clean) # Подготовка текста вопроса
            q_ids = tokenizer_utils.encode(prepared_q_clean)
            newline_idx = len(q_ids) # Индекс первого токена ответа = длина вопроса

            # Проверка: убедимся, что найденный индекс не выходит за пределы ids
            # (Это может случиться, если токенизация q_clean отличается от начала токенизации s)
            # Такая проверка не идеальна, но может помочь отловить расхождения.
            # Более надежно было бы всегда кодировать q и a отдельно.
            if newline_idx >= len(ids):
                 debug_messages.append(f"Warning: Calculated newline_idx ({newline_idx}) >= len(ids) ({len(ids)}). Skipping: Q='{q_clean[:50]}...' A='{a_clean[:50]}...'")
                 continue
            # Дополнительная проверка: совпадает ли токен в ids на позиции newline_idx-1 с последним токеном q_ids?
            if newline_idx > 0 and q_ids and ids[newline_idx-1] != q_ids[-1]:
                 debug_messages.append(f"Warning: Token mismatch at calculated split point ({newline_idx-1}). Skipping. Q_last={q_ids[-1]}, S_at_idx={ids[newline_idx-1]}")
                 # print(f"DEBUG: Q_ids: {q_ids}") # Отладка
                 # print(f"DEBUG: S_ids: {ids}")   # Отладка
                 continue
            # --- Конец определения индекса ---

            # Truncate or pad ids
            ids_truncated = ids[:max_seq_len_mp] if original_len > max_seq_len_mp else ids

            # Build input_ids
            input_tokens = ids_truncated[:-1] # All tokens except the last one (usually EOS)
            input_ids_final = np.full(max_seq_len_mp, pad_id, dtype=np.int64)
            len_input = min(len(input_tokens), max_seq_len_mp)
            input_ids_final[:len_input] = input_tokens[:len_input]

            # --- Two-stage logic for labels ---
            labels_final = np.full(max_seq_len_mp, ignore_index_mp, dtype=np.int64)
            if stage1_mode:
                # --- Stage 1: Only penultimate non-<eos> token is target ---
                original_len_trunc = len(ids_truncated)
                # Need at least [ ..., answer_token, eos ]
                if original_len_trunc >= 3:
                    eos_idx = original_len_trunc - 1
                    penult_idx = original_len_trunc - 2  # index of last answer token
                    label_pos = penult_idx - 1     # predict that answer token from previous input step
                    if 0 <= label_pos < max_seq_len_mp:
                        target_token_id = ids_truncated[penult_idx]
                        labels_final[label_pos] = target_token_id
                    else:
                        if len(debug_messages) < 10:
                            debug_messages.append("Warning: Calculated label_pos for stage1 outside max_seq_len.")
                else:
                    if len(debug_messages) < 10:
                        debug_messages.append("Warning: Sequence too short for stage1 mode (need >=3 tokens).")
            else:
                # --- Stage 2: Standard logic (all answer tokens after newline) ---
                # --- Stage 2: Standard logic (all answer tokens after calculated newline_idx) ---
                # Индекс первого токена ответа = newline_idx
                # Индекс в labels, куда записываем первый токен ответа = newline_idx (т.к. labels сдвинуты на 1)
                label_start_index = newline_idx
                answer_token_start_index_in_ids = newline_idx

                # Итерируемся по токенам ПОСЛЕ рассчитанного индекса вопроса в ids_truncated
                for j, tok_id in enumerate(ids_truncated[answer_token_start_index_in_ids:]):
                     label_pos = label_start_index + j
                     if label_pos < max_seq_len_mp:
                         # Устанавливаем ID текущего токена (tok_id) как цель для предыдущей позиции (label_pos)
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
            # Optionally add to debug_messages or just print
            debug_messages.append(f"ERROR processing item {i}: {e}")
            continue # Skip this item and continue with the next

    return out, debug_messages

# --- END OF FILE dataset_utils.py ---
