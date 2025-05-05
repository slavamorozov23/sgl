# --- START OF FILE math_train.py ---

import os
import re
import random
import traceback
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.text import Text
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import time
import sys
from typing import List, Tuple, Set
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
# from transformers import AutoTokenizer # Заменяем на tokenizer_utils
import tokenizer_utils
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR # Import schedulers

import config
from model import SpatialGraphTransformer
from train import train
from dataset_utils import process_batch_for_math_mp, init_math_worker
from text_utils import clean_text

console = Console()

class MathDataset(Dataset):
    """Handles loading, processing, caching, and sampling for the math dataset. Uses tokenizer_utils."""
    # def __init__(self, tokenizer, max_seq_len, split='train', cache_dir=".math_dataset_cache", stage1_mode=False): # Убираем tokenizer из аргументов
    def __init__(self, max_seq_len, split='train', cache_dir=".math_dataset_cache", stage1_mode=False):
        # self.tokenizer = tokenizer # Больше не храним экземпляр здесь
        self.max_seq_len = max_seq_len
        self.ignore_index = config.IGNORE_INDEX
        self.stage1_mode = stage1_mode
        os.makedirs(cache_dir, exist_ok=True)
        # Используем функцию из tokenizer_utils для получения безопасного имени
        safe_tokenizer_name = tokenizer_utils.get_safe_tokenizer_name()
        h5_fname = f"math_{split}_add_sub_tok_{safe_tokenizer_name}_seq{max_seq_len}_v3_nonl.h5"
        pt_fname = f"math_{split}_add_sub_tok_{safe_tokenizer_name}_seq{max_seq_len}_v3_nonl_stacked_tensor_cache.pt" # Новое имя для кэша со стакнутыми тензорами
        self.h5_cache_path = os.path.join(cache_dir, h5_fname)
        self.pt_cache_path = os.path.join(cache_dir, pt_fname) # Путь к тензорному кэшу

        self.all_data = None # Будет содержать словарь с большими тензорами/списками
        # self._original_raw_examples = [] # Больше не нужно
        load_cache = False
        loaded_from_pt = False # Флаг, что загрузились из .pt кэша

        # 1. Проверяем наличие тензорного кэша (.pt)
        if os.path.exists(self.pt_cache_path):
            console.log(f"Found PyTorch tensor cache at {self.pt_cache_path}, loading examples directly...")
            try:
                start_time = time.time()
                # Загружаем словарь с большими тензорами/списками
                self.all_data = torch.load(self.pt_cache_path)
                end_time = time.time()
                # Проверяем, что загрузился словарь с нужными ключами
                if not isinstance(self.all_data, dict) or not all(k in self.all_data for k in ['input_ids', 'labels', 'original_q', 'original_a']):
                    raise ValueError("Loaded .pt cache has incorrect format. Expected a dict with keys 'input_ids', 'labels', 'original_q', 'original_a'.")
                num_examples = self.all_data['input_ids'].shape[0]
                console.log(f"Loaded {num_examples} examples from optimized tensor cache in {end_time - start_time:.2f} seconds.")
                load_cache = True
                loaded_from_pt = True
            except Exception as e:
                console.print(f"[red]Error loading optimized tensor cache file {self.pt_cache_path}: {e}. Trying HDF5 cache...[/red]")
                self.all_data = None
                load_cache = False
                loaded_from_pt = False

        # 2. Если не загрузились из .pt, проверяем HDF5 кэш (.h5)
        if not loaded_from_pt and os.path.exists(self.h5_cache_path):
            console.log(f"Found HDF5 cache (v3_nonl format) at {self.h5_cache_path}, loading examples…")
            try:
                with h5py.File(self.h5_cache_path, 'r') as hf:
                    if 'input_ids' not in hf or 'labels' not in hf:
                         raise ValueError("HDF5 Cache file missing required datasets.")
                    ids_ds = hf['input_ids']
                    labels_ds = hf['labels']
                    originals_q_ds = hf.get('original_q', None)
                    originals_a_ds = hf.get('original_a', None)
                    total = ids_ds.shape[0]
                    console.log(f"Loading all {total} examples from HDF5 cache into memory...")

                    ids_np = ids_ds[:]
                    labels_np = labels_ds[:]
                    if originals_q_ds is not None and originals_a_ds is not None:
                        originals_q_np = originals_q_ds[:]
                        originals_a_np = originals_a_ds[:]
                    else:
                        originals_q_np = [None] * total
                        originals_a_np = [None] * total
                    console.log(f"Data loaded into NumPy arrays. Converting to tensors...")

                    loaded_examples = []
                    with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
                        task = progress.add_task("Creating tensor list from cache", total=total)
                        for i in range(total):
                            loaded_examples.append({
                                "input_ids": torch.tensor(ids_np[i], dtype=torch.long),
                                "labels": torch.tensor(labels_np[i], dtype=torch.long),
                                "original_q": originals_q_np[i],
                                "original_a": originals_a_np[i] # Пока оставляем байты, декодируем позже
                            })
                            progress.update(task, advance=1)
                    # self._all_examples = loaded_examples # Больше не храним список словарей

                    # --- ПРЕОБРАЗОВАНИЕ В СТАКНУТЫЕ ТЕНЗОРЫ И СПИСКИ ---
                    if loaded_examples:
                        console.log("Stacking loaded examples into large tensors/lists...")
                        try:
                            stacked_data = {
                                'input_ids': torch.stack([ex['input_ids'] for ex in loaded_examples]),
                                'labels': torch.stack([ex['labels'] for ex in loaded_examples]),
                                # Декодируем строки здесь один раз
                                'original_q': [ex['original_q'].decode('utf-8', errors='ignore') if isinstance(ex['original_q'], bytes) else str(ex['original_q']) for ex in loaded_examples],
                                'original_a': [ex['original_a'].decode('utf-8', errors='ignore') if isinstance(ex['original_a'], bytes) else str(ex['original_a']) for ex in loaded_examples]
                            }
                            self.all_data = stacked_data
                            console.log(f"Stacked {self.all_data['input_ids'].shape[0]} examples.")

                            # --- СОХРАНЕНИЕ ОПТИМИЗИРОВАННОГО КЭША ---
                            console.log(f"Saving stacked data to optimized PyTorch tensor cache at {self.pt_cache_path}...")
                            start_time = time.time()
                            torch.save(self.all_data, self.pt_cache_path)
                            end_time = time.time()
                            console.log(f"Optimized tensor cache saved successfully in {end_time - start_time:.2f} seconds.")
                            # --- КОНЕЦ СОХРАНЕНИЯ ---
                        except Exception as stack_err:
                            console.print(f"[red]Error stacking data or saving optimized cache: {stack_err}[/red]")
                            self.all_data = None # Сбрасываем, если была ошибка
                    else:
                            console.print("[yellow]No examples loaded from HDF5 cache.[/yellow]")
                            self.all_data = None
                    # --- КОНЕЦ ПРЕОБРАЗОВАНИЯ ---

                load_cache = True # Устанавливаем флаг, что загрузка из кэша (HDF5) произошла
            except Exception as e:
                console.print(f"[red]Error loading HDF5 cache file {self.h5_cache_path}: {e}. Reprocessing needed.[/red]")
                self.all_data = None
                load_cache = False

        # 3. Если не загрузились ни из .pt, ни из .h5, выполняем полную обработку
        if not load_cache:
            console.log(f"No valid cache found. Processing math dataset split='{split}' for cache (v3_nonl format)...")
            raw = None
            try:
                # Removed console.status context manager
                raw = load_dataset("math_dataset", "arithmetic__add_sub_multiple", split=split, cache_dir=config.PRETRAIN_LEARNING_DIR, trust_remote_code=True)
                console.log(f"Raw dataset '{split}' split loaded with {len(raw)} examples.")
            except Exception as e:
                console.print(f"[bold red]Failed to load raw dataset: {e}[/]")
                self.examples = []
                return

            console.log("Mapping and preprocessing math dataset using ProcessPoolExecutor (incremental HDF5 cache)...")
            raw_questions = raw['question']
            raw_answers = raw['answer']
            total_items = len(raw_questions)
            num_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
            console.log(f"Using {num_workers} workers for preprocessing.")
            chunk_size = 500

            # Определяем shape данных через маленький тестовый батч
            console.log("Processing a small batch to determine data shapes for HDF5...")
            # init_math_worker больше не принимает токенизатор, он получит его сам через tokenizer_utils
            # --- ДОБАВИТЬ ЛОГ ПЕРЕД СОЗДАНИЕМ ТЕСТОВОГО ПУЛА ---
            console.log("[DIAG] Creating test ProcessPoolExecutor...")
            with ProcessPoolExecutor(max_workers=1, initializer=init_math_worker, initargs=(None, max_seq_len, self.ignore_index)) as test_executor:
                # --- ДОБАВИТЬ ЛОГ ПОСЛЕ СОЗДАНИЯ ТЕСТОВОГО ПУЛА ---
                console.log("[DIAG] Test ProcessPoolExecutor created. Submitting test batch...")
                temp_batch = {'question': raw_questions[:chunk_size], 'answer': raw_answers[:chunk_size]}
                temp_future = test_executor.submit(process_batch_for_math_mp, temp_batch)
                temp_result, _ = temp_future.result()
                # --- ДОБАВИТЬ ЛОГ ПОСЛЕ ПОЛУЧЕНИЯ РЕЗУЛЬТАТА ТЕСТА ---
                console.log("[DIAG] Test batch result received.")
                if not temp_result or not temp_result.get('input_ids'):
                     console.print("[bold red]Failed to process even a small batch. Cannot determine HDF5 shapes. Exiting.[/bold red]")
                     self._all_examples = []
                     return
                sample_input_shape = np.asarray(temp_result['input_ids'][0]).shape
                sample_label_shape = np.asarray(temp_result['labels'][0]).shape
                console.log(f"Determined shapes: input_ids={sample_input_shape}, labels={sample_label_shape}")

            # Основная обработка с инкрементальным сохранением
            futures = []
            # init_math_worker больше не принимает токенизатор
            # --- ДОБАВИТЬ ЛОГ ПЕРЕД СОЗДАНИЕМ ОСНОВНОГО ПУЛА ---
            console.log("[DIAG] Creating main ProcessPoolExecutor...")
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_math_worker, initargs=(None, max_seq_len, self.ignore_index)) as executor:
                # --- ДОБАВИТЬ ЛОГ ПОСЛЕ СОЗДАНИЯ ОСНОВНОГО ПУЛА ---
                console.log(f"[DIAG] Main ProcessPoolExecutor created with {num_workers} workers. Submitting tasks...")
                for i in range(0, total_items, chunk_size):
                    batch = {'question': raw_questions[i:i+chunk_size], 'answer': raw_answers[i:i+chunk_size]}
                    futures.append(executor.submit(process_batch_for_math_mp, batch))
                # --- ДОБАВИТЬ ЛОГ ПОСЛЕ ОТПРАВКИ ЗАДАЧ ---
                console.log("[DIAG] All tasks submitted. Starting HDF5 cache creation and result processing...")

                console.log(f"Creating HDF5 cache (v4 incremental format) at {self.h5_cache_path} and writing incrementally...")
                current_write_pos = 0
                processed_count = 0
                try:
                    with h5py.File(self.h5_cache_path, 'w', libver='latest') as hf:
                        hf.create_dataset('input_ids', shape=(0, *sample_input_shape), maxshape=(None, *sample_input_shape), dtype=np.int64, compression='gzip')
                        hf.create_dataset('labels', shape=(0, *sample_label_shape), maxshape=(None, *sample_label_shape), dtype=np.int64, compression='gzip')
                        hf.create_dataset('original_q', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'), compression='gzip')
                        hf.create_dataset('original_a', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'), compression='gzip')

                        with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
                            task = progress.add_task(f"Processing and Saving '{split}' dataset", total=total_items)
                            # --- ДОБАВИТЬ ЛОГ ПЕРЕД ЦИКЛОМ ОБРАБОТКИ РЕЗУЛЬТАТОВ ---
                            console.log("[DIAG] Starting to process results with Progress bar...")

                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    result_data, debug_messages = future.result()
                                    for msg in debug_messages:
                                        console.print(msg)
                                    input_ids_list = result_data.get('input_ids', [])
                                    labels_list = result_data.get('labels', [])
                                    orig_q_list = result_data.get('original_q', [])
                                    orig_a_list = result_data.get('original_a', [])
                                    chunk_num_examples = len(input_ids_list)
                                    if chunk_num_examples > 0:
                                        input_ids_np = np.array(input_ids_list, dtype=np.int64)
                                        labels_np = np.array(labels_list, dtype=np.int64)
                                        originals_q_np = np.array(orig_q_list, dtype=h5py.string_dtype('utf-8'))
                                        originals_a_np = np.array(orig_a_list, dtype=h5py.string_dtype('utf-8'))
                                        new_size = current_write_pos + chunk_num_examples
                                        hf['input_ids'].resize((new_size, *sample_input_shape))
                                        hf['labels'].resize((new_size, *sample_label_shape))
                                        hf['original_q'].resize((new_size,))
                                        hf['original_a'].resize((new_size,))
                                        hf['input_ids'][current_write_pos:new_size] = input_ids_np
                                        hf['labels'][current_write_pos:new_size] = labels_np
                                        hf['original_q'][current_write_pos:new_size] = originals_q_np
                                        hf['original_a'][current_write_pos:new_size] = originals_a_np
                                        current_write_pos = new_size
                                        processed_count += chunk_num_examples
                                    progress.update(task, advance=chunk_num_examples)
                                except Exception as e:
                                    console.print(f"\n[red]Error processing or saving a batch result: {e}[/red]")
                                    progress.update(task, advance=chunk_size)
                    console.log(f"Finished processing. Total examples written to HDF5: {current_write_pos}")
                except Exception as e:
                    console.print(f"[bold red]Critical Error during HDF5 creation/writing: {e}[/bold red]")
                    if os.path.exists(self.h5_cache_path):
                        try:
                            os.remove(self.h5_cache_path)
                            console.print(f"[yellow]Removed incomplete HDF5 cache file: {self.h5_cache_path}[/yellow]")
                        except Exception as remove_e:
                            console.print(f"[red]Error removing incomplete HDF5 cache file: {remove_e}[/red]")
                    # self._all_examples = [] # Больше не используется
                    self.all_data = None
                    return

            # Загрузка из только что созданного HDF5 кэша (для последующего стакинга и сохранения в .pt)
            console.log(f"Loading data from the newly created HDF5 cache at {self.h5_cache_path} for stacking...")
            try:
                with h5py.File(self.h5_cache_path, 'r') as hf:
                    ids_ds = hf['input_ids']
                    labels_ds = hf['labels']
                    originals_q_ds = hf.get('original_q', None)
                    originals_a_ds = hf.get('original_a', None)
                    total = ids_ds.shape[0]
                    console.log(f"Loading all {total} examples from HDF5 cache into memory...")
                    ids_np = ids_ds[:]
                    labels_np = labels_ds[:]
                    if originals_q_ds is not None and originals_a_ds is not None:
                        originals_q_np = originals_q_ds[:]
                        originals_a_np = originals_a_ds[:]
                    else:
                        originals_q_np = [None] * total
                        originals_a_np = [None] * total
                    loaded_examples = []
                    with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
                        task = progress.add_task("Creating tensor list from cache", total=total)
                        for i in range(total):
                            q_str = originals_q_np[i].decode('utf-8') if originals_q_np[i] is not None and hasattr(originals_q_np[i], 'decode') else originals_q_np[i]
                            a_str = originals_a_np[i].decode('utf-8') if originals_a_np[i] is not None and hasattr(originals_a_np[i], 'decode') else originals_a_np[i]
                            loaded_examples.append({
                                "input_ids": torch.tensor(ids_np[i], dtype=torch.long),
                                "labels": torch.tensor(labels_np[i], dtype=torch.long),
                                "original_q": q_str,
                                "original_a": a_str # Строки уже декодированы
                            })
                            progress.update(task, advance=1)
                    # --- ПРЕОБРАЗОВАНИЕ В СТАКНУТЫЕ ТЕНЗОРЫ И СПИСКИ ---
                    if loaded_examples:
                        console.log("Stacking processed examples into large tensors/lists...")
                        try:
                            stacked_data = {
                                'input_ids': torch.stack([ex['input_ids'] for ex in loaded_examples]),
                                'labels': torch.stack([ex['labels'] for ex in loaded_examples]),
                                'original_q': [ex['original_q'] for ex in loaded_examples], # Строки уже готовы
                                'original_a': [ex['original_a'] for ex in loaded_examples]  # Строки уже готовы
                            }
                            self.all_data = stacked_data
                            console.log(f"Stacked {self.all_data['input_ids'].shape[0]} examples.")

                            # --- СОХРАНЕНИЕ ОПТИМИЗИРОВАННОГО КЭША ---
                            console.log(f"Saving stacked data to optimized PyTorch tensor cache at {self.pt_cache_path}...")
                            start_time = time.time()
                            torch.save(self.all_data, self.pt_cache_path)
                            end_time = time.time()
                            console.log(f"Optimized tensor cache saved successfully in {end_time - start_time:.2f} seconds.")
                            # --- КОНЕЦ СОХРАНЕНИЯ ---
                        except Exception as stack_err:
                            console.print(f"[red]Error stacking data or saving optimized cache: {stack_err}[/red]")
                            self.all_data = None
                    else:
                         console.print("[yellow]No examples loaded from newly created HDF5 cache.[/yellow]")
                         self.all_data = None
                    # --- КОНЕЦ ПРЕОБРАЗОВАНИЯ ---

            except Exception as e:
                console.print(f"[bold red]Error loading data from newly created HDF5 cache {self.h5_cache_path}: {e}. Dataset might be empty.[/bold red]")
                self.all_data = None

        # Вызываем resample только если данные были успешно загружены или созданы
        if self.all_data is not None:
             self.resample()
        else:
             console.print("[red]Dataset initialization failed. No examples loaded.[/red]")

    def resample(self):
        """Resamples indices based on config.DATASET_PERCENTAGE."""
        if self.all_data is None:
             self.examples = [] # Хранит индексы
             console.log("[yellow]No data loaded, cannot resample indices.[/yellow]")
             return

        total = self.all_data['input_ids'].shape[0]
        if total == 0:
            self.examples = []
            console.log("[yellow]No examples available to sample from.[/yellow]")
            return

        pct = config.DATASET_PERCENTAGE
        frac = max(0.001, min(1.0, pct / 100.0))
        all_indices = list(range(total))

        if frac < 1.0:
            num_to_keep = max(1, int(total * frac))
            self.examples = random.sample(all_indices, num_to_keep) # Сэмплируем индексы
            console.log(f"Sampled {pct}% ({len(self.examples)}/{total}) indices for the current epoch.")
        else:
            self.examples = all_indices # Используем все индексы
            console.log(f"Using all {total} indices for the current epoch (100%).")

        self.was_sampled = (frac < 1.0)

    def set_stage1_mode(self, mode: bool):
        self.stage1_mode = mode

    def __len__(self):
        # Возвращает количество сэмплированных индексов
        return len(self.examples)

    def __getitem__(self, idx):
        # idx - это индекс в списке сэмплированных индексов self.examples
        if not self.examples:
             raise IndexError("Dataset is not sampled or empty")
        if not 0 <= idx < len(self.examples):
             raise IndexError(f"Index {idx} out of bounds for current sample size {len(self.examples)}")
        if self.all_data is None:
             raise RuntimeError("Dataset not initialized properly, all_data is None.")

        # Получаем реальный индекс из списка сэмплированных
        real_idx = self.examples[idx]

        # Возвращаем данные по реальному индексу из больших тензоров/списков
        # Преобразование в тензоры не нужно, они уже тензоры
        return {
            "input_ids": self.all_data['input_ids'][real_idx],
            "labels": self.all_data['labels'][real_idx],
            "original_q": self.all_data['original_q'][real_idx], # Строка
            "original_a": self.all_data['original_a'][real_idx]  # Строка
        }


def evaluate(model, dataloader, crit, device):
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct_tokens = 0
    total_target_tokens = 0
    total_batches = 0

    if not dataloader or len(dataloader) == 0:
        console.log("[yellow]Validation dataloader is empty or None. Skipping evaluation.[/yellow]")
        return float('inf'), 0.0

    with torch.no_grad():
        with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), TextColumn("{task.completed}/{task.total} batches"), console=console) as progress:
            task = progress.add_task("Evaluating", total=len(dataloader))
            for batch in dataloader:
                if not isinstance(batch, dict) or 'input_ids' not in batch or 'labels' not in batch:
                    console.print(f"[yellow]Warning: Skipping invalid batch in evaluation: {type(batch)}[/yellow]")
                    progress.update(task, advance=1)
                    continue

                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                try:
                    # Unpack 5 return values, ignore the last two (aux losses) for evaluation
                    logits, _, _, _, *_ = model(input_ids)
                    loss = crit(logits.view(-1, logits.size(-1)), labels.view(-1))

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                    else:
                        console.print(f"[yellow]Warning: NaN/Inf loss during evaluation batch. Skipping batch loss.[/yellow]")

                    preds = logits.argmax(dim=-1)
                    mask = labels != config.IGNORE_INDEX
                    total_correct_tokens += ((preds == labels) & mask).sum().item()
                    total_target_tokens += mask.sum().item()
                    total_batches += 1
                except Exception as e:
                    console.print(f"[red]Error during evaluation batch: {e}[/red]")
                finally:
                    progress.update(task, advance=1)

    model.train()
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_acc = total_correct_tokens / total_target_tokens if total_target_tokens > 0 else 0.0
    return avg_loss, avg_acc


if __name__ == "__main__":
    try:
        script_start_time = time.time() # Запоминаем время старта
        console.rule("[bold blue]Initializing Math Training (v3_nonl Data Format)[/]")
        # Инициализация токенизатора через tokenizer_utils (вызов get_tokenizer() произойдет при первом использовании)
        # Просто убедимся, что он инициализируется и получим данные
        try:
            tokenizer_utils.get_tokenizer() # Вызовем для инициализации и проверки
        except Exception as e:
             console.print(f"[bold red]Failed to initialize tokenizer via tokenizer_utils: {e}[/]")
             exit(1)

        config.VOCAB_SIZE = tokenizer_utils.get_vocab_size()
        console.log(f"Tokenizer: {tokenizer_utils.get_tokenizer_name()}, Vocab size: {config.VOCAB_SIZE}")
        # Инициализация глобальных переменных для обработки батчей
        # init_math_worker(tokenizer, config.MAX_SEQ_LEN, config.IGNORE_INDEX) # Удаляем, init_math_worker должен сам получать токенизатор
        # Вместо этого, просто инициализируем worker-ы (если это все еще нужно)
        # Важно: init_math_worker в dataset_utils.py нужно будет обновить!
        init_math_worker(None, config.MAX_SEQ_LEN, config.IGNORE_INDEX) # Передаем None вместо токенизатора

        model = SpatialGraphTransformer(
            d_model=config.D_MODEL,
            dim_feedforward=config.DIM_FEEDFORWARD,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
            vocab_size=config.VOCAB_SIZE,
            rope_theta=getattr(config, 'ROPE_THETA', 10000.0)
        ).to(config.DEVICE)
        console.log(f"Loaded model with {model.num_cubes} cubes from JSON config.")
        console.log(f"Model initialized on {config.DEVICE}")

        # Calculate trainable parameters based on ROUTING_SAFETY_LIMIT
        global_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and not name.startswith('cubes.'))
        # Assuming all cubes have the same structure and trainable parameters
        if model.num_cubes > 0:
            modular_params_per_cube = sum(p.numel() for name, p in model.cubes[0].named_parameters() if p.requires_grad)
        else:
            modular_params_per_cube = 0
            console.print("[yellow]Warning: No cubes in the model. Modular parameters per cube is 0.[/yellow]")

        max_active_params = global_params + config.ROUTING_SAFETY_LIMIT * modular_params_per_cube
        console.log(f"Trainable parameters (max active based on ROUTING_SAFETY_LIMIT={config.ROUTING_SAFETY_LIMIT}): {max_active_params / 1_000_000:.2f} M")
        console.log(f"Global parameters: {global_params / 1_000_000:.2f} M")
        console.log(f"Modular parameters per cube: {modular_params_per_cube / 1_000_000:.2f} M")


        # --- Two-stage training flags ---
        stage1_complete = False
        stage1_target_accuracy = 0.90

        console.log("Loading/Processing Training Dataset...")
        # dataset = MathDataset(tokenizer, config.MAX_SEQ_LEN, split='train', stage1_mode=True) # Убираем tokenizer
        dataset = MathDataset(config.MAX_SEQ_LEN, split='train', stage1_mode=True)
        console.log("Loading/Processing Validation Dataset...")
        # val_dataset = MathDataset(tokenizer, config.MAX_SEQ_LEN, split='test', stage1_mode=True) # Убираем tokenizer
        val_dataset = MathDataset(config.MAX_SEQ_LEN, split='test', stage1_mode=True)

        if len(dataset) == 0:
            console.print("[bold red]Training dataset is empty. Exiting.[/]")
            exit(1)
        if len(val_dataset) == 0:
             console.print("[bold yellow]Warning: Validation dataset is empty.[/]")

        num_workers = 0
        pin_memory = (config.DEVICE.type == "cuda")
        prefetch_factor = 2 if num_workers > 0 else None

        # Use drop_last=True for training dataloader
        dl = DataLoader(
            dataset, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
            drop_last=True
        )
        val_dl = None
        if len(val_dataset) > 0:
            # No drop_last for validation
            val_dl = DataLoader(
                val_dataset, batch_size=config.PRETRAIN_BATCH_SIZE * 2, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor
            )

        console.log(f"Initial Training Dataloader: {len(dl)} batches (Batch Size: {config.PRETRAIN_BATCH_SIZE}, Drop Last: True)")
        if val_dl:
             console.log(f"Validation Dataloader: {len(val_dl)} batches (Batch Size: {config.PRETRAIN_BATCH_SIZE * 2})")
        else:
             console.log("Validation Dataloader: Empty")

        optim = torch.optim.AdamW(model.parameters(), lr=config.PRETRAIN_LEARNING_RATE, weight_decay=0.01)

        # --- НАЧАЛО: Настройка Warmup и Планировщика ---
        warmup_steps = config.WARMUP_STEPS
        try:
            steps_per_epoch = len(dl)
            total_training_steps = config.PRETRAIN_EPOCHS * steps_per_epoch
            main_scheduler_steps = total_training_steps - warmup_steps
            console.log(f"Scheduler: Warmup for {warmup_steps} steps.")
            if main_scheduler_steps > 0:
                console.log(f"           Then CosineAnnealing for {main_scheduler_steps} steps.")
            else:
                console.log(f"[yellow]Warning: Warmup steps ({warmup_steps}) >= total estimated steps ({total_training_steps}). Cosine Annealing might not activate effectively.[/yellow]")
                main_scheduler_steps = 1
        except TypeError:
            console.print("[yellow]Warning: Could not determine steps per epoch from DataLoader.")
            console.print(f"           Warmup will run for {warmup_steps} steps (from config).")
            console.print("           CosineAnnealing T_max will be approximate (using PRETRAIN_EPOCHS * 1000 estimate). Adjust if needed.")
            estimated_total_steps = config.PRETRAIN_EPOCHS * 1000
            main_scheduler_steps = max(1, estimated_total_steps - warmup_steps)

        scheduler_warmup = LinearLR(optim, start_factor=1e-9, end_factor=1.0, total_iters=warmup_steps)

        if config.USE_COSINE_ANNEALING:
            console.log(f"           Using CosineAnnealingLR for {main_scheduler_steps} steps after warmup (eta_min={1e-5}).")
            scheduler_main = CosineAnnealingLR(optim, T_max=main_scheduler_steps, eta_min=1e-5)
        else:
            console.log(f"           Using ConstantLR after warmup (Cosine Annealing disabled in config).")
            # Используем ConstantLR с фактором 1, чтобы LR оставался тем же, что и после warmup
            scheduler_main = ConstantLR(optim, factor=1.0, total_iters=main_scheduler_steps) # total_iters здесь не так важен, но нужен

        scheduler = SequentialLR(optim, schedulers=[scheduler_warmup, scheduler_main], milestones=[warmup_steps])
        # --- КОНЕЦ: Настройка Warmup и Планировщика ---

        crit = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        # Вывод времени перед началом обучения
        elapsed_init_time = time.time() - script_start_time
        console.log(f"Initialization and data loading took: {elapsed_init_time:.2f} seconds.")

        console.rule("[bold blue]Start Training[/]")
        best_val_loss = float('inf')
        for epoch in range(config.PRETRAIN_EPOCHS):
            console.rule(f"[bold]Epoch {epoch+1}/{config.PRETRAIN_EPOCHS} {'(Stage 2)' if stage1_complete else '(Stage 1)'}[/]")

            # --- Switch stage if needed ---
            dataset.set_stage1_mode(not stage1_complete)
            val_dataset.set_stage1_mode(not stage1_complete)

            # Resample training data for the epoch
            dataset.resample()
            # Recreate training dataloader
            dl = DataLoader(
                dataset, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                drop_last=True
            )
            console.log(f"Training Dataloader ready: {len(dl)} batches for epoch {epoch+1}")
            if len(dl) == 0:
                console.print("[yellow]Warning: Training dataloader is empty for this epoch after resampling. Skipping training phase.[/yellow]")
                continue

            avg_train_loss = train(model, dl, optim, crit, epoch, config.PRETRAIN_EPOCHS, config.DEVICE, scheduler)

            # Validation Phase
            avg_val_loss = float('inf')
            avg_val_acc = 0.0
            if val_dl and len(val_dl) > 0:
                console.log(f"--- Running Validation for Epoch {epoch+1} ---")
                avg_val_loss, avg_val_acc = evaluate(model, val_dl, crit, config.DEVICE)
                console.log(f"[Validation] Epoch {epoch+1}: Avg Loss = {avg_val_loss:.4f}, Avg Acc@Target = {avg_val_acc*100:.2f}%", style="bold blue")

                # --- Stage switch logic ---
                if not stage1_complete and avg_val_acc >= stage1_target_accuracy:
                    console.print(f"[bold green]Stage 1 complete! Accuracy {avg_val_acc*100:.2f}% >= {stage1_target_accuracy*100}%. Switching to Stage 2.[/]")
                    stage1_complete = True
                    # Optionally adjust LR or optimizer here

                if avg_val_loss < best_val_loss:
                    console.log(f"[bold green]Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...[/]", style="green")
                    best_val_loss = avg_val_loss
                    try:
                        os.makedirs(os.path.dirname(config.PRETRAINED_MODEL_SAVE_PATH), exist_ok=True)
                        torch.save(model.state_dict(), config.PRETRAINED_MODEL_SAVE_PATH)
                        console.log(f"Model saved to {config.PRETRAINED_MODEL_SAVE_PATH}")
                    except Exception as e:
                        console.print(f"[red]Error saving model: {e}[/red]")
                else:
                    console.log(f"Validation loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}).", style="yellow")
            else:
                 console.log("[yellow]No validation set or it's empty. Saving model checkpoint after epoch.[/yellow]")
                 try:
                    save_path = f"{config.PRETRAINED_MODEL_SAVE_PATH}.epoch{epoch+1}"
                    torch.save(model.state_dict(), save_path)
                    console.log(f"Model checkpoint saved to {save_path}")
                    if not val_dl:
                         torch.save(model.state_dict(), config.PRETRAINED_MODEL_SAVE_PATH)
                         console.log(f"Updated main model file: {config.PRETRAINED_MODEL_SAVE_PATH}")

                 except Exception as e:
                     console.print(f"[red]Error saving model checkpoint: {e}[/red]")


            # End-of-Epoch Sample Predictions
            console.log(f"--- End-of-Epoch Sample Predictions (Epoch {epoch+1}) ---")
            model.eval()
            num_test_samples = 3
            current_epoch_train_len = len(dataset)
            if current_epoch_train_len >= num_test_samples:
                indices_to_test = random.sample(range(current_epoch_train_len), num_test_samples)
                for test_i, idx in enumerate(indices_to_test):
                    console.print(f"\n--- Sample {test_i+1} (Index: {idx}) ---")
                    try:
                        example = dataset[idx]
                        q_orig_raw = example['original_q']
                        a_orig_raw = example['original_a']
                        # --- ДОБАВЛЕНО: Очистка Q и A для вывода/обработки в конце эпохи ---
                        q_orig = clean_text(q_orig_raw)
                        a_orig_clean = clean_text(a_orig_raw) # Название переменной оставляем для консистентности
                        # --- КОНЕЦ ДОБАВЛЕНИЯ ---
                        s_orig = f"{q_orig}\n{a_orig_clean}" # Используем очищенные версии
                        # initial_ids_approx = tokenizer.encode(s_orig, add_special_tokens=False)
                        prepared_s_orig = tokenizer_utils.prepare_text_for_encoding(s_orig) # Подготовка текста
                        initial_ids_approx = tokenizer_utils.encode(prepared_s_orig) # add_special_tokens=False по умолчанию
                        # if tokenizer.eos_token_id is not None and (not initial_ids_approx or initial_ids_approx[-1] != tokenizer.eos_token_id):
                        #     initial_ids_approx.append(tokenizer.eos_token_id)
                        eos_id = tokenizer_utils.get_eos_token_id()
                        if eos_id is not None and (not initial_ids_approx or initial_ids_approx[-1] != eos_id):
                             initial_ids_approx.append(eos_id)
                        # --- ИЗМЕНЕНИЕ: Определяем точку разделения по длине токенизированного вопроса ---
                        try:
                            # Токенизируем только очищенный вопрос, чтобы получить его длину
                            prepared_q_orig = tokenizer_utils.prepare_text_for_encoding(q_orig)
                            q_orig_ids = tokenizer_utils.encode(prepared_q_orig)
                            split_idx = len(q_orig_ids) # Индекс первого токена ответа = длина вопроса

                            # Проверяем, что индекс не выходит за пределы
                            if split_idx >= len(initial_ids_approx):
                                raise ValueError(f"Calculated split index {split_idx} is out of bounds for sequence length {len(initial_ids_approx)}")

                            # Разделяем последовательность
                            # Вопрос включает все токены до split_idx (не включая сам токен на split_idx, если он есть)
                            # Ответ начинается с токена на позиции split_idx
                            # Для генерации нам нужны ID вопроса как вход (current_input_ids)
                            # Для сравнения нам нужны ID истинного ответа (answer_ids_true)
                            question_ids = initial_ids_approx[:split_idx] # ID только вопроса
                            answer_ids_true = initial_ids_approx[split_idx:] # ID ответа, НАЧИНАЯ с токенов \n (_._)

                            # --- ИСПРАВЛЕНО: Вход для генерации - ТОЛЬКО вопрос ---
                            # Модель сама должна сгенерировать разделитель (\n -> _._), т.к. обучалась на этом
                            question_ids_for_input = question_ids
                            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                        except Exception as e:
                            console.print(f"[yellow]Warning: Error determining question/answer split: {e}. Skipping detailed test.[/yellow]")
                            console.print(f"Original Question: {q_orig}")
                            console.print(f"Original Answer: {a_orig_clean}")
                            console.print(f"Reconstructed Sequence (Approx): {tokenizer_utils.decode(initial_ids_approx)}")
                            continue
                        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

                        # --- УДАЛЕНЫ ДОП. DEBUG PRINTS ---

                        if not answer_ids_true:
                            console.print("[yellow]Warning: No answer tokens found after split point. Skipping detailed test.[/yellow]")
                            continue
                        # initial_full_str = tokenizer.decode(initial_ids_approx)
                        initial_full_str = tokenizer_utils.decode(initial_ids_approx) # Полная строка для информации
                        # enter_str = tokenizer.decode(question_ids_for_input) # Строка, подаваемая на вход генерации
                        enter_str = tokenizer_utils.decode(question_ids_for_input)
                        console.print(f"Initial Decoded (Q+A): {initial_full_str}")
                        console.print(f"Input for Generation (Q): {enter_str}") # Изменено для ясности

                        # --- Пошаговое предсказание токенов ---
                        console.print("--- Token-by-Token Prediction ---")
                        # Убедимся, что initial_ids_approx не пустой
                        if not initial_ids_approx:
                            console.print("[yellow]Warning: Tokenized sequence is empty. Skipping token prediction.[/yellow]")
                            continue

                        input_ids_tensor = torch.tensor([initial_ids_approx], dtype=torch.long).to(config.DEVICE)

                        with torch.no_grad():
                            # Проходим по последовательности до предпоследнего токена
                            # Ограничиваем длину, чтобы избежать слишком длинного вывода
                            max_pred_steps = min(len(initial_ids_approx) - 1, config.MAX_SEQ_LEN) # Используем MAX_SEQ_LEN как лимит шагов
                            for j in range(max_pred_steps):
                                # Ограничиваем входную последовательность для модели
                                current_input_ids = input_ids_tensor[:, :min(j+1, config.MAX_SEQ_LEN)]
                                # Убедимся, что индекс j+1 не выходит за пределы initial_ids_approx
                                if j + 1 >= len(initial_ids_approx):
                                    console.print(f"[yellow]Warning: Index {j+1} out of bounds for initial_ids_approx (len={len(initial_ids_approx)}). Stopping prediction.[/yellow]")
                                    break
                                true_next_token_id = initial_ids_approx[j+1]

                                try:
                                    # Получаем логиты от модели
                                    logits, *_ = model(current_input_ids)
                                    # Логиты для предсказания следующего токена (на последней позиции входа)
                                    next_token_logits = logits[:, -1, :]

                                    # Получаем вероятности и предсказанный токен
                                    probabilities = F.softmax(next_token_logits, dim=-1)
                                    confidence, predicted_next_token_id_tensor = torch.max(probabilities, dim=-1)
                                    predicted_next_token_id = predicted_next_token_id_tensor.item()
                                    confidence = confidence.item() * 100

                                    # Декодируем для вывода
                                    # Показываем только последние ~30 токенов контекста для читаемости
                                    context_tokens = initial_ids_approx[max(0, j+1-30):j+1]
                                    context_str = tokenizer_utils.decode(context_tokens)
                                    if j+1 > 30:
                                        context_str = "..." + context_str # Добавляем многоточие, если контекст урезан

                                    predicted_token_str = repr(tokenizer_utils.decode([predicted_next_token_id]))
                                    true_token_str = repr(tokenizer_utils.decode([true_next_token_id]))

                                    # Проверяем корректность
                                    is_correct = (predicted_next_token_id == true_next_token_id)
                                    result_marker = "[green]Correct[/]" if is_correct else "[red]Incorrect[/]"

                                    # Выводим результат
                                    console.print(f"Step {j+1}: Context: ...'{context_str}'")
                                    console.print(f"  -> Predict: {predicted_token_str} (Conf: {confidence:.2f}%) | True: {true_token_str} | {result_marker}")

                                    # Опционально: остановка, если предсказан EOS или достигнут лимит
                                    if eos_id is not None and predicted_next_token_id == eos_id and j > len(question_ids): # Останавливаемся после EOS, если он не сразу после вопроса
                                         console.print(f"  Predicted EOS. Stopping prediction for this sample.")
                                         break
                                except Exception as pred_err:
                                     console.print(f"[red]Error during token prediction step {j+1}: {pred_err}[/red]")
                                     console.print(traceback.format_exc())
                                     break # Прерываем цикл для этого примера при ошибке

                        # --- Конец пошагового предсказания ---
                    except Exception as sample_err:
                        console.print(f"[red]Error processing sample {test_i+1} (index {idx}): {sample_err}[/red]")
                        console.print(traceback.format_exc())
                    console.print("-" * 20)
            else:
                console.print(f"[yellow]Not enough samples ({current_epoch_train_len}) in the current epoch training dataset to run {num_test_samples} end-of-epoch tests.[/yellow]")
            model.train()

        console.rule("[bold blue]Training Finished[/]")
        console.log(f"Model training complete. Best validation loss: {best_val_loss:.4f}")
        if val_dl and best_val_loss != float('inf'):
             console.log(f"Best model based on validation loss saved to {config.PRETRAINED_MODEL_SAVE_PATH}")
        elif not val_dl:
             console.log(f"Final model saved to {config.PRETRAINED_MODEL_SAVE_PATH}")


    except KeyboardInterrupt:
        console.print("\n[bold yellow]Training interrupted by user (Ctrl+C).[/]")

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during training: {e}[/]")
        console.print(traceback.format_exc())
        exit(1)

# --- END OF FILE math_train.py ---
