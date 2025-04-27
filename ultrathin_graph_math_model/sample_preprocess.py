import os
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoTokenizer
import config
from dataset_utils import process_batch_for_math_mp, init_math_worker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

# === 1. Настройки ===
MODEL_NAME = getattr(config, 'MODEL_NAME', 'gpt2')
MAX_SEQ_LEN = getattr(config, 'MAX_SEQ_LEN', 256)
IGNORE_INDEX = getattr(config, 'IGNORE_INDEX', -100)

console.log(f"[bold cyan]MODEL_NAME:[/] {MODEL_NAME}, [bold cyan]MAX_SEQ_LEN:[/] {MAX_SEQ_LEN}, [bold cyan]IGNORE_INDEX:[/] {IGNORE_INDEX}")

# === 2. Токенайзер ===
with console.status("[bold green]Загрузка токенайзера…", spinner="dots"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
console.log(f"[green]Токенайзер загружен:[/] {tokenizer.__class__.__name__}")

# === 3. Глобальные переменные для process_batch_for_math_mp ===
import dataset_utils

dataset_utils.tokenizer_mp = tokenizer
dataset_utils.max_seq_len_mp = MAX_SEQ_LEN
dataset_utils.ignore_index_mp = IGNORE_INDEX

# === 4. Загрузка 3 примеров датасета ===
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
    task = progress.add_task("Загрузка датасета", total=1)
    raw = load_dataset("math_dataset", "arithmetic__add_or_sub", split="train", cache_dir=config.PRETRAIN_LEARNING_DIR, trust_remote_code=True)
    progress.update(task, advance=1)
import re
from text_utils import clean_text

examples = {k: raw[k][:3] for k in ('question', 'answer')}

console.rule("[bold yellow]RAW EXAMPLES")
for q, a in zip(examples['question'], examples['answer']):
    console.print(f"[bold]Q:[/] {q}\n[bold]A:[/] {a}\n---")

# === ЭТАП ОЧИСТКИ QA ===
cleaned_examples = {
    'question': [clean_text(q) for q in examples['question']],
    'answer': [clean_text(a) for a in examples['answer']]
}
console.rule("[bold yellow]CLEANED EXAMPLES")
for q, a in zip(cleaned_examples['question'], cleaned_examples['answer']):
    console.print(f"[bold]Q:[/] {q}\n[bold]A:[/] {a}\n---")

# === 5. Маппинг и токенизация ===
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
    task = progress.add_task("Маппинг и токенизация", total=3)
    processed, debug_messages = process_batch_for_math_mp(cleaned_examples, show_samples=True)
    progress.update(task, advance=3)

console.rule("[bold yellow]DEBUG MESSAGES")
for msg in debug_messages:
    console.log(msg)

# === 6. Преобразование в тензоры ===
input_ids, labels = [], []
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
    task = progress.add_task("Преобразование в тензоры", total=len(processed['input_ids']))
    for arr in processed['input_ids']:
        input_ids.append(torch.tensor(arr, dtype=torch.long))
        progress.update(task, advance=1)
    for arr in processed['labels']:
        labels.append(torch.tensor(arr, dtype=torch.long))

console.rule("[bold yellow]TOKENS (input_ids)")
for idx, ids in enumerate(input_ids):
    console.print(f"[bold]Example {idx+1}:[/] {ids.tolist()}")

# === 7. Обратное преобразование: input_ids -> токены -> текст ===
console.rule("[bold yellow]DECODED TEXTS")
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
    task = progress.add_task("input_ids → текст", total=len(input_ids))
    for idx, ids in enumerate(input_ids):
        arr = ids.tolist()
        if tokenizer.pad_token_id is not None:
            arr = [x for x in arr if x != tokenizer.pad_token_id]
        if tokenizer.eos_token_id is not None:
            arr = [x for x in arr if x != tokenizer.eos_token_id]
        text = tokenizer.decode(arr, skip_special_tokens=True)
        console.print(f"[bold]Example {idx+1}:[/] {text}")
        progress.update(task, advance=1)

# === 8. Labels -> текст (если нужно) ===
console.rule("[bold yellow]DECODED LABELS")
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
    task = progress.add_task("labels → текст", total=len(labels))
    for idx, lbl in enumerate(labels):
        arr = lbl.tolist()
        arr = [x for x in arr if x != IGNORE_INDEX]
        if tokenizer.pad_token_id is not None:
            arr = [x for x in arr if x != tokenizer.pad_token_id]
        if tokenizer.eos_token_id is not None:
            arr = [x for x in arr if x != tokenizer.eos_token_id]
        text = tokenizer.decode(arr, skip_special_tokens=True)
        console.print(f"[bold]Example {idx+1}:[/] {text}")
        progress.update(task, advance=1)