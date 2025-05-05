import os
import glob
import torch
import traceback
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import config
# from transformers import AutoTokenizer # Заменяем
import tokenizer_utils
from model import SpatialGraphTransformer
from datasets import load_dataset
import random
import ast
import argparse
import sys

console = Console()

def evaluate(model, dataloader, crit, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits, *_ = model(input_ids)
            loss = crit(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_count += input_ids.size(0)
    model.train()
    return total_loss / total_count if total_count > 0 else float('inf')

# def generate_math(model: torch.nn.Module,
#                   tokenizer, # Убираем аргумент
#                   prompt: str,
#                   max_seq_len: int = None,
#                   device: torch.device = None) -> str:
def generate_math(model: torch.nn.Module,
                  prompt_with_equals: str, # Переименован для ясности
                  max_new_tokens: int = 50, # Макс. кол-во новых токенов для генерации
                  max_seq_len: int = None,
                  device: torch.device = None) -> str:
    """
    Генерирует продолжение последовательности, начиная с prompt_with_equals.
    Ожидается, что prompt_with_equals уже содержит 'Question='.
    Возвращает декодированную строку сгенерированных токенов (ответ + eos).
    """
    model.eval()
    max_seq_len = max_seq_len or config.MAX_SEQ_LEN
    device = device or config.DEVICE

    # Подготовка входных данных
    prepared_s = tokenizer_utils.prepare_text_for_encoding(prompt_with_equals)
    ids = tokenizer_utils.encode(prepared_s)

    # Обрезка, если входная строка слишком длинная
    if len(ids) >= max_seq_len:
        ids = ids[:max_seq_len-1] # Оставляем место хотя бы для одного нового токена

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer_utils.get_eos_token_id()

    generated_ids = [] # Храним только сгенерированные ID

    with torch.no_grad():
        current_input_ids = input_ids
        for _ in range(max_new_tokens):
            # Проверяем, не превышена ли максимальная длина последовательности
            if current_input_ids.size(1) >= max_seq_len:
                console.print(f"[bold yellow]Warning:[/bold yellow] Reached max_seq_len ({max_seq_len}) during generation.")
                break

            logits, *_ = model(input_ids=current_input_ids)
            next_id_tensor = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_id = next_id_tensor.item()

            # Добавляем сгенерированный ID в список
            generated_ids.append(next_id)

            # Обновляем input_ids для следующего шага
            current_input_ids = torch.cat((current_input_ids, next_id_tensor), dim=1)

            # Проверяем на EOS
            if eos_id is not None and next_id == eos_id:
                break
        else: # Если цикл завершился без break (не сгенерирован EOS)
             console.print(f"[bold yellow]Warning:[/bold yellow] Reached max_new_tokens ({max_new_tokens}) without generating EOS.")


    # Декодируем только сгенерированные токены
    return tokenizer_utils.decode(generated_ids)

# def generate_math_variants(model, tokenizer, prompt, max_seq_len, device, num_variants):
def generate_math_variants(model, prompt, max_seq_len, device, num_variants):
    variants = []
    for _ in range(num_variants):
        # ans = generate_math(model, tokenizer, prompt, max_seq_len, device)
        ans = generate_math(model, prompt, max_seq_len, device) # Вызываем обновленную функцию
        # calculate probability of answer
        # input_ids = torch.tensor([tokenizer.encode(prompt + "=" + ans)], dtype=torch.long, device=device)
        prepared_full_input = tokenizer_utils.prepare_text_for_encoding(prompt + "=" + ans) # Подготовка текста
        input_ids = torch.tensor([tokenizer_utils.encode(prepared_full_input)], dtype=torch.long, device=device) # Используем tokenizer_utils
        with torch.no_grad():
            logits, *_ = model(input_ids=input_ids)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # prob = probs[0, -1, tokenizer.encode(ans)[-1]].item()
            prepared_ans = tokenizer_utils.prepare_text_for_encoding(ans) # Подготовка текста
            ans_encoded = tokenizer_utils.encode(prepared_ans) # Получаем ID последнего токена ответа
            if not ans_encoded:
                 console.print(f"[bold yellow]Warning: Could not encode generated answer '{ans}' for probability calculation.[/bold yellow]")
                 prob = 0.0 # Устанавливаем вероятность 0, если ответ не кодируется
            else:
                 ans_last_token_id = ans_encoded[-1]
                 prob = probs[0, -1, ans_last_token_id].item()
        variants.append((ans, prob))
    return variants

def test_model(args):
    console.rule("[bold blue]Testing Math Model[/]")
    # initialize tokenizer and model using tokenizer_utils
    try:
        tokenizer_utils.get_tokenizer() # Инициализируем и проверяем
    except Exception as e:
        console.print(f"[bold red]Failed to initialize tokenizer via tokenizer_utils: {e}[/]")
        return
    # Ручная настройка pad_token больше не нужна, это делается в tokenizer_utils
    console.log(f"Tokenizer vocab size: {tokenizer_utils.get_vocab_size()}")
    device = config.DEVICE
    model_path = config.PRETRAINED_MODEL_SAVE_PATH
    # load checkpoint state and derive model architecture
    # load only weights to avoid pickle warnings
    state = torch.load(model_path, map_location=device, weights_only=True)
    d_model = state['token_embedding.weight'].shape[1]
    vocab_size = state['token_embedding.weight'].shape[0]
    layer_idxs = {int(k.split('.')[1]) for k in state.keys() if k.startswith('cubes.')}
    num_cubes = max(layer_idxs) + 1
    dim_feedforward = state['cubes.0.w1.weight'].shape[0]
    # instantiate model with derived parameters
    model = SpatialGraphTransformer(
        num_cubes=num_cubes,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
        vocab_size=vocab_size,
        max_distance=config.MAX_DISTANCE
    ).to(device)
    console.log(f"Instantiated model [cubes={num_cubes}, d_model={d_model}, ff={dim_feedforward}, vocab={vocab_size}]")
    model.eval()
    console.log(f"Loading model weights from {model_path}")
    # Load pretrained weights, allowing for missing keys if any
    try:
        model.load_state_dict(state, strict=False)
        console.log("Model weights loaded successfully.")
    except Exception as e:
        console.print(f"[bold red]Error loading model weights: {e}[/]")
        console.print(traceback.format_exc())
        return

    # --- Evaluate mode ---
    if args.eval:
        from torch.utils.data import DataLoader
        from math_train import MathDataset
        import torch.nn as nn
        # val_dataset = MathDataset(tokenizer, config.MAX_SEQ_LEN, split='test') # MathDataset больше не принимает tokenizer
        val_dataset = MathDataset(config.MAX_SEQ_LEN, split='test')
        val_dl = DataLoader(
            val_dataset,
            batch_size=config.PRETRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(config.DEVICE.type=="cuda"),
            prefetch_factor=2 if 0>0 else None
        )
        crit = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        val_loss = evaluate(model, val_dl, crit, config.DEVICE)
        console.print(f"[Validation] Loss: {val_loss:.4f}", style="bold blue")
        return

    # --- Prompt mode ---
    if args.prompt:
        # В режиме --prompt генерируем ответ, добавляя '='
        prompt_with_equals = args.prompt + "="
        # Устанавливаем разумное количество новых токенов для генерации ответа
        max_new_tokens_prompt = 100
        out = generate_math(model, prompt_with_equals, max_new_tokens=max_new_tokens_prompt, max_seq_len=config.MAX_SEQ_LEN, device=config.DEVICE)
        console.print(f"Prompt: {args.prompt}")
        console.print(f"Model answer: {out.strip()}") # Убираем возможные пробелы в начале/конце
        return

    # --- Sample N random examples from validation set (default N=3) ---
    N = args.sample if args.sample else 3
    from datasets import load_dataset
    import ast
    raw = load_dataset(
        "math_dataset",
        "arithmetic__add_or_sub",
        split="test",
        cache_dir=config.PRETRAIN_LEARNING_DIR,
        trust_remote_code=True
    )
    total = len(raw)
    indices = random.sample(range(total), N)
    for idx in indices:
        # parse and clean dataset question
        raw_q = raw['question'][idx]
        console.print(f"[bold yellow]DEBUG:[/bold yellow] Raw question before cleaning: {repr(raw_q)}")
        if isinstance(raw_q, (bytes, bytearray)):
            q = raw_q.decode('utf-8').replace('\n', '').strip()
        elif isinstance(raw_q, str) and raw_q.startswith("b'"):
            try:
                tmp = ast.literal_eval(raw_q)
                q = tmp.decode('utf-8').replace('\n', '').strip()
            except Exception:
                q = raw_q.strip().strip("b'")
        else:
            q = raw_q.replace('\n', '').strip() if isinstance(raw_q, str) else str(raw_q)
        console.print(f"[bold yellow]DEBUG:[/bold yellow] Cleaned question after removing newlines: {repr(q)}")
        # parse and clean dataset answer
        raw_a = raw['answer'][idx]
        if isinstance(raw_a, (bytes, bytearray)):
            a_true = raw_a.decode('utf-8').replace('\n', ' ').strip()
        elif isinstance(raw_a, str) and raw_a.startswith("b'"):
            try:
                tmp = ast.literal_eval(raw_a)
                a_true = tmp.decode('utf-8').replace('\n', ' ').strip()
            except Exception:
                a_true = raw_a.strip().strip("b'")
        else:
            a_true = raw_a.replace('\n', ' ').strip() if isinstance(raw_a, str) else str(raw_a)
        console.print(f"[bold cyan]Question:[/bold cyan] {q}")
        # full input with actual answer and EOS
        # raw_full = q + "=" + a_true + tokenizer.eos_token
        eos_token = tokenizer_utils.get_tokenizer().eos_token # Получаем сам токен EOS
        raw_full = q + "=" + a_true + (eos_token if eos_token else "") # Добавляем, если он есть
        full_input = repr(raw_full.encode('utf-8'))
        console.print(f"Full Input Tensor (Cleaned): {full_input}")
        # masked input view up to EOS
        # ans_tokens = tokenizer.encode(a_true + tokenizer.eos_token)
        ans_tokens = tokenizer_utils.encode(a_true + (eos_token if eos_token else "")) # Используем tokenizer_utils
        mask_str = "[PREDICT]" * len(ans_tokens)
        masked_input = repr((q + "=").encode('utf-8')) + mask_str
        console.print(f"Masked Input (Model View): {masked_input}")
        # generate 5 answer variants with Rich progress bar
        num_variants = 5
        variants = []
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task("Generating answer variants", total=num_variants)
            for _ in range(num_variants):
                # Готовим вход для генерации: вопрос + знак равенства
                prompt_for_generation = q + "="
                # Генерируем только ответ (Answer + EOS)
                # Устанавливаем разумное ограничение на длину ответа, например 50 токенов
                max_new_tokens_test = 50
                generated_answer = generate_math(model, prompt_for_generation, max_new_tokens=max_new_tokens_test, max_seq_len=config.MAX_SEQ_LEN, device=device)

                # --- Вычисление вероятности (оставляем как есть, если нужно) ---
                # Собираем полный ввод: вопрос + '=' + сгенерированный ответ
                full_generated_sequence = prompt_for_generation + generated_answer
                prepared_full_input_var = tokenizer_utils.prepare_text_for_encoding(full_generated_sequence)
                ids = torch.tensor([tokenizer_utils.encode(prepared_full_input_var)], dtype=torch.long, device=device)

                # Получаем ID последнего токена сгенерированного ответа
                prepared_ans_var = tokenizer_utils.prepare_text_for_encoding(generated_answer)
                ans_encoded_var = tokenizer_utils.encode(prepared_ans_var)

                p = 0.0 # Вероятность по умолчанию
                if ans_encoded_var: # Если ответ не пустой и кодируется
                    # Убедимся, что ids не пустой и имеет достаточную длину
                    if ids.numel() > 0 and ids.size(1) > 1:
                         with torch.no_grad():
                             logits, *_ = model(input_ids=ids)
                             # Проверяем размерность logits перед вычислением softmax
                             if logits.numel() > 0 and logits.dim() == 3:
                                 probs = torch.nn.functional.softmax(logits, dim=-1)
                                 # Индекс последнего токена в исходной последовательности ids
                                 last_token_index = ids.size(1) - 1
                                 # ID последнего токена сгенерированного ответа
                                 ans_last_token_id = ans_encoded_var[-1]
                                 # Извлекаем вероятность последнего токена ответа
                                 # Убедимся, что индекс не выходит за границы
                                 if last_token_index < probs.size(1) and ans_last_token_id < probs.size(2):
                                      p = probs[0, last_token_index, ans_last_token_id].item()
                                 else:
                                      console.print(f"[bold yellow]Warning:[/bold yellow] Index out of bounds during probability calculation for '{generated_answer}'.")
                             else:
                                  console.print(f"[bold yellow]Warning:[/bold yellow] Invalid logits shape for probability calculation: {logits.shape}")
                    else:
                         console.print(f"[bold yellow]Warning:[/bold yellow] Empty or too short input tensor for probability calculation.")

                else:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Could not encode generated answer variant '{generated_answer}' for probability calculation.")

                variants.append((generated_answer.strip(), p)) # Сохраняем очищенный ответ
                progress.update(task, advance=1)
        # Выводим варианты сгенерированных ответов
        for i, (ans, p) in enumerate(variants, 1):
            console.log(f"[{i}] Generated: '{ans}' (Prob: {p*100:.2f}%)")
        console.print(f"[green]True answer:[/green] {a_true}")
        console.print("-"*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or interact with the math model.")
    parser.add_argument('--prompt', type=str, help='Custom math question to generate answer for')
    parser.add_argument('--eval', action='store_true', help='Evaluate loss on test set')
    parser.add_argument('--sample', type=int, help='Sample N random test examples')
    args = parser.parse_args()
    try:
        test_model(args)
    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user (Ctrl+C). Exiting.[/bold yellow]")
        sys.exit(0)
