import os
import glob
import torch
import traceback
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import config
from transformers import AutoTokenizer
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

def generate_math(model: torch.nn.Module,
                  tokenizer,
                  prompt: str,
                  max_seq_len: int = None,
                  device: torch.device = None) -> str:
    model.eval()
    max_seq_len = max_seq_len or config.MAX_SEQ_LEN
    device = device or config.DEVICE
    # prepare input "q=" (without initial EOS)
    s = f"{prompt}="
    ids = tokenizer.encode(s)
    # truncate
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_seq_len - len(ids)):
            logits, *_ = model(input_ids=input_ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    output_ids = input_ids[0].cpu().tolist()
    # extract answer after '='
    eq_id = tokenizer.encode("=")[0]
    if eq_id in output_ids:
        ans_ids = output_ids[output_ids.index(eq_id)+1:]
    else:
        ans_ids = output_ids[len(ids):]
    return tokenizer.decode(ans_ids)

def generate_math_variants(model, tokenizer, prompt, max_seq_len, device, num_variants):
    variants = []
    for _ in range(num_variants):
        ans = generate_math(model, tokenizer, prompt, max_seq_len, device)
        # calculate probability of answer
        input_ids = torch.tensor([tokenizer.encode(prompt + "=" + ans)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, *_ = model(input_ids=input_ids)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob = probs[0, -1, tokenizer.encode(ans)[-1]].item()
        variants.append((ans, prob))
    return variants

def test_model(args):
    console.rule("[bold blue]Testing Math Model[/]")
    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True)
    # ensure pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.log(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    device = config.DEVICE
    model_path = os.path.join(os.getcwd(), 'spatial_graph_math_model.pth')
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
        val_dataset = MathDataset(tokenizer, config.MAX_SEQ_LEN, split='test')
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
        out = generate_math(model, tokenizer, args.prompt, config.MAX_SEQ_LEN, config.DEVICE)
        console.print(f"Prompt: {args.prompt}")
        console.print(f"Model answer: {out}")
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
        raw_full = q + "=" + a_true + tokenizer.eos_token
        full_input = repr(raw_full.encode('utf-8'))
        console.print(f"Full Input Tensor (Cleaned): {full_input}")
        # masked input view up to EOS
        ans_tokens = tokenizer.encode(a_true + tokenizer.eos_token)
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
                ans = generate_math(model, tokenizer, q, config.MAX_SEQ_LEN, device)
                # compute probability
                ids = torch.tensor([tokenizer.encode(q + "=" + ans)], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, *_ = model(input_ids=ids)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    p = probs[0, -1, tokenizer.encode(ans)[-1]].item()
                variants.append((ans, p))
                progress.update(task, advance=1)
        for i, (ans, p) in enumerate(variants, 1):
            console.log(f"[{i}] '{ans}' ({p*100:.2f}%)")
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