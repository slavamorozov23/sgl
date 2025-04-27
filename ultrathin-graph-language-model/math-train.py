import os
import re
import random
import traceback
from rich.console import Console
from rich.progress import track
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config
from model import SpatialGraphTransformer
from train import train

console = Console()

PAD_TOKEN = '[PAD]'
EOS_TOKEN = '[EOS]'
UNK_TOKEN = '[UNK]'
math_chars = '0123456789+-= '
special_tokens = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
vocabulary = special_tokens + list(math_chars)
if len(vocabulary) != config.VOCAB_SIZE:
    console.print(f"[bold red]CRITICAL ERROR: vocab size mismatch {len(vocabulary)} != {config.VOCAB_SIZE}[/]")
    exit(1)

class MathTokenizer:
    def __init__(self, vocab):
        self.char_to_id = {c:i for i,c in enumerate(vocab)}
        self.id_to_char = {i:c for i,c in enumerate(vocab)}
        self._vocab_size = len(vocab)
        self.pad_token_id = self.char_to_id[PAD_TOKEN]
        self.eos_token_id = self.char_to_id[EOS_TOKEN]
        self.unk_token_id = self.char_to_id[UNK_TOKEN]
        self.name_or_path = config.TOKENIZER_NAME
    @property
    def vocab_size(self):
        return self._vocab_size
    def encode(self, text, add_special_tokens=False):
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]
    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self.id_to_char.get(i, UNK_TOKEN))
        return "".join(out)

def _convert_example(example):
    ids = example.get('input_ids', [])
    labels = example.get('labels', [])
    if not ids or not labels:
        return None
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

class MathDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len, split='train', cache_dir=".math_dataset_cache"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = config.IGNORE_INDEX
        os.makedirs(cache_dir, exist_ok=True)
        safe = re.sub(r'[\\/*?:"<>|]', '_', tokenizer.name_or_path)
        fname = f"math_{split}_tok_{safe}_seq{max_seq_len}.pt"
        self.cache_path = os.path.join(cache_dir, fname)

        all_examples = []
        load_cache = False
        if os.path.exists(self.cache_path):
            try:
                try:
                    all_examples = torch.load(self.cache_path, map_location='cpu', weights_only=True)
                except TypeError:
                    all_examples = torch.load(self.cache_path, map_location='cpu')
                console.log(f"Loaded {len(all_examples)} examples from cache.")
                load_cache = True
            except Exception as e:
                console.print(f"[red]Cache load error {self.cache_path}: {e}. Reprocessing.[/red]")

        if not load_cache:
            console.log(f"Processing math dataset split='{split}'…")
            raw = None
            with console.status("Loading raw math dataset…", spinner="dots"):
                try:
                    raw = load_dataset(
                        "math_dataset",
                        "arithmetic__add_or_sub",
                        split=split,
                        cache_dir=config.PRETRAIN_LEARNING_DIR,
                        trust_remote_code=True
                    )
                except Exception as e:
                    console.print(f"[red]Failed to load raw dataset: {e}[/red]")
            if raw is None:
                console.print("[bold red]No raw dataset. Exiting[/]")
                self.examples = []
                return

            fn_kwargs = {
                'tokenizer': tokenizer,
                'max_seq_len': max_seq_len,
                'ignore_index': self.ignore_index
            }
            num_proc = min(8, os.cpu_count()//2)
            with console.status("Mapping and preprocessing math dataset…", spinner="dots"):
                processed = raw.map(
                    process_batch_for_math,
                    batched=True,
                    num_proc=num_proc,
                    fn_kwargs=fn_kwargs,
                    remove_columns=raw.column_names
                )
            console.log("Converting to tensors…")
            examples = []
            for ex in track(processed, total=len(processed), description="To tensors…"):
                r = _convert_example(ex)
                if r:
                    examples.append(r)
            all_examples = examples
            console.log(f"Converted {len(all_examples)} examples.")
            with console.status(f"Saving cache to {self.cache_path}", spinner="dots"):
                try:
                    torch.save(all_examples, self.cache_path)
                    console.log(f"Saved cache to {self.cache_path}")
                except Exception as e:
                    console.print(f"[red]Error saving cache: {e}[/red]")

        total = len(all_examples)
        pct = config.DATASET_PERCENTAGE
        if pct<100 and total>0:
            keep = max(1, total*pct//100)
            self.examples = random.sample(all_examples, keep)
            console.log(f"Sampling {pct}% => {len(self.examples)}/{total}")
        else:
            self.examples = all_examples
            console.log(f"Using all {total} examples")

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def process_batch_for_math(batch, tokenizer, max_seq_len, ignore_index):
    out = {'input_ids':[], 'labels':[]}
    qs, ans = batch['question'], batch['answer']
    eq_id = tokenizer.encode("=")[0]
    for q,a in zip(qs,ans):
        s = f"{q}={a}{tokenizer.eos_token}"
        ids = tokenizer.encode(s)
        if len(ids)>max_seq_len:
            ids = ids[:max_seq_len]
            if ids[-1]!=tokenizer.eos_token_id:
                ids[-1]=tokenizer.eos_token_id
        arr = np.array(ids, dtype=np.int64)
        lbl = arr.copy()
        idxs = np.where(arr==eq_id)[0]
        if idxs.size>0:
            lbl[:idxs[0]+1]=ignore_index
        else:
            lbl[:]=ignore_index
        pad = max_seq_len-len(arr)
        if pad>0:
            arr = np.pad(arr, (0,pad), 'constant', constant_values=tokenizer.pad_token_id)
            lbl = np.pad(lbl, (0,pad), 'constant', constant_values=ignore_index)
        out['input_ids'].append(arr)
        out['labels'].append(lbl)
    return out

if __name__ == "__main__":
    try:
        console.rule("[bold blue]Initializing Math Training[/]")
        tokenizer = MathTokenizer(vocabulary)
        console.log(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        model = SpatialGraphTransformer(
            num_cubes=config.NUM_CUBES,
            d_model=config.D_MODEL,
            dim_feedforward=config.DIM_FEEDFORWARD,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
            vocab_size=config.VOCAB_SIZE,
            max_distance=config.MAX_DISTANCE
        ).to(config.DEVICE)
        console.log(f"Model on {config.DEVICE}")
        console.log(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

        dataset = MathDataset(tokenizer, config.MAX_SEQ_LEN, split='train')
        if len(dataset)==0:
            console.print("[bold red]Empty dataset. Exiting[/]")
            exit(1)

        workers = min(os.cpu_count()//2, 8)
        dl = DataLoader(
            dataset,
            batch_size=config.PRETRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=workers,
            pin_memory=(config.DEVICE.type=="cuda"),
            prefetch_factor=2 if workers>0 else None
        )
        console.log(f"Dataloader ready: {len(dl)} batches")

        optim = torch.optim.AdamW(model.parameters(), lr=config.PRETRAIN_LEARNING_RATE)
        crit = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

        console.rule("[bold blue]Start Training[/]")
        for epoch in range(config.PRETRAIN_EPOCHS):
            loss = train(model, dl, optim, crit, epoch, config.PRETRAIN_EPOCHS, config.DEVICE)
            console.log(f"Epoch {epoch+1}/{config.PRETRAIN_EPOCHS} avg loss: {loss:.4f}", style="bold green")
            try:
                torch.save(model.state_dict(), config.PRETRAINED_MODEL_SAVE_PATH)
            except Exception as e:
                console.print(f"[red]Error saving model: {e}[/red]")
        console.rule("[bold blue]Training Finished[/]")
        console.log(f"Model saved to {config.PRETRAINED_MODEL_SAVE_PATH}")

    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        console.print(traceback.format_exc())
        exit(1)