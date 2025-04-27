import os
import glob
import re
import traceback
import random
import torch
from rich.console import Console
from rich.progress import track
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import Dataset
from torch.nn import functional as F
import config

console = Console()

class TextDataset(Dataset):
    """
    Dataset class for pre-training (language modeling on raw text).
    Handles caching of processed data to speed up subsequent loads.
    """
    def __init__(self, file_paths, tokenizer, max_seq_len, cache_dir=".text_dataset_cache"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer_name_safe = re.sub(r'[\\/*?:"<>|]', '_', tokenizer.name_or_path)
        self.param_string = f"pretrain_tok_{tokenizer_name_safe}_seq{max_seq_len}"

        console.log(f"Processing/Loading pre-training files with cache config: {self.param_string}")
        for file_path in track(file_paths, description="Files..."):
            base_name = os.path.basename(file_path)
            cache_filename = f"{base_name}.{self.param_string}.pt"
            cache_path = os.path.join(cache_dir, cache_filename)

            cache_valid = False
            if os.path.exists(cache_path):
                try:
                    if os.path.getmtime(cache_path) >= os.path.getmtime(file_path):
                        cache_valid = True
                except OSError:
                    pass

            if cache_valid:
                try:
                    file_examples = torch.load(cache_path, weights_only=True)
                    self.examples.extend(file_examples)
                    continue
                except Exception as e:
                    console.print(f"[red]Error loading cache {cache_path}: {e}. Reprocessing {base_name}.[/red]")

            current_file_examples = []
            try:
                with console.status(f"Reading and cleaning {base_name}", spinner="dots"):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        full_text = f.read()
                    text_no_tags = re.sub(r'<[^>]+>', ' ', full_text)
                    text_no_placeholders = text_no_tags.replace(' @ @ @ @ @ @ @ @ @ @ ', ' ')
                    cleaned_text = ' '.join(text_no_placeholders.split())

                all_tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=False)
                if self.tokenizer.eos_token_id is not None:
                    all_tokens.append(self.tokenizer.eos_token_id)

                step = self.max_seq_len // 2 or 1
                indices = list(range(0, len(all_tokens) - self.max_seq_len, step))
                for i in track(indices, description=f"Chunking {base_name}", total=len(indices)):
                    chunk = all_tokens[i : i + self.max_seq_len + 1]
                    if len(chunk) == self.max_seq_len + 1:
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        current_file_examples.append({
                            "input_ids": input_ids,
                            "labels": labels
                        })

                if current_file_examples:
                    with console.status(f"Saving cache {cache_path}", spinner="dots"):
                        torch.save(current_file_examples, cache_path)
                self.examples.extend(current_file_examples)

            except Exception as file_e:
                console.print(f"[bold red]Error processing pre-training file {base_name}: {file_e}[/]")
                console.print(traceback.format_exc())

        console.log(f"Loaded/Processed {len(self.examples)} pre-training examples in total.")
        if len(self.examples) == 0 and file_paths:
            console.print("[bold red]Warning: No pre-training examples were created.[/]")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DialogDataset(Dataset):
    """
    Dataset class for fine-tuning on dialogue data (e.g., DailyDialog).
    Formats turns with special tokens and masks user input in labels.
    """
    def __init__(self, tokenizer, max_seq_len, split='train',
                 cache_dir=".dialog_dataset_cache", debug_limit=0):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = config.IGNORE_INDEX
        self.user_token_id = tokenizer.convert_tokens_to_ids(config.USER_TOKEN)
        self.assistant_token_id = tokenizer.convert_tokens_to_ids(config.ASSISTANT_TOKEN)
        self.eos_token_id = tokenizer.eos_token_id
        all_processed_examples = []
        debug_count = 0

        console.log(f"Loading and processing DailyDialog dataset ({split} split)...")
        try:
            with console.status(f"Loading DailyDialog [{split}]", spinner="dots"):
                daily_dialog_dataset = load_dataset(
                    "daily_dialog",
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
        except Exception as e:
            console.print(f"[bold red]Failed to load DailyDialog: {e}[/]")
            daily_dialog_dataset = []

        for conversation in track(daily_dialog_dataset, description="Dialogues..."):
            dialog_turns = conversation['dialog']
            for i in range(0, len(dialog_turns) - 1, 2):
                user_turn = dialog_turns[i].strip()
                assistant_turn = dialog_turns[i+1].strip()

                input_str = (
                    f"{config.USER_TOKEN} {user_turn} "
                    f"{config.ASSISTANT_TOKEN} {assistant_turn}"
                    f"{tokenizer.eos_token}"
                )
                input_ids_list = tokenizer.encode(input_str, add_special_tokens=False)
                original_length = len(input_ids_list)
                truncated = False
                if len(input_ids_list) > self.max_seq_len:
                    input_ids_list = input_ids_list[:self.max_seq_len]
                    truncated = True
                    if (input_ids_list[-1] != self.eos_token_id
                            and self.eos_token_id is not None):
                        input_ids_list[-1] = self.eos_token_id

                input_ids = torch.tensor(input_ids_list, dtype=torch.long)
                labels = input_ids.clone()

                try:
                    assistant_indices = (input_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]
                    if len(assistant_indices) > 0:
                        first_assistant_idx = assistant_indices[0].item()
                        labels[:first_assistant_idx + 1] = self.ignore_index
                    else:
                        if truncated and debug_count < debug_limit:
                            console.print(f"[yellow]DEBUG: ASSISTANT token truncated in example {input_str[:100]}...[/yellow]")
                            debug_count += 1
                        continue
                except Exception:
                    continue

                pad_len = self.max_seq_len - len(input_ids)
                if pad_len < 0:
                    continue
                if pad_len > 0:
                    input_ids = F.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                    labels = F.pad(labels, (0, pad_len), value=self.ignore_index)

                all_processed_examples.append({
                    "input_ids": input_ids,
                    "labels": labels
                })

        console.log(f"Initially processed {len(all_processed_examples)} examples.")
        pct = config.DATASET_PERCENTAGE
        if 1 <= pct < 100 and all_processed_examples:
            keep = max(1, int(len(all_processed_examples) * pct / 100.0))
            self.examples = random.sample(all_processed_examples, keep)
            console.log(f"Sampling {pct}% => {len(self.examples)} examples.")
        else:
            self.examples = all_processed_examples
            console.log(f"Using all {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]