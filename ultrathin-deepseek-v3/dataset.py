import torch
from torch.utils.data import Dataset
import os
import glob
from rich.console import Console
from rich.progress import track
import traceback
import re
import time
from datasets import load_dataset, Features, Value, Sequence
from torch.nn import functional as F
import random

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
        has_eos = self.tokenizer.eos_token_id is not None

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        tokenizer_name_safe = re.sub(r'[\\/*?:"<>|]', '_', tokenizer.name_or_path)
        self.param_string = f"pretrain_tok_{tokenizer_name_safe}_seq{max_seq_len}"

        console.log(f"Processing/Loading pre-training files with cache config: {self.param_string}")
        for file_path in track(file_paths, description="Processing/Loading pre-training files..."):
            base_name = os.path.basename(file_path)
            cache_filename = f"{base_name}.{self.param_string}.pt"
            cache_path = os.path.join(self.cache_dir, cache_filename)

            cache_valid = False
            if os.path.exists(cache_path):
                try:
                    if os.path.getmtime(cache_path) >= os.path.getmtime(file_path):
                        cache_valid = True
                except OSError:
                    pass

            if cache_valid:
                try:
                    # Load cached data, explicitly setting weights_only=True for security
                    file_examples = torch.load(cache_path, weights_only=True)
                    self.examples.extend(file_examples)
                    continue
                except Exception as e:
                    console.print(f"\n[red]Error loading pre-training cache file {cache_path}: {e}. Reprocessing {base_name}.[/red]", justify="right")
                    cache_valid = False

            current_file_examples = []
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()

                # Basic cleaning: remove HTML-like tags and specific placeholders
                text_no_tags = re.sub(r'<[^>]+>', ' ', full_text)
                text_no_placeholders = text_no_tags.replace(' @ @ @ @ @ @ @ @ @ @ ', ' ')
                # Normalize whitespace
                cleaned_text = ' '.join(text_no_placeholders.split())
                all_tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=False)

                # Append EOS token if the tokenizer has one
                if has_eos:
                     all_tokens.append(self.tokenizer.eos_token_id)

                # Use a step half the sequence length for overlapping chunks
                step = self.max_seq_len // 2
                if step == 0: step = 1 # Ensure step is at least 1

                # Create overlapping sequences for language modeling
                for i in range(0, len(all_tokens) - self.max_seq_len, step):
                     chunk = all_tokens[i : i + self.max_seq_len + 1]
                     # Ensure chunk has the exact required length (seq_len + 1 for label)
                     if len(chunk) == self.max_seq_len + 1:
                          input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                          labels = torch.tensor(chunk[1:], dtype=torch.long) # Labels are shifted input_ids
                          current_file_examples.append({
                              "input_ids": input_ids,
                              "labels": labels
                          })

                self.examples.extend(current_file_examples)

                if current_file_examples:
                    try:
                        torch.save(current_file_examples, cache_path)
                    except Exception as e:
                        console.print(f"\n[red]Error saving pre-training cache file {cache_path}: {e}[/red]", justify="right")

            except Exception as file_e:
                console.print(f"[bold red]Error processing pre-training file {file_path}: {file_e}[/]")
                console.print(traceback.format_exc())

        console.log(f"Loaded/Processed {len(self.examples)} pre-training examples in total.")
        if len(self.examples) == 0 and file_paths:
             console.print("[bold red]Warning: No pre-training examples were created.[/]")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the pre-processed example (dictionary of tensors)
        return self.examples[idx]


class DialogDataset(Dataset):
    """
    Dataset class for fine-tuning on dialogue data (e.g., DailyDialog).
    Formats turns with special tokens and masks user input in labels.
    Allows using a percentage of the dataset. Includes optional debugging output.
    """
    def __init__(self, tokenizer, max_seq_len, split='train', cache_dir=".dialog_dataset_cache", debug_limit=0): # Set debug_limit=0 to disable debug prints by default
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = config.IGNORE_INDEX
        self.user_token = config.USER_TOKEN
        self.assistant_token = config.ASSISTANT_TOKEN
        all_processed_examples = []
        self.debug_limit = debug_limit

        # Ensure special tokens exist in tokenizer
        try:
            self.user_token_id = tokenizer.convert_tokens_to_ids(self.user_token)
            self.assistant_token_id = tokenizer.convert_tokens_to_ids(self.assistant_token)
            if self.user_token_id == tokenizer.unk_token_id or self.assistant_token_id == tokenizer.unk_token_id:
                 raise ValueError("Special tokens (<USER>, <ASSISTANT>) not found in tokenizer vocabulary.")
        except Exception as e:
             console.print(f"[bold red]Error getting special token IDs: {e}.[/]")
             raise e

        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
             console.print("[bold yellow]Warning: EOS token ID not found in tokenizer.[/]")

        console.log(f"Loading and processing DailyDialog dataset ({split} split)...")
        try:
            daily_dialog_dataset = load_dataset(
                "daily_dialog",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True # Required for some datasets
            )
        except Exception as e:
            console.print(f"[bold red]Failed to load DailyDialog: {e}.[/]")
            console.print("[bold yellow]Attempting to continue without DailyDialog data.[/]")
            daily_dialog_dataset = [] # Use an empty list if loading fails

        debug_count = 0
        for conversation in track(daily_dialog_dataset, description=f"Processing {split} dialogues..."):
            dialog_turns = conversation['dialog']
            # Process pairs of turns (User -> Assistant)
            for i in range(0, len(dialog_turns) - 1, 2):
                user_turn = dialog_turns[i].strip()
                assistant_turn = dialog_turns[i+1].strip()

                input_str = f"{self.user_token} {user_turn} {self.assistant_token} {assistant_turn}{tokenizer.eos_token}"
                input_ids_list = tokenizer.encode(input_str, add_special_tokens=False)

                original_length = len(input_ids_list)
                truncated = False
                if len(input_ids_list) > self.max_seq_len:
                    input_ids_list = input_ids_list[:self.max_seq_len]
                    truncated = True
                    # Ensure the last token is EOS if truncated and EOS exists
                    if input_ids_list[-1] != self.eos_token_id and self.eos_token_id is not None:
                         input_ids_list[-1] = self.eos_token_id

                input_ids = torch.tensor(input_ids_list, dtype=torch.long)
                labels = input_ids.clone()

                try:
                     # Find the index of the first occurrence of the assistant token
                     assistant_indices = (input_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]
                     if len(assistant_indices) > 0:
                          first_assistant_idx = assistant_indices[0].item()
                          # Mask tokens up to and including the <ASSISTANT> token in the labels
                          labels[:first_assistant_idx + 1] = self.ignore_index
                     else:
                          # If assistant token is not found (e.g., truncated out)
                          if truncated and debug_count < self.debug_limit:
                              console.print(f"[yellow]DEBUG: Skipping example (ASSISTANT token truncated?):[/yellow]\nInput Str: {input_str[:200]}...\nInput IDs (truncated): {input_ids.tolist()}")
                              debug_count += 1
                          continue
                except Exception:
                     # Skip if any error occurs during token finding
                     continue

                pad_len = self.max_seq_len - len(input_ids)
                padded = False
                if pad_len < 0:
                    # Safeguard against unexpected negative padding length
                    continue
                if pad_len > 0:
                    input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                    labels = F.pad(labels, (0, pad_len), value=self.ignore_index)
                    padded = True

                # Add the example if it matches the max sequence length after processing
                if len(input_ids) == self.max_seq_len:
                     all_processed_examples.append({"input_ids": input_ids, "labels": labels})
                     # Optional Debug Output
                     if debug_count < self.debug_limit:
                         console.print("\n" + "="*10 + f" DEBUG EXAMPLE {debug_count+1} " + "="*10)
                         console.print(f"Original Str Len: {len(input_str)}, Tokenized Len: {original_length}, Truncated: {truncated}, Padded: {padded}")
                         console.print(f"Input Str (part): {input_str[:200]}...")
                         console.print(f"Input IDs: {input_ids.tolist()}")
                         console.print(f"Labels:    {labels.tolist()}")
                         decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
                         readable_labels = labels.clone()
                         readable_labels[readable_labels == self.ignore_index] = tokenizer.pad_token_id
                         decoded_labels = tokenizer.decode(readable_labels, skip_special_tokens=False)
                         console.print(f"Decoded Input:\n'''{decoded_input}'''")
                         console.print(f"Decoded Labels (Target):\n'''{decoded_labels}'''")
                         console.print("="*36)
                         debug_count += 1

        processed_examples_count = len(all_processed_examples)
        console.log(f"Initially processed {processed_examples_count} User->Assistant examples from {split} split.")

        # Sample a percentage of the data if specified in config
        percentage_to_use = config.FINETUNE_DATA_PERCENTAGE
        if not (1 <= percentage_to_use <= 100):
            console.print(f"[bold yellow]Warning: FINETUNE_DATA_PERCENTAGE ({percentage_to_use}) is outside the valid range [1, 100]. Using 100%.[/]")
            percentage_to_use = 100

        if percentage_to_use < 100 and processed_examples_count > 0:
            num_to_keep = max(1, int(processed_examples_count * (percentage_to_use / 100.0)))
            console.log(f"Sampling {percentage_to_use}% of the data ({num_to_keep} examples)...")
            self.examples = random.sample(all_processed_examples, num_to_keep)
            console.log(f"Using {len(self.examples)} examples for fine-tuning.")
        elif processed_examples_count > 0:
             self.examples = all_processed_examples
             console.log(f"Using all {processed_examples_count} processed examples for fine-tuning (100%).")
        else:
             self.examples = []
             # Handle the case where no examples were processed
             console.print(f"[bold red]Warning: No examples were created from {split} split. Cannot sample.[/]")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the processed example (dictionary containing padded/masked tensors)
        return self.examples[idx]