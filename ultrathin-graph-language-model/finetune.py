import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import os
from rich.console import Console
import traceback

from config import (
    TOKENIZER_NAME, D_MODEL, NHEAD,
    NUM_CUBES, MAX_DISTANCE,
    DIM_FEEDFORWARD, DROPOUT, MAX_SEQ_LEN, DEVICE,
    PRETRAIN_BATCH_SIZE, PRETRAIN_LEARNING_RATE, PRETRAIN_EPOCHS, PRETRAINED_MODEL_SAVE_PATH,
    DATASET_PERCENTAGE,
    USER_TOKEN, ASSISTANT_TOKEN, IGNORE_INDEX, ROUTING_SAFETY_LIMIT
)
from model import SpatialGraphTransformer
from dataset import DialogDataset
from train import train

console = Console()

if __name__ == "__main__":
    try:
        console.rule("[bold blue]Initializing Dialogue Fine-tuning (Spatial Graph Model)[/]")

        console.log(f"Loading tokenizer: {TOKENIZER_NAME}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            special_tokens_dict = {'additional_special_tokens': [USER_TOKEN, ASSISTANT_TOKEN]}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            console.log(f"Added {num_added_toks} special tokens: {USER_TOKEN}, {ASSISTANT_TOKEN}")

            if tokenizer.pad_token is None:
                 if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                 else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 console.log(f"Set PAD token to: {tokenizer.pad_token}")
            if tokenizer.eos_token is None:
                 if tokenizer.sep_token is not None: tokenizer.eos_token = tokenizer.sep_token
                 else: tokenizer.add_special_tokens({'eos_token': '[EOS]'})
                 console.log(f"Set EOS token to: {tokenizer.eos_token}")

            final_vocab_size = len(tokenizer)
            console.log(f"Tokenizer loaded and updated. New vocab size: {final_vocab_size}")

        except Exception as e:
            console.print(f"[bold red]Error loading or updating tokenizer: {e}[/]")
            console.print(traceback.format_exc())
            exit(1)

        console.log("Initializing Spatial Graph Transformer model structure for fine-tuning...")
        model = SpatialGraphTransformer(
            num_cubes=NUM_CUBES,
            d_model=D_MODEL,
            dim_feedforward=DIM_FEEDFORWARD,
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT,
            vocab_size=final_vocab_size,
            max_distance=MAX_DISTANCE
        ).to(DEVICE)
        console.log(f"Model initialized with final vocab size {final_vocab_size}")
        console.log(f"Architecture: {NUM_CUBES} Cubes, Max Distance: {MAX_DISTANCE}, Heads/Cube: {NHEAD}, Safety Limit: {ROUTING_SAFETY_LIMIT}")

        console.log("[bold blue]Skipping pre-trained weights. Fine-tuning from scratch on Daily Dialog dataset.[/]")

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.log(f"Model ready for fine-tuning on [bold green]{DEVICE}[/].")
        console.log(f"Trainable parameters: {num_params / 1_000_000:.2f} M")

        console.log("Loading fine-tuning (dialogue) data...")
        try:
            dataset = DialogDataset(tokenizer, MAX_SEQ_LEN, split='train', debug_limit=0)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize DialogDataset: {e}[/]")
            exit(1)

        if len(dataset) == 0:
             console.print("[bold red]Fine-tuning dataset is empty. Exiting.[/]")
             exit(1)
        dataloader = DataLoader(dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)
        num_batches = len(dataloader)
        if num_batches == 0:
            console.print("[bold red]Finetuning dataloader is empty. Check dataset processing and batch size.[/]")
            exit(1)

        optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        console.rule("[bold blue]Starting Dialogue Fine-tuning[/]")
        for epoch in range(PRETRAIN_EPOCHS):
            avg_loss = train(model, dataloader, optimizer, criterion, epoch, PRETRAIN_EPOCHS, DEVICE)
            console.log(f"Fine-tuning Epoch {epoch+1}/{PRETRAIN_EPOCHS} completed. Average Combined Loss: {avg_loss:.4f}", style="bold green")
            console.log(f"Saving model checkpoint to {PRETRAINED_MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), PRETRAINED_MODEL_SAVE_PATH)

        console.rule("[bold blue]Fine-tuning Finished[/]")
        console.log(f"Model saved to {PRETRAINED_MODEL_SAVE_PATH}")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Fine-tuning interrupted by user (Ctrl+C).[/]")
        exit()
    except Exception as main_e:
        console.print(f"[bold red]An unexpected error occurred during fine-tuning: {main_e}[/]")
        console.print(traceback.format_exc())
        exit(1)