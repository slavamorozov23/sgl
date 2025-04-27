import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import glob
import os
from rich.console import Console
from typing import Optional
import traceback

# Use pre-training specific parameters from config
from config import (
    TOKENIZER_NAME, D_MODEL, NHEAD, NUM_LAYERS,
    DIM_FEEDFORWARD, DROPOUT, MAX_SEQ_LEN, DEVICE,
    PRETRAIN_BATCH_SIZE, PRETRAIN_LEARNING_RATE, PRETRAIN_EPOCHS,
    PRETRAIN_LEARNING_DIR, PRETRAINED_MODEL_SAVE_PATH
)
from model import SimpleMQA_Transformer
from dataset import TextDataset # Only TextDataset needed for pre-training
from train import train # Only train function needed

console = Console()

if __name__ == "__main__":
    try:
        console.rule("[bold blue]Initializing Language Model Pre-training[/]")

        # 1. Load Tokenizer
        console.log(f"Loading tokenizer: {TOKENIZER_NAME}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if tokenizer.eos_token is None:
                if tokenizer.sep_token is not None: tokenizer.eos_token = tokenizer.sep_token
                else: tokenizer.add_special_tokens({'eos_token': '[EOS]'})

            VOCAB_SIZE = tokenizer.vocab_size # Base vocab size for pre-training model init
            console.log(f"Tokenizer loaded. Base vocab size: {VOCAB_SIZE}")
        except Exception as e:
            console.print(f"[bold red]Error loading tokenizer: {e}[/]")
            exit(1)

        if VOCAB_SIZE is None:
             console.print("[bold red]Error: VOCAB_SIZE not determined after tokenizer loading.[/]")
             exit(1)

        # 2. Load Pre-training Data
        console.log(f"Loading pre-training data from: {PRETRAIN_LEARNING_DIR}")
        learning_files = glob.glob(os.path.join(PRETRAIN_LEARNING_DIR, "*.txt"))
        if not learning_files:
            console.print(f"[bold red]No .txt files found in {PRETRAIN_LEARNING_DIR}[/]")
            exit(1)
        dataset = TextDataset(learning_files, tokenizer, MAX_SEQ_LEN)
        if len(dataset) == 0:
             console.print("[bold red]Pre-training dataset is empty. Exiting.[/]")
             exit(1)
        dataloader = DataLoader(dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)

        # 3. Initialize Model
        console.log("Initializing custom MQA transformer model for pre-training...")
        model = SimpleMQA_Transformer(
            d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD, max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT, vocab_size=VOCAB_SIZE
        ).to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.log(f"Model initialized on [bold green]{DEVICE}[/].")
        console.log(f"Trainable parameters (pre-training): {num_params / 1_000_000:.2f} M")

        # 4. Setup Optimizer and Criterion for Pre-training
        optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)

        # 5. Pre-training Loop
        console.rule("[bold blue]Starting Pre-training[/]")
        for epoch in range(PRETRAIN_EPOCHS):
            avg_loss = train(model, dataloader, optimizer, criterion, epoch, PRETRAIN_EPOCHS, DEVICE)
            console.log(f"Pre-training Epoch {epoch+1}/{PRETRAIN_EPOCHS} completed. Average Loss: {avg_loss:.4f}", style="bold green")
            console.log(f"Saving pre-trained model checkpoint to {PRETRAINED_MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), PRETRAINED_MODEL_SAVE_PATH)

        console.rule("[bold blue]Pre-training Finished[/]")
        console.log(f"Pre-trained model saved to {PRETRAINED_MODEL_SAVE_PATH}")

        console.rule("[bold blue]Pre-training Script Finished[/]")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Pre-training interrupted by user (Ctrl+C).[/]")
        exit()
    except Exception as main_e:
        console.print(f"[bold red]An unexpected error occurred during pre-training: {main_e}[/]")
        console.print(traceback.format_exc())
        exit(1)