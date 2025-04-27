import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from rich.console import Console
import traceback

import config
from model import SimpleMQA_Transformer
from dataset import DialogDataset
from train import train

console = Console()

if __name__ == "__main__":
    try:
        console.rule("[bold blue]Initializing Dialogue Fine-tuning[/]")

        # 1. Load Tokenizer & Add Special Tokens
        console.log(f"Loading tokenizer: {config.TOKENIZER_NAME}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
            special_tokens_dict = {'additional_special_tokens': [config.USER_TOKEN, config.ASSISTANT_TOKEN]}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            console.log(f"Added {num_added_toks} special tokens: {config.USER_TOKEN}, {config.ASSISTANT_TOKEN}")

            if tokenizer.pad_token is None:
                 if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
                 else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 console.log(f"Set PAD token to: {tokenizer.pad_token}")
            if tokenizer.eos_token is None:
                 if tokenizer.sep_token is not None: tokenizer.eos_token = tokenizer.sep_token
                 else: tokenizer.add_special_tokens({'eos_token': '[EOS]'})
                 console.log(f"Set EOS token to: {tokenizer.eos_token}")

            final_vocab_size = len(tokenizer) # Get vocab size AFTER adding tokens
            console.log(f"Tokenizer loaded and updated. New vocab size: {final_vocab_size}")

        except Exception as e:
            console.print(f"[bold red]Error loading or updating tokenizer: {e}[/]")
            console.print(traceback.format_exc())
            exit(1)

        # 2. Initialize Model with FINAL vocabulary size
        console.log("Initializing custom MQA transformer model structure...")
        model = SimpleMQA_Transformer(
            d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD, max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
            vocab_size=final_vocab_size # Use the final vocab size here
        ).to(config.DEVICE)
        console.log(f"Model initialized with final vocab size {final_vocab_size}")

        # 3. Load Pre-trained Weights (Filtering incompatible layers)
        console.log(f"Loading pre-trained weights from: {config.PRETRAINED_MODEL_SAVE_PATH}")
        try:
            # Load the state_dict from the pre-trained model file
            pretrained_state_dict = torch.load(config.PRETRAINED_MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True)
            # Get the state_dict of the current model (initialized with the final vocab size)
            current_model_state_dict = model.state_dict()

            # Create a new state_dict to load, filtering weights
            new_state_dict = {}
            loaded_count = 0
            skipped_layers_info = []
            warn_about_embeddings = False

            for name, param in pretrained_state_dict.items():
                if name in current_model_state_dict:
                    # Check if shapes match
                    if current_model_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                        loaded_count += 1
                    else:
                        # Shape mismatch: expected for embedding/output layers
                        skipped_layers_info.append(f"{name} (Shape Pretrained: {param.shape} vs Finetune: {current_model_state_dict[name].shape})")
                        if 'token_embedding' in name or 'fc_out' in name:
                             warn_about_embeddings = True
                        else:
                             # Mismatch in other layers is more concerning
                             console.print(f"[bold yellow]Warning:[/bold yellow] Shape mismatch for non-embedding/output layer '{name}'. Pretrained: {param.shape}, Current: {current_model_state_dict[name].shape}. Skipping.")
                else:
                    # Layer from pretrained not found in current model
                     skipped_layers_info.append(f"{name} (Not found in current model architecture)")
                     console.print(f"[bold yellow]Warning:[/bold yellow] Pretrained layer '{name}' not found in current model. Skipping.")

            # Load the filtered state_dict. strict=False allows missing/extra keys.
            model.load_state_dict(new_state_dict, strict=False)
            console.log(f"Pre-trained weights loaded for {loaded_count} layers.")

            if warn_about_embeddings:
                console.log("[yellow]Skipped loading embedding/output layers due to vocabulary size difference. These layers will be fine-tuned from their new initialization.[/yellow]")
            # Print other skipped layers if any occurred for different reasons
            other_skipped = [info for info in skipped_layers_info if not ('token_embedding' in info or 'fc_out' in info)]
            if other_skipped:
                 console.log(f"[yellow]Other skipped layers during loading:[/yellow] {other_skipped}")

            if loaded_count == 0 and os.path.exists(config.PRETRAINED_MODEL_SAVE_PATH):
                 console.print(f"[bold red]Warning: No layers were successfully loaded from {config.PRETRAINED_MODEL_SAVE_PATH}, although the file exists. Check architecture compatibility. Training from scratch.[/]")

        except FileNotFoundError:
             console.print(f"[bold yellow]Pre-trained model file not found at {config.PRETRAINED_MODEL_SAVE_PATH}. Starting fine-tuning from scratch.[/]")
             # Model keeps its random initialization
        except Exception as e:
             console.print(f"[bold red]Error loading pre-trained model weights: {e}.[/]")
             console.print(traceback.format_exc())
             exit(1) # Exit if loading fails unexpectedly

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.log(f"Model ready for fine-tuning on [bold green]{config.DEVICE}[/].")
        console.log(f"Trainable parameters: {num_params / 1_000_000:.2f} M")

        # 4. Load Fine-tuning Data
        console.log("Loading fine-tuning (dialogue) data...")
        try:
            # Pass the tokenizer which now includes special tokens
            dataset = DialogDataset(tokenizer, config.MAX_SEQ_LEN, split='train')
        except Exception as e:
            console.print(f"[bold red]Failed to initialize DialogDataset: {e}[/]")
            exit(1)

        if len(dataset) == 0:
             console.print("[bold red]Fine-tuning dataset is empty. Exiting.[/]")
             exit(1)
        dataloader = DataLoader(dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True)

        # 5. Setup Optimizer and Criterion for Fine-tuning
        optimizer = optim.AdamW(model.parameters(), lr=config.FINETUNE_LEARNING_RATE)
        # Use the configured IGNORE_INDEX for masking loss on padding and user input
        criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

        # 6. Fine-tuning Loop
        console.rule("[bold blue]Starting Dialogue Fine-tuning[/]")
        for epoch in range(config.FINETUNE_EPOCHS):
            avg_loss = train(model, dataloader, optimizer, criterion, epoch, config.FINETUNE_EPOCHS, config.DEVICE)
            console.log(f"Fine-tuning Epoch {epoch+1}/{config.FINETUNE_EPOCHS} completed. Average Loss: {avg_loss:.4f}", style="bold green")
            # Save after each epoch
            console.log(f"Saving fine-tuned model checkpoint to {config.FINETUNED_MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), config.FINETUNED_MODEL_SAVE_PATH)

        console.rule("[bold blue]Fine-tuning Finished[/]")
        console.log(f"Fine-tuned model saved to {config.FINETUNED_MODEL_SAVE_PATH}")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Fine-tuning interrupted by user (Ctrl+C).[/]")
        # Optionally save the model here if interrupted
        # console.log("Saving final model state before exiting...")
        # torch.save(model.state_dict(), config.FINETUNED_MODEL_SAVE_PATH + "_interrupted")
        exit()
    except Exception as main_e:
        console.print(f"[bold red]An unexpected error occurred during fine-tuning: {main_e}[/]")
        console.print(traceback.format_exc())
        exit(1)