import torch
from transformers import AutoTokenizer
import glob
import os
from rich.console import Console
import traceback

import config
from model import SpatialGraphTransformer
from generate import generate_text

console = Console()

def test_model():
    console.rule("[bold blue]Initializing Model Testing (Fine-tuned Spatial Graph Model)[/]")

    tokenizer_name = config.TOKENIZER_NAME
    model_save_path = config.PRETRAINED_MODEL_SAVE_PATH
    device = config.DEVICE
    testing_dir = config.TESTING_DIR
    max_new_tokens = config.MAX_OUTPUT_TOKENS_TEST
    max_seq_len = config.MAX_SEQ_LEN

    console.log(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        special_tokens_dict = {'additional_special_tokens': [config.USER_TOKEN, config.ASSISTANT_TOKEN]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        console.log(f"Added {num_added_toks} special tokens for testing.")

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.eos_token is None:
             if tokenizer.sep_token is not None: tokenizer.eos_token = tokenizer.sep_token
             else: tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        console.log(f"Tokenizer loaded and updated for testing. Vocab size: {len(tokenizer)}")

    except Exception as e:
        console.print(f"[bold red]Error loading or updating tokenizer: {e}[/]")
        console.print(traceback.format_exc())
        exit(1)

    console.log("Initializing Spatial Graph Transformer model structure for testing...")
    if not hasattr(tokenizer, 'vocab_size'):
         console.print("[bold red]Error: Tokenizer does not have vocab_size attribute.[/]")
         exit(1)

    model = SpatialGraphTransformer(
        num_cubes=config.NUM_CUBES,
        d_model=config.D_MODEL,
        dim_feedforward=config.DIM_FEEDFORWARD,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
        vocab_size=len(tokenizer),
        max_distance=config.MAX_DISTANCE
    ).to(device)

    model.eval()
    if hasattr(model, 'dropout_emb'): model.dropout_emb.p = 0.0
    for cube in model.cubes:
        if hasattr(cube, 'dropout1'): cube.dropout1.p = 0.0
        if hasattr(cube, 'dropout2'): cube.dropout2.p = 0.0
        if hasattr(cube.self_attn, 'attn_dropout'): cube.self_attn.attn_dropout.p = 0.0
        if hasattr(cube.self_attn, 'resid_dropout'): cube.self_attn.resid_dropout.p = 0.0

    console.log(f"Loading FINE-TUNED model weights from {model_save_path}...")
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        console.log("Fine-tuned model weights loaded successfully.")
    except FileNotFoundError:
         console.print(f"[bold red]Fine-tuned model file not found at {model_save_path}.[/]")
         exit(1)
    except Exception as e:
         console.print(f"[bold red]Could not load fine-tuned model from {model_save_path}: {e}.[/]")
         console.print(traceback.format_exc())
         exit(1)

    console.rule("[bold blue]Starting Testing Phase with Fine-tuned Model[/]")
    testing_files = glob.glob(os.path.join(testing_dir, "*.txt"))

    if not testing_files:
        console.print(f"[yellow]No .txt files found in {testing_dir}. No tests to run.[/]")
    else:
        for question_file in testing_files:
            try:
                with open(question_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                if prompt:
                    console.print(f"\n[bold cyan]File:[/bold cyan] {os.path.basename(question_file)}")
                    console.print(f"[bold cyan]Prompt:[/bold cyan]\n{prompt}")

                    generated_output = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        max_seq_len=max_seq_len
                    )
                    console.print(f"[bold magenta]Generated Answer:[/bold magenta]\n{generated_output}")
                    console.print("-" * 20)
                else:
                    console.print(f"[yellow]Skipping empty file: {question_file}[/]")
            except Exception as e:
                console.print(f"[bold red]Error processing testing file {question_file}: {e}[/]")
                console.print(traceback.format_exc())

    console.rule("[bold blue]Testing Finished[/]")

if __name__ == "__main__":
    try:
        test_model()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Script interrupted by user (Ctrl+C).[/]")
        exit(1)
    except Exception as main_e:
        console.print(f"[bold red]An unexpected error occurred in testing script: {main_e}[/]")
        console.print(traceback.format_exc())
        exit(1)