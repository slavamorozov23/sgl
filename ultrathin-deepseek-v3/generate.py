import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from typing import Optional, List, Tuple

import config

console = Console()

def generate_text(model: torch.nn.Module,
                  tokenizer: AutoTokenizer,
                  prompt: str,
                  max_new_tokens: int = 50,
                  device: str = "cpu",
                  max_seq_len: int = 128):
    """
    Generates text autoregressively using the custom model with RoPE, KV caching.
    Formats prompt for dialogue model and cleans the output.
    Stops if max_seq_len is reached or EOS/USER token is generated.
    Uses Temperature and Top-P sampling. Falls back to argmax if temperature is 0.
    """
    model.eval()

    formatted_prompt = f"{config.USER_TOKEN} {prompt} {config.ASSISTANT_TOKEN}"
    user_token_id = tokenizer.convert_tokens_to_ids(config.USER_TOKEN)

    prompt_ids = tokenizer.encode(
        formatted_prompt,
        return_tensors="pt",
        max_length=max_seq_len, # Truncate prompt
        truncation=True,
        add_special_tokens=False
    ).to(device)

    if prompt_ids.shape[1] >= max_seq_len:
         console.print(f"[yellow]Warning: Input prompt after formatting and truncation is already at max_seq_len ({prompt_ids.shape[1]}/{max_seq_len}). No room for generation.[/yellow]")
         return ""

    generated_ids = prompt_ids
    batch_size = generated_ids.shape[0] # Should be 1 for typical inference

    console.log(f"Generating text from formatted prompt (tokenized length: {prompt_ids.shape[1]})... (Temp: {config.SAMPLING_TEMPERATURE}, Top-P: {config.SAMPLING_TOP_P})")

    past_kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None

    with torch.no_grad():
        progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console)
        task_id = progress.add_task("Generating...", total=max_new_tokens)
        with progress:
            for i in range(max_new_tokens):

                current_seq_len = generated_ids.shape[1]
                if current_seq_len >= max_seq_len:
                    console.log(f"[yellow]Warning: Reached maximum sequence length ({max_seq_len}). Stopping generation.[/yellow]")
                    progress.update(task_id, completed=max_new_tokens)
                    break

                # Use only the last token for input when using KV cache
                input_token_ids_step = generated_ids if past_kv_cache is None else generated_ids[:, -1:]

                logits, present_kv_cache = model(input_ids=input_token_ids_step, past_kv_cache=past_kv_cache)
                past_kv_cache = present_kv_cache # Update cache for next iteration

                # Get logits for the very last token prediction
                next_token_logits = logits[:, -1, :]

                # --- Sampling Logic ---
                if config.SAMPLING_TEMPERATURE <= 0:
                    # Deterministic: Use argmax if temperature is 0 or less
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    # Apply temperature scaling
                    next_token_logits = next_token_logits / config.SAMPLING_TEMPERATURE
                    # Get probabilities
                    probs = F.softmax(next_token_logits, dim=-1)

                    # Apply Top-P (Nucleus) Sampling if enabled
                    if config.SAMPLING_TOP_P > 0.0 and config.SAMPLING_TOP_P < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        # Create mask for tokens to remove (cumulative prob > top_p)
                        sorted_indices_to_remove = cumulative_probs > config.SAMPLING_TOP_P
                        # Shift the mask right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0 # Always keep the most probable token

                        # Map the mask back to the original indices
                        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

                        # Apply the mask by setting probabilities of removed tokens to 0
                        probs[indices_to_remove] = 0.0

                        # Renormalize the probabilities
                        # Add a small epsilon to prevent division by zero in edge cases
                        renorm_probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-9)

                        # Sample using multinomial
                        next_token_id = torch.multinomial(renorm_probs, num_samples=1)
                    else:
                        # Sample from the temperature-adjusted distribution without Top-P
                        next_token_id = torch.multinomial(probs, num_samples=1)
                # --- End Sampling Logic ---

                # Append the chosen token
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
                progress.update(task_id, advance=1)

                # Check for stopping conditions (EOS or USER token)
                if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                    console.log(f"EOS token generated (ID: {tokenizer.eos_token_id}).")
                    progress.update(task_id, completed=max_new_tokens)
                    break
                if next_token_id.item() == user_token_id:
                     console.log(f"USER token generated (ID: {user_token_id}). Stopping generation.")
                     progress.update(task_id, completed=max_new_tokens)
                     generated_ids = generated_ids[:, :-1] # Remove the generated USER token
                     break
            else: # Executed if the loop completes without break
                 progress.update(task_id, completed=max_new_tokens)

    # --- Debugging Output ---
    console.print("\n" + "="*10 + " DEBUG INFO " + "="*10)
    console.print(f"Final Generated IDs shape: {generated_ids.shape}")
    console.print(f"Final Generated IDs: {generated_ids[0].tolist()}")
    try:
        raw_decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        console.print(f"Raw Decoded Text (before cleaning):\n'''\n{raw_decoded_text}\n'''")
    except Exception as decode_err:
        console.print(f"[bold red]Error during raw decoding for debug: {decode_err}[/]")
    console.print("="*32 + "\n")
    # --- End Debugging Output ---

    # --- Decode and Clean Output ---
    full_generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # Find the start of the assistant's response
    assistant_response_start_index = full_generated_text.rfind(config.ASSISTANT_TOKEN)
    if assistant_response_start_index != -1:
        # Take text after the last ASSISTANT token
        assistant_response = full_generated_text[assistant_response_start_index + len(config.ASSISTANT_TOKEN):]
    else:
        # Fallback: try to remove the original prompt (less reliable)
        prompt_decoded_len = len(tokenizer.decode(prompt_ids[0], skip_special_tokens=False))
        assistant_response = full_generated_text[prompt_decoded_len:]
        console.print("[yellow]Warning: Could not find ASSISTANT_TOKEN in generated text. Output might include parts of the prompt.[/yellow]")


    # Remove special tokens from the assistant's response
    if tokenizer.eos_token:
        assistant_response = assistant_response.replace(tokenizer.eos_token, "")
    if config.USER_TOKEN:
        # Prevent removing user token if it was legitimately generated within the response
        # This cleanup focuses on removing structural tokens added by the script
        pass # Typically USER token generation stops the process anyway
    if tokenizer.pad_token:
         assistant_response = assistant_response.replace(tokenizer.pad_token, "")

    # Remove leading/trailing whitespace
    assistant_response = assistant_response.strip()
    # Remove potential leading space often left after ASSISTANT_TOKEN
    assistant_response = assistant_response.lstrip()

    return assistant_response