# --- START OF FILE dataset_utils.py ---

import numpy as np
import random
from text_utils import clean_text
from multiprocessing import current_process

tokenizer_mp = None
max_seq_len_mp = None
ignore_index_mp = None

def init_math_worker(tok, seq_len, ignore_idx):
    """Initializes global variables for ProcessPoolExecutor workers."""
    global tokenizer_mp, max_seq_len_mp, ignore_index_mp
    tokenizer_mp = tok
    max_seq_len_mp = seq_len
    ignore_index_mp = ignore_idx

def process_batch_for_math_mp(batch, show_samples=False, stage1_mode=False):
    """
    Processes a batch of questions and answers for the math task.
    Now concatenates question and answer with '\\n', logs inputs, and masks the last token.
    """
    global tokenizer_mp, max_seq_len_mp, ignore_index_mp
    out = {'input_ids': [], 'labels': [], 'original_q': [], 'original_a': []}
    debug_messages = []


    # Prepare IDs for newline and EOS
    try:
        newline_id = tokenizer_mp.encode('\n', add_special_tokens=False)[0]
    except Exception:
        newline_id = -999
    eos_id = tokenizer_mp.eos_token_id
    pad_id = tokenizer_mp.pad_token_id or eos_id

    qs, ans = batch['question'], batch['answer']

    # Initial display of 3 random cleaned examples if requested
    if show_samples and len(qs) > 0:
        sample_indices = random.sample(range(len(qs)), min(3, len(qs)))
        print("Sample cleaned examples:")
        for idx in sample_indices:
            q_clean = clean_text(qs[idx])
            a_clean = clean_text(ans[idx])
            print(f"Q: {q_clean}")
            print(f"A: {a_clean}")
            print("---")

    for q, a in zip(qs, ans):
        # Clean question and answer using shared function
        q = clean_text(q)
        a_clean = clean_text(a)

        # Concatenate with newline
        s = f"{q}\n{a_clean}"


        # Tokenize
        ids = tokenizer_mp.encode(s, add_special_tokens=False)
        # Add EOS token if missing
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(eos_id)

        # --- НАЧАЛО НОВОЙ ЛОГИКИ input_ids/labels ---
        original_len = len(ids)
        if original_len < 2:
            debug_messages.append(f"Warning: Skipping short sequence (len {original_len}): {s[:50]}...")
            continue
        # Find delimiter index
        try:
            newline_idx = ids.index(newline_id)
        except ValueError:
            debug_messages.append(f"Warning: Newline token not found in sequence. Skipping: {s[:50]}..." )
            continue
        # Truncate or pad ids
        ids_truncated = ids[:max_seq_len_mp] if original_len > max_seq_len_mp else ids
        # Build input_ids
        input_tokens = ids_truncated[:-1]
        input_ids_final = np.full(max_seq_len_mp, pad_id, dtype=np.int64)
        len_input = min(len(input_tokens), max_seq_len_mp)
        input_ids_final[:len_input] = input_tokens[:len_input]

        # --- Two-stage logic for labels ---
        labels_final = np.full(max_seq_len_mp, ignore_index_mp, dtype=np.int64)
        if stage1_mode:
            # --- Stage 1: Only penultimate non-<eos> token is target ---
            original_len = len(ids_truncated)
            # Need at least [ ..., answer_token, eos ]
            if original_len >= 3:
                eos_idx = original_len - 1
                penult_idx = original_len - 2  # index of last answer token
                label_pos = penult_idx - 1     # predict that answer token from previous input step
                if 0 <= label_pos < max_seq_len_mp:
                    target_token_id = ids_truncated[penult_idx]
                    labels_final[label_pos] = target_token_id
                else:
                    if len(debug_messages) < 10:
                        debug_messages.append("Warning: Calculated label_pos for stage1 outside max_seq_len.")
            else:
                if len(debug_messages) < 10:
                    debug_messages.append("Warning: Sequence too short for stage1 mode (need >=3 tokens).")
        else:
            # --- Stage 2: Standard logic (all answer tokens after newline) ---
            answer_tokens = ids_truncated[newline_idx+1:]
            label_start = newline_idx
            if answer_tokens:
                for i, tok in enumerate(answer_tokens):
                    pos = label_start + i
                    if pos < max_seq_len_mp:
                        labels_final[pos] = tok
                    else:
                        if len(debug_messages) < 10:
                            debug_messages.append(
                                f"Warning: Answer tokens truncated for labels (max_seq_len: {max_seq_len_mp})."
                            )
                        break
        # Validate
        if len(input_ids_final) != max_seq_len_mp or len(labels_final) != max_seq_len_mp:
            if len(debug_messages) < 10:
                debug_messages.append("Warning: Length mismatch after processing. Skipping example.")
            continue
        # Append to output
        out['input_ids'].append(input_ids_final)
        out['labels'].append(labels_final)
        out['original_q'].append(q)
        out['original_a'].append(a_clean)
        # --- КОНЕЦ НОВОЙ ЛОГИКИ input_ids/labels ---

    return out, debug_messages

# --- END OF FILE dataset_utils.py ---
