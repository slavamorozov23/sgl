import torch
import os

# --- Settings and Hyperparameters ---
# Architecture
TOKENIZER_NAME = "bert-base-uncased"
D_MODEL = 768
VOCAB_SIZE = None # Determined at runtime
NHEAD = 12
NUM_LAYERS = 8
DIM_FEEDFORWARD = D_MODEL * 4
DROPOUT = 0.1
MAX_SEQ_LEN = 128

# Pre-Training
PRETRAIN_BATCH_SIZE = 16
PRETRAIN_LEARNING_RATE = 1e-4
PRETRAIN_EPOCHS = 5
PRETRAIN_LEARNING_DIR = "now-text-2024-super-cuted"
PRETRAINED_MODEL_SAVE_PATH = "small_llm_pretrained_model.pth"

# Fine-tuning
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 8
FINETUNE_LEARNING_RATE = 5e-5
FINETUNE_DATA_PERCENTAGE = 15 # (1-100) Percentage of dialogue dataset to use for fine-tuning
FINETUNED_MODEL_SAVE_PATH = "small_llm_finetuned_dialog_model.pth"
USER_TOKEN = "<USER>"
ASSISTANT_TOKEN = "<ASSISTANT>"
IGNORE_INDEX = -100 # Default ignore_index for CrossEntropyLoss

# Common / Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing / Generation
MAX_OUTPUT_TOKENS_TEST = 150
TESTING_DIR = "ultrathin-deepseek-v3/tests-super-cuted"
# --- Generation Sampling Parameters ---
SAMPLING_TEMPERATURE = 0.8 # Temperature for sampling. Lower values make the output more deterministic. Must be > 0. Set to 0 for argmax.
SAMPLING_TOP_P = 0.9       # Nucleus sampling probability threshold. 1.0 disables it. Must be > 0 and <= 1.0.