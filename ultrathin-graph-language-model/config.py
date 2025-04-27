import torch
import os
import numpy as np

# Architecture
TOKENIZER_NAME = "character-math"
D_MODEL = 256
VOCAB_SIZE = 17 # 10 digits + 3 ops (+-=) + space + 3 special (PAD, EOS, UNK)
NHEAD = 8
NUM_CUBES = 15
MAX_DISTANCE = 2.0
DIM_FEEDFORWARD = D_MODEL*2
DROPOUT = 0.1
MAX_SEQ_LEN = 64
MIN_PATH_LENGTH_RATIO = 0.6

# Pre-Training (Parameters for math-train)
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LEARNING_RATE = 1e-4
PRETRAIN_EPOCHS = 30
PRETRAIN_LEARNING_DIR = "deepmind_math_arithmetic"
PRETRAINED_MODEL_SAVE_PATH = "spatial_graph_math_model.pth"

# Fine-Tuning Specific (Not used in math-train)
DATASET_PERCENTAGE = 10
USER_TOKEN = "<USER>"
ASSISTANT_TOKEN = "<ASSISTANT>"
IGNORE_INDEX = -100 # Used for masking labels

# Common / Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing / Generation
MAX_OUTPUT_TOKENS_TEST = 20
TESTING_DIR = "ultrathin-graph-language-model/tests-math"
SAMPLING_TEMPERATURE = 0.0 # Use deterministic generation for math
SAMPLING_TOP_P = 0.0 # No nucleus sampling needed

# Spatial Graph Specific
EXIT_TOKEN_INDEX = -1
ROUTING_SAFETY_LIMIT = NUM_CUBES * 5
MAX_CUBES_PER_PATH = NUM_CUBES * 4

# Penalty strength for adaptive bias update (route penalization)
ROUTE_PENALTY_LAMBDA = 0.1

# Bias-update rate for adaptive bias strategy
ROUTING_BIAS_UPDATE_RATE = 0.01
ROUTING_EPSILON_GREEDY = 0.05

# Visualization Paths
VIZ_POSITIONS_FILE = "viz_cube_positions_math.json"
VIZ_PATHS_FILE = "viz_training_paths_math.jsonl"
VIZ_START_CANDIDATES_FILE = "viz_start_candidates_math.json"