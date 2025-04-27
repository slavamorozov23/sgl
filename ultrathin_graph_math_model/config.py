import torch
import os
import numpy as np

# Architecture
TOKENIZER_NAME = "gpt2"
D_MODEL = 256
VOCAB_SIZE = None
NHEAD = 8
#NUM_CUBES = 10  # No longer needed; loaded from JSON
#MAX_DISTANCE = 2.0  # No longer used; neighbors defined in JSON
DIM_FEEDFORWARD = D_MODEL*4
DROPOUT = 0.33
MAX_SEQ_LEN = 256  # reduced for speed
MIN_PATH_LENGTH_RATIO = 0.3
MLA_D_C = D_MODEL  # Dimension of latent space for Multi-Latent Attention (d_c ≪ d_h·n_h)
MLA_COMPRESS_QUERY = False  # Optional query compression in MLA

# Pre-Training (Parameters for math-train)
PRETRAIN_BATCH_SIZE = 8  # larger batch for quick convergence
PRETRAIN_LEARNING_RATE = 3e-4
PRETRAIN_EPOCHS = 10  # fewer epochs for fast tests
PRETRAIN_LEARNING_DIR = "deepmind_math_arithmetic"
PRETRAINED_MODEL_SAVE_PATH = "spatial_graph_math_model.pth"

DATASET_PERCENTAGE = 0.1
IGNORE_INDEX = -100 # Used for masking labels

# Common / Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing / Generation
MAX_OUTPUT_TOKENS_TEST = 20
SAMPLING_TEMPERATURE = 0.8  # use stochastic sampling to avoid degenerate repetitions
SAMPLING_TOP_P = 0.9  # enable nucleus sampling
REPETITION_PENALTY = 1.1  # >1.0 to penalize repeated tokens (1.0 = no penalty)
REPETITION_WINDOW = -1    # consider all previous tokens; set >0 to limit window length

# Spatial Graph Specific
POINTS_JSON_PATH = "points.json"  # Path to graph configuration JSON
EXIT_TOKEN_INDEX = -1
ROUTING_SAFETY_LIMIT = 20
MAX_CUBES_PER_PATH = 3
ROUTING_TEMPERATURE = 0.5  # sharper routing distribution

# Weight for the MoE-style Load Balancing auxiliary loss
LOAD_BALANCING_LOSS_WEIGHT = 0.05  # Start with small value, e.g. 0.01 or 0.001

# Auxiliary Routing Loss
AUX_LOSS_WEIGHT = 0.05 # Weight for the auxiliary routing loss #0.01

# Penalty strength for adaptive bias update (route penalization)
# Removed ROUTE_PENALTY_LAMBDA as adaptive biases are removed.

# Bias-update rate for adaptive bias strategy
# Removed ROUTING_BIAS_UPDATE_RATE as adaptive biases are removed.
ROUTING_EPSILON_GREEDY = 0.2  # more exploration

# Visualization Paths
VIZ_POSITIONS_FILE = "viz_cube_positions_math.json"
VIZ_PATHS_FILE = "viz_training_paths_math.jsonl"
VIZ_START_CANDIDATES_FILE = "viz_start_candidates_math.json"