import torch
import os
import numpy as np

# Architecture
TOKENIZER_NAME = "gpt2" # t5-base для математики, bert-base-uncased легковесный общего назначения, AnReu/math_albert лучший
NHEAD = 8
HEAD_DIM = 64  # размер одной головы (dₕ)
D_MODEL = NHEAD * HEAD_DIM  # общее скрытое пространство (~800)
VOCAB_SIZE = None
#NUM_CUBES = 10  # No longer needed; loaded from JSON
#MAX_DISTANCE = 2.0  # No longer used; neighbors defined in JSON
DIM_FEEDFORWARD = 2048  # FFN hidden size ≈2.5×d_model for extra-small model
DROPOUT = 0.1
MAX_SEQ_LEN = 128  # reduced for speed
MIN_PATH_LENGTH_RATIO = 0.3
MLA_D_C = 256  # Dimension of latent space for KV compression in MLA
MLA_QUERY_D_C = 768  # Dimension of latent space for query compression (d'_c)
MLA_DECOUPLED_KEY_DIM = 64  # Dimension for decoupled keys after RoPE (dʰᴿ)
MLA_COMPRESS_QUERY = True  # Optional query compression in MLA

# Pre-Training (Parameters for math-train)
PRETRAIN_BATCH_SIZE = 16  # larger batch for quick convergence
PRETRAIN_LEARNING_RATE = 5e-5
PRETRAIN_EPOCHS = 8  # fewer epochs for fast tests
PRETRAIN_LEARNING_DIR = os.path.join(os.path.dirname(__file__), "..", "deepmind_math_arithmetic")
PRETRAINED_MODEL_SAVE_PATH = "output_models/spatial_graph_math_model.pth"
WARMUP_STEPS = 500  # Количество шагов (батчей) для линейного разогрева LR
DATASET_PERCENTAGE = 1 # 5%, или 100000 примеров, будет лучшим значением, дял поулчения хорошей модели
IGNORE_INDEX = -100 # Used for masking labels

# Common / Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing / Generation
MAX_OUTPUT_TOKENS_TEST = 20
SAMPLING_TEMPERATURE = 0.1  # use stochastic sampling to avoid degenerate repetitions
SAMPLING_TOP_P = 0.9  # enable nucleus sampling
REPETITION_PENALTY = 1.1  # >1.0 to penalize repeated tokens (1.0 = no penalty)
REPETITION_WINDOW = -1    # consider all previous tokens; set >0 to limit window length

# Spatial Graph Specific
POINTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "points.json")  # Path to graph configuration JSON
EXIT_TOKEN_INDEX = -1
ROUTING_SAFETY_LIMIT = 10
MAX_CUBES_PER_PATH = 3
ROUTING_TEMPERATURE = 1  # sharper routing distribution

# Weight for the MoE-style Load Balancing auxiliary loss
LOAD_BALANCING_LOSS_WEIGHT = 0.01  # Start with small value, e.g. 0.01 or 0.001

# Auxiliary Routing Loss
AUX_LOSS_WEIGHT = 0.05 # Weight for the auxiliary routing loss #0.01

# Penalty strength for adaptive bias update (route penalization)
# Removed ROUTE_PENALTY_LAMBDA as adaptive biases are removed.

# Bias-update rate for adaptive bias strategy
# Removed ROUTING_BIAS_UPDATE_RATE as adaptive biases are removed.
ROUTING_EPSILON_GREEDY = 0.01 # more exploration

# Visualization Paths
