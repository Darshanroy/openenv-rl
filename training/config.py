import os

# Model Settings
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "training_outputs"

# Environment Server
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:8000")

# Training Hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 128
USE_VLLM = False # False by default for CPU compatibility
MAX_TURNS = 6

# Logging & Evaluation
LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "evaluation_report.csv")
WANDB_PROJECT = "openenv-csa"
LOG_TO_WANDB = False
