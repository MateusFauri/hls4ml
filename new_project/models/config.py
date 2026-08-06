from dataclasses import dataclass
from pathlib import Path

# ======================================================
# Paths
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
DATA_DIR = PARENT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DATASET_PATH = DATA_DIR / "sat-6-full.mat"

# ======================================================
# Dataset
# ======================================================

INPUT_SHAPE = (28, 28, 4)
NUM_CLASSES = 6
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42

# ======================================================
# Quantization
# ======================================================

BIT_LIST = [8, 4, 2, 1]

# ======================================================
# Models
# ======================================================

MODELS = [
    "t1",
    "t2", 
]

# ======================================================
# Training
# ======================================================

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int


TRAIN_CONFIGS = {
    "best": TrainingConfig(
        learning_rate=1e-4,
        batch_size=512,
        epochs=50,
    ),
    "paper": TrainingConfig(
        learning_rate=3e-4,
        batch_size=512,
        epochs=30,
    ),
}

LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
METRICS = ["accuracy"]

EARLY_STOPPING = {
    "monitor": "val_accuracy",
    "mode": "max",
    "patience": 30,
    "restore_best_weights": True,
}

# ======================================================
# Output files
# ======================================================

MODEL_FILE = "model.h5"
METRICS_FILE = "metrics.csv"
HISTORY_PLOT = "history.png"
CONFUSION_MATRIX_PLOT = "confusion_matrix.png"

# ======================================================
# Plots
# ======================================================
FIGURE_SIZE = (12, 5)
CONFUSION_MATRIX_SIZE = (7, 7)
DPI = 300

# ======================================================
# Initialization
# ======================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Dataset constants
# ======================================================

TEST_SIZE = 0.20
VALIDATION_SPLIT = 0.15
NORMALIZATION_FACTOR = 255.0
DATASET_TRANSPOSE = (3, 0, 1, 2)
