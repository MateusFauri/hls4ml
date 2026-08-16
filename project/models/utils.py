from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from config import RANDOM_SEED, RESULTS_DIR


# ======================================================
# Reproducibility
# ======================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# Output directories
# ======================================================

def create_output_dir(
    model_name: str,
    bits: int,
    config_name: str,
) -> Path:
    """Create and return the output directory for an experiment."""

    output_dir = (
        RESULTS_DIR
        / model_name
        / f"{bits}bits"
        / config_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir