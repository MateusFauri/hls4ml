from dataclasses import dataclass
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config import (
    DATASET_PATH,
    NUM_CLASSES,
    TEST_SIZE,
    RANDOM_SEED,
    NORMALIZATION_FACTOR,
    DATASET_TRANSPOSE,
    VALIDATION_SPLIT,
)


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray

    x_validation: np.ndarray
    y_validation: np.ndarray

    x_test: np.ndarray
    y_test: np.ndarray


def load_sat6() -> Dataset:
    """Load and preprocess the SAT-6 dataset."""

    data = loadmat(DATASET_PATH)

    images = data["train_x"]
    labels = data["train_y"]

    images = images.transpose(DATASET_TRANSPOSE)
    images = images.astype(np.float32) / NORMALIZATION_FACTOR

    labels = labels.argmax(axis=0)
    labels = to_categorical(labels, NUM_CLASSES)

    # 80% train/validation | 20% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels.argmax(axis=1),
    )

    # 85% train | 15% validation (dos 80%)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train_val.argmax(axis=1),
    )

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )