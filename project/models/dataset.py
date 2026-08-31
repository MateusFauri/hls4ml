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

def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images to the range [0, 1]."""
    return images.astype(np.float32) / NORMALIZATION_FACTOR

def to_signed_integers(images: np.ndarray, bits: int) -> np.ndarray:
    """Convert unsigned image values to signed integers with the given bit width."""
    if bits <= 0:
        raise ValueError("bits must be greater than 0")

    dtype = getattr(np, f"int{bits}")
    return images.astype(dtype) 

def load_sat6(
    normalize_images: bool = False,
    signed_bits: int | None = None,
) -> Dataset:
    """Load and preprocess the SAT-6 dataset."""

    data = loadmat(DATASET_PATH)

    if signed_bits is not None:
        images = to_signed_integers(data["train_x"], signed_bits)
    else:   
        images = data["train_x"]

    labels = data["train_y"]
    images = images.transpose(DATASET_TRANSPOSE)

    if normalize_images:
        images = normalize_images(images)

    labels = labels.argmax(axis=0)
    labels = to_categorical(labels, NUM_CLASSES)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels.argmax(axis=1),
    )

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