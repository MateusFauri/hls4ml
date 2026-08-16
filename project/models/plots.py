from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from config import (
    CONFUSION_MATRIX_PLOT,
    HISTORY_PLOT,
    FIGURE_SIZE,
    DPI,
)


# ======================================================
# History
# ======================================================

def plot_history(
    history,
    output_dir: Path,
):

    fig, ax = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
    )

    # Accuracy
    ax[0].plot(
        history.history["accuracy"],
        label="Train",
    )

    ax[0].plot(
        history.history["val_accuracy"],
        label="Validation",
    )

    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss
    ax[1].plot(
        history.history["loss"],
        label="Train",
    )

    ax[1].plot(
        history.history["val_loss"],
        label="Validation",
    )

    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / HISTORY_PLOT,
        dpi=DPI,
    )

    plt.close(fig)


# ======================================================
# Confusion Matrix
# ======================================================

def plot_confusion_matrix(
    evaluation,
    output_dir: Path,
):

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=evaluation.confusion_matrix
    ).plot(
        ax=ax,
        colorbar=False,
        values_format="d",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / CONFUSION_MATRIX_PLOT,
        dpi=DPI,
    )

    plt.close(fig)


# ======================================================
# Summary
# ======================================================

def generate_plots(
    training,
    evaluation,
):

    plot_history(
        training.history,
        training.output_dir,
    )

    plot_confusion_matrix(
        evaluation,
        training.output_dir,
    )