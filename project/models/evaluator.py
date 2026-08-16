from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from config import METRICS_FILE
from keras.models import Sequential
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ======================================================
# Evaluation Result
# ======================================================

@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray


# ======================================================
# Evaluation
# ======================================================

def evaluate_model(
    model: Sequential,
    dataset,
) -> EvaluationResult:

    predictions = model.predict(
        dataset.x_test,
        verbose=0,
    )

    y_true = np.argmax(dataset.y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    return EvaluationResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        recall=recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        f1_score=f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
        ),
    )



def save_metrics(
    result: EvaluationResult,
    output_dir: Path,
) -> None:

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ],
        "Value": [
            result.accuracy,
            result.precision,
            result.recall,
            result.f1_score,
        ],
    })

    metrics.to_csv(
        output_dir / METRICS_FILE,
        index=False,
    )