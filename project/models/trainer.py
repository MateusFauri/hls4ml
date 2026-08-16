from dataclasses import dataclass
from pathlib import Path

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf

from config import (
    RESULTS_DIR,
    LOSS,
    METRICS,
    MODEL_FILE,
    MODEL_FILE_JSON,
    EARLY_STOPPING,
    WEIGHTS_FILE
)


# ======================================================
# Training Result
# ======================================================

@dataclass
class TrainingResult:
    history: object
    loss: float
    accuracy: float
    output_dir: Path


# ======================================================
# Output directory
# ======================================================

def create_output_dir(
    model_name: str,
    bits: int,
    config_name: str,
) -> Path:

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


# ======================================================
# Train
# ======================================================

def train_model(
    model: Sequential,
    dataset,
    config,
    output_dir: Path,
) -> TrainingResult:

    model.compile(
        optimizer=Adam(
            learning_rate=config.learning_rate
        ),
        loss=LOSS,
        metrics=METRICS,
    )

    callbacks = [
        EarlyStopping(**EARLY_STOPPING)
    ]

    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(
            dataset.x_validation,
            dataset.y_validation,
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(
        dataset.x_test,
        dataset.y_test,
        verbose=0,
    )

    model.save(output_dir / MODEL_FILE)
    model.save_weights(output_dir / WEIGHTS_FILE)

    with open(output_dir / MODEL_FILE_JSON, "w") as f:
        f.write(model.to_json())
    
    tf.keras.utils.plot_model(
        model,
        to_file=output_dir / "architecture.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
    )


    return TrainingResult(
        history=history,
        loss=loss,
        accuracy=accuracy,
        output_dir=output_dir,
    )