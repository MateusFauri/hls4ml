from dataclasses import dataclass
from typing import Callable

from keras import Sequential
from keras.layers import (
    BatchNormalization,
    Flatten,
    Input,
)
from qkeras import (
    QActivation,
    QConv2D,
    QDense,
)

from config import INPUT_SHAPE, NUM_CLASSES
from quantizers import build_quantizers


# ======================================================
# Model definition
# ======================================================

@dataclass(frozen=True)
class ModelDefinition:
    name: str
    builder: Callable[[int], Sequential]


# ======================================================
# Topology 1
# ======================================================

def build_t1(bits: int) -> Sequential:

    q = build_quantizers(bits)

    model = Sequential()

    model.add(Input(shape=INPUT_SHAPE))

    model.add(
        QConv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QConv2D(
            filters=64,
            kernel_size=(1, 1),
            padding="valid",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(Flatten())

    model.add(
        QDense(
            128,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QDense(
            NUM_CLASSES,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            activation="softmax",
        )
    )

    return model


# ======================================================
# Topology 2
# ======================================================

def build_t2(bits: int) -> Sequential:

    q = build_quantizers(bits)

    model = Sequential()

    model.add(Input(shape=INPUT_SHAPE))

    model.add(
        QConv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QConv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QConv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QConv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(Flatten())

    model.add(
        QDense(
            128,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
        )
    )
    model.add(BatchNormalization())
    model.add(QActivation(q.activation))

    model.add(
        QDense(
            NUM_CLASSES,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            activation="softmax",
        )
    )

    return model


# ======================================================
# Available models
# ======================================================

MODEL_BUILDERS = {
    "t1": ModelDefinition(
        name="t1",
        builder=build_t1,
    ),
    "t2": ModelDefinition(
        name="t2",
        builder=build_t2,
    ),
}


# ======================================================
# Factory
# ======================================================

def build_model(model_name: str, bits: int) -> Sequential:

    try:
        return MODEL_BUILDERS[model_name].builder(bits)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_BUILDERS.keys())}"
        ) from exc