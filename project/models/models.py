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
from quantizers_configurations import build_quantizers


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

    model.add(Input(shape=INPUT_SHAPE, name="input"))

    model.add(
        QConv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_1",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_1"))
    model.add(QActivation(q.activation, name="activation_1"))

    model.add(
        QConv2D(
            filters=64,
            kernel_size=(1, 1),
            padding="valid",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_2",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_2"))
    model.add(QActivation(q.activation, name="activation_2"))

    model.add(Flatten(name="flatten"))

    model.add(
        QDense(
            128,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="dense_1",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_3"))
    model.add(QActivation(q.activation, name="activation_3"))

    model.add(
        QDense(
            NUM_CLASSES,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            activation="softmax",
            name="dense_output",
        )
    )

    return model


# ======================================================
# Topology 2
# ======================================================

def build_t2(bits: int) -> Sequential:

    q = build_quantizers(bits)

    model = Sequential()

    model.add(Input(shape=INPUT_SHAPE, name="input"))

    model.add(
        QConv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_1",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_1"))
    model.add(QActivation(q.activation, name="activation_1"))

    model.add(
        QConv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_2",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_2"))
    model.add(QActivation(q.activation, name="activation_2"))

    model.add(
        QConv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_3",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_3"))
    model.add(QActivation(q.activation, name="activation_3"))

    model.add(
        QConv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="conv2d_4",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_4"))
    model.add(QActivation(q.activation, name="activation_4"))

    model.add(Flatten(name="flatten"))

    model.add(
        QDense(
            128,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            name="dense_1",
        )
    )
    model.add(BatchNormalization(name="batch_normalization_5"))
    model.add(QActivation(q.activation, name="activation_5"))

    model.add(
        QDense(
            NUM_CLASSES,
            kernel_quantizer=q.weights,
            bias_quantizer=q.weights,
            activation="softmax",
            name="dense_output",
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
    #"t2": ModelDefinition(
    #    name="t2",
    #    builder=build_t2,
    #),
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