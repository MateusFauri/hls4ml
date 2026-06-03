import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, 
    MaxPooling2D, 
    Flatten, 
    ReLU, 
    Conv2D, 
    Dense, 
    Softmax, 
    Dropout
)
from qkeras import QConv2D, QDense, QActivation
from qkeras import quantized_bits, quantized_relu


def build_t1_float():

    inputs = Input(shape=(28, 28, 4), name='cnn_input')

    x = Conv2D(
        16,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)

    x = ReLU(name='relu1')(x)

    x = MaxPooling2D((2,2), name='pool1')(x)

    x = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv2'
    )(x)

    x = ReLU(name='relu2')(x)

    x = MaxPooling2D((2,2), name='pool2')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(
        64,
        use_bias=False,
        name='fc1'
    )(x)

    x = ReLU(name='relu3')(x)

    outputs = Dense(
        6,
        use_bias=False,
        name='fc2'
    )(x)

    #outputs = Softmax(name='softmax')(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name='t1_float'
    )

    return model


def build_t2_float():

    inputs = Input(shape=(28, 28, 4), name='cnn_input')

    x = Conv2D(
        16,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)

    x = ReLU(name='relu1')(x)

    x = Conv2D(
        16,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv2'
    )(x)

    x = ReLU(name='relu2')(x)

    x = MaxPooling2D((2,2), name='pool1')(x)

    x = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv3'
    )(x)

    x = ReLU(name='relu3')(x)

    x = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv4'
    )(x)

    x = ReLU(name='relu4')(x)

    x = MaxPooling2D((2,2), name='pool2')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(
        128,
        use_bias=False,
        name='fc1'
    )(x)

    x = ReLU(name='relu5')(x)

    outputs = Dense(
        6,
        use_bias=False,
        name='fc2'
    )(x)

    #outputs = Softmax(name='softmax')(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name='t2_float'
    )

    return model




def build_t1_qkeras(w_bits=8, a_bits=8):

    inputs = Input(shape=(28, 28, 4))

    x = QConv2D(
        16,
        kernel_size=3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(inputs)

    x = QActivation(quantized_relu(a_bits, 0))(x)
    x = MaxPooling2D(2)(x)

    x = QConv2D(
        32,
        3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)

    x = QDense(
        64,
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)

    x = QDense(
        6,
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    return Model(inputs, x, name="t1_qkeras")



def build_t2_qkeras(w_bits=8, a_bits=8):

    inputs = Input(shape=(28, 28, 4))

    x = QConv2D(
        16,
        3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(inputs)

    x = QActivation(quantized_relu(a_bits, 0))(x)

    x = QConv2D(
        16,
        3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)
    x = MaxPooling2D(2)(x)

    x = QConv2D(
        32,
        3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)

    x = QConv2D(
        32,
        3,
        padding='same',
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)

    x = QDense(
        128,
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    x = QActivation(quantized_relu(a_bits, 0))(x)

    x = QDense(
        6,
        use_bias=False,
        kernel_quantizer=quantized_bits(w_bits, 0, alpha=1)
    )(x)

    return Model(inputs, x, name="t2_qkeras")