import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from qkeras import QConv2D, QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

def build_t1(img_size=32, num_bands=4, num_classes=6):
    inputs = layers.Input(shape=(img_size, img_size, num_bands), name="input_32x32x4")
 
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv1_3x3_s2",
    )(inputs)
    x = layers.Activation("relu", name="conv1_relu")(x)
 
    x = layers.Conv2D(
        filters=8,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name="conv2_1x1_s1",
    )(x)
    x = layers.Activation("relu", name="conv2_relu")(x)
 
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(num_classes, name="dense_output")(x)
    outputs = layers.Activation("softmax", name="softmax_output")(x)
 
    model = models.Model(inputs=inputs, outputs=outputs, name="t1")
    return model
 


def get_weight_quantizer(bit_width):
    return quantized_bits(bits=bit_width - 1, integer=0, alpha=1, keep_negative=True)
 
 
def get_act_quantizer(bit_width):
    return quantized_relu(bits=bit_width - 1, integer=0)
 
 
def _maybe_uniform_init(bit_quantization):
    if bit_quantization == 2:
        return keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
    return "glorot_uniform"
 
 
def build_t1_quantizedCNN(bit_quantization, input_shape=(32, 32, 4)):
    w_quant = get_weight_quantizer(bit_quantization)
    a_quant = get_act_quantizer(bit_quantization)
    kernel_init = _maybe_uniform_init(bit_quantization)
 
    inputs = keras.Input(shape=input_shape, name="input_layer")
 
    x = QConv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_quantizer=w_quant,
        #kernel_initializer=kernel_init,
        name="conv1",
    )(inputs)
    x = QActivation(a_quant, name="relu1")(x)
 
    x = QConv2D(
        filters=8,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_quantizer=w_quant,
        #kernel_initializer=kernel_init,
        name="conv2",
    )(x)
    x = QActivation(a_quant, name="relu2")(x)
 
    x = layers.Flatten(name="flatten")(x)
 
    outputs = QDense(
        units=6,
        use_bias=False,
        kernel_quantizer=w_quant,
        #kernel_initializer=kernel_init,
        name="fc1",
    )(x)
 
    return keras.Model(inputs=inputs, outputs=outputs, name=f"t1_quantizedCNN_{bit_quantization}b")
 
 
def build_t2_quantizedCNN(bit_quantization, input_shape=(32, 32, 4)):
    w_quant = get_weight_quantizer(bit_quantization)
    a_quant = get_act_quantizer(bit_quantization)
    kernel_init = _maybe_uniform_init(bit_quantization)
 
    inputs = keras.Input(shape=input_shape, name="input_layer")
 
    channels = [16, 8, 16, 8]
    x = inputs
    for i, out_ch in enumerate(channels, start=1):
        x = QConv2D(
            filters=out_ch,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_quantizer=w_quant,
            kernel_initializer=kernel_init,
            name=f"conv{i}",
        )(x)
        x = QActivation(a_quant, name=f"relu{i}")(x)
 
    x = layers.Flatten(name="flatten")(x)
 
    outputs = QDense(
        units=6,
        use_bias=False,
        kernel_quantizer=w_quant,
        kernel_initializer=kernel_init,
        name="fc1",
    )(x)
 
    return keras.Model(inputs=inputs, outputs=outputs, name=f"t2_quantizedCNN_{bit_quantization}b")
 
