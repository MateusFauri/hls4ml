from tensorflow.keras import layers, models, optimizers
from qkeras import QConv2D, QDense, QActivation, QBatchNormalization, quantized_bits, quantized_relu


WEIGHT_QUANT = "quantized_bits(8,0,alpha=1)"
BIAS_QUANT = "quantized_bits(8,0,alpha=1)"
ACT_QUANT = "quantized_relu(8,0)"
BN_QUANT = "quantized_bits(8,0,alpha=1)"


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
 

def build_t1_qkeras(pad_size=32, num_bands=4, num_classes=6):
    inputs = layers.Input(shape=(pad_size, pad_size, num_bands), name="input_32x32x4")
 
    x = QConv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_quantizer=WEIGHT_QUANT,
        bias_quantizer=BIAS_QUANT,
        use_bias=False,
        name="conv1_3x3_s2",
    )(inputs)
    #x = QBatchNormalization(
    #    beta_quantizer=BN_QUANT,
    #    gamma_quantizer=BN_QUANT,
    #    mean_quantizer=BN_QUANT,
    #    variance_quantizer=BN_QUANT,
    #    name="conv1_bn",
    #)(x)
    x = QActivation(ACT_QUANT, name="conv1_relu")(x)
 
    x = QConv2D(
        filters=8,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_quantizer=WEIGHT_QUANT,
        bias_quantizer=BIAS_QUANT,
        use_bias=False,
        name="conv2_1x1_s1",
    )(x)
    #x = QBatchNormalization(
    #    beta_quantizer=BN_QUANT,
    #    gamma_quantizer=BN_QUANT,
    #    mean_quantizer=BN_QUANT,
    #    variance_quantizer=BN_QUANT,
    #    name="conv2_bn",
    #)(x)
    x = QActivation(ACT_QUANT, name="conv2_relu")(x)
 
    x = layers.Flatten(name="flatten")(x)
    x = QDense(
        num_classes,
        kernel_quantizer=WEIGHT_QUANT,
        bias_quantizer=BIAS_QUANT,
        name="dense_output",
    )(x)
 
    outputs = layers.Activation("softmax", name="softmax_output")(x)
 
    model = models.Model(inputs=inputs, outputs=outputs, name="t1_qkeras")
    return model
