import tensorflow as tf
from keras.layers import Activation
from keras.models import Model, Sequential
from keras.layers import Input, MaxPooling2D,Conv2D,Dense, Activation, Flatten, Dense, BatchNormalization
from qkeras import QConv2D, QActivation, QDense, quantized_bits, quantized_relu


def build_t1(img_size=32, num_bands=4, num_classes=6):
    model = Sequential()
    model.add( 
        Conv2D(
            input_shape=(img_size, img_size, num_bands),
            filters=20,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name="conv1",
        )
    )
    model.add(Activation("relu", name="relu1"))

    model.add(
        Conv2D(
            filters=8,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="conv2",
        )
    )
    model.add(Activation("relu", name="relu2"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(num_classes, name="output_dense"))
    model.add(Activation("softmax", name="softmax_output"))
    return model
 
def build_t1_quantized(img_size=32, num_bands=4, num_classes=6, bits=8):
    model = Input(shape=(img_size, img_size, num_bands))
    model.add(
        QConv2D(
            input_shape=(img_size, img_size, num_bands),
            filters=20,
            kernel_size=3,
            strides=2,
            padding="same",
            name="conv1",
            kernel_quantizer=quantized_bits(bits, 0, alpha=1),
        )
    )
    model.add(QActivation(activation=quantized_relu(bits), name="relu1"))

    model.add(
        QConv2D(
            filters=8,
            kernel_size=1,
            strides=1,
            padding="same",
            name="conv2",
            kernel_quantizer=quantized_bits(bits, 0, alpha=1),
        )
    )
    model.add(QActivation(activation=quantized_relu(bits), name="relu2"))

    model.add(Flatten(name="flatten"))
    model.add(
        QDense(
            num_classes,
            name="output_dense",
            kernel_quantizer=quantized_bits(bits, 0, alpha=1),
        )
    )
    model.add(Activation("softmax", name="softmax"))

    return model



def build_qkeras_model(
    input_shape,
    n_classes,
    filters_per_conv_layer,
    neurons_per_dense_layer=None,
    bits=8,
    **kwargs,
):
    neurons_per_dense_layer = neurons_per_dense_layer or []

    kernel_size = kwargs.get('kernel_size', (3, 3))
    pool_size = kwargs.get('pool_size', (2, 2))
    batch_norm = kwargs.get('batch_norm', False)
    max_pool = kwargs.get('max_pool', False)
    use_bias = kwargs.get('use_bias', False)

    kernel_quantizer = f'quantized_bits({bits},0,alpha=1)'
    bias_quantizer = kernel_quantizer if use_bias else None

    x_in = Input(shape=input_shape)
    x = x_in

    for i, f in enumerate(filters_per_conv_layer):
        x = QConv2D(
            int(f),
            kernel_size=kernel_size,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            use_bias=use_bias,
            name=f'conv_{i}',
        )(x)
        if batch_norm:
            x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = QActivation(f'quantized_relu({bits})', name=f'conv_act_{i}')(x)
        if max_pool:
            x = MaxPooling2D(pool_size=pool_size, name=f'pool_{i}')(x)

    x = Flatten()(x)

    for i, n in enumerate(neurons_per_dense_layer):
        x = QDense(
            n,
            kernel_quantizer=kernel_quantizer,
            use_bias=False,
            name=f'dense_{i}',
        )(x)
        if batch_norm:
            x = BatchNormalization(name=f'bn_dense_{i}')(x)
        x = QActivation(f'quantized_relu({bits})', name=f'dense_act_{i}')(x)

    x = QDense(
        n_classes,
        kernel_quantizer=kernel_quantizer,
        use_bias=use_bias,
        bias_quantizer=bias_quantizer,
        name='output_dense',
    )(x)
    x_out = Activation('softmax', name='output_softmax')(x)

    return Model(inputs=[x_in], outputs=[x_out], name='qkeras_cnn')