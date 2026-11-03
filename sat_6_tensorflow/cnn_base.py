import tensorflow as tf
from keras.layers import Activation
from keras.models import Model, Sequential
from keras.layers import Input, MaxPooling2D,Conv2D,Dense, Activation, Flatten, Dense, BatchNormalization, ZeroPadding2D
from qkeras import QConv2D, QActivation, QDense, quantized_bits, quantized_relu


def _t1_bloco(x, conv, idx, batch_norm, max_pool, ativacao):
    """conv -> [pool] -> [BN] -> ReLU, na mesma ordem para as versões float e QKeras."""
    x = conv(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{idx}")(x)
    if batch_norm:
        x = BatchNormalization(name=f"bn{idx}")(x)
    return ativacao(name=f"relu{idx}")(x)


def build_t1(input_shape=(32, 32, 4), n_classes=6, batch_norm=False, max_pool=False, softmax=False):
    """Topologia t1 do paper FINN (Ely et al., SBCCI 2024; cnns_classes.t1_quantizedCNN):

    conv 3x3/2 4->20 (padding 1) -> ReLU -> conv 1x1 20->8 -> ReLU -> dense 2048->6, sem bias.
    O padding 1 explícito + 'valid' reproduz o padding=1 do PyTorch (o 'same' do Keras com
    stride 2 coloca o zero só no fim). batch_norm/max_pool adicionam BN/pooling depois de cada
    conv (não existem no paper). Como no FINN (top-1/LabelSelect sobre a saída da dense), a
    saída é em logits: treinar com CategoricalCrossentropy(from_logits=True). softmax=True
    adiciona a camada de softmax.
    """
    x_in = Input(shape=input_shape)
    x = ZeroPadding2D(padding=1, name="pad1")(x_in)
    x = _t1_bloco(
        x,
        Conv2D(20, kernel_size=3, strides=2, padding="valid", use_bias=False, name="conv1"),
        1, batch_norm, max_pool, lambda name: Activation("relu", name=name),
    )
    x = _t1_bloco(
        x,
        Conv2D(8, kernel_size=1, strides=1, padding="valid", use_bias=False, name="conv2"),
        2, batch_norm, max_pool, lambda name: Activation("relu", name=name),
    )
    x = Flatten(name="flatten")(x)
    x = Dense(n_classes, use_bias=False, name="output_dense")(x)
    if softmax:
        x = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x], name="t1_cnn")


def build_t1_quantized(
    input_shape=(32, 32, 4),
    n_classes=6,
    bits=8,
    batch_norm=False,
    max_pool=False,
    softmax=False,
):
    """Mesma t1 de build_t1, com pesos e ativações quantizados em `bits` (QKeras).

    Pesos: quantized_bits(bits, 0, alpha=1) ~ CommonWeightQuant do FINN (inteiro com sinal,
    escala fixa 1.0). Ativações: quantized_relu(bits) (o QuantReLU do FINN aprende a escala;
    aqui a faixa é fixa em [0, 1)).
    """
    w_q = quantized_bits(bits, 0, alpha=1)
    relu_q = lambda name: QActivation(quantized_relu(bits), name=name)

    x_in = Input(shape=input_shape)
    x = ZeroPadding2D(padding=1, name="pad1")(x_in)
    x = _t1_bloco(
        x,
        QConv2D(20, kernel_size=3, strides=2, padding="valid", use_bias=False,
                kernel_quantizer=w_q, name="conv1"),
        1, batch_norm, max_pool, relu_q,
    )
    x = _t1_bloco(
        x,
        QConv2D(8, kernel_size=1, strides=1, padding="valid", use_bias=False,
                kernel_quantizer=w_q, name="conv2"),
        2, batch_norm, max_pool, relu_q,
    )
    x = Flatten(name="flatten")(x)
    x = QDense(n_classes, use_bias=False, kernel_quantizer=w_q, name="output_dense")(x)
    if softmax:
        x = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x], name=f"t1_w{bits}")


def build_base_model(input_shape = (32, 32, 4), n_classes=6):

    x_in = Input(shape=input_shape)
    x = x_in

    x = Conv2D(
        filters=16,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)

    x = Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name="conv2",  
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)

    x = Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name="conv3",  
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Activation("relu", name="relu3")(x)

    x = Flatten(name="flatten")(x)

    x = Dense(32, name="dense1")(x)
    x = BatchNormalization(name="bn4")(x)
    x = Activation("relu", name="relu4")(x)

    x = Dense(n_classes, name="output_dense")(x)
    x_out = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x_out], name="t1_cnn")


def build_qkeras_model(
    input_shape,
    n_classes,
    filters_per_conv_layer,
    neurons_per_dense_layer=None,
    bits=8,
    use_integer=False,
    **kwargs,
):
    neurons_per_dense_layer = neurons_per_dense_layer or []

    kernel_size = kwargs.get('kernel_size', (3, 3))
    pool_size = kwargs.get('pool_size', (2, 2))
    batch_norm = kwargs.get('batch_norm', False)
    max_pool = kwargs.get('max_pool', False)
    use_bias = kwargs.get('use_bias', False)
    use_integer = kwargs.get('use_integer', False)


    if use_integer:
        kernel_quantizer = f'quantized_bits({bits},{bits - 1},keep_negative=True,alpha=1)'
    else:
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
        if max_pool:
            x = MaxPooling2D(pool_size=pool_size, name=f'pool_{i}')(x)
        if batch_norm:
            x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = QActivation(f'quantized_relu({bits})', name=f'conv_act_{i}')(x)

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
