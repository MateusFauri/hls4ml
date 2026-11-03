import tensorflow as tf
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Conv2D, Dense, Activation, Flatten, BatchNormalization, ZeroPadding2D
from qkeras import QConv2D, QConv2DBatchnorm, QActivation, QDense, quantized_bits, quantized_relu


def build_t1(input_shape = (32, 32, 4), n_classes=6, max_pool=False, batch_norm=False, softmax=False):
    x_in = Input(shape=input_shape)
    x = ZeroPadding2D(padding=1, name="pad1")(x_in)

    x = Conv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding="valid",
        use_bias=False,
        name="conv1",
    )(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    if batch_norm:
        x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)

    x = Conv2D(
        filters=8,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        name="conv2",  
    )(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    if batch_norm:
        x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)
    
    x = Flatten(name="flatten")(x)
    x = Dense(n_classes, use_bias=False, name="output_dense")(x)
    if softmax:
        x = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x], name="t1_cnn")


def input_quantizer(bits=8, signed_input=False):
    # hls4ml ignora keep_negative=False em quantized_bits (vira fixed<8,1>, perde 1 bit),
    # então a entrada sem sinal usa quantized_relu, que é identidade para x >= 0 e vira ufixed<8,0>
    return f"quantized_bits({bits},0,alpha=1)" if signed_input else f"quantized_relu({bits})"


def weight_quantizer(bits=8):
    return f"quantized_bits({bits},0,alpha=1)"


# bias vai para o acumulador, então usa mais bits que os pesos
BIAS_QUANTIZER = "quantized_bits(16,6,alpha=1)"


def qconv(x, filters, kernel_size, strides, padding, bits, batch_norm, name):
    if batch_norm:
        return QConv2DBatchnorm(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_quantizer=weight_quantizer(bits), bias_quantizer=BIAS_QUANTIZER, name=name,
        )(x)
    return QConv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=False,
        kernel_quantizer=weight_quantizer(bits), name=name,
    )(x)


def build_t1_quantizade(input_shape = (32, 32, 4), n_classes=6, bits=8, max_pool=False, batch_norm=False, signed_input=False, softmax=False):
    # Todos os tensores usam escala potência de 2, então em hardware o datapath é inteiro:
    #   entrada  ap_ufixed<8,0> (x/256)          == uint8   | signed_input: ap_fixed<8,1> ((x-128)/128) == int8
    #   pesos    ap_fixed<8,1>                   == int8 * 2^-7
    #   relu     ap_ufixed<8,0>                  == uint8 * 2^-8
    #   bias     ap_fixed<16,7>  (só com batch_norm, vindo da BN dobrada; fica no acumulador)
    x_in = Input(shape=input_shape)
    x = QActivation(input_quantizer(8, signed_input), name="input_quant")(x_in)

    x = ZeroPadding2D(padding=1, name="pad1")(x)
    x = qconv(x, 20, 3, 2, "valid", bits, batch_norm, "conv1")
    x = QActivation(f"quantized_relu({bits})", name="relu1")(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = qconv(x, 8, 1, 1, "valid", bits, batch_norm, "conv2")
    x = QActivation(f"quantized_relu({bits})", name="relu2")(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = Flatten(name="flatten")(x)
    x = QDense(n_classes, use_bias=False, kernel_quantizer=weight_quantizer(bits), name="output_dense")(x)
    if softmax:
        x = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x], name="t1_cnn")


def prepare_input(X_raw, model):
    """Normaliza X (uint8 0..255) conforme a quantização de entrada do modelo."""
    X_raw = X_raw.astype('float32')
    names = [l.name for l in model.layers]
    if 'input_quant' not in names:
        return X_raw / 255.0  # modelos antigos, sem quantização de entrada
    if 'quantized_bits' in str(model.get_layer('input_quant').quantizer):
        return (X_raw - 128.0) / 128.0  # int8 com sinal -> ap_fixed<8,1>
    return X_raw / 256.0  # uint8 -> ap_ufixed<8,0>


def set_input_precision(hls_config, model):
    """Faz a porta de entrada do hls4ml ter 8 bits em vez do padrão fixed<16,6>."""
    if 'input_quant' not in [l.name for l in model.layers]:
        return
    signed = 'quantized_bits' in str(model.get_layer('input_quant').quantizer)
    hls_config['LayerName'][model.layers[0].name]['Precision']['result'] = 'fixed<8,1>' if signed else 'ufixed<8,0>'
