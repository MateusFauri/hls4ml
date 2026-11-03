import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Conv2D, Dense, Activation, Flatten, BatchNormalization
from keras.initializers import Constant, VarianceScaling
from qkeras import QConv2D, QConv2DBatchnorm, QActivation, QDense


# Treino explicitamente inteiro: todo tensor da rede é um inteiro, e a única escala é um
# shift (2^k) por canal, aprendido pelo QKeras (alpha='auto_po2'). Não há normalização da entrada.
#
#   entrada  x   : uint8 bruto 0..255                         -> ap_uint<8>  | signed_input: x-128 -> ap_int<8>
#   pesos    w   : int8 [-127,127], escala 2^k por canal       -> ap_int<8> + ApplyAlpha (shift)
#   acumul.  acc : int32 = sum(x * w)
#   bias     b   : int16 no domínio da saída
#   relu     y   : uint8 = clip(round(acc * 2^k + b), 0, 255)  -> ap_uint<8>
#   logits       : acc * 2^k + b, com b em passos de 2^-8 (sem softmax no FPGA basta o argmax)
#
# Com bits=4 pesos e ativações viram int4/uint4 (0..15); a entrada continua uint8.

def input_quantizer(bits=8, signed_input=False):
    # quantized_relu(b, b) -> ufixed<b,b> == uint; quantized_bits(b, b-1) -> fixed<b,b> == int
    # (quantized_bits com keep_negative=False não serve: o hls4ml sempre gera tipo com sinal)
    return f"quantized_bits({bits},{bits - 1},alpha=1)" if signed_input else f"quantized_relu({bits},{bits})"


def weight_quantizer(bits=8):
    # integer=bits-1 -> a mantissa é o próprio inteiro; auto_po2 escolhe o shift 2^k de cada canal
    return f"quantized_bits({bits},{bits - 1},symmetric=1,alpha='auto_po2')"


def act_quantizer(bits=8):
    return f"quantized_relu({bits},{bits})"


HIDDEN_BIAS_QUANTIZER = "quantized_bits(16,15,alpha=1)"  # int16, mesma escala da saída uint
OUTPUT_BIAS_QUANTIZER = "quantized_bits(16,7,alpha=1)"   # logits: int16 * 2^-8


def kernel_init(in_bits, out_bits):
    # Glorot reescalado por 2^(out_bits - in_bits): a saída inicial cabe em 0..2^out_bits-1.
    # Só muda algo na conv1 com bits=4 (entrada uint8 -> saída uint4); sem isso quase toda relu
    # satura em 15 e o gradiente morre
    return VarianceScaling(scale=4.0 ** (out_bits - in_bits), mode="fan_avg", distribution="uniform")


def qconv(x, filters, kernel_size, strides, in_bits, bits, batch_norm, name):
    if batch_norm:
        # BN dobrada nos pesos (inteiros) durante o treino. A BN normaliza para ~N(0,1),
        # então gamma começa em 2^(bits-3) para a saída usar a faixa do uint
        return QConv2DBatchnorm(
            filters, kernel_size, strides=strides, padding="same",
            kernel_quantizer=weight_quantizer(bits), bias_quantizer=HIDDEN_BIAS_QUANTIZER,
            gamma_initializer=Constant(2.0 ** (bits - 3)), name=name,
        )(x)
    return QConv2D(
        filters, kernel_size, strides=strides, padding="same", use_bias=False,
        kernel_quantizer=weight_quantizer(bits), kernel_initializer=kernel_init(in_bits, bits), name=name,
    )(x)


def build_t1(input_shape = (32, 32, 4), n_classes=6, max_pool=False, batch_norm=False):

    x_in = Input(shape=input_shape)
    x = x_in

    x = Conv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding="same",
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
        padding="same",
        use_bias=False,
        name="conv2",
    )(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    if batch_norm:
        x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(n_classes, name="output_dense")(x)
    x_out = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x_out], name="t1_cnn")


def build_t1_int(input_shape = (32, 32, 4), n_classes=6, bits=8, max_pool=False, batch_norm=False, signed_input=False):
    x_in = Input(shape=input_shape)
    # entrada sempre 8 bits (o dado é uint8); com bits=4, quantized_relu(4,4) cortaria tudo acima de 15
    x = QActivation(input_quantizer(8, signed_input), name="input_quant")(x_in)

    # conv -> (bn) -> relu -> pool: relu e max-pool são monótonos, então o pool depois da relu
    # dá o mesmo resultado e compara inteiros de 8 bits em vez do acumulador
    x = qconv(x, 20, 3, 2, 8, bits, batch_norm, "conv1")
    x = QActivation(act_quantizer(bits), name="relu1")(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = qconv(x, 8, 1, 1, bits, bits, batch_norm, "conv2")
    x = QActivation(act_quantizer(bits), name="relu2")(x)
    if max_pool:
        x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = Flatten(name="flatten")(x)
    x = QDense(
        n_classes, kernel_quantizer=weight_quantizer(bits), bias_quantizer=OUTPUT_BIAS_QUANTIZER,
        kernel_initializer=kernel_init(bits, bits), name="output_dense",
    )(x)
    x_out = Activation("softmax", name="softmax_output")(x)
    return Model(inputs=[x_in], outputs=[x_out], name="t1_cnn")


def prepare_input(X_raw, model):
    """Entrada inteira: uint8 bruto, ou x-128 (int8) se o modelo tiver entrada com sinal."""
    X = X_raw.astype("float32")
    if "input_quant" in [l.name for l in model.layers] and _signed_input(model):
        return X - 128.0
    return X


def set_input_precision(hls_config, model):
    """Porta de entrada do hls4ml como inteiro de 8 bits em vez do padrão fixed<16,6>,
    e as relus como inteiro sem sinal COM saturação."""
    names = [l.name for l in model.layers]
    if "input_quant" not in names:
        return
    bits = model.get_layer("input_quant").quantizer.bits
    hls_config["LayerName"][model.layers[0].name]["Precision"]["result"] = (
        f"ap_int<{bits}>" if _signed_input(model) else f"ap_uint<{bits}>"
    )
    # o hls4ml gera uint<b> para quantized_relu(b,b), mas ap_uint não tem saturação: o cast trunca e
    # dá a volta (256 -> 0). ufixed<b,b,RND_CONV,SAT> é o mesmo inteiro, mas satura em 2^b-1 como no QKeras
    for layer in model.layers:
        if isinstance(layer, QActivation) and layer.name != "input_quant":
            b = layer.quantizer.bits
            hls_config["LayerName"][layer.name]["Precision"]["result"] = f"ufixed<{b},{b},RND_CONV,SAT>"


def _signed_input(model):
    return "quantized_bits" in str(model.get_layer("input_quant").quantizer)


def fix_alpha_precision(hls_model, logits_precision="fixed<20,10,RND,SAT>"):
    """Precisão das camadas ApplyAlpha (o shift 2^k por canal) que o hls4ml insere após cada conv/dense.

    O hls4ml não sabe inferir a precisão quando a escala é potência de 2 e cai no padrão
    fixed<16,6,TRN,WRAP>, que dá overflow (as ativações vão até 255). Aqui cada uma recebe o tipo
    exato de acc * 2^k + b, e a que alimenta o softmax recebe logits_precision.
    Chamar depois de convert_from_keras_model e antes de compile/build.
    """
    from hls4ml.model.types import FixedPrecisionType
    from hls4ml.backends.fpga.fpga_types import APTypeConverter  # tipos ap_* do Vivado/Vitis

    layers = list(hls_model.get_layers())
    for i, layer in enumerate(layers):
        if layer.class_name != "ApplyAlpha":
            continue
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if nxt is not None and nxt.class_name == "Softmax":
            prec = hls_model.config.backend.convert_precision_string(logits_precision)
        else:
            inp = layer.get_input_variable().type.precision
            k = np.log2(np.asarray(layer.weights["scale"].data, dtype=np.float64)).round().astype(int)
            bias = layer.weights["bias"].type.precision
            frac = max(int(-k.min()) + inp.width - inp.integer, bias.width - bias.integer, 0)
            integer = max(inp.integer + int(k.max()), bias.integer) + 1
            prec = FixedPrecisionType(width=integer + frac, integer=integer, signed=True)
        prec = APTypeConverter().convert(prec)
        layer.types["result_t"].precision = prec
        layer.get_output_variable().type.precision = prec


# ---------------------------------------------------------------------------------------------
# Exportação dos inteiros e inferência só com aritmética inteira (numpy), para conferir que o
# modelo treinado é de fato inteiro: deve bater exatamente com model.predict e com o hls4ml.
# ---------------------------------------------------------------------------------------------

def _int_weights(layer):
    """(pesos int, expoente k por canal, bias) de uma camada QConv2D/QConv2DBatchnorm/QDense."""
    if isinstance(layer, QConv2DBatchnorm):
        kernel, bias = layer.get_folded_weights()
    else:
        kernel = layer.kernel
        bias = layer.bias if layer.use_bias else tf.zeros(kernel.shape[-1])
    kq = layer.kernel_quantizer_internal
    w = kq(kernel).numpy().astype(np.float64)
    scale = np.reshape(np.asarray(kq.scale, dtype=np.float64), -1)  # 2^k por canal (já inclui a mantissa)
    w_int = np.round(w / scale)
    assert np.allclose(w_int * scale, w), f"{layer.name}: pesos não são inteiro * 2^k"
    k = np.log2(scale).round().astype(int)
    b = layer.bias_quantizer_internal(bias).numpy().astype(np.float64) if layer.bias_quantizer_internal else np.asarray(bias, np.float64)
    return w_int.astype(np.int64), k, b


def export_integer_model(model):
    """Lista de operações com os inteiros e shifts de cada camada (o que vai para o FPGA)."""
    ops = []
    for layer in model.layers:
        if isinstance(layer, (QConv2D, QConv2DBatchnorm)):
            w, k, b = _int_weights(layer)
            ops.append(dict(op="conv", name=layer.name, w=w, k=k, b=b,
                            strides=layer.strides[0], padding=layer.padding))
        elif isinstance(layer, QDense):
            w, k, b = _int_weights(layer)
            ops.append(dict(op="dense", name=layer.name, w=w, k=k, b=b))
        elif isinstance(layer, QActivation) and layer.name != "input_quant":
            ops.append(dict(op="relu", name=layer.name, bits=layer.quantizer.bits))
        elif isinstance(layer, MaxPooling2D):
            ops.append(dict(op="maxpool", name=layer.name, size=layer.pool_size[0]))
        elif isinstance(layer, Flatten):
            ops.append(dict(op="flatten", name=layer.name))
    return ops


def _conv_int(x, w, stride, padding):
    # x: (N,H,W,C) int64, w: (kh,kw,C,F) int64 -> acumulador int64, mesmo "same" do TF
    kh, kw = w.shape[:2]
    if padding == "same":
        h, wd = x.shape[1:3]
        ph = max((-(-h // stride) - 1) * stride + kh - h, 0)
        pw = max((-(-wd // stride) - 1) * stride + kw - wd, 0)
        x = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]
    return np.einsum("nhwcij,ijcf->nhwf", win, w, optimize=True)


def _requant(acc, k, b):
    # acc * 2^k + b em racional exato (numerador inteiro / 2^shift)
    shift = max(int(-k.min()), 0) + 8  # +8 para o bias da saída (passo 2^-8)
    num = acc * (2 ** (k + shift)).astype(np.int64) + np.round(b * 2 ** shift).astype(np.int64)
    return num, shift


def integer_inference(ops, X_int):
    """Inferência só com inteiros. Retorna os logits (float64 exato) antes do softmax."""
    x = X_int.astype(np.int64)
    num, shift = None, 0
    for op in ops:
        if op["op"] == "conv":
            num, shift = _requant(_conv_int(x, op["w"], op["strides"], op["padding"]), op["k"], op["b"])
        elif op["op"] == "dense":
            num, shift = _requant(x @ op["w"], op["k"], op["b"])
        elif op["op"] == "relu":
            # round half-to-even (igual ao tf.round do QKeras e ao RND_CONV do hls4ml), satura em 2^bits-1
            q = num / 2.0 ** shift
            x = np.clip(np.round(q), 0, 2 ** op["bits"] - 1).astype(np.int64)
            num = None
        elif op["op"] == "maxpool":
            s = op["size"]
            n, h, w, c = x.shape
            x = x[:, : h // s * s, : w // s * s].reshape(n, h // s, s, w // s, s, c).max(axis=(2, 4))
        elif op["op"] == "flatten":
            x = x.reshape(len(x), -1)
    return num / 2.0 ** shift
