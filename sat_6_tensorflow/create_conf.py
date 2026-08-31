from math import ceil, log2
import hls4ml
from cnn_base import build_t1_quantizedCNN

BITS = 8
GUARD_BITS = 1

model = build_t1_quantizedCNN(bit_quantization=BITS, input_shape=(32, 32, 4))

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name',
    backend='Vitis',
)

config['Model']['Strategy'] = 'Resource'
config['Model']['ReuseFactor'] = 1
config['Model']['Precision'] = f'ap_fixed<{BITS},0,AP_RND_CONV,AP_SAT>'


def accumulator_precision(fan_in, frac_bits=BITS, guard_bits=GUARD_BITS):
    """
    Smallest accumulator that is *guaranteed* not to overflow when summing `fan_in`
    signed 8-bit fixed-point products, plus a small guard margin, with saturation
    as a second line of defense.
    """
    int_bits = ceil(log2(fan_in)) + guard_bits
    total_bits = int_bits + frac_bits
    return f'ap_fixed<{total_bits},{int_bits},AP_RND_CONV,AP_SAT>'


# fan_in = in_channels * kernel_h * kernel_w for Conv2D, in_features for Dense
layer_fan_in = {
    'conv1': 4 * 3 * 3,     # 36
    'conv2': 20 * 1 * 1,    # 20
    'fc1':   8 * 16 * 16,   # 2048  <- the layer most at risk of overflow
}

for layer_name, fan_in in layer_fan_in.items():
    prec = accumulator_precision(fan_in)
    config['LayerName'][layer_name]['accum_t'] = prec
    config['LayerName'][layer_name]['result_t'] = prec
    # Force weights to the exact 8-bit signed format expected (matches
    # quantized_bits(8, 0, alpha=1, keep_negative=True) used when building the model)
    config['LayerName'][layer_name]['weight_t'] = f'ap_fixed<{BITS},0,AP_RND_CONV,AP_SAT>'

# QActivation (ReLU) layers -> unsigned 8-bit, matches quantized_relu(8, 0)
for layer_name in ('relu1', 'relu2'):
    config['LayerName'][layer_name]['Precision'] = f'ap_ufixed<{BITS},0,AP_RND_CONV,AP_SAT>'

# fc1 has no activation after it (raw logits) — give the *output* a little extra
# integer headroom purely for the final result_t so logits used for argmax don't
# saturate; this does NOT change the 8-bit weight/activation quantization anywhere
# upstream, only how the final accumulated logits are represented.
config['LayerName']['fc1']['result_t'] = accumulator_precision(
    layer_fan_in['fc1'], frac_bits=BITS, guard_bits=GUARD_BITS + 1
)

print(config)

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_t1_quantizedCNN_8b',
    part='xczu7ev-ffvc1156-2-e',   # replace with your target FPGA part
)

hls_model.compile()

if __name__ == '__main__':
    import numpy as np

    x = np.random.uniform(-1, 1, size=(1, 32, 32, 4)).astype('float32')
    y_keras = model.predict(x)
    y_hls = hls_model.predict(x)
    print('Keras output:', y_keras)
    print('hls4ml output:', y_hls)
    print('Max abs diff:', np.max(np.abs(y_keras - y_hls)))