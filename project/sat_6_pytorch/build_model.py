import numpy as np
import torch
import pprint
import onnx
import hls4ml
from torch.utils.data import DataLoader
from project.sat_6_pytorch.utils.utils import read_object
from hls4ml.model.profiling import numerical
from onnx import numpy_helper
from project.sat_6_pytorch.cnn_base import t1_quantizedCNN
from sklearn.metrics import accuracy_score


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def trace_layers(config):
    config['Model']['Trace'] = True
    for layer in config['LayerName']:
        config['LayerName'][layer]['Trace'] = True

def set_config(config):
    trace_layers(config)
    config['LayerName']['global_in']['Precision'] = 'ap_uint<8>'

    for layer in ['Conv_0', 'Conv_1', 'MatMul_0']:
        config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<8,1>'
        config['LayerName'][layer]['Precision']['accum']  = 'ap_fixed<24,12>'
        config['LayerName'][layer]['Precision']['result']  = 'ap_fixed<16,6>'
        config['LayerName'][layer]['Strategy'] = 'Resource'

    config['LayerName']['MatMul_0']['ReuseFactor'] = 64
    config['LayerName']['Conv_0']['ReuseFactor'] = 8
    config['LayerName']['Conv_1']['ReuseFactor'] = 8

    for layer in ['Relu_0', 'Relu_1']:
        config['LayerName'][layer]['Precision']['result'] = 'ap_ufixed<8,0>'

    for layer in ['Quant_0', 'Quant_1', 'Quant_2']:
        config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<8,1>'

    for layer in ['Quant_3', 'Quant_4']:
        config['LayerName'][layer]['Precision']['result'] = 'ap_ufixed<8,0>'

    return config


BUILD = False
batch_size = 512
test_dataset_output_name = "./dataset/test_dataset.pkl"
onnx_model_path = "./hls4ml_onnxs/sat6-cnn-t1w8.onnx"
pytorch_model_path = "./pytorch_models/sat6-cnn-t1.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_test = read_object(test_dataset_output_name)
loader_test = DataLoader(dataset_test, batch_size, shuffle=False)

X_test = dataset_test.tensors[0].numpy().astype(np.float32)
y_test = dataset_test.tensors[1].numpy().astype(np.int32)

model = t1_quantizedCNN(8)
model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
model.eval()

with torch.no_grad():
    y_ref = model(torch.FloatTensor(X_test)).numpy()


onnx_model = onnx.load(onnx_model_path)

config = hls4ml.utils.config_from_onnx_model(
    onnx_model,
    granularity='name',
    backend='Vitis',
    default_precision='ap_fixed<16,6>'
)

config = set_config(config)

hls_model = hls4ml.converters.convert_from_onnx_model(
    onnx_model,
    output_dir='./hls4ml_prjs/hls4ml_prj_base_t1',
    io_type='io_stream',
    backend='Vitis',
    hls_config=config,
    part='xc7z020clg400-1',
)
hls_model.compile()



y_hls = hls_model.predict(np.ascontiguousarray(X_test, dtype=np.float32))

print('Accuracy (Pytorch): {}'.format(accuracy_score(y_test, np.argmax(y_ref, axis=1))))
print('Accuracy (HLS): {}'.format(accuracy_score(y_test, np.argmax(y_hls, axis=1))))


if BUILD: 
    hls_model.build(csim=False, synth=True, cosim=False, export=True)

