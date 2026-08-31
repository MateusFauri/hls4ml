import os
import math
import yaml
import numpy as np
import tensorflow as tf
import hls4ml

from qkeras.utils import _add_supported_quantized_objects


KERAS_DIR = "models/results"
CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"
PART = "xc7z020clg400-1"
CLOCK_PERIOD = 10
BACKEND = "Vitis"
IO_TYPE = "io_stream"

BIT_WIDTH = 8

INT_LAYERS = ["conv2d_1", "conv2d_2", "dense_1", "dense_output"]

co = {}
_add_supported_quantized_objects(co)


def get_fan_in(model, layer_name):
    """Calcula o fan-in de uma camada Conv2D/Dense a partir dos pesos."""
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()[0]

    if weights.ndim == 4:
        kh, kw, in_ch, _ = weights.shape
        return kh * kw * in_ch
    elif weights.ndim == 2:
        in_features, _ = weights.shape
        return in_features
    else:
        raise ValueError(f"Formato de peso inesperado em '{layer_name}': {weights.shape}")


def compute_accum_bits(model, layer_name, weight_bits=BIT_WIDTH, act_bits=BIT_WIDTH):
    fan_in = get_fan_in(model, layer_name)
    extra_bits = max(1, math.ceil(math.log2(fan_in)))
    return weight_bits + act_bits + extra_bits


def run(bits, experiment):

    os.makedirs(CONFIG_DIR, exist_ok=True)

    for qbits in bits:
        print("=" * 70)
        print(f"Generating YAML ({qbits} bits) — integer-only")
        print("=" * 70)

        model_path = os.path.join(
            KERAS_DIR, experiment, f"{qbits}bits", "best", "model.h5"
        )

        if not os.path.exists(model_path):
            print("Model not found.\n")
            continue

        model = tf.keras.models.load_model(model_path, custom_objects=co)

        precision = f"int<{qbits}>"

        hls_config = hls4ml.utils.config_from_keras_model(
            model,
            granularity="name",
            default_precision=precision,
        )

        hls_config["Model"]["ReuseFactor"] = 1
        hls_config["Model"]["Strategy"] = "Latency"
        hls_config["Model"]["BramFactor"] = 4096
        hls_config["Model"]["TraceOutput"] = True

        for layer_name, layer_cfg in hls_config["LayerName"].items():
            layer_cfg["Trace"] = True

        for layer_name in INT_LAYERS:
            if layer_name not in hls_config["LayerName"]:
                print(f"  [aviso] camada '{layer_name}' não encontrada no modelo, pulando.")
                continue

            accum_bits = compute_accum_bits(model, layer_name, qbits, qbits)

            hls_config["LayerName"][layer_name]["Precision"] = {
                "weight": f"int<{qbits}>",
                "bias": f"int<{qbits}>",
                "accum": f"int<{accum_bits}>",
                "result": f"int<{qbits}>",
            }

        if "dense_output" in hls_config["LayerName"]:
            hls_config["LayerName"]["dense_output"]["Precision"]["result"] = f"int<{2 * qbits}>"

        if "network_input" in hls_config["LayerName"]:
            hls_config["LayerName"]["network_input"]["Precision"] = {
                "result": f"int<{qbits}>"
            }

        softmax_layers = [k for k in hls_config["LayerName"] if k.endswith("_softmax")]
        for sm_layer in softmax_layers:
            hls_config["LayerName"][sm_layer]["Precision"] = {
                "result": "ap_fixed<18,8>"
            }

        config = {
            "ModelName": f"{experiment}_best_{qbits}b",
            "Backend": BACKEND,
            "OutputDir": os.path.join(OUTPUT_DIR, experiment, f"{qbits}bits"),
            "ProjectName": f"{experiment}_best_{qbits}b",
            "KerasH5": model_path,
            "InputData": os.path.join(
                "data", "hls", experiment, f"{qbits}bits", "input_features.dat"
            ),
            "OutputPredictions": os.path.join(
                "data", "hls", experiment, f"{qbits}bits", "output_predictions.dat"
            ),
            "Part": PART,
            "ClockPeriod": CLOCK_PERIOD,
            "IOType": IO_TYPE,
            "HLSConfig": hls_config,
        }

        yaml_path = os.path.join(CONFIG_DIR, f"{experiment}_{qbits}b.yaml")

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"Saved: {yaml_path}\n")


def main():
    run(bits=[8], experiment="t1")


if __name__ == "__main__":
    main()