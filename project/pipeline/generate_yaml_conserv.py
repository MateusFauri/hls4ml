import os
import yaml
import hls4ml
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)


def build_hls_config(model: tf.keras.Model, reuse_factor: int = 1) -> dict:

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        default_precision="ap_fixed<8,1>",
    )

    config["Model"]["Strategy"] = "Latency"
    config["Model"]["ReuseFactor"] = reuse_factor

    for layer_cfg in config["LayerName"].values():
        layer_cfg["Trace"] = True

    return config


def widen_accumulators(config: dict, accum_precision: str = "ap_int<24>") -> dict:
    for layer_name, layer_cfg in config["LayerName"].items():
        if "Precision" in layer_cfg and "accum" in layer_cfg["Precision"]:
            layer_cfg["Precision"]["accum"] = accum_precision
    return config


def generate_yaml(bits: int, experiment: str = "t1"):
    model_path = f"models/results/{experiment}/{bits}bits/model.h5"
    model = tf.keras.models.load_model(model_path, custom_objects=co)

    hls_config = build_hls_config(model, reuse_factor=1)
    hls_config = widen_accumulators(hls_config, accum_precision="ap_int<24>")

    config = {
        "ModelName": f"{experiment}_best_{bits}b",
        "Backend": "Vitis",
        "OutputDir": f"outputs/{experiment}/{bits}bits",
        "ProjectName": f"{experiment}_best_{bits}b",
        "KerasH5": model_path,
        "InputData": f"data/hls/{experiment}/{bits}bits/input_features.dat",
        "OutputPredictions": f"data/hls/{experiment}/{bits}bits/output_predictions.dat",
        "Part": "xc7z020clg400-1",
        "ClockPeriod": 10,
        "IOType": "io_stream",
        "HLSConfig": hls_config,
    }

    yaml_path = f"configs/{experiment}_{bits}b.yaml"
    os.makedirs("configs", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Config gerada: {yaml_path}")


if __name__ == "__main__":
    for bits in [8, 4, 2, 1]:
        generate_yaml(bits)