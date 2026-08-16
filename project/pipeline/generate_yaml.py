import os
import yaml
import tensorflow as tf

from qkeras.utils import _add_supported_quantized_objects


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
KERAS_DIR = "models/results"
CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"
PART = "xc7z020clg400-1"
CLOCK_PERIOD = 10
BACKEND = "Vitis"
IO_TYPE = "io_stream"

co = {}
_add_supported_quantized_objects(co)


def run(bits, experiment):

    os.makedirs(CONFIG_DIR, exist_ok=True)

    for qbits in bits:
        print("=" * 70)
        print(f"Generating YAML ({qbits} bits)")
        print("=" * 70)

        model_path = os.path.join(
            KERAS_DIR,
            experiment,
            f"{qbits}bits",
            "best",
            "model.h5"
        )

        if not os.path.exists(model_path):
            print("Model not found.\n")
            continue

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=co
        )

        precision = f"ap_uint<{qbits}>"
        accum = f"ap_uint<{qbits*2}>"


        layer_name = {}

        for layer in model.layers:

            layer_name[layer.name] = {
                "Trace": True
            }

        hls_config = {
            "Model": {
                "Strategy": "Latency",
                "ReuseFactor": 1,
                "Precision": {
                    "default": precision,
                    "weight": precision,
                    "bias": precision,
                    "result": precision,
                    "accum": accum
                }
            },
            "LayerType": {
                "Conv2D": {
                    "Strategy": "Latency",
                    "ReuseFactor": 1,
                    "Precision": {
                        "weight": precision,
                        "bias": precision,
                        "result": precision,
                        "accum": accum
                    }
                },
                "Dense": {
                    "Strategy": "Latency",
                    "ReuseFactor": 1,
                    "Precision": {
                        "weight": precision,
                        "bias": precision,
                        "result": precision,
                        "accum": accum
                    }
                }
            },
            "LayerName": layer_name
        }
        config = {
            "ModelName":
                f"{experiment}_best_{qbits}b",
            "Backend":
                BACKEND,
            "OutputDir":
                os.path.join(
                    OUTPUT_DIR,
                    experiment,
                    f"{qbits}bits"
                ),
            "ProjectName":
                f"{experiment}_best_{qbits}b",
            "KerasH5":
                model_path,
            "InputData":
                os.path.join(
                    "data",
                    "hls",
                    experiment,
                    f"{qbits}bits",
                    "input_features.dat"
                ),
            "OutputPredictions":
                os.path.join(
                    "data",
                    "hls",
                    experiment,
                    f"{qbits}bits",
                    "output_predictions.dat"
                ),
            "Part":
                PART,
            "ClockPeriod":
                CLOCK_PERIOD,
            "IOType":
                IO_TYPE,
            "HLSConfig":
                hls_config
        }

        yaml_path = os.path.join(
            CONFIG_DIR,
            f"{experiment}_{qbits}b.yaml"
        )

        with open(yaml_path, "w") as f:
            yaml.dump(
                config,
                f,
                sort_keys=False
            )

        print(f"Saved: {yaml_path}\n")



def main():
    run(
        bits=[8,4,2,1],
        experiment="t1"
    )


if __name__ == "__main__":
    main()