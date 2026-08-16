import os
import yaml

import numpy as np
import pandas as pd

import tensorflow as tf
import hls4ml

from sklearn.metrics import accuracy_score

from qkeras.utils import _add_supported_quantized_objects


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

CONFIG_DIR = "configs"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


# ==========================================================
# CUSTOM OBJECTS
# ==========================================================

co = {}
_add_supported_quantized_objects(co)


# ==========================================================
# RELATÓRIO
# ==========================================================

results = []


# ==========================================================
# LOOP
# ==========================================================
def run(bits, experiment):
    for bit in bits:

        print("=" * 80)
        print(f"{bit} bits")
        print("=" * 80)

        yaml_path = os.path.join(
            CONFIG_DIR,
            f"{experiment}_{bit}b.yaml"
        )

        if not os.path.exists(yaml_path):

            print("YAML not found.\n")
            continue

        with open(yaml_path, "r") as f:

            config = yaml.safe_load(f)

        print("Loading keras model...")

        model = tf.keras.models.load_model(
            config["KerasH5"],
            custom_objects=co
        )

        print("Loading input data...")

        X = np.loadtxt(
            config["InputData"]
        )

        print("Loading keras predictions...")

        keras_predictions = np.loadtxt(
            config["OutputPredictions"]
        )

        print("Converting model...")

        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config["HLSConfig"],
            backend=config["Backend"],
            output_dir=config["OutputDir"],
            part=config["Part"],
            clock_period=config["ClockPeriod"],
            io_type=config["IOType"],
            project_name=config["ProjectName"]
        )

        print("Compiling...")

        hls_model.compile()

        print("Running inference...")

        hls_predictions = hls_model.predict(X)

        keras_labels = np.argmax(
            keras_predictions,
            axis=1
        )

        hls_labels = np.argmax(
            hls_predictions,
            axis=1
        )

        agreement = np.mean(
            keras_labels == hls_labels
        )

        mae = np.mean(
            np.abs(
                keras_predictions -
                hls_predictions
            )
        )

        mse = np.mean(
            (
                keras_predictions -
                hls_predictions
            ) ** 2
        )

        max_error = np.max(
            np.abs(
                keras_predictions -
                hls_predictions
            )
        )

        results.append({

            "bits": bit,

            "agreement": agreement,

            "mae": mae,

            "mse": mse,

            "max_error": max_error

        })

        print(f"Agreement : {agreement:.6f}")
        print(f"MAE       : {mae:.8f}")
        print(f"MSE       : {mse:.8f}")
        print(f"Max Error : {max_error:.8f}")

        print()


    # ==========================================================
    # RELATÓRIO
    # ==========================================================

    report = pd.DataFrame(results)

    report.to_csv(

        os.path.join(
            REPORT_DIR,
            f"{experiment}_{bit}b_conversion_report.csv"
        ),

        index=False

    )

    print("=" * 80)
    print("Conversion finished.")
    print("=" * 80)

    print(report)

def main():

    run(
        bits=[8,4,2,1],
        experiment="t1"
    )


if __name__ == "__main__":
    main()