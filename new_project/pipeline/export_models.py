import os
import shutil

import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects


# ==========================================================
# CONFIGURAÇÕES PADRÃO
# ==========================================================

RESULTS_DIR = "results"
OUTPUT_DIR = "keras"


# ==========================================================
# CUSTOM OBJECTS
# ==========================================================

co = {}
_add_supported_quantized_objects(co)


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def run(bits, experiment):

    print("=" * 80)
    print("EXPORTING MODELS")
    print("=" * 80)

    for bit in bits:

        print(f"\n{bit} bits")

        source_model = os.path.join(
            RESULTS_DIR,
            experiment,
            f"{bit}bits",
            "models",
            "best_model.h5"
        )

        if not os.path.exists(source_model):

            print("Model not found.")
            continue

        output_folder = os.path.join(
            OUTPUT_DIR,
            experiment,
            f"{bit}bits"
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        destination_model = os.path.join(
            output_folder,
            "model.h5"
        )

        shutil.copy2(
            source_model,
            destination_model
        )

        model = tf.keras.models.load_model(
            destination_model,
            custom_objects=co
        )

        json_path = os.path.join(
            output_folder,
            "model.json"
        )

        with open(json_path, "w") as f:
            f.write(model.to_json())

        print("Done.")

    print("\nFinished exporting models.")


# ==========================================================
# EXECUÇÃO DIRETA
# ==========================================================

def main():

    run(
        bits=[8, 4, 2, 1],
        experiment="t1"
    )


if __name__ == "__main__":
    main()