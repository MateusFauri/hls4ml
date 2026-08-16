import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects


KERAS_DIR = "models/results"
DATASET = "data/sat-6-full.mat"
N_SAMPLES = 1000


co = {}
_add_supported_quantized_objects(co)


print("Loading dataset...")

data = sio.loadmat(DATASET)

X_test = data["test_x"]
y_test = data["test_y"]

X_test = X_test.astype(np.float32) / 255.0

X_test = np.transpose(
    X_test,
    (3, 0, 1, 2)
)

y_test = np.transpose(
    y_test,
    (1, 0)
)

if N_SAMPLES is not None:

    X_test = X_test[:N_SAMPLES]
    y_test = y_test[:N_SAMPLES]

print(f"Samples: {len(X_test)}")


def run(bits, experiment):

    for bit in bits:

        print("=" * 70)
        print(f"{bit} bits")
        print("=" * 70)

        model_path = os.path.join(
            KERAS_DIR,
            experiment,
            f"{bit}bits",
            "best",
            "model.h5"
        )

        if not os.path.exists(model_path):

            print("Model not found.")
            continue

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=co
        )

        output_dir = os.path.join(
            KERAS_DIR,
            experiment,
            f"{bit}bits"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        print("Running inference...")

        predictions = model.predict(
            X_test,
            verbose=0
        )

        input_file = os.path.join(
            output_dir,
            "input_features.dat"
        )

        prediction_file = os.path.join(
            output_dir,
            "output_predictions.dat"
        )

        print("Saving input_features.dat...")

        np.savetxt(
            input_file,
            X_test.reshape(
                X_test.shape[0],
                -1
            ),
            fmt="%.8f"
        )

        print("Saving output_predictions.dat...")

        np.savetxt(
            prediction_file,
            predictions,
            fmt="%.8f"
        )

        print("Done.\n")


    print("=" * 70)
    print("Finished.")
    print("=" * 70)


def main():

    run(
        bits=[8,4,2,1],
        experiment="t1"
    )


if __name__ == "__main__":
    main()