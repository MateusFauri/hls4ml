import re
import shutil
import argparse
import hls4ml
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hls4ml.model.profiling import numerical
from hls4ml.converters import keras_v2_to_hls

from pathlib import Path

def load_data_profilling():
    X_test = pd.read_csv('data/X_test_sat6.csv')

    #X_test = np.transpose(X_test, (3, 0, 1, 2))
    #X_test = X_test.astype(np.float32) / 255.0

    return X_test.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profiling",
        action='store_true',
        help='Enable Profiling'
    )

    parser.add_argument(
        "--keras_config",
        action='store_true',
        help='Enable keras_config'
    )

    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Generate only frontend/project files without synthesis flow",
    )

    parser.add_argument(
        "--config",
        default='keras',
        help=(
            "Configuration YAML filename inside src/conf "
            "(without .yaml extension)"
        ),
    )

    parser.add_argument(
        "--no-csim",
        action="store_true",
        help="Disable C simulation",
    )

    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Disable synthesis",
    )

    parser.add_argument(
        "--no-cosim",
        action="store_true",
        help="Disable co-simulation",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable export",
    )

    return parser.parse_args()


def main():

    args = parse_args()
    config_file_name = args.config
    config_path = f"hls_conf/{config_file_name}.yaml"

    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["ModelName"]
    model = f"models/keras/{model_name}.keras"
    png_output_path = f"{model_name}.png"

    output_dir = config["OutputDir"]
    csim = not args.no_csim
    synth = not args.no_synth
    cosim = not args.no_cosim
    export = not args.no_export


    print("[INFO] Cleaning old project...")
    shutil.rmtree(output_dir, ignore_errors=True)

    print("[INFO] Generating hls4ml project...")

    if args.keras_config:
        hls_model = hls4ml.converters.keras_v2_to_hls(config)
    else:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config["HLSConfig"],
            output_dir=config["OutputDir"],
            backend=config["Backend"],
            part=config["Part"],
            clock_period=config["ClockPeriod"],
            io_type=config["IOType"],
            project_name=config["ProjectName"],
        )

    if args.profiling:
        print("[INFO] Profiling enable")
        X = load_data_profilling()
        plots = numerical(model=model, hls_model=hls_model, X=X)
        plt.show()

    print("[INFO] Generating model plot...")
    hls4ml.utils.plot_model(
        hls_model,
        show_shapes=True,
        show_precision=True,
        to_file=png_output_path,
    )

    if args.frontend_only:
        print("[INFO] Frontend-only mode enabled.")

        print("[INFO] Writing project...")
        hls_model.write()

        print("[INFO] Done.")
        return

    #print("[INFO] Linking existing project...")
    #hls_model = hls4ml.converters.link_existing_project(output_dir)

    print("[INFO] Compiling...")
    hls_model.compile()

    print("[INFO] Build configuration:")
    print(f"  - csim  = {csim}")
    print(f"  - synth = {synth}")
    print(f"  - cosim = {cosim}")
    print(f"  - export = {export}")

    print("[INFO] Running synthesis/build flow...")
    hls_model.build(
        csim=csim,
        synth=synth,
        cosim=cosim,
        export=export,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()