"""
python main.py \
    --base-dir t1 t2 t3 \
    --reuse-factor 1 2 \
    --strategy Resource Latency \
    --skip-synthesis
    --fpga-part xck26-sfvc784-2LV-c

python main.py \
    --base-dir t1 t2 \
    --reuse-factor 1 2 \
    --strategy Resource Latency \
    --io-type io_stream \
    --clock-period-ns 10 \
    --continue-on-error
"""

import argparse
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import hls4ml
from qkeras.utils import load_qmodel
from sklearn.metrics import accuracy_score
from hls4ml.model.profiling import numerical


def load_pickle_data(path):
    print(f"Lendo {path} ...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]

def trace_layers(config):
    config["Model"]["TraceOutput"] = True

    for layer in config["LayerName"]:
        config["LayerName"][layer]["Trace"] = True

def print_dict(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * max(1, 20 - len(key) - 2 * indent) + str(value))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa conversão, análise numérica e síntese hls4ml para uma ou várias configurações."
    )

    parser.add_argument(
        "--base-dir",
        nargs="+",
        required=True,
        help="Um ou mais diretórios de modelos dentro de models/. Ex.: --base-dir t1 t2 t3",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Se uma configuração falhar, registra o erro e segue para a próxima.",
    )

    parser.add_argument(
        "--reuse-factor",
        type=int,
        nargs="+",
        default=[1],
        help="Um ou mais ReuseFactor. Ex.: --reuse-factor 1 2 4",
    )

    parser.add_argument(
        "--strategy",
        nargs="+",
        default=["Resource"],
        choices=["Resource", "Latency"],
        help="Uma ou mais estratégias. Ex.: --strategy Resource Latency",
    )

    parser.add_argument(
        "--io-type",
        nargs="+",
        default=["io_stream"],
        choices=["io_stream", "io_parallel"],
        help="Um ou mais tipos de I/O.",
    )

    parser.add_argument(
        "--clock-period-ns",
        type=float,
        nargs="+",
        default=[10],
        help="Um ou mais períodos de clock em ns. Ex.: --clock-period-ns 5 10",
    )

    parser.add_argument(
        "--fpga-part", 
        default="xc7z020clg400-1",
        help="Part da FPGA. Ex.: xc7z020clg400-1",
    )

    parser.add_argument(
        "--backend",
        default="Vitis",
        help="Backend do hls4ml. Ex.: Vitis",
    )

    parser.add_argument(
        "--train-pickle",
        default="./dataset/sat6_train.pkl",
        help="Arquivo pickle de treinamento.",
    )

    parser.add_argument(
        "--test-pickle",
        default="./dataset/sat6_test.pkl",
        help="Arquivo pickle de teste.",
    )

    parser.add_argument(
        "--model-name",
        default="model",
        help="Nome do arquivo .h5 dentro de models/BASE_DIR.",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=3000,
        help="Número de amostras usadas na avaliação de acurácia.",
    )

    parser.add_argument(
        "--numerical-size",
        type=int,
        default=2000,
        help="Número de amostras usadas no numerical().",
    )

    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Não executa hls_model.build(..., synth=True, vsynth=True).",
    )

    parser.add_argument(
        "--skip-numerical",
        action="store_true",
        help="Não executa análise numerical().",
    )

    return parser.parse_args()

def make_output_dir(base_dir, io_type, reuse_factor, strategy, clock_period_ns):
    dir = 'projects'
    os.makedirs(dir, exist_ok=True)

    if clock_period_ns == 10:
        name = f"prj_{base_dir}_{io_type}_rf{reuse_factor}_{strategy}"
    else:
        clock_str = str(clock_period_ns).replace(".", "p")
        name = f"prj_{base_dir}_{io_type}_rf{reuse_factor}_{strategy}_cp{clock_str}"

    output_dir = os.path.join(".",dir, name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_model_plot(hls_model, output_dir):
    path = os.path.join(output_dir, "model_plot.png")
    try:
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=path,
        )
        print(f"Modelo salvo em: {path}")
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar plot_model: {e}")

def save_numerical_plot(qmodel, hls_model, X_test, output_dir, numerical_size):
    X_num = np.ascontiguousarray(X_test[:numerical_size])

    plots = numerical(
        model=qmodel,
        hls_model=hls_model,
        X=X_num,
        plot="boxplot",
    )

    nomes = [
        "numerical_weights_keras.png",
        "numerical_weights_hls.png",
        "numerical_activations_keras.png",
        "numerical_activations_hls.png",
    ]

    if isinstance(plots, (tuple, list)):
        for fig, nome in zip(plots, nomes):
            if fig is None:
                continue
            path = os.path.join(output_dir, nome)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"Salvo: {path}")
    else:
        for i, num in enumerate(plt.get_fignums()):
            path = os.path.join(output_dir, nomes[i] if i < len(nomes) else f"numerical_{i}.png")
            plt.figure(num).savefig(path, bbox_inches="tight", dpi=150)
            print(f"Salvo: {path}")

    plt.close("all")

def save_accuracy(qmodel, hls_model, X_test, y_test, output_dir, test_size):
    X_test_reduced = np.ascontiguousarray(X_test[:test_size])
    Y_test_reduced = np.asarray(y_test[:test_size])

    y_qkeras = qmodel.predict(X_test_reduced)
    y_pred_hls = hls_model.predict(X_test_reduced)

    accuracy_qkeras = accuracy_score(
        np.argmax(Y_test_reduced, axis=1),
        np.argmax(y_qkeras, axis=1),
    )

    accuracy_hls = accuracy_score(
        np.argmax(Y_test_reduced, axis=1),
        np.argmax(y_pred_hls, axis=1),
    )

    print(f"Accuracy base: {accuracy_qkeras}")
    print(f"Accuracy after hls4ml: {accuracy_hls}")

    accuracy_file = os.path.join(output_dir, "accuracy.txt")

    with open(accuracy_file, "w") as f:
        f.write(f"Accuracy base: {accuracy_qkeras:.6f}\n")
        f.write(f"Accuracy after hls4ml: {accuracy_hls:.6f}\n")

    return accuracy_qkeras, accuracy_hls


def run_configuration(
    qmodel,
    X_test,
    y_test,
    base_dir,
    reuse_factor,
    strategy,
    io_type,
    clock_period_ns,
    fpga_part,
    backend,
    test_size,
    numerical_size,
    skip_synthesis,
    skip_numerical,
):
    print("\n" + "=" * 80)
    print("CONFIGURAÇÃO")
    print(f"BASE_DIR       = {base_dir}")
    print(f"ReuseFactor    = {reuse_factor}")
    print(f"Strategy       = {strategy}")
    print(f"IOType         = {io_type}")
    print(f"ClockPeriod    = {clock_period_ns} ns")
    print(f"FPGA Part      = {fpga_part}")
    print(f"Backend        = {backend}")
    print("=" * 80)

    output_dir = make_output_dir(
        base_dir=base_dir,
        io_type=io_type,
        reuse_factor=reuse_factor,
        strategy=strategy,
        clock_period_ns=clock_period_ns,
    )

    print(f"OUTPUT_DIR = {output_dir}")

    hls_config = hls4ml.utils.config_from_keras_model(
        qmodel,
        granularity="name",
        backend=backend,
        default_reuse_factor=reuse_factor,
    )

    hls_config["LayerName"]["output_dense"]["Precision"]["result"] = (
        "fixed<16,6,RND,SAT>"
    )
    hls_config["Model"]["Strategy"] = strategy

    trace_layers(hls_config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=hls_config,
        output_dir=output_dir,
        part=fpga_part,
        clock_period=clock_period_ns,
        io_type=io_type,
        backend=backend,
    )

    hls_model.compile()

    save_model_plot(hls_model, output_dir)
    if not skip_numerical:
        save_numerical_plot(
            qmodel,
            hls_model,
            X_test,
            output_dir,
            numerical_size,
        )

    save_accuracy(
        qmodel,
        hls_model,
        X_test,
        y_test,
        output_dir,
        test_size,
    )

    if not skip_synthesis:
        print("\nIniciando síntese HLS...")
        hls_model.build(csim=False, synth=True, vsynth=True)
        print("Síntese concluída.")

    return output_dir


def main():
    args = parse_args()

    train_pickle_path = args.train_pickle
    test_pickle_path = args.test_pickle

    load_pickle_data(train_pickle_path)
    X_test, y_test = load_pickle_data(test_pickle_path)
    X_test = np.ascontiguousarray(X_test.astype("float32") / 255.0)

    model_paths = {}
    for base_dir in args.base_dir:
        path = f"models/{base_dir}/{args.model_name}.h5"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Modelo não encontrado: {path}")
        model_paths[base_dir] = path

    configurations = list(
        itertools.product(
            args.reuse_factor,
            args.strategy,
            args.io_type,
            args.clock_period_ns,
        )
    )

    total_runs = len(args.base_dir) * len(configurations)

    print(f"\nModelos: {len(args.base_dir)} ({', '.join(args.base_dir)})")
    print(f"Configurações por modelo: {len(configurations)}")
    print(f"Total de execuções: {total_runs}")

    results = []
    run_index = 0

    for base_dir in args.base_dir:
        final_path = model_paths[base_dir]

        print("\n" + "#" * 80)
        print(f"MODELO: {base_dir}")
        print("#" * 80)
        print(f"Carregando modelo: {final_path}")

        try:
            qmodel = load_qmodel(final_path)
        except Exception as e:
            if not args.continue_on_error:
                raise
            print(f"[ERRO] Falha ao carregar {final_path}: {e}")
            for reuse_factor, strategy, io_type, clock_period_ns in configurations:
                run_index += 1
                results.append(
                    {
                        "base_dir": base_dir,
                        "reuse_factor": reuse_factor,
                        "strategy": strategy,
                        "io_type": io_type,
                        "clock_period_ns": clock_period_ns,
                        "output_dir": None,
                        "error": f"load_qmodel: {e}",
                    }
                )
            continue

        for reuse_factor, strategy, io_type, clock_period_ns in configurations:
            run_index += 1
            print(f"\n>>> Execução {run_index}/{total_runs} (modelo {base_dir})")

            result = {
                "base_dir": base_dir,
                "reuse_factor": reuse_factor,
                "strategy": strategy,
                "io_type": io_type,
                "clock_period_ns": clock_period_ns,
                "output_dir": None,
                "error": None,
            }

            try:
                result["output_dir"] = run_configuration(
                    qmodel=qmodel,
                    X_test=X_test,
                    y_test=y_test,
                    base_dir=base_dir,
                    reuse_factor=reuse_factor,
                    strategy=strategy,
                    io_type=io_type,
                    clock_period_ns=clock_period_ns,
                    fpga_part=args.fpga_part,
                    backend=args.backend,
                    test_size=args.test_size,
                    numerical_size=args.numerical_size,
                    skip_synthesis=args.skip_synthesis,
                    skip_numerical=args.skip_numerical,
                )
            except Exception as e:
                if not args.continue_on_error:
                    raise
                result["error"] = str(e)
                print(f"[ERRO] {base_dir} rf={reuse_factor} {strategy} {io_type} "
                      f"{clock_period_ns}ns: {e}")
                plt.close("all")

            results.append(result)

    print("\n" + "=" * 80)
    print("TODAS AS EXECUÇÕES CONCLUÍDAS")
    print("=" * 80)

    for i, r in enumerate(results, start=1):
        status = "OK  " if r["error"] is None else "ERRO"
        print(
            f"{i:02d} [{status}] "
            f"model={r['base_dir']}, "
            f"rf={r['reuse_factor']}, "
            f"strategy={r['strategy']}, "
            f"io={r['io_type']}, "
            f"clock={r['clock_period_ns']} ns "
            f"-> {r['output_dir'] or r['error']}"
        )


if __name__ == "__main__":
    main()