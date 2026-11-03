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

python main.py \
    --base-dir t1 \
    --rf-min-mid-max \
    --strategy Resource Latency \
    --io-type io_stream \
    --clock-period-ns 10 \
    --continue-on-error \
    --max-threads 1

python t1.py \
    --base-dir t1 t1_4 t1_8 t1_batch_norm_max_pool \
    --rf-min-mid-max \
    --strategy Resource Latency \
    --dense-strategy Resource \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1

# mesmo throughput do FINN (FPS medidos na PYNQ-Z1 no paper)
python t1.py \
    --base-dir t1 t1_4 t1_8 \
    --target-fps 543 6512 16271 \
    --strategy Resource Latency \
    --dense-strategy Resource \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1
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

from vivado_utils import add_vivado_timing


# (nome do parâmetro de build(), padrão, descrição)
BUILD_OPTIONS = [
    ("reset", False, "Recria o projeto HLS do zero"),
    ("csim", True, "Simulação C"),
    ("synth", True, "Síntese HLS (csynth)"),
    ("cosim", True, "Co-simulação RTL (requer synth)"),
    ("validation", False, "Compara saídas csim x cosim"),
    ("export", False, "Exporta o IP"),
    ("vsynth", True, "Síntese lógica no Vivado (requer synth)"),
    ("fifo_opt", False, "Otimização de profundidade das FIFOs"),
]

def estimate_layer_cycles(info, rf):
    """Ciclos por imagem de uma camada io_stream com ReuseFactor `rf` (Vitis, nnet_*_stream.h).

    - Conv2D (line buffer): PIPELINE II=RF por pixel de *entrada* (já com padding).
    - PointwiseConv2D (1x1): II=RF por pixel de saída.
    - Dense: lê a entrada (um pixel por ciclo) e depois roda com II=RF.
    Bateu com o csynth da t1 em ~2% para RF > 1. Com RF=1 o line buffer da conv 3x3 ficou em
    ~9 ciclos/pixel nas sínteses rfmin, então o estimado é otimista nessa faixa.
    """
    if "Pointwise" in info["type"]:
        return info["out_pixels"] * rf
    if "Conv" in info["type"]:
        return info["in_pixels"] * rf
    return info["in_pixels"] + rf


def rf_profile_for_fps(layers, target_fps, clock_period_ns):
    """Igual ao SetFolding do FINN (step_target_fps_parallelization): para cada camada, o maior
    RF válido cujo custo estimado cabe em clock / target_fps ciclos. Assim todas as camadas
    ficam abaixo do alvo, sem paralelismo sobrando nas que não são gargalo.
    """
    alvo = 1e9 / (target_fps * clock_period_ns)
    profile = {}
    print(f"\nPerfil para {target_fps:g} FPS (alvo: {alvo:,.0f} ciclos/imagem @ {clock_period_ns} ns):")
    pior = 0
    for info in layers:
        ok = [rf for rf in info["valid_rf"] if estimate_layer_cycles(info, rf) <= alvo]
        rf = max(ok) if ok else info["valid_rf"][0]
        ciclos = estimate_layer_cycles(info, rf)
        pior = max(pior, ciclos)
        profile[info["name"]] = rf
        aviso = "" if ok else "  <- não atinge o alvo nem com o menor RF"
        print(f"  {info['name']} ({info['type']}): RF={rf}, ~{ciclos:,.0f} ciclos{aviso}")
    print(f"  FPS estimado: {1e9 / (pior * clock_period_ns):,.0f}")
    return profile


def get_layer_rf_profiles(
    qmodel,
    backend_name,
    fpga_part,
    io_type,
    clock_period_ns,
    strategy,
    base_dir,
    target_fps=None,
):
    """Calcula RFs por camada usando as dimensões internas do hls4ml.

    Primeiro converte o modelo para um ModelGraph temporário com RF=1. Depois
    usa backend.get_layer_mult_size(layer), a mesma função do backend que
    extrai n_in/n_out das camadas Dense/Conv para a validação de RF.

    Retorna os perfis "min"/"mid"/"max" e, para cada FPS em target_fps, um perfil
    "fps<N>" (ver rf_profile_for_fps).
    """
    backend_obj = hls4ml.backends.get_backend(backend_name)
    profiles = {"min": {}, "mid": {}, "max": {}}
    report = []

    probe_config = hls4ml.utils.config_from_keras_model(
        qmodel,
        granularity="name",
        backend=backend_name,
        default_reuse_factor=1,
    )
    if "output_dense" in probe_config.get("LayerName", {}):
        probe_config["LayerName"]["output_dense"]["Precision"]["result"] = (
            "fixed<16,6,RND,SAT>"
        )
    probe_config["Model"]["Strategy"] = strategy

    clock_str = str(clock_period_ns).replace(".", "p")
    probe_dir = os.path.join(
        ".", "t1_projects", "_rf_profile_probe",
        f"{base_dir}_{io_type}_{strategy}_cp{clock_str}",
    )
    probe_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=probe_config,
        output_dir=probe_dir,
        part=fpga_part,
        clock_period=clock_period_ns,
        io_type=io_type,
        backend=backend_name,
    )

    # largura do stream (canais) que chega em cada camada, para estimar a leitura da Dense
    canais = 1
    for layer in probe_model.get_layers():
        try:
            n_in, n_out = backend_obj.get_layer_mult_size(layer)
        except (Exception,):
            n_in = None

        if n_in is not None:
            valid_rf = backend_obj.get_valid_reuse_factors(
                n_in=int(n_in),
                n_out=int(n_out),
            )
            if not valid_rf:
                print(
                    f"[AVISO] Nenhum RF válido retornado para "
                    f"{layer.name} ({layer.class_name}); camada ignorada."
                )
            else:
                min_rf = valid_rf[0]
                max_rf = valid_rf[-1]
                midpoint = (min_rf + max_rf) / 2
                mid_rf = min(valid_rf, key=lambda rf: (abs(rf - midpoint), rf))

                profiles["min"][layer.name] = min_rf
                profiles["mid"][layer.name] = mid_rf
                profiles["max"][layer.name] = max_rf

                h_in = layer.get_attr("in_height") or 1
                w_in = layer.get_attr("in_width") or 1
                h_out = layer.get_attr("out_height") or 1
                w_out = layer.get_attr("out_width") or 1
                report.append({
                    "name": layer.name,
                    "type": layer.class_name,
                    "n_in": int(n_in),
                    "n_out": int(n_out),
                    "min_rf": min_rf,
                    "mid_rf": mid_rf,
                    "max_rf": max_rf,
                    "n_valid": len(valid_rf),
                    "valid_rf": valid_rf,
                    "in_pixels": (
                        h_in * w_in if "Conv" in layer.class_name
                        else max(1, int(n_in) // canais)
                    ),
                    "out_pixels": h_out * w_out,
                })

        for attr in ("n_filt", "n_chan"):
            if layer.get_attr(attr):
                canais = int(layer.get_attr(attr))
                break

    print("\nRFs calculados por camada (dimensões internas do hls4ml):")
    for row in report:
        print(
            f"  {row['name']} ({row['type']}): "
            f"n_in={row['n_in']}, n_out={row['n_out']}, "
            f"min={row['min_rf']}, mid={row['mid_rf']}, "
            f"max={row['max_rf']} ({row['n_valid']} RFs válidos)"
        )

    if not report:
        raise RuntimeError(
            "Nenhuma camada Dense/Conv compatível foi encontrada no ModelGraph "
            "temporário; não foi possível calcular os perfis de RF."
        )

    for fps in target_fps or []:
        profiles[fps_profile_name(fps)] = rf_profile_for_fps(report, fps, clock_period_ns)

    return profiles


def fps_profile_name(fps):
    return f"fps{fps:g}"


def apply_layer_rf_profile(hls_config, layer_rf_profile):
    """Aplica RF individual às camadas presentes na configuração hls4ml."""
    for layer_name, rf in layer_rf_profile.items():
        if layer_name not in hls_config.get("LayerName", {}):
            print(
                f"[AVISO] Camada {layer_name!r} não encontrada em "
                "hls_config['LayerName']; RF específico não aplicado."
            )
            continue

        hls_config["LayerName"][layer_name]["ReuseFactor"] = int(rf)


def apply_max_threads(output_dir, max_threads):
    """Insere `set_param general.maxThreads N` nos scripts Tcl do projeto.

    build_prj.tcl roda no vitis_hls; vivado_synth.tcl roda num processo
    Vivado separado (exec vivado), que não herda o parâmetro, então os dois
    precisam ser alterados. O `catch` evita erro caso a ferramenta não aceite.
    """
    line = f"catch {{set_param general.maxThreads {int(max_threads)}}}\n"
    for script in ("build_prj.tcl", "vivado_synth.tcl"):
        path = os.path.join(output_dir, script)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            lines = f.readlines()
        lines = [l for l in lines if "general.maxThreads" not in l]
        # Insere logo após o primeiro `source ... project.tcl`.
        idx = next(
            (i + 1 for i, l in enumerate(lines) if "project.tcl" in l and l.lstrip().startswith("source")),
            0,
        )
        lines.insert(idx, line)
        with open(path, "w") as f:
            f.writelines(lines)
        print(f"maxThreads={max_threads} aplicado em {path}")


def apply_dense_strategy(qmodel, hls_config, dense_strategy):
    """Sobrescreve a Strategy só das camadas Dense/QDense.

    Em Latency o hls4ml desenrola a Dense inteira (n_in * n_out multiplicações)
    independentemente do RF; com Flatten grande isso derruba o Vitis HLS.
    """
    for layer in qmodel.layers:
        if "Dense" not in layer.__class__.__name__:
            continue
        if layer.name not in hls_config.get("LayerName", {}):
            print(f"[AVISO] Camada {layer.name!r} não encontrada em hls_config['LayerName'].")
            continue
        hls_config["LayerName"][layer.name]["Strategy"] = dense_strategy
        print(f"Strategy de {layer.name} = {dense_strategy}")


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
        "--rf-min-mid-max",
        action="store_true",
        help=(
            "Converte a rede temporariamente para obter as dimensões internas do ModelGraph, "
            "calcula RFs válidos por camada e executa três variantes (mínimo, meio e máximo). "
            "Quando ativo, substitui --reuse-factor."
        ),
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Um ou mais FPS alvo. Para cada um, escolhe por camada o maior RF válido que ainda "
            "atinge o FPS (como o target_fps do FINN). Ex.: --target-fps 543 6512 16271. "
            "Pode ser combinado com --rf-min-mid-max; substitui --reuse-factor."
        ),
    )

    parser.add_argument(
        "--strategy",
        nargs="+",
        default=["Resource"],
        choices=["Resource", "Latency"],
        help="Uma ou mais estratégias. Ex.: --strategy Resource Latency",
    )

    parser.add_argument(
        "--dense-strategy",
        default=None,
        choices=["Resource", "Latency"],
        help=(
            "Sobrescreve a estratégia só das camadas Dense. Ex.: --strategy Latency "
            "--dense-strategy Resource deixa as convs em Latency e a Dense em Resource."
        ),
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
        help="Não executa hls_model.build() (ignora as opções de build).",
    )

    # Opções repassadas para hls_model.build(). Cada uma aceita --X / --no-X.
    build_group = parser.add_argument_group("build", "Etapas de hls_model.build()")
    for name, default, desc in BUILD_OPTIONS:
        build_group.add_argument(
            f"--{name.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            default=default,
            help=f"{desc} (padrão: {default})",
        )

    parser.add_argument(
        "--max-threads",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help=(
            "Adiciona `set_param general.maxThreads N` em build_prj.tcl e "
            "vivado_synth.tcl antes do build para reduzir o consumo de memória. "
            "Sem valor usa 1. Ex.: --max-threads ou --max-threads 2"
        ),
    )

    parser.add_argument(
        "--skip-numerical",
        action="store_true",
        help="Não executa análise numerical().",
    )

    return parser.parse_args()

def make_output_dir(base_dir, io_type, reuse_factor, strategy, clock_period_ns, rf_profile=None, dense_strategy=None):
    projects_dir = "t1_projects"
    os.makedirs(projects_dir, exist_ok=True)

    # No modo de perfis, identifica o diretório pelo perfil, não pelo RF global
    # de fallback (1): rfmin, rfmed ou rfmax.
    profile_labels = {"min": "rfmin", "mid": "rfmed", "max": "rfmax"}
    if rf_profile is None:
        rf_label = f"rf{reuse_factor}"
    else:
        rf_label = profile_labels.get(rf_profile, rf_profile)

    name = f"prj_{base_dir}_{io_type}_{rf_label}_{strategy}"
    # Só marca no nome quando a Dense difere do resto da rede.
    if dense_strategy is not None and dense_strategy != strategy:
        name += f"_dense{dense_strategy}"
    if clock_period_ns != 10:
        clock_str = str(clock_period_ns).replace(".", "p")
        name += f"_cp{clock_str}"

    output_dir = os.path.join(".", projects_dir, name)
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
    build_kwargs=None,
    dense_strategy=None,
    rf_profile=None,
    layer_rf_profile=None,
    max_threads=None,
):
    print("\n" + "=" * 80)
    print("CONFIGURAÇÃO")
    print(f"BASE_DIR       = {base_dir}")
    print(f"ReuseFactor    = {reuse_factor}")
    if rf_profile is not None:
        print(f"RF profile     = {rf_profile}")
    print(f"Strategy       = {strategy}")
    if dense_strategy is not None:
        print(f"Dense Strategy = {dense_strategy}")
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
        rf_profile=rf_profile,
        dense_strategy=dense_strategy,
    )

    print(f"OUTPUT_DIR = {output_dir}")

    hls_config = hls4ml.utils.config_from_keras_model(
        qmodel,
        granularity="name",
        backend=backend,
        default_reuse_factor=reuse_factor,
    )

    if layer_rf_profile is not None:
        apply_layer_rf_profile(hls_config, layer_rf_profile)

    hls_config["LayerName"]["output_dense"]["Precision"]["result"] = (
        "fixed<16,6,RND,SAT>"
    )
    hls_config["Model"]["Strategy"] = strategy
    if dense_strategy is not None:
        apply_dense_strategy(qmodel, hls_config, dense_strategy)

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
        add_vivado_timing(output_dir)
        if max_threads is not None:
            apply_max_threads(output_dir, max_threads)
        print("\nIniciando síntese HLS...")
        print(f"Opções de build: {build_kwargs}")
        hls_model.build(**build_kwargs)
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

    build_kwargs = {name: getattr(args, name) for name, _, _ in BUILD_OPTIONS}

    use_profiles = args.rf_min_mid_max or bool(args.target_fps)
    reuse_factors = [1] if use_profiles else args.reuse_factor
    configurations = list(
        itertools.product(
            reuse_factors,
            args.strategy,
            args.io_type,
            args.clock_period_ns,
        )
    )

    profile_names = ["min", "mid", "max"] if args.rf_min_mid_max else []
    profile_names += [fps_profile_name(f) for f in args.target_fps or []]
    profile_names = profile_names or [None]
    total_runs = len(args.base_dir) * len(configurations) * len(profile_names)

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
                for rf_profile in profile_names:
                    run_index += 1
                    results.append(
                        {
                            "base_dir": base_dir,
                            "reuse_factor": reuse_factor,
                            "rf_profile": rf_profile,
                            "strategy": strategy,
                            "io_type": io_type,
                            "clock_period_ns": clock_period_ns,
                            "output_dir": None,
                            "error": f"load_qmodel: {e}",
                        }
                    )
            continue

        for reuse_factor, strategy, io_type, clock_period_ns in configurations:
            layer_rf_profiles = None
            if use_profiles:
                print(
                    f"Calculando RFs válidos por camada para {base_dir} "
                    f"(backend={args.backend}, io={io_type}, strategy={strategy}) ..."
                )
                layer_rf_profiles = get_layer_rf_profiles(
                    qmodel=qmodel,
                    backend_name=args.backend,
                    fpga_part=args.fpga_part,
                    io_type=io_type,
                    clock_period_ns=clock_period_ns,
                    strategy=strategy,
                    base_dir=base_dir,
                    target_fps=args.target_fps,
                )

            for rf_profile in profile_names:
                run_index += 1
                print(
                    f"\n>>> Execução {run_index}/{total_runs} "
                    f"(modelo {base_dir}, RF profile={rf_profile or 'manual'})"
                )

                result = {
                    "base_dir": base_dir,
                    "reuse_factor": reuse_factor,
                    "rf_profile": rf_profile,
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
                        max_threads=args.max_threads,
                        build_kwargs=build_kwargs,
                        dense_strategy=args.dense_strategy,
                        rf_profile=rf_profile,
                        layer_rf_profile=(
                            layer_rf_profiles[rf_profile]
                            if rf_profile is not None
                            else None
                        ),
                    )
                except Exception as e:
                    if not args.continue_on_error:
                        raise
                    result["error"] = str(e)
                    print(
                        f"[ERRO] {base_dir} rf={reuse_factor} "
                        f"profile={rf_profile} {strategy} {io_type} "
                        f"{clock_period_ns}ns: {e}"
                    )
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
            f"rf_profile={r.get('rf_profile')}, "
            f"strategy={r['strategy']}, "
            f"io={r['io_type']}, "
            f"clock={r['clock_period_ns']} ns "
            f"-> {r['output_dir'] or r['error']}"
        )


if __name__ == "__main__":
    main()