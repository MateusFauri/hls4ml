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

python main.py \
    --base-dir t1 t1_4 t1_8 t1_maxpool_bn t1_4_maxpool_bn t1_8_maxpool_bn \
    --rf-min-mid-max \
    --strategy Resource Latency \
    --dense-strategy Resource \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1 \

python main.py \
    --base-dir t1 t1_4 t1_8 t1_maxpool_bn t1_4_maxpool_bn t1_8_maxpool_bn \
    --target-fps 543 6512 16271 \
    --strategy Resource Latency \
    --dense-strategy Resource \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1 \

# Resource nos FPS do FINN com o modelo de custo corrigido (utils.estimate_layer_cycles)
python main.py \
    --base-dir t1 t1_4 t1_8 t1_maxpool_bn t1_4_maxpool_bn t1_8_maxpool_bn \
    --target-fps 500 5000 543 6512 16271 50000 \
    --strategy Resource \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1 > logs/resource.log 2>&1

# tudo em Latency (Dense também), numa pasta à parte
python main.py \
    --base-dir t1_4 t1_8 t1_4_maxpool_bn t1_8_maxpool_bn t1 t1_maxpool_bn \
    --rf-min-mid-max --target-fps 543 6512 16271 50000 \
    --strategy Latency \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1 \
    --projects-dir t1_projects_latency > logs/t1_projects_latency.log 2>&1

# só a síntese extra com Distributed Arithmetic (convs em DA, Dense em Resource RF=4096)
python main.py \
    --base-dir t1_4 t1_8 \
    --only-da \
    --io-type io_stream \
    --continue-on-error \
    --max-threads 1
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import hls4ml
from qkeras.utils import load_qmodel
from cnn_base import prepare_input, set_input_precision
from vivado_utils import add_vivado_timing
from utils import (
    fps_profile_name,
    get_layer_rf_profiles,
    load_pickle_data,
    print_dict,
    trace_layers,
)
from sklearn.metrics import accuracy_score

from hls4ml.model.profiling import numerical


BUILD_OPTIONS = [
    ("reset", False, "Recria o projeto HLS do zero"),
    ("csim", True, "Simulação C"),
    ("synth", True, "Síntese HLS (csynth)"),
    ("cosim", False, "Co-simulação RTL (requer synth)"),
    ("validation", False, "Compara saídas csim x cosim"),
    ("export", False, "Exporta o IP"),
    ("vsynth", True, "Síntese lógica no Vivado (requer synth)"),
    ("fifo_opt", False, "Otimização de profundidade das FIFOs"),
]

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
        
    Isto é útil para reduzir o consumo de memória do Vivado HLS em máquinas com poucos núcleos.
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


def apply_dense_strategy(qmodel, hls_config, dense_strategy, reuse_factor=None):
    """Sobrescreve a Strategy (e opcionalmente o RF) só das camadas Dense/QDense.

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
        if reuse_factor is not None:
            hls_config["LayerName"][layer.name]["ReuseFactor"] = int(reuse_factor)
        print(f"Strategy de {layer.name} = {dense_strategy}"
              + (f" (RF={reuse_factor})" if reuse_factor is not None else ""))


def usar_solver_da_leve():
    """Troca as opções que o hls4ml passa ao da4ml (hard_dc=2, search_all_decompose_dc=True) por
    hard_dc=-1 e search_all_decompose_dc=False. O hls4ml 1.3 não expõe essas opções; com elas a
    dense 2048->6 de 8 bits otimiza em ~11 min com ~5,4 GB, em vez de passar de 27 GB."""
    from da4ml.trace import FixedVariableArray

    original = FixedVariableArray.from_kif.__func__

    def from_kif(cls, *args, solver_options=None, **kwargs):
        opcoes = dict(solver_options or {})
        opcoes.update(hard_dc=-1, search_all_decompose_dc=False)
        return original(cls, *args, solver_options=opcoes, **kwargs)

    FixedVariableArray.from_kif = classmethod(from_kif)
    print("da4ml: solver leve (hard_dc=-1, search_all_decompose_dc=False)")


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
        "--projects-dir",
        default="t1_projects",
        help="Pasta onde os projetos são criados (padrão: t1_projects).",
    )

    parser.add_argument(
        "--da",
        action="store_true",
        help=(
            "Roda uma síntese a mais por modelo com Strategy distributed_arithmetic (da4ml, RF=1), "
            "para cada --io-type e --clock-period-ns. Modelos sem quantização são pulados."
        ),
    )

    parser.add_argument(
        "--only-da",
        action="store_true",
        help="Roda só a síntese com DA (implica --da e ignora --strategy/--reuse-factor/perfis).",
    )

    parser.add_argument(
        "--da-dense-strategy",
        default="Resource",
        choices=["DA", "Resource", "Latency"],
        help=(
            "Estratégia da Dense na síntese DA. 'DA' põe a Dense 2048->6 também em DA: funciona na "
            "t1_4, mas na t1_8 o da4ml passa de 27 GB de memória. Padrão: Resource."
        ),
    )

    parser.add_argument(
        "--da-dense-rf",
        type=int,
        default=4096,
        help=(
            "RF da Dense quando --da-dense-strategy != DA. A Dense leva ~256 + RF ciclos; 4096 fica "
            "abaixo do II da conv1 em DA (~4624), sem limitar o FPS. Padrão: 4096."
        ),
    )

    parser.add_argument(
        "--da-solver",
        default="padrao",
        choices=["padrao", "leve"],
        help=(
            "Opções do da4ml. 'padrao' = as do hls4ml (hard_dc=2, busca todas as decomposições). "
            "'leve' = hard_dc=-1 sem a busca: usa bem menos memória (permite a dense de 8 bits em DA), "
            "mas a rede de somas pode ficar mais profunda. Acrescenta _daleve ao nome da pasta."
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

# pasta onde ficam os projetos; main() troca pelo --projects-dir
PROJECTS_DIR = "t1_projects"


def make_output_dir(base_dir, io_type, reuse_factor, strategy, clock_period_ns, rf_profile=None, dense_strategy=None, suffix=""):
    projects_dir = PROJECTS_DIR
    os.makedirs(projects_dir, exist_ok=True)

    profile_labels = {"min": "rfmin", "mid": "rfmed", "max": "rfmax"}
    if rf_profile is None:
        rf_label = f"rf{reuse_factor}"
    else:
        rf_label = profile_labels.get(rf_profile, rf_profile)

    name = f"prj_{base_dir}_{io_type}_{rf_label}_{strategy}"
    if dense_strategy is not None and dense_strategy != strategy:
        name += f"_dense{dense_strategy}"
    if clock_period_ns != 10:
        clock_str = str(clock_period_ns).replace(".", "p")
        name += f"_cp{clock_str}"
    name += suffix

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
    dense_rf=None,
    suffix="",
):
    """strategy="DA" usa distributed_arithmetic (RF=1); dense_strategy/dense_rf valem só para a Dense."""
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
        suffix=(str(dense_rf) if dense_rf is not None else "") + suffix,
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
    set_input_precision(hls_config, qmodel)
    hls_config["Model"]["Strategy"] = "distributed_arithmetic" if strategy == "DA" else strategy
    if dense_strategy is not None:
        apply_dense_strategy(qmodel, hls_config, dense_strategy, dense_rf)

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
    global PROJECTS_DIR
    args = parse_args()
    PROJECTS_DIR = args.projects_dir

    train_pickle_path = args.train_pickle
    test_pickle_path = args.test_pickle

    load_pickle_data(train_pickle_path)
    X_test_raw, y_test = load_pickle_data(test_pickle_path)

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
    if args.only_da:
        args.da = True
        configurations = []
    da_configurations = list(itertools.product(args.io_type, args.clock_period_ns)) if args.da else []
    if args.da and args.da_solver == "leve":
        usar_solver_da_leve()
    if args.da_dense_strategy == "DA":
        da_dense_strategy, da_dense_rf = None, None
    else:
        da_dense_strategy, da_dense_rf = args.da_dense_strategy, args.da_dense_rf
    total_runs = len(args.base_dir) * (len(configurations) * len(profile_names) + len(da_configurations))

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

        X_test = np.ascontiguousarray(prepare_input(X_test_raw, qmodel))

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
                    projects_dir=args.projects_dir,
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

        # síntese extra com Distributed Arithmetic: RF=1 em todas as camadas, sem perfis de RF
        if da_configurations and "input_quant" not in [l.name for l in qmodel.layers]:
            print(f"[AVISO] {base_dir} não é quantizado: DA com fixed<16,6> fica enorme, pulando.")
            da_configurations_modelo = []
        else:
            da_configurations_modelo = da_configurations
        for io_type, clock_period_ns in da_configurations_modelo:
            run_index += 1
            print(f"\n>>> Execução {run_index}/{total_runs} (modelo {base_dir}, Distributed Arithmetic)")
            result = {
                "base_dir": base_dir,
                "reuse_factor": 1,
                "rf_profile": None,
                "strategy": "DA",
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
                    reuse_factor=1,
                    strategy="DA",
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
                    dense_strategy=da_dense_strategy,
                    dense_rf=da_dense_rf,
                    suffix="_daleve" if args.da_solver == "leve" else "",
                )
            except Exception as e:
                if not args.continue_on_error:
                    raise
                result["error"] = str(e)
                print(f"[ERRO] {base_dir} DA {io_type} {clock_period_ns}ns: {e}")
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