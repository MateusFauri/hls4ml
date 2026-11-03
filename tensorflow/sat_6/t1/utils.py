"""Funções compartilhadas pelos notebooks (0_pickle_dataset, 1_train_t1_qkeras, 2_synth_t1)
e pelo main.py."""

import os
import pickle

import numpy as np
import pandas as pd

IMG_SIZE = 28
PAD_SIZE = 32
NUM_BANDS = 4


# ---------------------------------------------------------------- dataset

def load_sat6_csv(x_path, y_path, n_samples=None):
    print(f"Lendo {x_path} ...")
    X = pd.read_csv(x_path, header=None, nrows=n_samples).values
    print(f"Lendo {y_path} ...")
    y = pd.read_csv(y_path, header=None, nrows=n_samples).values

    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, NUM_BANDS).astype(np.float32)
    y = y.astype(np.float32)
    return X, y


def pad_to_32x32(X):
    """Zero-padding em numpy: (N, 28, 28, 4) -> (N, 32, 32, 4)."""
    pad = (PAD_SIZE - IMG_SIZE) // 2
    return np.pad(
        X,
        pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


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


# ---------------------------------------------------------------- reuse factor por camada

def estimate_layer_cycles(info, rf, strategy="Resource"):
    """Ciclos por imagem de uma camada io_stream com ReuseFactor `rf` (Vitis, nnet_*_stream.h).

    - Conv2D em Latency (line buffer): PIPELINE II=RF por pixel de *entrada* (já com padding).
    - Conv2D em Resource: o laço não tem PIPELINE e a multiplicação (RF ciclos) só roda nos pixels
      que fecham uma janela (os de saída). Medido na cosim da t1: ~4 ciclos por pixel sem saída e
      ~RF+9 por pixel de saída. Com RF=1 o HLS faz pipeline do laço com II 8 em todos os pixels.
      O csynth não vê isso e informa RF+9 em todos os pixels de entrada (até 4,5x pessimista na
      conv1, que tem stride 2); por isso o FPS dessas sínteses só é confiável na cosim.
    - PointwiseConv2D (1x1): II=RF por pixel de saída (+4 de controle quando RF > 1).
    - Dense: lê a entrada (um pixel por ciclo) e depois roda com II=RF.
    """
    if "Pointwise" in info["type"]:
        return info["out_pixels"] * (rf + 4 if rf > 1 else 1)
    if "Conv" in info["type"]:
        if strategy != "Resource":
            return info["in_pixels"] * rf
        if rf == 1:
            return info["in_pixels"] * 8
        return (info["in_pixels"] - info["out_pixels"]) * 4 + info["out_pixels"] * (rf + 9)
    return info["in_pixels"] + rf


def rf_profile_for_fps(layers, target_fps, clock_period_ns, strategy="Resource"):
    """Igual ao SetFolding do FINN (step_target_fps_parallelization): para cada camada, o maior
    RF válido cujo custo estimado cabe em clock / target_fps ciclos. Assim todas as camadas
    ficam abaixo do alvo, sem paralelismo sobrando nas que não são gargalo. Se nenhum RF atinge
    o alvo, usa o de menor custo estimado (em Resource, RF=1 não é o mais rápido na conv).
    """
    alvo = 1e9 / (target_fps * clock_period_ns)
    profile = {}
    print(f"\nPerfil para {target_fps:g} FPS (alvo: {alvo:,.0f} ciclos/imagem @ {clock_period_ns} ns):")
    pior = 0
    for info in layers:
        custo = lambda rf: estimate_layer_cycles(info, rf, strategy)
        ok = [rf for rf in info["valid_rf"] if custo(rf) <= alvo]
        rf = max(ok) if ok else min(info["valid_rf"], key=lambda r: (custo(r), -r))
        ciclos = custo(rf)
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
    projects_dir="t1_projects",
):
    """Calcula RFs por camada usando as dimensões internas do hls4ml.

    Primeiro converte o modelo para um ModelGraph temporário com RF=1. Depois
    usa backend.get_layer_mult_size(layer), a mesma função do backend que
    extrai n_in/n_out das camadas Dense/Conv para a validação de RF.

    Retorna os perfis "min"/"mid"/"max" e, para cada FPS em target_fps, um perfil
    "fps<N>" (ver rf_profile_for_fps).
    """
    import hls4ml

    from cnn_base import set_input_precision

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
    set_input_precision(probe_config, qmodel)
    probe_config["Model"]["Strategy"] = strategy

    clock_str = str(clock_period_ns).replace(".", "p")
    probe_dir = os.path.join(
        ".", projects_dir, "_rf_profile_probe",
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
            f"max={row['max_rf']} ({row['n_valid']} RFs válidos: {row['valid_rf']})"
        )

    if not report:
        raise RuntimeError(
            "Nenhuma camada Dense/Conv compatível foi encontrada no ModelGraph "
            "temporário; não foi possível calcular os perfis de RF."
        )

    for fps in target_fps or []:
        profiles[fps_profile_name(fps)] = rf_profile_for_fps(report, fps, clock_period_ns, strategy)

    return profiles


def fps_profile_name(fps):
    return f"fps{fps:g}"
