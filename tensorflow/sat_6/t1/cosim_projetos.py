"""Roda só a co-simulação C/RTL (cosim) em projetos já sintetizados pelo main.py, sem refazer o HLS.

Diferenças em relação ao build(cosim=True) do hls4ml:
- usa imagens reais do teste no testbench (o padrão do hls4ml são 5 imagens de zeros);
- cosim_design -trace_level none (o build_prj.tcl usa "all", que grava a forma de onda de todos os
  sinais e deixa a simulação muito lenta nos designs grandes);
- limite de threads e timeout por projeto;
- compara a saída do RTL com a do QKeras.

O resultado fica em myproject_prj/solution1/sim/report/myproject_cosim.rpt, que o relatorio.py lê
(cosim_lat_*, cosim_ii_*, fps_cosim).

python cosim_projetos.py t1_projects/prj_t1_4_io_stream_rf1_DA_denseResource4096
python cosim_projetos.py "t1_projects/prj_t1_4_io_stream_fps*_Resource"
python cosim_projetos.py "t1_projects/prj_t1_*_rf1_DA*" --n-imagens 3 --timeout 7200 > logs/cosim.log 2>&1
"""

import argparse
import glob
import subprocess
import time
from pathlib import Path

import numpy as np

TCL = "cosim_only.tcl"
COSIM_RPT = "myproject_prj/solution1/sim/report/myproject_cosim.rpt"

# Abre o projeto/solução existentes (sem -reset, então a síntese é reaproveitada) e roda só a cosim.
TCL_TEMPLATE = """\
source project.tcl
catch {{set_param general.maxThreads {threads}}}
open_project ${{project_name}}_prj
set_top ${{project_name}}
add_files firmware/${{project_name}}.cpp -cflags "-std=c++0x"
add_files -tb ${{project_name}}_test.cpp -cflags "-std=c++0x -DRTL_SIM"
add_files -tb firmware/weights
add_files -tb tb_data
open_solution "solution1"
set_part $part
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default
cosim_design -trace_level none -rtl verilog
exit
"""


def base_dir_do_projeto(pasta):
    """prj_t1_4_maxpool_bn_io_stream_... -> t1_4_maxpool_bn (pasta do modelo em models/)."""
    nome = pasta.name.removeprefix("prj_")
    return nome.split("_io_")[0]


def escrever_tb(pasta, X, y_ref):
    """Formato lido pelo myproject_test.cpp: uma imagem por linha, valores separados por espaço."""
    tb = pasta / "tb_data"
    tb.mkdir(exist_ok=True)
    np.savetxt(tb / "tb_input_features.dat", X.reshape(len(X), -1), fmt="%.10g")
    np.savetxt(tb / "tb_output_predictions.dat", y_ref.reshape(len(y_ref), -1), fmt="%.10g")


def comparar_saida(pasta, y_ref):
    """Compara tb_data/rtl_cosim_results.log com a saída do QKeras."""
    log = pasta / "tb_data" / "rtl_cosim_results.log"
    if not log.is_file():
        return "sem rtl_cosim_results.log"
    y_rtl = np.loadtxt(log, ndmin=2)
    if y_rtl.shape != y_ref.shape:
        return f"formato diferente: RTL {y_rtl.shape} x QKeras {y_ref.shape}"
    iguais = int((y_rtl.argmax(1) == y_ref.argmax(1)).sum())
    return f"classe igual em {iguais}/{len(y_ref)}; maior |RTL - QKeras| = {np.abs(y_rtl - y_ref).max():.4g}"


def resumo_rpt(pasta):
    from relatorio import parse_cosim

    rpt = pasta / COSIM_RPT
    return parse_cosim(rpt.read_text()) if rpt.is_file() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projetos", nargs="+", help="pastas de projeto (aceita glob entre aspas)")
    ap.add_argument("--n-imagens", type=int, default=3,
                    help="imagens de teste no testbench (>= 2 para medir o intervalo). Padrão: 3")
    ap.add_argument("--max-threads", type=int, default=1, help="set_param general.maxThreads. Padrão: 1")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="segundos por projeto. Padrão: 3 h")
    ap.add_argument("--forcar", action="store_true", help="refaz projetos que já têm myproject_cosim.rpt")
    ap.add_argument("--test-pickle", default="./dataset/sat6_test.pkl")
    args = ap.parse_args()

    pastas = sorted({Path(p) for padrao in args.projetos for p in (glob.glob(padrao) or [padrao])})
    pastas = [p for p in pastas if p.is_dir()]
    if not pastas:
        print("Nenhuma pasta de projeto encontrada.")
        return

    # imports pesados só depois de validar os argumentos
    from qkeras.utils import load_qmodel

    from cnn_base import prepare_input
    from utils import load_pickle_data

    X_raw, _ = load_pickle_data(args.test_pickle)
    X_raw = X_raw[:args.n_imagens]
    modelos = {}
    resultados = []

    for i, pasta in enumerate(pastas, start=1):
        print("\n" + "=" * 80 + f"\n[{i}/{len(pastas)}] {pasta}\n" + "=" * 80)
        if not (pasta / "myproject_prj" / "solution1" / "syn").is_dir():
            print("[PULADO] sem síntese HLS (myproject_prj/solution1/syn)")
            resultados.append((pasta.name, "sem síntese", ""))
            continue
        if (pasta / COSIM_RPT).is_file() and not args.forcar:
            print("[PULADO] já tem cosim (use --forcar para refazer)")
            resultados.append((pasta.name, "já existia", str(resumo_rpt(pasta))))
            continue

        base = base_dir_do_projeto(pasta)
        if base not in modelos:
            modelos[base] = load_qmodel(f"models/{base}/model.h5")
        qmodel = modelos[base]
        X = np.ascontiguousarray(prepare_input(X_raw, qmodel))
        y_ref = qmodel.predict(X, verbose=0)
        escrever_tb(pasta, X, y_ref)
        (pasta / TCL).write_text(TCL_TEMPLATE.format(threads=args.max_threads))

        inicio = time.time()
        try:
            with open(pasta / "cosim_only.log", "w") as log:
                proc = subprocess.run(["vitis-run", "--tcl", TCL, "--mode", "hls"], cwd=pasta,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "ok" if proc.returncode == 0 else f"erro (código {proc.returncode}, ver cosim_only.log)"
        except subprocess.TimeoutExpired:
            status = f"timeout ({args.timeout} s)"
        duracao = time.time() - inicio

        rpt = resumo_rpt(pasta)
        comparacao = comparar_saida(pasta, y_ref) if status == "ok" else ""
        print(f"{status} em {duracao / 60:.1f} min | {rpt} | {comparacao}")
        resultados.append((pasta.name, f"{status}, {duracao / 60:.1f} min", f"{rpt} {comparacao}"))

    print("\n" + "=" * 80 + "\nRESUMO\n" + "=" * 80)
    for nome, status, detalhe in resultados:
        print(f"{nome}: {status} {detalhe}")
    print("\nRode `python relatorio.py <pasta>` para incluir cosim_lat_*, cosim_ii_* e fps_cosim no resumo.csv.")


if __name__ == "__main__":
    main()
