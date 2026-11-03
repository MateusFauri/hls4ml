"""Ajustes no vivado_synth.tcl gerado pelo hls4ml.

O script original só roda synth_design + report_utilization, sem restrição de clock, então o
Vivado não reporta timing. add_vivado_timing() cria a restrição do ap_clk (com o clock_period
do project.tcl) antes da síntese e grava vivado_timing.rpt depois dela.
"""

import os

MARCA = "# hls4ml-timing"

ANTES_SYNTH = f"""{MARCA}
set xdc [file join $tcldir vivado_clock.xdc]
set f [open $xdc w]
puts $f "create_clock -period $clock_period -name ap_clk \\[get_ports ap_clk\\]"
close $f
read_xdc $xdc
"""

DEPOIS_UTIL = f"""{MARCA}
report_timing_summary -max_paths 10 -file vivado_timing.rpt
"""


def add_vivado_timing(output_dir):
    """Insere restrição de clock e report_timing_summary no vivado_synth.tcl (idempotente)."""
    path = os.path.join(output_dir, "vivado_synth.tcl")
    if not os.path.isfile(path):
        print(f"[AVISO] {path} não encontrado; timing do Vivado não configurado.")
        return False
    with open(path) as f:
        linhas = f.readlines()
    if any(MARCA in l for l in linhas):
        return True

    saida = []
    for linha in linhas:
        if linha.lstrip().startswith("synth_design"):
            saida.append(ANTES_SYNTH)
        saida.append(linha)
        if linha.lstrip().startswith("report_utilization"):
            saida.append(DEPOIS_UTIL)
    with open(path, "w") as f:
        f.writelines(saida)
    print(f"Timing do Vivado configurado em {path}")
    return True
