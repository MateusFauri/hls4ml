"""Compara as sínteses hls4ml da t1 (saída do relatorio.py) com os resultados FINN do paper
"Image Classification CNNs using the FINN Engine for SRAM-based APSoC in Satellite Applications"
(Ely et al., SBCCI 2024, Fig. 6).

python relatorio.py t1_projects
python compara_finn.py \
    --resumo relatorios_t1_projects/resumo.csv \
    --finn dados_paper/fig6_recursos_finn.csv \
    --out comparacao_finn

Fontes:
- Área: Vivado dos dois lados. FINN = só o acelerador (lut_finn/ff_finn/bram_finn, do
  finn_design_partition_util.rpt), sem AXI/DMA, lido do gráfico; hls4ml = vivado_synth.rpt do IP.
- FPS FINN = medido na placa (inclui DMA/driver). FPS hls4ml = 1e9 / (II * T): II medido na
  cosim (cosim_projetos.py; o csynth só entra se faltar a cosim); T do vivado_timing.rpt (o clock
  alvo se o timing fechou, senão o período atingido). O csynth superestima o II das convs em
  Resource (ver estimate_layer_cycles em utils.py). Sem vivado_timing.rpt, usa o clock alvo.
- O FINN não reporta latência nem DSP; essas colunas ficam vazias para ele.
- As topologias podem diferir: por isso a tabela traz MACs por inferência e eficiência
  normalizada (MMAC/s por kLUT).
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

# MACs por inferência das topologias do paper (Fig. 3), entrada 32x32x4.
# t1: conv 3x3/2 4->20 (16x16) + conv 1x1 20->8 (16x16) + dense 2048->6
MACS_PAPER = {"t1": 16 * 16 * 36 * 20 + 16 * 16 * 20 * 8 + 2048 * 6}

# paleta categórica (ordem fixa) e tinta dos eixos
COR = {
    "FINN": "#2a78d6",
    "hls4ml Latency (dense Resource)": "#eb6834",
    "hls4ml Resource": "#1baf7a",
    "hls4ml DA": "#7a4fc9",
}
COR_REF = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#fcfcfb"

def carregar_hls(caminho):
    df = pd.read_csv(caminho)
    df["extra"] = df["extra"].fillna("")
    df = df[df["rede"] == "t1"].copy()

    extra = df["extra"].fillna("")
    # variantes de FIFO/BRAM do 2_synth_t1.ipynb (_fifo64, _fifoopt, _bf1000) ficam fora da comparação
    variante_mem = extra.str.contains(r"(?:^|_)(?:fifo\w*|bf\d+)(?:_|$)")
    df, extra = df[~variante_mem].copy(), extra[~variante_mem]
    bits = df["quant"].str.extract(r"(\d+)bits")[0]
    df["bits"] = bits.where(bits.notna(), "float")
    bn = extra.str.contains("batch_norm|maxpool_bn|mp_bn|bn_mp")
    df.loc[bn, "bits"] = df.loc[bn, "bits"] + "_bn"

    df["framework"] = "hls4ml"
    df["serie"] = df["tipo"].map({
        "latency": "hls4ml Latency (dense Resource)",
        "resource": "hls4ml Resource",
        "da": "hls4ml DA",
    })
    df["rotulo"] = df["rf"].str.replace(r"^(rf|fps)", "", regex=True)
    df["config"] = df["rf"] + " / " + df["serie"].str.replace("hls4ml ", "")
    tem_viv = df["fps_vivado"].notna() if "fps_vivado" in df else pd.Series(False, index=df.index)
    df["fps_csynth"] = df["fps_vivado"].where(tem_viv, df["fps_clk_alvo"]) if "fps_vivado" in df else df["fps_clk_alvo"]
    df["fps_fonte"] = tem_viv.map({True: "II csynth + clock Vivado", False: "II csynth + clock alvo"})
    tem_cosim = df["fps_cosim"].notna() if "fps_cosim" in df else pd.Series(False, index=df.index)
    df["fps"] = df["fps_cosim"].where(tem_cosim, df["fps_csynth"]) if "fps_cosim" in df else df["fps_csynth"]
    df.loc[tem_cosim, "fps_fonte"] = "II cosim + clock Vivado"
    periodo = df["clk_target_ns"]
    if "viv_periodo_ns" in df:
        periodo = df[["clk_target_ns", "viv_periodo_ns"]].max(axis=1).where(tem_viv, periodo)
    # na cosim só a 1ª imagem não espera na fila de entrada, então a latência real é a mínima
    lat_cyc = df["cosim_lat_min"].where(tem_cosim, df["lat_max_cyc"]) if "cosim_lat_min" in df else df["lat_max_cyc"]
    df["lut"] = df["viv_LUT"]
    df["ff"] = df["viv_FF"]
    df["bram"] = df["viv_BRAM_tile"]
    df["dsp"] = df["viv_DSP"]
    df["lat_us"] = lat_cyc * periodo / 1e3
    df["acc_hls4ml"] = df["acc_segundo_valor"]
    df["macs"] = df["macs_por_inferencia"]
    df["cabe_no_fpga"] = (
        (df["viv_LUT"] <= df["viv_LUT_disp"])
        & (df["viv_FF"] <= df["viv_FF_disp"])
        & (df["viv_BRAM_tile"] <= df["viv_BRAM_tile_disp"])
        & (df["viv_DSP"] <= df["viv_DSP_disp"])
    )
    return df


def carregar_finn(caminho, topologia, bits):
    df = pd.read_csv(caminho)
    df = df[(df["topologia"] == topologia) & df["bits_quant"].astype(str).isin(bits)].copy()
    df["framework"] = "FINN"
    df["serie"] = "FINN"
    df["bits"] = df["bits_quant"].astype(str)
    df["rotulo"] = (df["fps_alvo"] / 1000).map(lambda v: f"{v:g}k")
    df["config"] = df["config"] + " alvo " + df["fps_alvo"].astype(str) + " FPS"
    df["fps"] = df["fps_obtido"]
    df["fps_fonte"] = "medido na placa"
    df["lut"] = df["lut_finn"]
    df["ff"] = df["ff_finn"]
    df["bram"] = df["bram_finn"]
    df["lut_sistema"] = df["lut_total"]
    df["ff_sistema"] = df["ff_total"]
    df["bram_sistema"] = df["bram_total"]
    df["macs"] = MACS_PAPER.get(topologia)
    return df


def metricas(df):
    df["mmacs_s"] = df["macs"] * df["fps"] / 1e6
    df["fps_por_klut"] = df["fps"] / (df["lut"] / 1e3)
    df["mmacs_s_por_klut"] = df["mmacs_s"] / (df["lut"] / 1e3)
    return df


def _fps_alvo(rf):
    """'fps6512' -> 6512.0 (sínteses geradas com main.py --target-fps)."""
    try:
        return float(str(rf)[3:]) if str(rf).startswith("fps") else None
    except ValueError:
        return None


def parear(hls, finn):
    """Para cada ponto FINN, as sínteses hls4ml de mesmos bits (sem BN/pooling) que o representam.

    1. Se houver sínteses com --target-fps igual ao FPS obtido ou ao alvo do FINN (1%), usa
       todas elas (uma por estratégia).
    2. Senão, a de FPS mais próximo (escala log), preferindo as que cabem no FPGA.
    """
    linhas = []
    for _, f in finn.iterrows():
        cand = hls[hls["bits"] == f["bits"]]
        if cand.empty:
            continue
        alvo = cand["rf"].map(_fps_alvo)
        mesmo_alvo = alvo.map(lambda a: a is not None and any(
            abs(a - v) <= 0.01 * v for v in (f["fps_obtido"], f["fps_alvo"])))
        if mesmo_alvo.any():
            escolhidos, criterio = cand[mesmo_alvo], "mesmo FPS alvo"
        else:
            if cand["cabe_no_fpga"].any():
                cand = cand[cand["cabe_no_fpga"]]
            dist = (cand["fps"].map(math.log10) - math.log10(f["fps"])).abs()
            escolhidos, criterio = cand.loc[[dist.idxmin()]], "FPS mais próximo"
        for _, h in escolhidos.iterrows():
            linhas.append(_linha_par(f, h, criterio))
    return pd.DataFrame(linhas)


def _linha_par(f, h, criterio):
    return {
        "finn": f["config"],
        "hls4ml": h["config"],
        "criterio": criterio,
        "fps_fonte_hls4ml": h["fps_fonte"],
        "fps_finn": f["fps"],
        "fps_hls4ml": h["fps"],
        "lut_finn": f["lut"],
        "lut_hls4ml": h["lut"],
        "ff_finn": f["ff"],
        "ff_hls4ml": h["ff"],
        "bram_finn": f["bram"],
        "bram_hls4ml": h["bram"],
        "dsp_hls4ml": h["dsp"],
        "lat_us_hls4ml": h["lat_us"],
        "hls4ml_cabe_no_fpga": bool(h["cabe_no_fpga"]),
        "razao_fps": h["fps"] / f["fps"],
        "razao_lut": h["lut"] / f["lut"],
        "razao_ff": h["ff"] / f["ff"],
        "razao_bram": h["bram"] / f["bram"],
        "razao_mmacs_s_por_klut": h["mmacs_s_por_klut"] / f["mmacs_s_por_klut"],
        "rf_por_camada": h["rf_por_camada"],
    }


def _estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EIXO)
    ax.tick_params(colors=INK2, labelsize=8)


def _cabe(r):
    v = r.get("cabe_no_fpga")
    return True if pd.isna(v) else bool(v)


# ---------------------------------------------------------------- gráficos
# Três grupos, cada um na sua subpasta:
#   1_finn_vs_hls4ml : FINN x hls4ml, 4 e 8 bits, sem BN/pooling
#   2_bn_pooling     : hls4ml 4 e 8 bits, com x sem BN + pooling
#   3_quantizacao    : hls4ml sem quantização x 8 x 4 bits, sem BN/pooling
# Em todos: marcador/barra vazada = não cabe no xc7z020.

ESTRATEGIAS = [("hls4ml Resource", "Resource"), ("hls4ml Latency (dense Resource)", "Latency")]
COR_BITS = {"float": "#898781", "8": "#7a4fc9", "4": "#eda100"}
NOME_BITS = {"float": "sem quantização", "8": "8 bits", "4": "4 bits"}


def _fmt(v):
    if pd.isna(v):
        return ""
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.1f}" if v != int(v) else f"{v:.0f}"


def _rotulo_perfil(p):
    a = _fps_alvo(p)
    return f"alvo {a:g} FPS" if a is not None else p


def _uma(df, **filtros):
    """Primeira linha de df que casa com todos os filtros (ou None)."""
    m = pd.Series(True, index=df.index)
    for k, v in filtros.items():
        m &= df[k] == v
    d = df[m]
    return None if d.empty else d.iloc[0]


def _face(cor, hatch):
    return _clarear(cor, 0.55) if hatch else cor


def _eixo_log(ax, fmt=None):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: (fmt or _fmt)(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _barras(ax, categorias, series, col, log=False, limite=None, titulo=None, fmt=_fmt):
    """series: [(nome, cor, hatch, f(categoria) -> linha ou None)]. Uma barra por série em cada categoria."""
    _estilo(ax)
    n = len(series)
    larg = 0.8 / n
    vertical = n * len(categorias) > 10
    for k, (nome, cor, hatch, pega) in enumerate(series):
        for i, cat in enumerate(categorias):
            r = pega(cat)
            if r is None or pd.isna(r.get(col)):
                continue
            ok = _cabe(r)
            x = i - 0.4 + larg * (k + 0.5)
            ax.bar(x, r[col], larg * 0.92, color=_face(cor, hatch) if ok else SUPERFICIE,
                   edgecolor=cor, linewidth=1.4, hatch=hatch, zorder=3)
            ax.annotate(fmt(r[col]), (x, r[col]), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=INK2,
                        rotation=90 if vertical else 0)
        ax.bar(0, 0, 0, color=_face(cor, hatch), edgecolor=cor, hatch=hatch, label=nome)
    if log:
        _eixo_log(ax)
    if limite:
        ax.axhline(limite, color=INK2, linestyle="--", linewidth=1, zorder=2)
        ax.annotate("limite xc7z020", (1, limite), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.margins(y=0.18)
    ax.set_xticks(range(len(categorias)))
    ax.set_xticklabels([c[1] if isinstance(c, tuple) else str(c) for c in categorias],
                       fontsize=8, color=INK2)
    if titulo:
        ax.set_title(titulo, fontsize=10, color=INK, loc="left")


def _figura_barras(paineis, categorias, series, titulo, saida, limites, ncols=2):
    """paineis: [(coluna, título, log)]. Legenda única no topo."""
    nrows = math.ceil(len(paineis) / ncols)
    fig, eixos = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.9 * nrows),
                              facecolor=SUPERFICIE, squeeze=False)
    for ax, (col, nome, log) in zip(eixos.flat, paineis):
        _barras(ax, categorias, series, col, log=log, limite=limites.get(col), titulo=nome)
    for ax in list(eixos.flat)[len(paineis):]:
        ax.set_visible(False)
    _titulo_legenda(fig, eixos[0][0], titulo, len(series), 3.9 * nrows)
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def _titulo_legenda(fig, ax, titulo, n, altura_pol):
    """Título e legenda única no topo, sem sobrepor os eixos."""
    topo = 1 - 0.75 / altura_pol
    h, l = ax.get_legend_handles_labels()
    fig.suptitle(titulo, fontsize=12, color=INK, x=0.01, y=1 - 0.12 / altura_pol, ha="left", va="top")
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.42 / altura_pol), ncol=min(n, 4),
               frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, topo))


def _xy_alvo(d):
    """Sínteses geradas com --target-fps: (FPS alvo, FPS obtido), ordenadas pelo alvo."""
    d = d.assign(x=d["rf"].map(_fps_alvo)).dropna(subset=["x"])
    return d.drop_duplicates("x").sort_values("x")


def _figura_alvo(paineis, titulo, saida):
    """FPS obtido x FPS alvo (log-log), um painel por item de paineis: (título, [(nome, cor, marcador,
    linha, df com colunas x e fps)]). Na diagonal o design entrega exatamente o alvo."""
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.2 * len(paineis), 4.8),
                              facecolor=SUPERFICIE, squeeze=False)
    for ax, (nome_p, series) in zip(eixos[0], paineis):
        _estilo(ax)
        todos = pd.concat([d[["x", "fps"]] for *_, d in series])
        lo, hi = todos.min().min() / 1.5, todos.max().max() * 1.5
        ax.plot([lo, hi], [lo, hi], color=INK2, linestyle=":", linewidth=1, zorder=1)
        ax.annotate("obtido = alvo", (hi / 1.6, hi / 1.6), rotation=45, ha="right", va="bottom",
                    fontsize=7, color=INK2, rotation_mode="anchor")
        for nome, cor, marcador, linha, d in series:
            ax.plot(d["x"], d["fps"], color=cor, linestyle=linha, linewidth=1.4, zorder=2)
            for _, r in d.iterrows():
                ok = _cabe(r)
                ax.scatter(r["x"], r["fps"], s=44, marker=marcador, zorder=3,
                           facecolor=cor if ok else SUPERFICIE, edgecolor=cor, linewidth=1.5)
            ax.plot([], [], color=cor, marker=marcador, linestyle=linha, label=nome)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _eixo_log(ax)
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(v)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("FPS alvo", color=INK2, fontsize=9)
        ax.set_ylabel("FPS obtido (cosim ou csynth)", color=INK2, fontsize=9)
        ax.set_title(nome_p, fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK2)
    fig.suptitle(titulo, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def _figura_eficiencia(paineis, categorias, titulo, saida):
    """FPS por kLUT em barras. paineis: [(título, series no formato de _barras)]."""
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.2 * len(paineis), 4.2),
                              facecolor=SUPERFICIE, squeeze=False)
    for ax, (nome_p, series) in zip(eixos[0], paineis):
        _barras(ax, categorias, series, "fps_por_klut", titulo=nome_p)
        ax.set_ylabel("FPS por mil LUTs (maior = melhor)", color=INK2, fontsize=9)
    _titulo_legenda(fig, eixos[0][0], titulo, len(paineis[0][1]), 4.2)
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def _perfil_finn(h, fr):
    """Perfil hls4ml cujo FPS alvo é o FPS medido do FINN (1%)."""
    for p in h["rf"].unique():
        a = _fps_alvo(p)
        if a is not None and abs(a - fr["fps_obtido"]) <= 0.01 * fr["fps_obtido"]:
            return p
    return None


def _figura_razoes(blocos, saida, titulo="hls4ml / FINN no mesmo FPS alvo"):
    """Mapa de calor hls4ml / FINN. blocos: [(título, [(rótulo da linha, {métrica: razão})])].
    A cor mostra quem leva vantagem: FPS maior é melhor, área menor é melhor."""
    from matplotlib.colors import LinearSegmentedColormap

    mets = [("fps", "FPS"), ("lut", "LUT"), ("ff", "FF"), ("bram", "BRAM")]
    cmap = LinearSegmentedColormap.from_list("vant", ["#c8453b", "#fcfcfb", "#1f9e5a"])
    lim = math.log2(8)
    n_linhas = max(len(l) for _, l in blocos)
    fig, eixos = plt.subplots(1, len(blocos), figsize=(5.2 * len(blocos), 0.55 * n_linhas + 2.2),
                              facecolor=SUPERFICIE, squeeze=False)
    for ax, (nome_b, linhas) in zip(eixos[0], blocos):
        vant = [[(1 if c == "fps" else -1) * math.log2(v[c]) for c, _ in mets] for _, v in linhas]
        ax.imshow(vant, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i, (_, v) in enumerate(linhas):
            for j, (c, _) in enumerate(mets):
                ax.text(j, i, f"{v[c]:.2f}x", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(mets)))
        ax.set_xticklabels([n for _, n in mets], fontsize=9, color=INK2)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(linhas)))
        ax.set_yticklabels([n for n, _ in linhas], fontsize=8, color=INK2)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_xticks([x - 0.5 for x in range(1, len(mets))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(linhas))], minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
        ax.set_title(nome_b, fontsize=11, color=INK, loc="left", pad=22)
    fig.suptitle(f"{titulo}   (verde = vantagem hls4ml, vermelho = vantagem FINN)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def grupo_finn(hls, finn, limites, perfis, pasta):
    """1. FINN x hls4ml nos mesmos bits, sem BN/pooling. O hls4ml usa o perfil cujo FPS alvo é
    o FPS medido do FINN (543 / 6512 / 16271)."""
    pasta.mkdir(parents=True, exist_ok=True)
    blocos, paineis_alvo, paineis_ef = [], [], []
    for bits in sorted(set(finn["bits"]) & set(hls["bits"])):
        f = finn[finn["bits"] == bits].sort_values("fps_alvo")
        h = hls[hls["bits"] == bits]
        linha_finn = {fr["fps_alvo"]: fr for _, fr in f.iterrows()}
        cats = [(fr["fps_alvo"], f"FINN alvo {fr['fps_alvo']:g}\n(hls4ml alvo {fr['fps_obtido']:g})")
                for _, fr in f.iterrows()]
        series = [("FINN (só acelerador)", COR["FINN"], None, lambda c, lf=linha_finn: lf[c[0]])]
        for serie, nome in ESTRATEGIAS:
            series.append((f"hls4ml {nome}", COR[serie], None,
                           lambda c, s=serie, h=h, lf=linha_finn: _uma(h, serie=s, rf=_perfil_finn(h, lf[c[0]]))))
        _figura_barras([("fps", "FPS (escala log)", True), ("lut", "LUT", False),
                        ("ff", "Flip-flops", False), ("bram", "BRAM (tiles 36Kb)", False)],
                       cats, series, f"t1 {bits} bits: FINN x hls4ml no mesmo FPS alvo",
                       pasta / f"barras_{bits}bits.png", limites)
        paineis_ef.append((f"{bits} bits", series))

        linhas = []
        for _, fr in f.iterrows():
            for serie, nome in ESTRATEGIAS:
                r = _uma(h, serie=serie, rf=_perfil_finn(h, fr))
                if r is not None:
                    linhas.append((f"alvo {fr['fps_alvo']:g} · {nome}",
                                   {c: r[c] / fr[c] for c in ("fps", "lut", "ff", "bram")}))
        blocos.append((f"{bits} bits", linhas))

        paineis_alvo.append((f"{bits} bits", [("FINN", COR["FINN"], "o", "-", f.assign(x=f["fps_alvo"]))]
                             + [(f"hls4ml {nome}", COR[s], m, "-", _xy_alvo(h[h["serie"] == s]))
                                for (s, nome), m in zip(ESTRATEGIAS, ("D", "s"))]))
    _figura_razoes(blocos, pasta / "razoes.png")
    _figura_alvo(paineis_alvo, "t1: FPS obtido x FPS alvo (FINN: alvo do SetFolding; hls4ml: alvo do main.py --target-fps)",
                 pasta / "alvo_vs_obtido.png")
    cats_ef = [(c[0], f"FINN alvo {c[0]:g}\n(hls4ml no FPS medido do FINN)") for c in cats]
    _figura_eficiencia(paineis_ef, cats_ef, "t1: eficiência de área (FPS por mil LUTs) no mesmo FPS alvo",
                       pasta / "eficiencia.png")


def grupo_bn_pooling(hls, limites, perfis, pasta):
    """2. hls4ml com x sem BN + pooling, por quantização."""
    pasta.mkdir(parents=True, exist_ok=True)
    cats = [(p, _rotulo_perfil(p)) for p in perfis]
    paineis_alvo, paineis_ef = [], []
    for bits in ("4", "8"):
        variantes = [(bits, "sem BN/pooling", None, "-"), (bits + "_bn", "com BN + pooling", "///", "--")]
        if not all((hls["bits"] == v).any() for v, *_ in variantes):
            continue
        series = []
        for serie, nome in ESTRATEGIAS:
            for v, vnome, hatch, _ in variantes:
                series.append((f"{nome}, {vnome}", COR[serie], hatch,
                               lambda c, s=serie, v=v: _uma(hls, bits=v, serie=s, rf=c[0])))
        _figura_barras([("fps", "FPS (escala log)", True), ("lut", "LUT", False),
                        ("ff", "Flip-flops", False), ("bram", "BRAM (tiles 36Kb)", False)],
                       cats, series, f"hls4ml t1 {bits} bits: com (hachurado) x sem BN + pooling, mesmo FPS alvo",
                       pasta / f"barras_{bits}bits.png", limites, ncols=2)
        paineis_ef.append((f"{bits} bits", series))
        paineis_alvo.append((f"{bits} bits", [
            (f"{nome}, {vnome}", COR[serie] if hatch is None else _clarear(COR[serie], 0.35),
             m, linha, _xy_alvo(hls[(hls["bits"] == v) & (hls["serie"] == serie)]))
            for (serie, nome), m in zip(ESTRATEGIAS, ("D", "s")) for v, vnome, hatch, linha in variantes]))
    _figura_alvo(paineis_alvo, "hls4ml t1: FPS obtido x FPS alvo, com (tracejado) x sem BN + pooling",
                 pasta / "alvo_vs_obtido.png")
    _figura_eficiencia(paineis_ef, cats, "hls4ml t1: FPS por mil LUTs, com (hachurado) x sem BN + pooling",
                       pasta / "eficiencia.png")
    _figura_acuracia(hls, [("4", "4 bits"), ("4_bn", "4 bits + BN + pool"),
                           ("8", "8 bits"), ("8_bn", "8 bits + BN + pool")],
                     "Acurácia hls4ml (teste): com x sem BN + pooling", pasta / "acuracia.png")


def _rotulo_da(r):
    """'Tudo em DA (RF=1)' ou 'Convs DA (RF=1) + Dense Resource (RF=4096)' a partir do extra do relatorio.py.
    O DA só existe com RF=1; o RF do rótulo híbrido é o da Dense, que não está em DA."""
    m = re.search(r"dense([A-Za-z]+)(\d+)", str(r["extra"]) if pd.notna(r["extra"]) else "")
    return "Tudo em DA (RF=1)" if m is None else f"Convs DA (RF=1) + Dense {m.group(1)} (RF={m.group(2)})"


def grupo_da(hls, finn, limites, pasta):
    """4. Distributed Arithmetic (RF=1, sintetizado com main.py --da) x as sínteses compatíveis:
    mesmo io_stream e RF=1 em todas as camadas (perfil rfmin, Latency e Resource), a síntese hls4ml
    no mesmo FPS alvo do FINN (fps16271) e o FINN de maior throughput (alvo 50000)."""
    da = hls[hls["tipo"] == "da"]
    if da.empty:
        print("Sem sínteses DA (rode main.py --da); grupo 4_da pulado.")
        return
    pasta.mkdir(parents=True, exist_ok=True)
    blocos = []
    for bits in sorted(set(da["bits"])):
        h = hls[hls["bits"] == bits]
        configs = []
        for _, r in da[da["bits"] == bits].iterrows():
            nome = _rotulo_da(r)
            # uma hachura por variante híbrida: Dense Resource RF alto, Resource RF=1, Latency RF=1
            hatch = (None if nome.startswith("Tudo") else "xx" if "Latency" in nome
                     else "///" if "(RF=1)" not in nome.split("Dense")[1] else "..")
            configs.append((nome, COR["hls4ml DA"], hatch, r))
        for serie, nome in reversed(ESTRATEGIAS):
            r = _uma(h, serie=serie, rf="rfmin")
            if r is not None:
                configs.append((f"{nome} RF=1", COR[serie], None, r))
        r = _uma(h, serie="hls4ml Latency (dense Resource)", rf="fps16271")
        if r is not None:
            configs.append(("Latency no FPS do FINN (alvo 16271)", _clarear(COR["hls4ml Latency (dense Resource)"]), None, r))
        f = finn[(finn["bits"] == bits.split("_")[0]) & (finn["fps_alvo"] == 50000)]
        if not f.empty:
            configs.append(("FINN alvo 50000 (só acelerador)", COR["FINN"], None, f.iloc[0]))

        series = [(nome, cor, hatch, lambda c, r=r: r) for nome, cor, hatch, r in configs]
        _figura_barras([("fps", "FPS", False), ("lut", "LUT", False),
                        ("ff", "Flip-flops", False), ("bram", "BRAM (tiles 36Kb)", False),
                        ("lat_us", "Latência (µs)", False), ("fps_por_klut", "FPS por mil LUTs", False)],
                       [(0, "")], series,
                       f"t1 {NOME_BITS.get(bits, bits)}: Distributed Arithmetic x RF=1 (io_stream) x FINN",
                       pasta / f"barras_{bits}bits.png", limites, ncols=3)

        if not f.empty:
            fr = f.iloc[0]
            blocos.append((NOME_BITS.get(bits, bits), [
                (nome, {c: r[c] / fr[c] for c in ("fps", "lut", "ff", "bram")})
                for nome, _, _, r in configs if not nome.startswith("FINN")]))
    if blocos:
        _figura_razoes(blocos, pasta / "razoes_finn.png", titulo="hls4ml / FINN alvo 50000")


def grupo_quantizacao(hls, limites, perfis, pasta):
    """3. hls4ml sem quantização x 8 x 4 bits, sem BN/pooling, uma figura por estratégia."""
    pasta.mkdir(parents=True, exist_ok=True)
    cats = [(p, _rotulo_perfil(p)) for p in perfis]
    bits_l = [b for b in ("float", "8", "4") if (hls["bits"] == b).any()]
    paineis_alvo, paineis_ef = [], []
    for serie, nome in ESTRATEGIAS:
        series = [(NOME_BITS[b], COR_BITS[b], None,
                   lambda c, b=b, serie=serie: _uma(hls, bits=b, serie=serie, rf=c[0])) for b in bits_l]
        _figura_barras([("fps", "FPS (escala log)", True), ("lut", "LUT", False),
                        ("ff", "Flip-flops", False), ("bram", "BRAM (tiles 36Kb)", False),
                        ("dsp", "DSP", False), ("lat_us", "Latência (µs, escala log)", True)],
                       cats, series, f"hls4ml t1 {nome}: sem quantização x 8 x 4 bits (sem BN/pooling)",
                       pasta / f"barras_{nome.lower()}.png", limites, ncols=2)
        paineis_ef.append((nome, series))
        paineis_alvo.append((nome, [(NOME_BITS[b], COR_BITS[b], "o", "-",
                                     _xy_alvo(hls[(hls["bits"] == b) & (hls["serie"] == serie)]))
                                    for b in bits_l]))
    _figura_alvo(paineis_alvo, "hls4ml t1: FPS obtido x FPS alvo por quantização (vazado = não cabe)",
                 pasta / "alvo_vs_obtido.png")
    _figura_eficiencia(paineis_ef, cats, "hls4ml t1: FPS por mil LUTs por quantização",
                       pasta / "eficiencia.png")
    _figura_acuracia(hls, [(b, NOME_BITS[b]) for b in bits_l],
                     "Acurácia hls4ml (teste) por quantização", pasta / "acuracia.png")


def _clarear(cor, f=0.5):
    r, g, b = (int(cor[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in (r, g, b))


def _figura_acuracia(hls, variantes, titulo, saida):
    vals = [(nome, hls.loc[hls["bits"] == b, "acc_hls4ml"].dropna()) for b, nome in variantes]
    vals = [(nome, v.iloc[0] * 100) for nome, v in vals if not v.empty]
    fig, ax = plt.subplots(figsize=(1.6 * len(vals) + 2, 3.6), facecolor=SUPERFICIE)
    _estilo(ax)
    ax.bar(range(len(vals)), [v for _, v in vals], 0.6, color=COR["FINN"], zorder=3)
    for i, (_, v) in enumerate(vals):
        ax.annotate(f"{v:.2f}%", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([n for n, _ in vals], fontsize=8, color=INK2)
    menor = min(v for _, v in vals)
    ax.set_ylim(math.floor(menor) - 2, 100)
    ax.set_ylabel("acurácia (%)", color=INK2, fontsize=9)
    ax.set_title(titulo, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)


def _md(df, colunas, fmt):
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "---|" * len(colunas)]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in colunas) + " |")
    return "\n".join(linhas)


def escrever_md(dados, par, macs_hls, topologia, saida):
    n = lambda v: "-" if pd.isna(v) else f"{v:,.0f}".replace(",", ".")
    x = lambda v: "-" if pd.isna(v) else f"{v:.2f}x"
    f1 = lambda v: "-" if pd.isna(v) else f"{v:.1f}"
    sim = lambda v: "sim" if v else "**não**"

    cols_tab = ["framework", "bits", "config", "fps", "fps_fonte", "lat_us", "lut", "ff", "bram",
                "dsp", "macs", "mmacs_s_por_klut", "acc_hls4ml", "cabe_no_fpga"]
    tab = dados.sort_values(["bits", "framework", "fps"])
    fmt_tab = {"fps": n, "lat_us": f1, "lut": n, "ff": n, "bram": f1, "dsp": n, "macs": n,
               "mmacs_s_por_klut": f1,
               "acc_hls4ml": lambda v: "-" if pd.isna(v) else f"{v * 100:.2f}%",
               "cabe_no_fpga": lambda v: "-" if pd.isna(v) else sim(v)}
    cols_par = ["finn", "hls4ml", "criterio", "fps_finn", "fps_hls4ml", "fps_fonte_hls4ml", "razao_fps", "razao_lut", "razao_ff",
                "razao_bram", "razao_mmacs_s_por_klut", "hls4ml_cabe_no_fpga", "rf_por_camada"]
    fmt_par = {"fps_finn": n, "fps_hls4ml": n, "razao_fps": x, "razao_lut": x, "razao_ff": x,
               "razao_bram": x, "razao_mmacs_s_por_klut": x, "hls4ml_cabe_no_fpga": sim}

    macs_txt = "\n".join(f"- hls4ml {b}: {n(m)} MACs/inferência" for b, m in macs_hls.items())
    texto = f"""# Comparação hls4ml x FINN ({topologia})

## Avisos de comparabilidade

- **Área**: síntese do Vivado nos dois lados. FINN = só o acelerador (`*_finn`, do
  `finn_design_partition_util.rpt`, sem AXI/DMA), lido do gráfico da Fig. 6 (precisão de
  ~500 LUT/FF e ~0,5 BRAM). hls4ml = `vivado_synth.rpt` do IP.
- **FPS FINN** = medido na placa (inclui DMA). **FPS hls4ml** = `1e9 / (II · T)`: II medido
  na cosim (espaçamento entre saídas, `cosim_projetos.py`) quando existe, senão o do csynth; T
  do `vivado_timing.rpt` (o clock alvo se o timing fechou, senão o período atingido). A coluna
  `fps_fonte` diz qual foi usado. Nas convs em Resource o csynth cobra RF ciclos em todo pixel
  de entrada, mas o RTL só gasta isso nos pixels de saída: o II do csynth chega a ser 4x o real.
- **Latência hls4ml** = latência da 1ª imagem na cosim (ou a do csynth) × o mesmo T. O FINN não
  reporta **latência** nem **DSP**.
- **Topologia**: FINN {topologia} = {n(MACS_PAPER.get(topologia))} MACs/inferência.
{macs_txt}
  Quando os MACs diferem, compare pela coluna `mmacs_s_por_klut`.

## Pares hls4ml x FINN (mesma quantização, sem BN/pooling)

`criterio`: "mesmo FPS alvo" = síntese gerada com `main.py --target-fps` no FPS do FINN;
"FPS mais próximo" = a síntese existente de FPS mais próximo (preferindo as que cabem).

{_md(par, cols_par, fmt_par)}

## Todas as configurações

{_md(tab, cols_tab, fmt_tab)}

## Gráficos

Barras e pontos vazados = síntese que não cabe no xc7z020.

- `1_finn_vs_hls4ml/`: FINN x hls4ml, 4 e 8 bits, sem BN/pooling, no mesmo FPS alvo
  - `barras_<b>bits.png`: FPS, LUT, FF e BRAM lado a lado
  - `razoes.png`: hls4ml / FINN por métrica (verde = vantagem hls4ml)
  - `alvo_vs_obtido.png`: FPS obtido x FPS alvo
  - `eficiencia.png`: FPS por mil LUTs
- `2_bn_pooling/`: hls4ml 4 e 8 bits com x sem BN + pooling
  - `barras_<b>bits.png`, `alvo_vs_obtido.png`, `eficiencia.png`, `acuracia.png`
- `4_da/`: Distributed Arithmetic (`main.py --da`) x RF=1 (Latency e Resource, io_stream) x hls4ml no
  FPS do FINN x FINN alvo 50000
  - `barras_<b>bits.png`, `razoes_finn.png`
- `3_quantizacao/`: hls4ml sem quantização x 8 x 4 bits, sem BN/pooling
  - `barras_resource.png`, `barras_latency.png`, `alvo_vs_obtido.png`, `eficiencia.png`, `acuracia.png`
"""
    (saida / "comparacao.md").write_text(texto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumo", default="relatorios_t1_projects/resumo.csv",
                    help="resumo.csv do relatorio.py")
    ap.add_argument("--finn", default="dados_paper/fig6_recursos_finn.csv",
                    help="CSV com os dados da Fig. 6 do paper")
    ap.add_argument("--topologia", default="t1", help="topologia do paper a comparar")
    ap.add_argument("--bits", nargs="+", default=["4", "8"], help="quantizações FINN a comparar")
    ap.add_argument("--out", default="comparacao_finn")
    ap.add_argument("--perfis", nargs="+", default=["fps543", "fps6512", "fps16271", "fps50000"],
                    help="perfis de RF nos gráficos de barras (fps500/fps5000 repetem os RFs de 543/6512)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    resolver = lambda p: Path(p) if Path(p).exists() else base / p
    saida = Path(args.out)
    saida.mkdir(parents=True, exist_ok=True)

    hls = metricas(carregar_hls(resolver(args.resumo)))
    finn = metricas(carregar_finn(resolver(args.finn), args.topologia, args.bits))
    if hls.empty or finn.empty:
        print("Sem dados para comparar (verifique --resumo e --topologia).")
        return

    limites = {
        "lut": hls["viv_LUT_disp"].max(),
        "ff": hls["viv_FF_disp"].max(),
        "bram": hls["viv_BRAM_tile_disp"].max(),
        "dsp": hls["viv_DSP_disp"].max(),
    }
    colunas = ["framework", "serie", "bits", "config", "rotulo", "rf", "fps", "fps_fonte", "fps_csynth",
               "viv_wns_ns", "fps_clk_alvo", "lat_us",
               "lut", "ff", "bram", "dsp", "lut_sistema", "ff_sistema", "bram_sistema",
               "macs", "mmacs_s", "fps_por_klut", "mmacs_s_por_klut", "acc_hls4ml",
               "cabe_no_fpga", "rf_por_camada"]
    dados = pd.concat([hls, finn], ignore_index=True).reindex(columns=colunas)
    dados["cabe_no_fpga"] = dados["cabe_no_fpga"].astype("boolean")
    par = parear(hls, finn)

    dados.to_csv(saida / "tabela_completa.csv", index=False)
    par.to_csv(saida / "pareamento_finn.csv", index=False)

    grupo_finn(hls, finn, limites, args.perfis, saida / "1_finn_vs_hls4ml")
    grupo_bn_pooling(hls, limites, args.perfis, saida / "2_bn_pooling")
    grupo_quantizacao(hls, limites, args.perfis, saida / "3_quantizacao")
    grupo_da(hls, finn, limites, saida / "4_da")

    macs_hls = hls.groupby("bits")["macs"].first().to_dict()
    escrever_md(dados, par, macs_hls, args.topologia, saida)

    pd.set_option("display.width", 200)
    print(par[["finn", "hls4ml", "criterio", "fps_finn", "fps_hls4ml", "razao_lut", "razao_ff", "razao_bram",
               "razao_mmacs_s_por_klut", "hls4ml_cabe_no_fpga"]].round(2).to_string(index=False))
    print(f"\nSaída em {saida.resolve()} (comparacao.md, tabela_completa.csv, pareamento_finn.csv, *.png)")


if __name__ == "__main__":
    main()
