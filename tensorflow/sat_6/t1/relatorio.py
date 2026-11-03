import argparse
import csv
import re
from pathlib import Path

ARQUIVOS = {
    "vivado_synth.rpt": "vivado_synth.rpt",
    "vivado_timing.rpt": "vivado_timing.rpt",
    "accuracy.txt": "accuracy.txt",
    "myproject_csynth.rpt": "myproject_prj/solution1/syn/report/myproject_csynth.rpt",
}
PARAMETERS_H = "firmware/parameters.h"
# só existe nos projetos em que o cosim_projetos.py (ou build cosim=True) rodou
COSIM_RPT = "myproject_prj/solution1/sim/report/myproject_cosim.rpt"


PADRAO_NOVO = re.compile(
    r"^(?P<rede>base|paper|t1)"
    r"(?:_(?P<bits>\d+))?(?:_(?P<variante>[a-z_]+?))?"
    r"_io_(?P<io>stream|parallel)"
    r"_(?:rf(?P<rf>\d+|min|med|max)|(?P<fps>fps[\d.]+))"
    r"_(?P<tipo>latency|resource|da)"
    r"(?P<resto>_.*)?$",
    re.I,
)


def classificar(nome_pasta):
    """Retorna dict com rede, quant, rf, tipo, io, clock_ns, extra a partir do nome da pasta."""
    n = nome_pasta[len("prj_"):] if nome_pasta.startswith("prj_") else nome_pasta

    m = PADRAO_NOVO.match(n)
    if m:
        bits = m.group("bits")
        quant = f"quantizado_{bits}bits" if bits else "nao_quantizado"
        cp = re.search(r"cp(\d+)", n)
        extra = []
        if m.group("variante"):
            extra.append(m.group("variante"))
        if m.group("resto"):
            extra.append(m.group("resto").strip("_").replace("without_pooling", "nopool"))

        return {
            "rede": m.group("rede").lower(),
            "quant": quant,
            "rf": f"rf{m.group('rf')}" if m.group("rf") else m.group("fps"),
            "tipo": m.group("tipo").lower(),
            "io": f"io_{m.group('io').lower()}",
            "clock_ns": float(cp.group(1)) if cp else 10.0,
            "extra": "_".join(extra),
        }
    
    elif n.startswith("base"):
        rede, quant = "base", "nao_quantizado"
    elif n.startswith("t1"):
        rede, quant = "t1", "nao_quantizado"
    elif n.startswith("paper"):
        rede, quant = "paper", "nao_quantizado"
    else:
        rede, quant = "desconhecida", "desconhecido"

    rf = re.search(r"rf(\d+)", n)
    tipo = re.search(r"(latency|resource)", n, re.I)
    cp = re.search(r"cp(\d+)", n)
    io = re.search(r"io_(stream|parallel)", n)

    extra = []
    if "without_pooling" in n:
        extra.append("nopool")
    if cp:
        extra.append(f"cp{cp.group(1)}")

    return {
        "rede": rede,
        "quant": quant,
        "rf": f"rf{rf.group(1)}" if rf else "rfNA",
        "tipo": tipo.group(1).lower() if tipo else "tipoNA",
        "io": io.group(0) if io else "",
        "clock_ns": float(cp.group(1)) if cp else 10.0,
        "extra": "_".join(extra),
    }


def ler(path):
    try:
        return Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return None


def parse_vivado(txt):
    """Extrai linhas de utilização do vivado_synth.rpt."""
    r = {}
    if not txt:
        return r
    chaves = {
        "Slice LUTs*": "viv_LUT",
        "Slice Registers": "viv_FF",
        "Block RAM Tile": "viv_BRAM_tile",
        "DSPs": "viv_DSP",
    }
    for linha in txt.splitlines():
        cols = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(cols) >= 6 and cols[0] in chaves:
            r[chaves[cols[0]]] = cols[1]
            r[chaves[cols[0]] + "_pct"] = cols[-1]
            r[chaves[cols[0]] + "_disp"] = cols[-2]
    return r


def parse_timing(txt, clock_ns):
    """WNS do report_timing_summary do Vivado -> período e Fmax atingidos após a síntese."""
    if not txt:
        return {}
    linhas = txt.splitlines()
    for i, linha in enumerate(linhas):
        if "WNS(ns)" in linha and "TNS(ns)" in linha:
            for prox in linhas[i + 1:i + 5]:
                m = re.match(r"\s*(-?\d+\.\d+)\s", prox)
                if m:
                    wns = float(m.group(1))
                    periodo = clock_ns - wns
                    return {
                        "viv_wns_ns": f"{wns:.3f}",
                        "viv_periodo_ns": f"{periodo:.3f}",
                        "viv_fmax_mhz": f"{1e3 / periodo:.1f}",
                        "viv_timing_ok": wns >= 0,
                    }
    return {}


def parse_cosim(txt):
    """Linha Verilog do myproject_cosim.rpt: status, latência e intervalo medidos (ciclos)."""
    if not txt:
        return {}
    for linha in txt.splitlines():
        cols = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(cols) >= 8 and cols[0].lower() == "verilog":
            nomes = ["cosim_lat_min", "cosim_lat_avg", "cosim_lat_max",
                     "cosim_ii_min", "cosim_ii_avg", "cosim_ii_max", "cosim_total_cyc"]
            return {"cosim_status": cols[1], **dict(zip(nomes, cols[2:9]))}
    return {}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_csynth(txt):
    """Extrai clock, latência, II e utilização estimada do csynth.rpt."""
    r = {}
    if not txt:
        return r

    m = re.search(r"\|\s*ap_clk\s*\|\s*([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns\s*\|", txt)
    if m:
        r["clk_target_ns"] = m.group(1)
        r["clk_estimado_ns"] = m.group(2)

    # tabela de latência (primeira linha de dados após o cabeçalho)
    idx = txt.find("Latency (cycles)")
    if idx >= 0:
        pad = re.compile(
            r"\|\s*([\d?]+)\s*\|\s*([\d?]+)\s*\|\s*([\d.?]+\s*\w*)\s*\|\s*([\d.?]+\s*\w*)\s*\|"
            r"\s*([\d?]+)\s*\|\s*([\d?]+)\s*\|\s*(\S[^|]*?)\s*\|"
        )
        for linha in txt[idx:].splitlines()[1:12]:
            mm = pad.search(linha)
            if mm:
                (r["lat_min_cyc"], r["lat_max_cyc"], r["lat_min_abs"], r["lat_max_abs"],
                 r["ii_min"], r["ii_max"], r["pipeline"]) = mm.groups()
                break

    u = txt.find("Utilization Estimates")
    if u >= 0:
        sub = txt[u:].splitlines()
        header = None
        for linha in sub:
            if "BRAM_18K" in linha and "LUT" in linha:
                header = [c.strip() for c in linha.strip().strip("|").split("|")]
            elif header and re.match(r"\|\s*Total\s*\|", linha):
                vals = [c.strip() for c in linha.strip().strip("|").split("|")]
                for h, v in zip(header[1:], vals[1:]):
                    r[f"hls_{h}"] = v
                break
    return r


def parse_acc(txt):
    """Pega os números decimais do accuracy.txt (raw fica no arquivo)."""
    if not txt:
        return {}
    nums = re.findall(r"\d+\.\d+", txt)
    r = {"acc_primeiro_valor": nums[0]} if nums else {}
    if len(nums) > 1:
        r["acc_segundo_valor"] = nums[1]
    return r


def parse_parameters(txt):
    """RF/strategy efetivamente sintetizados por camada, a partir do firmware/parameters.h.

    Cada bloco de camada começa com um comentário '// nome_da_camada'; usa o primeiro
    reuse_factor/strategy/n_in/n_out/out_height/out_width encontrados nas structs do bloco.
    Em conv, a primeira struct é a *_mult (n_in = kernel*canais, n_out = filtros).
    """
    camadas = {}
    if not txt:
        return camadas
    atual = None
    for linha in txt.splitlines():
        c = re.match(r"^//\s*(\w+)\s*$", linha)
        if c:
            atual = c.group(1)
            camadas[atual] = {}
            continue
        if atual is None:
            continue
        s = re.match(r"^struct\s+\w+\s*:\s*nnet::(\w+)", linha)
        if s:
            camadas[atual].setdefault("tipo", s.group(1))
        rf = re.search(r"reuse_factor\s*=\s*(\d+);", linha)
        if rf:
            camadas[atual].setdefault("rf", int(rf.group(1)))
        st = re.search(r"strategy\s*=\s*nnet::(\w+);", linha)
        if st:
            camadas[atual].setdefault("strategy", st.group(1))
        dim = re.search(r"\b(n_in|n_out|out_height|out_width)\s*=\s*(\d+);", linha)
        if dim:
            camadas[atual].setdefault(dim.group(1), int(dim.group(2)))
    for c in camadas.values():
        if _tem_mult(c.get("tipo")) and "n_in" in c and "n_out" in c:
            c["macs"] = c["n_in"] * c["n_out"] * c.get("out_height", 1) * c.get("out_width", 1)
    return {k: v for k, v in camadas.items() if "rf" in v}


def _tem_mult(tipo):
    return any(t in (tipo or "") for t in ("dense", "conv"))


def resumo_rf(camadas):
    """Colunas rf_<camada>/strategy_<camada> para camadas com multiplicadores (conv/dense)."""
    r = {}
    pares = []
    macs = 0
    for nome, c in camadas.items():
        if not _tem_mult(c.get("tipo")):
            continue
        r[f"rf_{nome}"] = c["rf"]
        if "strategy" in c:
            r[f"strategy_{nome}"] = c["strategy"]
        pares.append(f"{nome}={c['rf']}")
        macs += c.get("macs", 0)
    if pares:
        r["rf_por_camada"] = ";".join(pares)
    if macs:
        r["macs_por_inferencia"] = macs
    return r


def calc_fps(info, hls, timing=None):
    """FPS = 1 / (II * período). II (ciclos) vem do csynth; o período, do clock alvo, do estimado
    pelo HLS e, se houver vivado_timing.rpt, do Vivado (fps_vivado: alvo se o timing fechou,
    senão o período atingido)."""
    ii = _num(hls.get("ii_max"))
    per = _num(hls.get("clk_target_ns")) or info["clock_ns"]
    per_est = _num(hls.get("clk_estimado_ns"))
    per_viv = _num((timing or {}).get("viv_periodo_ns"))
    out = {}
    if ii and per and per_viv:
        out["fps_vivado"] = f"{1e9 / (ii * max(per, per_viv)):.2f}"
    if ii and per:
        out["fps_clk_alvo"] = f"{1e9 / (ii * per):.2f}"
    if ii and per_est:
        out["fps_clk_estimado"] = f"{1e9 / (ii * per_est):.2f}"
    lat = _num(hls.get("lat_max_cyc"))
    if lat and per:
        out["fps_1_sobre_latencia"] = f"{1e9 / (lat * per):.2f}"
    return out


def resumir_pasta(pasta):
    """Lê os relatórios de um projeto. Retorna (info do nome, resumo, conteúdos dos arquivos, camadas)."""
    pasta = Path(pasta)
    info = classificar(pasta.name)
    conteudos = {k: ler(pasta / rel) for k, rel in ARQUIVOS.items()}

    viv = parse_vivado(conteudos["vivado_synth.rpt"])
    hls = parse_csynth(conteudos["myproject_csynth.rpt"])
    timing = parse_timing(conteudos["vivado_timing.rpt"],
                          _num(hls.get("clk_target_ns")) or info["clock_ns"])
    acc = parse_acc(conteudos["accuracy.txt"])
    fps = calc_fps(info, hls, timing)
    camadas = parse_parameters(ler(pasta / PARAMETERS_H))
    rfs = resumo_rf(camadas)

    cosim_txt = ler(pasta / COSIM_RPT)
    cosim = parse_cosim(cosim_txt)
    if cosim_txt is not None:
        conteudos["myproject_cosim.rpt"] = cosim_txt
    # O "Interval" do relatório é o tempo entre dois inícios: com FIFOs que cabem uma imagem inteira
    # ele só mede a leitura da entrada (1024 ciclos). O throughput real é o espaçamento entre as
    # saídas: (ciclo em que a última imagem termina - latência da primeira) / (n - 1).
    n = len((ler(pasta / "tb_data" / "rtl_cosim_results.log") or "").split("\n")) - 1
    total, lat0 = _num(cosim.get("cosim_total_cyc")), _num(cosim.get("cosim_lat_min"))
    per = _num(hls.get("clk_target_ns")) or info["clock_ns"]
    if total and lat0 and n >= 2 and per:
        ii_ef = (total - lat0) / (n - 1)
        cosim["cosim_n_imagens"] = n
        cosim["cosim_ii_efetivo"] = f"{ii_ef:.0f}"
        cosim["fps_cosim"] = f"{1e9 / (ii_ef * max(per, _num(timing.get('viv_periodo_ns')) or 0)):.2f}"
    return info, {**viv, **timing, **hls, **acc, **fps, **rfs, **cosim}, conteudos, camadas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", default=None,
                    help="pasta com os projetos de síntese (padrão: t1_projects)")
    ap.add_argument("--projects", default=None, help="mesmo que o argumento posicional")
    ap.add_argument("--out", default=None,
                    help="pasta de saída (padrão: relatorios, ou relatorios_<pasta> se a pasta != projects)")
    args = ap.parse_args()

    raiz = Path(args.pasta or args.projects or "t1_projects")
    # caminho relativo inexistente no cwd: tenta ao lado deste script
    if not raiz.is_absolute() and not raiz.exists():
        raiz = Path(__file__).resolve().parent / raiz
    if not raiz.is_dir():
        print(f"Pasta não encontrada: {raiz}")
        return

    if args.out:
        saida = Path(args.out)
    elif raiz.name == "projects":
        saida = raiz.parent / "relatorios"
    else:
        saida = raiz.parent / f"relatorios_{raiz.name}"
    saida.mkdir(parents=True, exist_ok=True)

    linhas_csv = []
    pastas = sorted(p for p in raiz.iterdir() if p.is_dir())
    if not pastas:
        print(f"Nenhuma pasta em {raiz.resolve()}")
        return

    for pasta in pastas:
        info, resumo, conteudos, camadas = resumir_pasta(pasta)

        partes = [info["rede"], info["quant"], info["rf"], info["tipo"]]
        if info["extra"]:
            partes.append(info["extra"])
        nome_saida = "_".join(partes) + ".txt"

        with open(saida / nome_saida, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"PASTA: {pasta.name}\n")
            for k in ("rede", "quant", "rf", "tipo", "io", "clock_ns", "extra"):
                f.write(f"{k}: {info[k]}\n")
            f.write("\nRESUMO EXTRAÍDO (n/d = não encontrado)\n")
            f.write("-" * 80 + "\n")
            for k, v in resumo.items():
                f.write(f"{k}: {v}\n")
            f.write("\nREUSE FACTOR POR CAMADA (firmware/parameters.h)\n")
            f.write("-" * 80 + "\n")
            if camadas:
                f.write(f"{'camada':<28}{'rf':>8}  {'strategy':<10}{'macs':>10}  {'tipo'}\n")
                for nome, c in camadas.items():
                    f.write(f"{nome:<28}{c['rf']:>8}  {c.get('strategy', '-'):<10}"
                            f"{c.get('macs', '-'):>10}  {c.get('tipo', '-')}\n")
            else:
                f.write("[parameters.h não encontrado]\n")
            f.write("\nNOTA: FPS = 1e9 / (II_max * período_ns); II = initiation interval do csynth. "
                    "fps_vivado usa o período do vivado_timing.rpt (alvo se WNS >= 0).\n")
            for nome, txt in conteudos.items():
                f.write("\n" + "=" * 80 + f"\nARQUIVO: {nome}\n" + "=" * 80 + "\n")
                f.write(txt if txt is not None else "[ARQUIVO NÃO ENCONTRADO]\n")
                if txt is not None and not txt.endswith("\n"):
                    f.write("\n")

        linhas_csv.append({**info, **resumo})
        faltando = [k for k, v in conteudos.items() if v is None]
        print(f"[ok] {pasta.name} -> {nome_saida}" + (f"  (faltando: {', '.join(faltando)})" if faltando else ""))

    campos = []
    for l in linhas_csv:
        for k in l:
            if k not in campos:
                campos.append(k)
    with open(saida / "resumo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"\n{len(linhas_csv)} pastas processadas. Saída em {saida.resolve()} (inclui resumo.csv)")


if __name__ == "__main__":
    main()