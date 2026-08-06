"""
# Executa tudo
python main.py --all

# Apenas exporta modelos
python main.py --export

# Exporta + gera dados
python main.py --export --test-data

# Converte para hls4ml
python main.py --convert

# Apenas síntese
python main.py --build

# Pipeline completo sem síntese
python main.py --export --test-data --yaml --convert

# Apenas 8 bits
python main.py --all --bits 8

# Apenas 4 e 2 bits
python main.py --all --bits 4 2
"""

import argparse

import export_models
import generate_test_data
import generate_yaml
import convert_models
import build_projects


DEFAULT_BITS = [8, 4, 2, 1]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Pipeline hls4ml"
    )

    parser.add_argument(

        "--experiment",

        default="t1",

        help="Nome do experimento"

    )

    parser.add_argument(

        "--bits",

        nargs="+",

        type=int,

        default=DEFAULT_BITS,

        help="Bits a serem utilizados"

    )

    parser.add_argument(

        "--export",

        action="store_true",

        help="Exportar modelos"

    )

    parser.add_argument(

        "--test-data",

        action="store_true",

        help="Gerar arquivos DAT"

    )

    parser.add_argument(

        "--yaml",

        action="store_true",

        help="Gerar YAML"

    )

    parser.add_argument(

        "--convert",

        action="store_true",

        help="Converter para hls4ml"

    )

    parser.add_argument(

        "--build",

        action="store_true",

        help="Executar síntese"

    )

    parser.add_argument(

        "--all",

        action="store_true",

        help="Executar pipeline completo"

    )

    return parser.parse_args()


def main():

    args = parse_args()

    bits = args.bits

    experiment = args.experiment

    print("=" * 80)
    print("HLS4ML PIPELINE")
    print("=" * 80)

    if args.all:

        print("\n[1/5] Exportando modelos")
        export_models.run(bits, experiment)

        print("\n[2/5] Gerando dados")
        generate_test_data.run(bits, experiment)

        print("\n[3/5] Gerando YAML")
        generate_yaml.run(bits, experiment)

        print("\n[4/5] Convertendo")
        convert_models.run(bits, experiment)

        print("\n[5/5] Build")
        build_projects.run(bits, experiment)

        print("\nPipeline concluído.")
        return


    if args.export:

        export_models.run(bits, experiment)

    if args.test_data:

        generate_test_data.run(bits, experiment)

    if args.yaml:

        generate_yaml.run(bits, experiment)

    if args.convert:

        convert_models.run(bits, experiment)

    if args.build:

        build_projects.run(bits, experiment)


if __name__ == "__main__":

    main()