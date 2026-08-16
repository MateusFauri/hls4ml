from config import (
    BIT_LIST,
    MODELS,
    TRAIN_CONFIGS,
)

from dataset import load_sat6
from evaluator import (
    evaluate_model,
    save_metrics,
)
from models import build_model
from plots import generate_plots
from trainer import train_model
from utils import (
    create_output_dir,
    set_seed,
)


def run_experiment(
    dataset,
    model_name,
    bits,
    config_name,
    config,
):
    print(
        f"\n[{model_name.upper()}] "
        f"{bits} bits | {config_name}"
    )

    output_dir = create_output_dir(
        model_name=model_name,
        bits=bits,
        config_name=config_name,
    )

    model = build_model(
        model_name=model_name,
        bits=bits,
    )

    training = train_model(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
    )

    evaluation = evaluate_model(
        model=model,
        dataset=dataset,
    )

    save_metrics(
        evaluation,
        output_dir,
    )

    generate_plots(
        training,
        evaluation,
    )

    print(
        f"Accuracy: {evaluation.accuracy:.4f}"
    )


def main():

    set_seed()

    dataset = load_sat6()

    for model_name in MODELS:

        print(f"\n{'='*70}")
        print(f"Running {model_name.upper()}")
        print(f"{'='*70}")

        for bits in BIT_LIST:

            for config_name, config in TRAIN_CONFIGS.items():

                run_experiment(
                    dataset,
                    model_name,
                    bits,
                    config_name,
                    config,
                )


if __name__ == "__main__":
    main()