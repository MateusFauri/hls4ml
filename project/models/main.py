import os
import numpy as np
import tensorflow as tf

from plots import plot_training, plot_model
from models import (
    build_t1_float,
    build_t2_float,
)
from train import (
    load_sat6,
    train_model,
    evaluate_model
)

DATASET_PATH = 'data/sat-6-full.mat'
OUTPUT_DIR = 'models'
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 3e-4


def save_model(model, model_name):

    keras_path = f'{OUTPUT_DIR}/keras/{model_name}.keras'


    json_path = f'{OUTPUT_DIR}/keras/{model_name}.json'
    weights_path = f'{OUTPUT_DIR}/keras/{model_name}_weights.h5'

    with open(json_path, 'w') as json_file:
        json_file.write(model.to_json())

    model.save_weights(weights_path)
    model.save(keras_path)

    print(f'\nModel architecture saved: {json_path}')
    print(f'Model weights saved: {weights_path}')
    print(f'\nModel saved: {keras_path}')


def main():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x_train, y_train, x_test, y_test = load_sat6(
        DATASET_PATH
    )

    model = build_t1_float()

    history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        LEARNING_RATE,
        BATCH_SIZE,
        EPOCHS
    )

    plot_training(
        history,
        model.name,
        dir_name = 'images'
    )

    plot_model(
        model,
        f'images/{model.name}.png'
    )

    evaluate_model(
        model,
        x_test,
        y_test,
        model.name,
        'images'
    )

    save_model(
        model,
        model.name,
    )

if __name__ == "__main__":
    main()