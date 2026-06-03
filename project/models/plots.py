import matplotlib.pyplot as plt
import tensorflow as tf


def plot_training(history, model_name,dir_name):

    # LOSS
    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(['train', 'validation'])

    plt.grid(True)

    plt.savefig(f'{dir_name}/{model_name}_loss.png')

    plt.close()

     # ACCURACY
    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend(['train', 'validation'])

    plt.grid(True)

    plt.savefig(f'{dir_name}/{model_name}_accuracy.png')

    plt.close()


def plot_model(model, path_file, show_shapes=True, show_dtype=True, show_layer_names=True):
    tf.keras.utils.plot_model(
        model,
        to_file=path_file,
        show_shapes=show_shapes,
        show_dtype=show_dtype,
        show_layer_names=show_layer_names,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=False,
        show_trainable=False,
    )