import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical

def load_sat6(mat_path):

    data = scipy.io.loadmat(mat_path)

    x_train = data['train_x']
    y_train = data['train_y']

    x_test = data['test_x']
    y_test = data['test_y']

    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = np.argmax(y_train, axis=0).reshape(-1).astype(np.int32)
    y_test = np.argmax(y_test, axis=0).reshape(-1).astype(np.int32)

    return (
        x_train,
        y_train,
        x_test,
        y_test
    )


def train_model(model, x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
        ]
    )

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    return history


def evaluate_model(model, x_test, y_test, model_name, output_dir):

    predictions = model.predict(x_test)

    preds = np.argmax(predictions, axis=1)

    print('\nClassification Report\n')

    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(f'{model_name} - Confusion Matrix')

    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png')

    plt.close()