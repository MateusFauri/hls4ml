import os
import gc
import itertools
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from qkeras import *
from qkeras import quantizers
from qkeras.utils import _add_supported_quantized_objects

co = {}

_add_supported_quantized_objects(co)

# =====================================================
# CONFIGURAÇÕES
# =====================================================

BIT_LIST =  [8,4,2,1]
LEARNING_RATES = [3e-4, 1e-4, 1e-3]
BATCH_SIZES = [256, 512, 1024]
EPOCHS_LIST = [100]

RESULTS_DIR = "results_sem_batchnorm"
EARLY_STOPPING_PATIENCE = 20

os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================
# DATASET
# =====================================================

print("Loading SAT-6...")

data = sio.loadmat('../data/sat-6-full.mat')


X_train = data['train_x']
y_train = data['train_y']

X_test = data['test_x']
y_test = data['test_y']

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

X_train = np.transpose(X_train, (3, 0, 1, 2))
X_test = np.transpose(X_test, (3, 0, 1, 2))

y_train = np.transpose(y_train, (1, 0))
y_test = np.transpose(y_test, (1, 0))


X_test_final = X_test
y_test_final = y_test


X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(y_train, axis=1)
)

y_val_labels = np.argmax(y_val, axis=1)
y_test_labels = np.argmax(y_test_final, axis=1)

# =====================================================
# QUANTIZERS
# =====================================================

def get_quantizers(qbits):

    if qbits == 1:

        quant_w = quantizers.binary(alpha=1)
        quant_a = quantizers.binary(alpha=1)

    else:

        quant_w = quantizers.quantized_bits(
            qbits,
            0,
            alpha=1
        )

        quant_a = quantizers.quantized_bits(
            qbits,
            0,
            alpha=1
        )

    return quant_w, quant_a


# =====================================================
# MODEL
# =====================================================

def build_qmodel(qbits):

    quant_w, quant_a = get_quantizers(qbits)

    model = models.Sequential()

    model.add(
        layers.Input(shape=(28, 28, 4))
    )

    model.add(
        QConv2D(
            32,
            (3, 3),
            strides=2,
            padding="same",
            kernel_quantizer=quant_w,
            bias_quantizer=quant_w
        )
    )

    model.add(layers.BatchNormalization())
    model.add(QActivation(quant_a))

    model.add(
        QConv2D(
            64,
            (1, 1),
            padding="valid",
            kernel_quantizer=quant_w,
            bias_quantizer=quant_w
        )
    )

    model.add(layers.BatchNormalization())
    model.add(QActivation(quant_a))

    model.add(layers.Flatten())

    model.add(
        QDense(
            128,
            kernel_quantizer=quant_w,
            bias_quantizer=quant_w
        )
    )

    model.add(layers.BatchNormalization())
    model.add(QActivation(quant_a))

    model.add(
        QDense(
            6,
            kernel_quantizer=quant_w,
            bias_quantizer=quant_w,
            activation="softmax"
        )
    )

    return model


# =====================================================
# PLOTS
# =====================================================


def plot_training(history, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # Accuracy
    ax[0].plot(epochs, history.history["accuracy"], label="Train")
    ax[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(True)
    ax[0].legend()

    # Loss
    ax[1].plot(epochs, history.history["loss"], label="Train")
    ax[1].plot(epochs, history.history["val_loss"], label="Validation")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# TREINO
# =====================================================

all_results = []
best_models = {}

for bits in BIT_LIST:

    print("\n" + "=" * 80)
    print(f"Testing {bits} bits")
    print("=" * 80)

    bit_dir = os.path.join(RESULTS_DIR, f"{bits}bits")
    model_dir = os.path.join(bit_dir, "models")
    plot_dir = os.path.join(bit_dir, "plots")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    best_acc = -np.inf
    best_info = None

    experiments = list(
        itertools.product(
            LEARNING_RATES,
            BATCH_SIZES,
            EPOCHS_LIST
        )
    )

    total = len(experiments)

    for idx, (lr, batch_size, epochs) in enumerate(experiments, start=1):

        print(
            f"\n[{idx}/{total}] "
            f"Bits={bits} | "
            f"LR={lr:.0e} | "
            f"Batch={batch_size} | "
            f"Epochs={epochs}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

        model = build_qmodel(bits)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        experiment_name = (
            f"lr{lr:.0e}_"
            f"bs{batch_size}_"
            f"ep{epochs}"
        )

        checkpoint_path = os.path.join(
            model_dir,
            experiment_name + ".h5"
        )

        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=0
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                checkpoint,
                early_stop
            ],
            verbose=1
        )

        val_loss, val_acc = model.evaluate(
            X_val,
            y_val,
            verbose=0
        )

        train_acc = history.history["accuracy"][-1]
        train_loss = history.history["loss"][-1]

        best_epoch = (
            np.argmax(
                history.history["val_accuracy"]
            ) + 1
        )

        generalization_gap = (
            train_acc - val_acc
        )

        plot_training(
            history,
            os.path.join(
                plot_dir,
                experiment_name + ".png"
            )
        )

        all_results.append({
            "bits": bits,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "generalization_gap": generalization_gap
        })

        print(
            f"Train Acc : {train_acc:.4f}"
        )

        print(
            f"Val Acc   : {val_acc:.4f}"
        )

        if val_acc > best_acc:
            print(">>> Novo melhor modelo!")

            best_acc = val_acc
            best_model_path = os.path.join(
                model_dir,
                "best_model.h5"
            )

            shutil.copy2(
                checkpoint_path,
                best_model_path
            )

            model_size = (
                os.path.getsize(best_model_path)
                / (1024 ** 2)
            )

            best_info = {
                "path": best_model_path,
                "bits": bits,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "best_epoch": best_epoch,
                "generalization_gap": generalization_gap,
                "model_size_mb": model_size
            }

        del history
        del model

        tf.keras.backend.clear_session()
        gc.collect()

    best_models[bits] = best_info

    print("\n" + "-" * 80)
    print(f"Melhor configuração ({bits} bits)")
    print("-" * 80)
    print(f"Learning Rate : {best_info['lr']}")
    print(f"Batch Size    : {best_info['batch_size']}")
    print(f"Epochs        : {best_info['epochs']}")
    print(f"Val Accuracy  : {best_info['val_accuracy']:.4f}")
    print(f"Val Loss      : {best_info['val_loss']:.4f}")
    print(f"Best Epoch    : {best_info['best_epoch']}")
    print(f"Gap           : {best_info['generalization_gap']:.4f}")
    print(f"Model Size    : {best_info['model_size_mb']:.2f} MB")


# =====================================================
# AVALIAÇÃO DOS MELHORES MODELOS
# =====================================================
for bits in BIT_LIST:

    print("\n" + "="*60)
    print(f"BEST MODEL - {bits} BITS")
    print("="*60)

    info = best_models[bits]

    model = tf.keras.models.load_model(
        info["path"],
        custom_objects=co
    )

    y_pred = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(
        y_test_labels,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test_labels,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test_labels,
        y_pred,
        average="macro"
    )

    print(f"Learning Rate : {info['lr']}")
    print(f"Batch Size    : {info['batch_size']}")
    print(f"Epochs        : {info['epochs']}")

    print(f"Validation Acc: {info['val_accuracy']:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1            : {f1:.4f}")

# =====================================================
# RESULTS DATAFRAME
# =====================================================

results_df = pd.DataFrame(all_results)

ranking = results_df.sort_values(
    "val_accuracy",
    ascending=False
)

ranking.to_csv(
    os.path.join(
        RESULTS_DIR,
        "ranking.csv"
    ),
    index=False
)

# =====================================================
# PLOTS POR QUANTIZAÇÃO
# =====================================================

for bits in BIT_LIST:

    subset = (
        results_df[results_df.bits == bits]
        .copy()
        .sort_values("val_accuracy", ascending=False)
    )

    plot_dir = os.path.join(
        RESULTS_DIR,
        f"{bits}bits",
        "plots"
    )

    subset["experiment"] = (
        "LR=" + subset.lr.astype(str)
        + "\nBS=" + subset.batch_size.astype(str)
        + "\nEP=" + subset.epochs.astype(str)
    )

    ####################################################
    # Ranking
    ####################################################

    plt.figure(figsize=(12,7))

    plt.barh(
        subset["experiment"],
        subset["val_accuracy"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Validation Accuracy")
    plt.title(f"Ranking - {bits} bits")

    plt.grid(axis="x")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "ranking.png"
        ),
        dpi=300
    )

    plt.close()



    ####################################################
    # Accuracy x Batch
    ####################################################

    plt.figure(figsize=(8,6))

    for lr in sorted(subset.lr.unique()):

        tmp = subset[
            subset.lr == lr
        ].sort_values("batch_size")

        plt.plot(
            tmp.batch_size,
            tmp.val_accuracy,
            marker="o",
            linewidth=2,
            label=f"LR={lr:.0e}"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Validation Accuracy")

    plt.title(
        f"Accuracy x Batch Size ({bits} bits)"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            "accuracy_vs_batch.png"
        ),
        dpi=300
    )

    plt.close()


# =====================================================
# HEATMAPS
# =====================================================

for bits in BIT_LIST:

    subset = results_df[
        results_df.bits == bits
    ]

    pivot = subset.pivot_table(
        values='val_accuracy',
        index='batch_size',
        columns='lr',
        aggfunc='max'
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f"
    )

    plt.title(f"Validation Accuracy ({bits} bits)")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{bits}bits",
            "plots",
            "heatmap.png"
        )
    )

    plt.close()


# =====================================================
# TESTE FINAL
# =====================================================

final_results = []

for bits in BIT_LIST:

    print(f"\nFinal test for {bits} bits")

    model = tf.keras.models.load_model(
        best_models[bits]["path"],
        custom_objects=co
    )

    loss, acc = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Accuracy: {acc:.4f}")

    test_loss, test_acc = model.evaluate(
        X_test_final,
        y_test_final,
        verbose=0
    )

    y_pred = model.predict(
        X_test_final,
        verbose=0
    )

    y_pred_labels = np.argmax(
        y_pred,
        axis=1
    )

    precision = precision_score(
        y_test_labels,
        y_pred_labels,
        average="macro"
    )

    recall = recall_score(
        y_test_labels,
        y_pred_labels,
        average="macro"
    )

    f1 = f1_score(
        y_test_labels,
        y_pred_labels,
        average="macro"
    )

    print(
        f"Accuracy: {test_acc:.4f} | "
        f"F1: {f1:.4f}"
    )

    final_results.append({
        "bits": bits,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

pd.DataFrame(final_results).to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_test_results.csv"
    ),
    index=False
)