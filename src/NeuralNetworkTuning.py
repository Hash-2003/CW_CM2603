import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_nn_preprocessor,
    prepare_nn_data,
)

SEED = 2025

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(input_dim, units1, units2, lr, dropout_rate, l2_reg):
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units2, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tune_neural_network():

    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_nn_preprocessor(X_train)
    X_train_nn, X_test_nn = prepare_nn_data(preprocessor, X_train, X_test)

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_nn, y_train, test_size=0.2, random_state=2025, stratify=y_train
    )

    architectures = [(32, 16), (64, 32)]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    dropout_rates = [0.0, 0.1]       # mild dropout
    l2_regs = [0.0, 1e-4]            # mild L2 regularisation

    best_f1 = -1
    best_config = None
    best_model = None

    print("===== Neural Network Hyperparameter Tuning =====\n")

    for units1, units2 in architectures:
        for lr in learning_rates:
            for batch in batch_sizes:
                for dr in dropout_rates:
                    for l2r in l2_regs:

                        print(f"\nTesting config:")
                        print(f"   units=({units1},{units2}), lr={lr}, batch={batch}, dropout={dr}, l2={l2r}")

                        model = build_model(
                            input_dim=X_train_nn.shape[1],
                            units1=units1,
                            units2=units2,
                            lr=lr,
                            dropout_rate=dr,
                            l2_reg=l2r
                        )

                        callbacks = [
                            keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=2,
                                restore_best_weights=True
                            )
                        ]

                        model.fit(
                            X_train_sub, y_train_sub,
                            validation_data=(X_val, y_val),
                            epochs=15,
                            batch_size=batch,
                            callbacks=callbacks,
                            verbose=0
                        )

                        y_pred_val = (model.predict(X_val).ravel() >= 0.5).astype(int)
                        f1 = f1_score(y_val, y_pred_val)

                        print(f"   F1-score: {f1:.4f}")

                        if f1 > best_f1:
                            best_f1 = f1
                            best_model = model
                            best_config = {
                                "units1": units1,
                                "units2": units2,
                                "learning_rate": lr,
                                "batch_size": batch,
                                "dropout_rate": dr,
                                "l2_reg": l2r,
                            }

    print("\n===== Best Configuration Found =====")
    print(best_config)
    print("Best F1-score:", best_f1)

    return best_model, best_config


if __name__ == "__main__":
    best_model, best_params = tune_neural_network()
    print("\nBest parameters (for final NN model):", best_params)
