import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_nn_preprocessor,
    prepare_nn_data,
)


def build_model(input_dim, units1, units2, lr):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu"),
        keras.layers.Dense(units2, activation="relu"),
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

    # Validation split for tuning
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_nn, y_train, test_size=0.2, random_state=2025, stratify=y_train
    )

    # Hyperparameter search space
    architectures = [(32, 16), (64, 32), (128, 64)]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]

    best_f1 = -1
    best_config = None
    best_model = None

    print("Starting Neural Network hyperparameter tuning...\n")

    for units1, units2 in architectures:
        for lr in learning_rates:
            for batch in batch_sizes:

                print(f"Testing config: units=({units1},{units2}), lr={lr}, batch={batch}")

                model = build_model(
                    input_dim=X_train_nn.shape[1],
                    units1=units1,
                    units2=units2,
                    lr=lr
                )

                model.fit(
                    X_train_sub, y_train_sub,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=batch,
                    verbose=0
                )

                y_pred_val = (model.predict(X_val).ravel() >= 0.5).astype(int)
                f1 = f1_score(y_val, y_pred_val)

                print("F1 on validation:", f1)

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {
                        "units1": units1,
                        "units2": units2,
                        "learning_rate": lr,
                        "batch_size": batch,
                    }
                    best_model = model

    print("\nBest Neural Network configuration:", best_config)
    print("Best F1-score:", best_f1)

    return best_model, best_config

if __name__ == "__main__":
    best_model, best_params = tune_neural_network()
    print("\nBest parameters (for final NN model):", best_params)

