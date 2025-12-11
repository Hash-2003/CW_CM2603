from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_nn_preprocessor,
    prepare_nn_data,
)

def build_nn_model(
    input_dim: int,
    hidden_units1: int = 32,
    hidden_units2: int = 16,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build a feed-forward neural network (MLP) for binary churn prediction.

    Architecture:
        Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)

    Parameters
    ----------
    input_dim : int
        Number of input features after preprocessing.
    hidden_units1 : int
        Number of units in the first hidden layer.
    hidden_units2 : int
        Number of units in the second hidden layer.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    model : keras.Model
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units1, activation="relu"),
            layers.Dense(hidden_units2, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # binary output
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_nn_experiment(
    epochs: int = 20,
    batch_size: int = 32,
    debug: bool = False,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, keras.Model]:
    """
    Full workflow for the Neural Network model:

      1. Load and clean data (from preprocess.py)
      2. Split into train/test
      3. Build NN preprocessing and transform to dense arrays
      4. Build and train the NN model
      5. Predict probabilities and binary labels on the test set
      6. (Optionally) print evaluation metrics for debugging

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for training.
    debug : bool
        If True, prints evaluation metrics using evaluation.py.
        If False, returns results silently (for evaluation.py to handle).
    threshold : float
        Classification threshold for converting probabilities to 0/1.

    Returns
    -------
    y_test : np.ndarray
        True labels for the test set.
    y_pred_binary : np.ndarray
        Predicted binary labels (0/1) for the test set.
    y_prob : np.ndarray
        Predicted probabilities (between 0 and 1) for the positive class.
    model : keras.Model
        Trained Keras model (can be reused or saved).
    """
    print("Loading and preparing data (NN)...")
    X, y = load_and_prepare_data()

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building NN preprocessing pipeline...")
    preprocessor = build_nn_preprocessor(X_train)

    print("Transforming data for NN (to dense arrays)...")
    X_train_nn, X_test_nn = prepare_nn_data(preprocessor, X_train, X_test)

    input_dim = X_train_nn.shape[1]
    print(f"Input dimension after preprocessing: {input_dim}")

    print("Building Neural Network model...")
    model = build_nn_model(input_dim=input_dim)

    print("Training Neural Network model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_nn,
        y_train,
        validation_data=(X_test_nn, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Generating predictions on test set...")
    y_prob = model.predict(X_test_nn).ravel()
    y_pred_binary = (y_prob >= threshold).astype(int)

    if debug:
        acc = accuracy_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary, zero_division=0)
        rec = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)

        print("=== Neural Network (debug evaluation) ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

    # return values so evaluation.py can use them later
    return y_test.to_numpy(), y_pred_binary, y_prob, model

if __name__ == "__main__":
    # Debug run for you during development. Safe to ignore/remove later.
    run_nn_experiment(debug=True)