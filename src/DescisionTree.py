from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_tree_preprocessor,
)


def build_decision_tree_model(
    X_train: pd.DataFrame,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_tree_preprocessor(X_train)

    tree_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("tree", tree_clf),
        ]
    )

    return model


def run_tree_experiment(
    max_depth: int | None = None,
    min_samples_split: int = 2,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:

    print("Loading and preparing data...")
    X, y = load_and_prepare_data()

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building Decision Tree model...")
    model = build_decision_tree_model(
        X_train=X_train,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    if debug:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("=== Decision Tree (debug evaluation) ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

    return y_test.to_numpy(), y_pred, y_prob, model


if __name__ == "__main__":
    # Debug run
    run_tree_experiment(debug=True)
