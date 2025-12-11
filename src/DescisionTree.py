from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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
) -> Tuple:

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
