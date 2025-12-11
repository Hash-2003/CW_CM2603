import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#load original csv
def load_and_prepare_data(csv_path: str = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(csv_path)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Split features
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})  # convert to numeric

    return X, y

#train and test split
def split_data(X, y, test_size: float = 0.2, random_state: int = 2025):
    """
    Train/test split to preserve churn imbalance.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def build_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    preprocessing pipeline for the Decision Tree model.
    """
    numeric = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return preprocessor


def build_nn_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    preprocessing pipeline for the Neural Network model.
    """
    numeric = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return preprocessor

def prepare_nn_data(preprocessor: ColumnTransformer, X_train, X_test):
    """
    Fit the preprocessor on training data and transform both train and test

    """
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    if hasattr(X_train_trans, "toarray"):
        X_train_nn = X_train_trans.toarray()
        X_test_nn = X_test_trans.toarray()
    else:
        X_train_nn = np.asarray(X_train_trans)
        X_test_nn = np.asarray(X_test_trans)

    return X_train_nn, X_test_nn