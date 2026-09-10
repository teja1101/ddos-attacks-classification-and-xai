from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_export import save_model_artifacts


DATA_PATH = Path("final-dataset.arff")
TARGET_COL = "PKT_CLASS"
RANDOM_STATE = 42


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
    return df


def train_and_export() -> dict[str, str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw_data, _ = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(raw_data)
    df = _decode_bytes_columns(df)
    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    random_state=RANDOM_STATE,
                    max_iter=1500,
                    tol=1e-3,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    paths = save_model_artifacts(model, output_dir=".", base_name="ddos_model")
    print(f"Saved model artifacts: {paths}")
    return paths


if __name__ == "__main__":
    train_and_export()
