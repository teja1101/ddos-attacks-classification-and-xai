from __future__ import annotations

import csv
import os
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from werkzeug.utils import secure_filename

from model_export import save_model_artifacts


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"MODEL_PATH does not exist: {env_path}")

    default_candidates = [Path("ddos_model.joblib"), Path("ddos_model.pkl")]
    for path in default_candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No model file found. Expected ddos_model.joblib or ddos_model.pkl in current directory."
    )


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

TARGET_DEFAULT = "PKT_CLASS"
FEATURE_DESCRIPTIONS = {
    "SRC_ADD": "Source address",
    "DES_ADD": "Destination address",
    "PKT_ID": "Packet identifier",
    "FROM_NODE": "Source node ID",
    "TO_NODE": "Destination node ID",
    "PKT_TYPE": "Packet type",
    "PKT_SIZE": "Packet size (bytes)",
    "FLAGS": "Packet/TCP flags",
    "FID": "Flow identifier",
    "SEQ_NUMBER": "Sequence number",
    "NUMBER_OF_PKT": "Number of packets in flow",
    "NUMBER_OF_BYTE": "Number of bytes in flow",
    "NODE_NAME_FROM": "Source node name",
    "NODE_NAME_TO": "Destination node name",
    "PKT_IN": "Incoming packets",
    "PKT_OUT": "Outgoing packets",
    "PKT_R": "Packet receive metric",
    "PKT_DELAY_NODE": "Node processing delay",
    "PKT_RATE": "Packet rate",
    "BYTE_RATE": "Byte rate",
    "PKT_AVG_SIZE": "Average packet size",
    "UTILIZATION": "Link utilization",
    "PKT_DELAY": "End-to-end packet delay",
    "PKT_SEND_TIME": "Packet send timestamp",
    "PKT_RESEVED_TIME": "Packet received timestamp",
    "FIRST_PKT_SENT": "First packet sent time",
    "LAST_PKT_RESEVED": "Last packet received time",
}

MODEL = None
MODEL_PATH = None
MODEL_ERROR = None
try:
    MODEL_PATH = _resolve_model_path()
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL_ERROR = str(exc)


def _predict_rows(X: np.ndarray):
    if MODEL is None:
        raise RuntimeError(f"Model not loaded: {MODEL_ERROR}")
    preds = MODEL.predict(X)
    probs = None
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)
    return preds, probs


def _expected_feature_names() -> list[str]:
    if MODEL is None:
        return []
    if hasattr(MODEL, "feature_names_in_"):
        return list(MODEL.feature_names_in_)
    preprocess = getattr(getattr(MODEL, "named_steps", {}), "get", lambda _: None)("preprocess")
    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
        return list(preprocess.feature_names_in_)
    return []


def _coerce_value(value: str):
    val = value.strip()
    if val == "" or val.lower() in {"null", "none", "nan"}:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return val


def _prepare_single_input(raw_input: str):
    values = next(csv.reader([raw_input], skipinitialspace=True), [])
    values = [_coerce_value(v) for v in values if v.strip() != ""]
    if not values:
        raise ValueError("Please provide at least one feature value.")

    feature_names = _expected_feature_names()
    if feature_names:
        if len(values) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: model expects {len(feature_names)} values, received {len(values)}."
            )
        return pd.DataFrame([values], columns=feature_names)
    return np.array(values, dtype=object).reshape(1, -1)


def _prepare_named_form_input(form_data, feature_names: list[str]):
    if not feature_names:
        raise ValueError("Model feature names are unavailable.")
    values = [_coerce_value(form_data.get(f"f__{name}", "")) for name in feature_names]
    non_empty_count = sum(
        1 for name in feature_names if str(form_data.get(f"f__{name}", "")).strip() != ""
    )
    if non_empty_count == 0:
        raise ValueError("Please enter feature values in either comma-separated input or named fields.")
    return pd.DataFrame([values], columns=feature_names)


def _read_example_feature_row() -> str | None:
    data_path = Path("final-dataset.arff")
    if not data_path.exists():
        return None

    in_data = False
    with data_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if s.lower() == "@data":
                    in_data = True
                continue
            if not s or s.startswith("%"):
                continue
            row = next(csv.reader([s], skipinitialspace=True), [])
            if len(row) < 2:
                continue
            return ",".join(row[:-1])
    return None


def _read_example_rows_by_class() -> dict[str, str]:
    data_path = Path("final-dataset.arff")
    if not data_path.exists():
        return {}

    collected: dict[str, str] = {}
    fallback: dict[str, str] = {}
    feature_names = _expected_feature_names()
    target_classes = {str(c) for c in getattr(MODEL, "classes_", [])} if MODEL is not None else set()
    in_data = False
    with data_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if s.lower() == "@data":
                    in_data = True
                continue
            if not s or s.startswith("%"):
                continue

            row = next(csv.reader([s], skipinitialspace=True), [])
            if len(row) < 2:
                continue
            cls = str(row[-1]).strip()
            if target_classes and cls not in target_classes:
                continue
            row_values = row[:-1]
            row_string = ",".join(row_values)
            if cls not in fallback:
                fallback[cls] = row_string

            if MODEL is not None and feature_names and cls not in collected and len(row_values) == len(feature_names):
                try:
                    sample_df = pd.DataFrame([row_values], columns=feature_names)
                    pred = str(MODEL.predict(sample_df)[0])
                    if pred == cls:
                        collected[cls] = row_string
                except Exception:
                    pass

            if target_classes and len(collected) == len(target_classes):
                break
            if not target_classes and len(collected) >= 8:
                break
    if target_classes:
        for cls in target_classes:
            if cls not in collected and cls in fallback:
                collected[cls] = fallback[cls]
    return collected or fallback


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
    return df


def _load_uploaded_dataset(file_obj, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_obj)
    if ext == ".arff":
        raw_data, _ = arff.loadarff(file_obj)
        return _decode_bytes_columns(pd.DataFrame(raw_data))
    raise ValueError("Unsupported file type. Upload .csv or .arff")


def _train_pipeline(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    data = df.dropna(subset=[target_col]).copy()
    if data.empty:
        raise ValueError("Dataset is empty after removing rows with missing target.")

    X = data.drop(columns=[target_col])
    y = data[target_col].astype(str)

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

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", SGDClassifier(loss="log_loss", random_state=42, max_iter=1500, tol=1e-3)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    accuracy = float(pipeline.score(X_test, y_test))
    return pipeline, accuracy, len(X), len(X.columns)


def _reload_model_from_disk():
    global MODEL, MODEL_PATH, MODEL_ERROR
    MODEL_PATH = _resolve_model_path()
    MODEL = joblib.load(MODEL_PATH)
    MODEL_ERROR = None


@app.route("/", methods=["GET", "POST"])
def predict():
    result = None
    confidence = None
    class_probs = None
    error = MODEL_ERROR
    raw_input = ""
    feature_names = _expected_feature_names()
    field_values = {name: "" for name in feature_names}
    selected_mode = ""

    if request.method == "POST":
        selected_mode = request.form.get("input_mode", "").strip()
        raw_input = request.form.get("features", "").strip()
        field_values = {name: request.form.get(f"f__{name}", "") for name in feature_names}
        try:
            if selected_mode == "csv":
                X = _prepare_single_input(raw_input)
            elif selected_mode == "form":
                X = _prepare_named_form_input(request.form, feature_names)
            elif raw_input:
                X = _prepare_single_input(raw_input)
            else:
                X = _prepare_named_form_input(request.form, feature_names)
            pred, probs = _predict_rows(X)
            pred = pred[0]
            result = int(pred) if isinstance(pred, (np.integer, int)) else pred

            if probs is not None:
                confidence = float(np.max(probs[0]))
                classes = getattr(MODEL, "classes_", [])
                if len(classes) == len(probs[0]):
                    class_probs = sorted(
                        [(str(cls), float(score)) for cls, score in zip(classes, probs[0])],
                        key=lambda item: item[1],
                        reverse=True,
                    )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        model_path=str(MODEL_PATH) if MODEL_PATH else "Not loaded",
        result=result,
        confidence=confidence,
        class_probs=class_probs,
        error=error,
        raw_input=raw_input,
        expected_features=len(feature_names) or "Unknown",
        feature_names=feature_names,
        field_values=field_values,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        selected_mode=selected_mode,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    error = MODEL_ERROR
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            error = "Please upload a CSV file."
        else:
            try:
                df = pd.read_csv(file)
                feature_names = _expected_feature_names()
                X = df
                if feature_names:
                    missing = [name for name in feature_names if name not in df.columns]
                    if missing:
                        raise ValueError(
                            f"CSV is missing required columns: {', '.join(missing[:6])}"
                            + ("..." if len(missing) > 6 else "")
                        )
                    X = df[feature_names].copy()

                preds, probs = _predict_rows(X)
                out = df.copy()
                out["prediction"] = preds
                if probs is not None:
                    out["confidence"] = np.max(probs, axis=1)

                csv_buffer = StringIO()
                out.to_csv(csv_buffer, index=False)
                return Response(
                    csv_buffer.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"},
                )
            except Exception as exc:
                error = str(exc)

    return render_template(
        "batch.html",
        model_path=str(MODEL_PATH) if MODEL_PATH else "Not loaded",
        error=error,
        expected_features=len(_expected_feature_names()) or "Unknown",
        feature_names=_expected_feature_names(),
        feature_descriptions=FEATURE_DESCRIPTIONS,
    )


@app.route("/model-info")
def model_info():
    model_type = type(MODEL).__name__ if MODEL is not None else "Not loaded"
    n_features = getattr(MODEL, "n_features_in_", "Unknown") if MODEL is not None else "Unknown"
    classes = getattr(MODEL, "classes_", "Unknown") if MODEL is not None else "Unknown"
    has_proba = hasattr(MODEL, "predict_proba") if MODEL is not None else False
    return render_template(
        "model_info.html",
        model_path=str(MODEL_PATH) if MODEL_PATH else "Not loaded",
        model_type=model_type,
        n_features=n_features,
        classes=classes,
        has_proba=has_proba,
        model_error=MODEL_ERROR,
        feature_names=_expected_feature_names(),
        feature_descriptions=FEATURE_DESCRIPTIONS,
    )


@app.route("/help")
def help_page():
    return render_template(
        "help.html",
        model_path=str(MODEL_PATH) if MODEL_PATH else "Not loaded",
        model_error=MODEL_ERROR,
    )


@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    message = None
    error = None
    metrics = None
    target_col = TARGET_DEFAULT

    if request.method == "POST":
        target_col = request.form.get("target_col", TARGET_DEFAULT).strip() or TARGET_DEFAULT
        file = request.files.get("dataset")
        if not file or not file.filename:
            error = "Please upload a dataset file."
        else:
            safe_name = secure_filename(file.filename)
            try:
                df = _load_uploaded_dataset(file, safe_name)
                trained_model, accuracy, row_count, feature_count = _train_pipeline(df, target_col)
                save_model_artifacts(trained_model, output_dir=".", base_name="ddos_model")
                _reload_model_from_disk()
                message = "Model retrained and saved successfully."
                metrics = {
                    "accuracy": accuracy,
                    "rows": row_count,
                    "feature_count": feature_count,
                    "target_col": target_col,
                }
            except (ValueError, KeyError) as exc:
                error = str(exc)

    return render_template(
        "retrain.html",
        model_path=str(MODEL_PATH) if MODEL_PATH else "Not loaded",
        error=error,
        message=message,
        metrics=metrics,
        target_col=target_col,
        model_error=MODEL_ERROR,
    )


@app.route("/example-row")
def example_row():
    options = _read_example_rows_by_class()
    row = next(iter(options.values()), None) if options else _read_example_feature_row()
    if not row:
        return jsonify({"error": "No example row available from dataset."}), 404
    values = next(csv.reader([row], skipinitialspace=True), [])
    return jsonify({"features": row, "count": len(values)})


@app.route("/example-options")
def example_options():
    options = _read_example_rows_by_class()
    if not options:
        return jsonify({"error": "No class-wise sample rows available."}), 404
    return jsonify({"options": options})


@app.route("/batch-template")
def batch_template():
    feature_names = _expected_feature_names()
    if not feature_names:
        return Response("Model feature names unavailable.", status=400, mimetype="text/plain")

    template_df = pd.DataFrame(columns=feature_names)
    sample_row = _read_example_feature_row()
    if sample_row:
        values = next(csv.reader([sample_row], skipinitialspace=True), [])
        if len(values) == len(feature_names):
            template_df.loc[0] = values

    csv_buffer = StringIO()
    template_df.to_csv(csv_buffer, index=False)
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_template.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
