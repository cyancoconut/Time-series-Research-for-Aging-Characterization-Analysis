"""Train a RandomForest classifier to replace HDBSCAN + cluster_filter.

Reads every per-segment CSV in `<working_path>/with_features_post_labeled/`,
treats the `target` column as ground truth, runs leave-one-cell-out CV to
report per-class precision/recall, then refits on all data and writes:

    models/vtc_classifier.joblib       — the fitted estimator
    models/vtc_classifier_meta.json    — feature columns, classes, training cells

Usage (from src/):
    python -m cluster.train_classifier /path/to/battery_config.json
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FEATURE_COLS = [
    "Voltage_mean", "Voltage_std", "Voltage_max", "Voltage_min", "Voltage_range",
    "Current_mean", "Current_std", "Current_max", "Current_min", "Current_range",
    "Temperature_mean",
    "Duration_minutes", "Duration_quartile",
    "abs_Current_mean",
]

# Final-labeler classes only. Anything else (stringified cluster ints) → "-1".
KNOWN_LABELS = {"CAP", "PUL", "PUL*RES", "qOCV_DCH", "qOCV_CHA", "-1"}


def _load_cell_csvs(working_path: str) -> pd.DataFrame:
    pattern = os.path.join(working_path, "with_features_post_labeled", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No labeled CSVs under {pattern}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["cell"] = os.path.basename(f).replace(".csv", "")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["target"] = df["target"].astype(str)
    df.loc[~df["target"].isin(KNOWN_LABELS), "target"] = "-1"

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS)
    return df


def _new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _loco_cv(df: pd.DataFrame) -> None:
    cells = sorted(df["cell"].unique())
    y_true_all, y_pred_all = [], []
    for held in cells:
        train = df[df["cell"] != held]
        test = df[df["cell"] == held]
        if test.empty:
            continue
        model = _new_model()
        model.fit(train[FEATURE_COLS], train["target"])
        pred = model.predict(test[FEATURE_COLS])
        y_true_all.append(test["target"].to_numpy())
        y_pred_all.append(pred)
        acc = (pred == test["target"].to_numpy()).mean()
        logging.info(f"LOCO {held}: n={len(test)} acc={acc:.3f}")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    print("\n=== Leave-one-cell-out classification report ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))


def train(config_path: str, model_out: str, meta_out: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    working_path = cfg["working_path"]

    df = _load_cell_csvs(working_path)
    logging.info(f"Loaded {len(df)} segments from {df['cell'].nunique()} cells")
    logging.info(f"Class counts:\n{df['target'].value_counts().to_string()}")

    _loco_cv(df)

    model = _new_model()
    model.fit(df[FEATURE_COLS], df["target"])

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)

    meta = {
        "feature_columns": FEATURE_COLS,
        "classes": sorted(df["target"].unique().tolist()),
        "training_cells": sorted(df["cell"].unique().tolist()),
        "n_segments": int(len(df)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "estimator": "RandomForestClassifier",
        "estimator_params": model.get_params(),
    }
    # joblib estimator_params may contain non-JSON values; coerce
    meta["estimator_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                 for k, v in meta["estimator_params"].items()}
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Wrote model -> {model_out}")
    logging.info(f"Wrote meta  -> {meta_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segment classifier")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--model-out", default="../models/vtc_classifier.joblib")
    parser.add_argument("--meta-out", default="../models/vtc_classifier_meta.json")
    args = parser.parse_args()

    train(args.config, args.model_out, args.meta_out)
