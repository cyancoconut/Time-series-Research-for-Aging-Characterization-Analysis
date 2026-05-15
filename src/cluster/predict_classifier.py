"""Inference replacement for HDBSCAN + cluster_filter.

Given a trained RandomForest (see train_classifier.py) and the per-segment
feature table produced by feature_extraction.create_features, predict the
segment label and translate it back to the cluster-tagged form the
downstream calculation step expects.
"""

import json
import logging
import os

import joblib
import pandas as pd

# Map classifier's final labels -> cluster-tagged labels the calculate/ step
# matches on. update_capacity refines CAP* -> CAP, update_pulse refines PUL*
# (including PUL*RES re-identified by _filter_pulse_group), update_qOCV
# refines QOCV* -> qOCV_DCH / qOCV_CHA based on sign.
_LABEL_TO_TAGGED = {
    "CAP": "CAP*",
    "PUL": "PUL*",
    "PUL*RES": "PUL*",
    "qOCV_DCH": "QOCV*",
    "qOCV_CHA": "QOCV*",
    "-1": "-1",
}


def _load(model_path: str, meta_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Classifier metadata not found: {meta_path}")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def predict_targets(X_features: pd.DataFrame, model_path: str, meta_path: str) -> pd.DataFrame:
    """Replace the `target` column in X_features with classifier predictions
    (translated to cluster-tagged form). Returns a new DataFrame; input is not
    modified."""
    model, meta = _load(model_path, meta_path)
    feature_cols = meta["feature_columns"]

    missing = set(feature_cols) - set(X_features.columns)
    if missing:
        raise ValueError(f"X_features missing required columns: {missing}")

    X_out = X_features.copy()
    X = X_out[feature_cols].copy()

    # RandomForest tolerates NaN poorly; rows with NaN features are flagged -1.
    na_mask = X.isna().any(axis=1)
    preds = pd.Series(index=X_out.index, dtype=object)
    if na_mask.any():
        preds.loc[na_mask] = "-1"
        logging.info(f"classifier: {int(na_mask.sum())} rows with NaN features → '-1'")
    if (~na_mask).any():
        preds.loc[~na_mask] = model.predict(X.loc[~na_mask])

    X_out["target"] = preds.map(_LABEL_TO_TAGGED).fillna("-1").astype(object)
    return X_out
