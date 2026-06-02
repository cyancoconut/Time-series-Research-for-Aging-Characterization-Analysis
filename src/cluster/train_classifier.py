"""Train a RandomForest classifier to replace HDBSCAN + cluster_filter.

Reads every per-segment CSV in `<working_path>/with_features_post_labeled/`
(produced by the HDBSCAN pipeline, post-#37 so it carries *every* segment incl.
the prep / SOC-adjust leftovers and a `cluster_id` column), treats the `target`
column as ground truth, weak-labels the obvious leftover families (PREP_CHA /
PREP_DCH), runs leave-one-cell-out CV to report per-class precision/recall, then
refits on all data and writes:

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

# Current profile + voltage *edges* + duration only. Voltage_mean/std (the
# voltage curve *shape*) and Temperature_mean (absolute °C) are intentionally
# excluded: on VTC they carry no unique signal (LOCO 0.9748 -> 0.9745 when
# removed, per-class unchanged) and they are exactly the chemistry-/sensor-bound
# features that would not transfer to a different cell type. What remains is
# scale- and chemistry-portable: the current features are C-rate normalized
# (÷ Nom_Capacity), the voltage edges are window-normalized (÷ (V_max - V_min)),
# and the voltage *range* is the SoC-swing proxy that — together with duration —
# separates a brief pulse from a full CAP discharge. This is the feature basis
# for one model that can run across cell types.
FEATURE_COLS = [
    "Current_mean", "Current_std", "Current_max", "Current_min", "Current_range",
    "abs_Current_mean",
    "Voltage_max", "Voltage_min", "Voltage_range",
    "Duration_minutes", "Duration_quartile",
    "prev_end_voltage_norm",
]

# Final-labeler classes carried straight through from the HDBSCAN pipeline.
# Anything else (stringified raw cluster ints — the leftover families) is a
# candidate for weak-labeling, then falls back to "-1" (OTHER).
KNOWN_LABELS = {"CAP", "PUL", "PUL*RES", "qOCV_DCH", "qOCV_CHA", "-1"}


def bootstrap_leftover_labels(df: pd.DataFrame, min_prep_current: float = 0.1) -> pd.DataFrame:
    """Weak-label the obvious leftover (non-final-labeled) segments by scale-free
    signature. Modifies and returns ``df``.

    - ``PREP_CHA``: a charge (positive current) at a non-trivial rate
      (``|Current_mean| >= min_prep_current``, i.e. faster than ~C/10) that ends
      near the top of charge (``Voltage_max`` normalized > 0.95) — the full prep
      charge before a CAP DCH. The current floor excludes slow C/20 top-offs /
      CV holds and off-spec qOCV charges, which stay OTHER.
    - ``SOC_ADJUST``: a partial charge or discharge at moderate current
      (``|Current_mean| >= min_prep_current``) that ends at an *intermediate*
      SoC — it does not reach the top rail (charge: ``Voltage_max <= 0.95``) or
      the bottom rail (discharge: ``Voltage_min > 0.05``). These set the cell to
      the SoC the next pulse expects. PREP_CHA takes priority on any overlap.

    Everything else leftover stays ``"-1"`` (OTHER) — e.g. slow C/20 off-spec
    qOCV sweeps and full non-CAP discharges. Known final labels are not touched.
    CAP-vs-discharge discrimination is carried by the ``prev_end_voltage_norm``
    feature rather than a dedicated leftover label.
    """
    leftover = ~df["target"].isin(KNOWN_LABELS)
    abs_current = df["Current_mean"].abs()
    moderate = abs_current >= min_prep_current

    soc_adjust = leftover & moderate & (
        ((df["Current_mean"] > 0) & (df["Voltage_max"] <= 0.95))   # partial charge
        | ((df["Current_mean"] < 0) & (df["Voltage_min"] > 0.05))  # partial discharge
    )
    prep_cha = leftover & (df["Current_mean"] > 0) & moderate & (df["Voltage_max"] > 0.95)

    df.loc[leftover, "target"] = "-1"
    df.loc[soc_adjust, "target"] = "SOC_ADJUST"
    df.loc[prep_cha, "target"] = "PREP_CHA"   # PREP_CHA overrides SOC_ADJUST on overlap
    return df


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
    # Merge restore pulses into the pulse class: predict_classifier maps both
    # PUL and PUL*RES back to "PUL*", and update_pulse re-identifies PUL*RES
    # downstream by proc-num adjacency/sign. So the classifier only needs to
    # recognize "pulse"; keeping them apart just adds a hard confusion pair.
    df.loc[df["target"] == "PUL*RES", "target"] = "PUL"
    df = bootstrap_leftover_labels(df)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. If 'prev_end_voltage_norm' is "
            "absent, re-run the pipeline so create_features regenerates the "
            "with_features_post_labeled CSVs with the context feature."
        )

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
