"""Train a RandomForest classifier to replace HDBSCAN + cluster_filter.

Reads every per-segment CSV in `<working_path>/with_features_post_labeled/`
(produced by the HDBSCAN pipeline, post-#37 so it carries *every* segment incl.
the prep / SOC-adjust leftovers and a `cluster_id` column), treats the `target`
column as ground truth, weak-labels the obvious leftover families (PREP_CHA /
PREP_DCH), runs leave-one-cell-out CV to report per-class precision/recall, then
refits on all data and writes the model.

Label source (``--labels`` / config ``classifier_label_source``):
- ``target`` (default): the flow above — HDBSCAN final labels + leftover bootstrap.
- ``llm``: train on the free-form ``llm_label`` column written by
  ``interpret_clusters`` (no taxonomy collapse, no bootstrap); segments with no
  ``llm_label`` are dropped. ``predict_classifier`` maps the free-form prediction
  back to CAP*/PUL*/QOCV* at inference. ``_meta.json["label_source"]`` records it.

Outputs (stem from config `type_cell`, timestamped so runs
are kept side by side and never overwritten; also uploaded to MinIO when the
config's `upload_to` includes minio, untagged under
`<minio_prefix>/60_classifier/models/`):

    <working_path>/60_classifier/models/<type_cell>_classifier_<ts>.joblib    — the fitted estimator
    <working_path>/60_classifier/models/<type_cell>_classifier_<ts>_meta.json — feature columns, classes, training cells

Usage (from src/):
    python -m cluster.train_classifier /path/to/battery_config.json
"""

import argparse
import glob
import io
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from util import io_router
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


def _iter_cell_csvs(cfg: dict, source: str):
    """Yield ``(cell_name, DataFrame)`` for every with_features_post_labeled CSV.

    Routed by ``download_from`` (``source``): ``local`` globs
    ``<working_path>/with_features_post_labeled/*.csv``; ``minio`` lists and
    fetches the tagged objects via :mod:`util.io_router`.
    """
    if source == "minio":
        client = io_router.make_minio_client(cfg)
        names = io_router.list_x_silver_cells(client, cfg)
        if not names:
            raise FileNotFoundError(
                f"No labeled CSVs under "
                f"{cfg['minio_prefix']}/{io_router.UPLOAD_PREFIX_TAG}/with_features_post_labeled/"
            )
        for name in names:
            data = io_router.fetch_x_silver_bytes(client, cfg, name)
            yield name.replace(".csv", ""), pd.read_csv(io.BytesIO(data))
        return

    pattern = os.path.join(cfg["working_path"], "with_features_post_labeled", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No labeled CSVs under {pattern}")
    for f in files:
        yield os.path.basename(f).replace(".csv", ""), pd.read_csv(f)


def _load_cell_csvs(cfg: dict, source: str, label_source: str = "target") -> pd.DataFrame:
    frames = []
    for cell, df in _iter_cell_csvs(cfg, source):
        df["cell"] = cell
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if label_source == "llm":
        # Train directly on the LLM's free-form cluster names (the `llm_label`
        # column written by interpret_clusters). No collapse to the pipeline
        # taxonomy: the classifier learns the LLM vocabulary as-is, and
        # predict_classifier maps the prediction back to CAP*/PUL*/QOCV* at
        # inference. The LLM already merges restore/test pulses and qOCV
        # DCH/CHA, so no PUL*RES merge or bootstrap is needed here.
        if "llm_label" not in df.columns:
            raise ValueError(
                "label source 'llm' requested but no 'llm_label' column found. "
                "Run `python -m cluster.interpret_clusters <cfg>` first so the "
                "with_features_post_labeled CSVs carry the llm_* columns."
            )
        df["target"] = df["llm_label"].astype("string").str.strip()
        before = len(df)
        df = df[
            df["target"].notna()
            & (df["target"] != "")
            & (df["target"].str.lower() != "nan")
        ].copy()
        dropped = before - len(df)
        if dropped:
            logging.info(f"dropped {dropped} segments with no llm_label (uninterpreted)")
        df["target"] = df["target"].astype(str)
    else:
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


def _stamp_path(path: str, ts: str) -> str:
    """Inject `_<ts>` before the file extension (keeps runs from overwriting)."""
    root, ext = os.path.splitext(path)
    return f"{root}_{ts}{ext}"


def _resolve_out_paths(cfg: dict, model_out, meta_out, ts: str):
    """Build timestamped, cell-type-aware output paths.

    The trained model is a pipeline artifact, so by default it lands under the
    data folder (``<working_path>/60_classifier/models/``), mirroring the MinIO
    ``<prefix>/60_classifier/models/`` layout — not in the code workspace. Falls
    back to ``../models/`` only when ``working_path`` is unset. The stem is
    derived from the config's ``type_cell`` (e.g. ``vtc_classifier_<ts>.joblib``);
    when ``--model-out`` / ``--meta-out`` are given, the user's stem is kept but
    still gets a ``_<ts>`` suffix so nothing is clobbered.
    """
    cell = str(cfg.get("type_cell", "model")).lower()
    working_path = cfg.get("working_path")
    base = (
        os.path.join(working_path, "60_classifier", "models")
        if working_path
        else "../models"
    )
    if model_out is None:
        model_out = os.path.join(base, f"{cell}_classifier_{ts}.joblib")
    else:
        model_out = _stamp_path(model_out, ts)
    if meta_out is None:
        meta_out = os.path.join(base, f"{cell}_classifier_{ts}_meta.json")
    else:
        meta_out = _stamp_path(meta_out, ts)
    return model_out, meta_out


def train(config_path: str, model_out=None, meta_out=None, label_source=None) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    source = cfg.get("download_from", "local")
    label_source = label_source or cfg.get("classifier_label_source", "target")
    if label_source not in ("target", "llm"):
        raise ValueError(f"Unknown label source {label_source!r} (expected 'target' or 'llm')")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model_out, meta_out = _resolve_out_paths(cfg, model_out, meta_out, ts)

    df = _load_cell_csvs(cfg, source, label_source)
    logging.info(
        f"Loaded {len(df)} segments from {df['cell'].nunique()} cells "
        f"({source}, label_source={label_source})"
    )
    logging.info(f"Class counts:\n{df['target'].value_counts().to_string()}")

    _loco_cv(df)

    model = _new_model()
    model.fit(df[FEATURE_COLS], df["target"])

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)

    meta = {
        "feature_columns": FEATURE_COLS,
        "label_source": label_source,
        "classes": sorted(df["target"].unique().tolist()),
        "training_cells": sorted(df["cell"].unique().tolist()),
        "n_segments": int(len(df)),
        "type_cell": cfg.get("type_cell"),
        "run_timestamp": ts,
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

    # MinIO upload follows `upload_to` (untagged, under
    # <prefix>/60_classifier/models/), mirroring how exports / aging_status.html
    # sit directly under the prefix.
    if io_router.writes_minio(cfg):
        client = io_router.make_minio_client(cfg)
        with open(model_out, "rb") as f:
            io_router._upload_bytes(
                client, cfg, f"60_classifier/models/{os.path.basename(model_out)}",
                f.read(), include_tag=False,
            )
        with open(meta_out, "rb") as f:
            io_router._upload_bytes(
                client, cfg, f"60_classifier/models/{os.path.basename(meta_out)}",
                f.read(), include_tag=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segment classifier")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--model-out", default=None,
                        help="Override model path stem; a _<timestamp> suffix is always added. "
                             "Default: <working_path>/60_classifier/models/<type_cell>_classifier_<timestamp>.joblib")
    parser.add_argument("--meta-out", default=None,
                        help="Override meta path stem; a _<timestamp> suffix is always added. "
                             "Default: <working_path>/60_classifier/models/<type_cell>_classifier_<timestamp>_meta.json")
    parser.add_argument("--labels", choices=["target", "llm"], default=None,
                        help="Training target source: 'target' (HDBSCAN final labels + "
                             "leftover bootstrap, default) or 'llm' (the free-form llm_label "
                             "column from interpret_clusters). Overrides config "
                             "'classifier_label_source'.")
    args = parser.parse_args()

    train(args.config, args.model_out, args.meta_out, args.labels)
