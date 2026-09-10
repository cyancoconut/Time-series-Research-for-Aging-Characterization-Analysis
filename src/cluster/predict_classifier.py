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
# matches on. update_capacity refines CAP* -> CAP, update_qOCV refines
# QOCV* -> qOCV_DCH / qOCV_CHA based on sign.
#
# PUL maps to the intermediate "PUL*" so update_pulse runs downstream: it
# re-identifies restore pulses (PUL*RES) by proc-num adjacency/sign and applies
# the duration check. The classifier itself does NOT need to distinguish PUL
# from PUL*RES — they are merged into PUL for training — because that split is
# recomputed here at the calculate/ stage.
#
# PREP_CHA / SOC_ADJUST pass through unchanged: the calculate/ step only touches
# CAP*/PUL*/QOCV* rows, so these informational labels survive into the GOLD
# `target` column without affecting any numeric (capacity/pulse) result.
_LABEL_TO_TAGGED = {
    "CAP": "CAP*",
    "PUL": "PUL*",
    "qOCV_DCH": "QOCV*",
    "qOCV_CHA": "QOCV*",
    "PREP_CHA": "PREP_CHA",
    "SOC_ADJUST": "SOC_ADJUST",
    "-1": "-1",
}


def _map_llm_label_to_tagged(
    label: str, abs_crate, cap_rate, cap_tol: float, qocv_rate=None, qocv_tol: float = 0.2
) -> str:
    """Map a free-form LLM cluster name (`full_discharge_c2`, `pulse`, `qocv_c20`,
    `mixed_*`, …) to the cluster-tagged form calculate/ matches on.

    Only the labels that drive a numeric result are translated; everything else
    (partial_*, rest, artifact, …) passes through unchanged into the GOLD
    `target` (informational, no numeric effect — like PREP_CHA today).

    - ``*pulse*`` -> ``PUL*``   (update_pulse re-splits test/restore downstream)
    - ``*qocv*``  -> ``QOCV*``  (update_qOCV re-splits DCH/CHA by sign)
    - bare ``cap`` -> ``CAP*``  (resolved-space label: when the classifier was
      trained on the canonicalized label space — see
      ``train_classifier._canonicalize_llm_labels`` — full discharges in the
      capacity band arrive pre-resolved as ``cap`` / ``qocv``, so they map
      straight through rather than via the measured-rate branch below).
    - a ``full_charge`` / ``full_discharge`` is resolved by its **measured**
      C-rate (``abs_Current_mean``, already ÷ Nom_Capacity) — the same signal the
      rule filters key on — rather than trusting the LLM's full-vs-qocv call or
      re-parsing the label's crate suffix:
        * ``abs_crate <= qocv_rate * (1 + qocv_tol)`` -> ``QOCV*`` (a full sweep
          at/below the configured quasi-OCV C-rate, charge or discharge).
        * else a ``full_discharge`` within ``cap_rate * (1 ± cap_tol)`` -> ``CAP*``.
      ``qOCV_CRate`` << ``cap_rate``, so the two bands never overlap; qocv is
      checked first. With either rate unset that branch is skipped.
    """
    low = label.lower()
    if "pulse" in low:
        return "PUL*"
    if "qocv" in low:
        return "QOCV*"
    if low == "cap":
        return "CAP*"
    if ("full_discharge" in low or "full_charge" in low) and abs_crate is not None:
        ac = abs(abs_crate)
        if qocv_rate and ac <= qocv_rate * (1 + qocv_tol):
            return "QOCV*"
        if (
            "full_discharge" in low
            and cap_rate
            and cap_rate * (1 - cap_tol) <= ac <= cap_rate * (1 + cap_tol)
        ):
            return "CAP*"
    return label


def _load(model_path: str, meta_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Classifier metadata not found: {meta_path}")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def predict_targets(
    X_features: pd.DataFrame,
    model_path: str,
    meta_path: str,
    cap_rate=None,
    cap_tol: float = 0.05,
    qocv_rate=None,
    qocv_tol: float = 0.2,
) -> pd.DataFrame:
    """Replace the `target` column in X_features with classifier predictions
    (translated to cluster-tagged form) and add a `cluster_id` column holding
    the raw (untagged) predicted label. Returns a new DataFrame; input is not
    modified.

    Two label spaces are supported, distinguished by ``meta["label_source"]``:

    - ``"target"`` (default / legacy): the model predicts the fixed pipeline
      vocabulary (CAP/PUL/qOCV_*/…) and :data:`_LABEL_TO_TAGGED` maps it.
    - ``"llm"``: the model predicts free-form LLM cluster names; each prediction
      is mapped by :func:`_map_llm_label_to_tagged`, which resolves a full
      charge/discharge by its measured C-rate: ``qocv_rate`` (config
      ``qocv_crate``) tags a full sweep at/below the quasi-OCV rate as QOCV*, and
      ``cap_rate`` (config ``cap_rate``) tags a full discharge at the capacity
      rate as CAP*. With a rate unset, the corresponding tag is never produced.
    """
    model, meta = _load(model_path, meta_path)
    feature_cols = meta["feature_columns"]
    label_source = meta.get("label_source", "target")

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

    # cluster_id = the raw classifier label, mirroring how the HDBSCAN path
    # stores its raw integer cluster here. Captured before _LABEL_TO_TAGGED
    # collapses qOCV_DCH/qOCV_CHA -> QOCV* and PUL -> PUL*, so the DCH/CHA and
    # restore distinctions are preserved for training provenance.
    X_out["cluster_id"] = preds.fillna("-1").astype(object)

    if label_source == "llm":
        if not cap_rate:
            logging.warning(
                "classifier label_source='llm' but cap_rate is unset — no segment "
                "can be tagged CAP* (capacity will not be computed)"
            )
        crate = X_out["abs_Current_mean"] if "abs_Current_mean" in X_out else None
        tagged = []
        for idx, lbl in preds.fillna("-1").items():
            c = float(crate.loc[idx]) if crate is not None and pd.notna(crate.loc[idx]) else None
            tagged.append(
                _map_llm_label_to_tagged(str(lbl), c, cap_rate, cap_tol, qocv_rate, qocv_tol)
            )
        X_out["target"] = pd.Series(tagged, index=preds.index).astype(object)
    else:
        X_out["target"] = preds.map(_LABEL_TO_TAGGED).fillna("-1").astype(object)
    return X_out
