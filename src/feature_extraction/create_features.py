import pandas as pd
import numpy as np
from util.bronze_column_filter import bronze_column_filter
from feature_extraction import classification
import duckdb


def prev_end_voltage_norm(dismembered_df, V_max, V_min):
    """Per-segment context feature: end-of-segment voltage of the most recent
    non-PAU predecessor, normalized on the same scale as Voltage features:
    ``(V - V_min) / (V_max - V_min)`` → 0 = bottom rail, 1 = top rail.

    Mirrors ``post_cluster_filter.previous_voltage`` exactly (walk back up to 4
    procedure steps, skip PAU stubs, read the predecessor's *last* row voltage)
    so a learned classifier sees the same signal the CAP rule keys on: a true
    CAP discharge is preceded by a fully charged segment, a prep discharge is
    not. Returns ``{ID: value}``; ``-1`` when no predecessor is found.
    """
    v_window = V_max - V_min
    id_summary = (
        dismembered_df.groupby("ID", sort=False)
        .agg(first_target=("target", "first"), last_voltage=("Voltage", "last"))
        .to_dict("index")
    )
    out = {}
    for seg_id in id_summary:
        value = -1.0
        try:
            group, proc = seg_id.split("_")[0], int(seg_id.split("_")[1])
            for step in range(1, 5):
                prev = id_summary.get(f"{group}_{proc - step}")
                if prev is None:
                    continue
                if str(prev["first_target"]) == "PAU":
                    continue
                value = (prev["last_voltage"] - V_min) / v_window
                break
        except Exception:
            value = -1.0
        out[seg_id] = value
    return out


def feature_extraction(
    dismembered_df, feature_columns, V_max, V_min, V_nom, Nom_Capacity
):
    ## feature extraction
    df_unlabeled = dismembered_df[dismembered_df["target"] == -1]
    df_labeled = dismembered_df[dismembered_df["target"] != -1]

    feature_extraction = classification.FeatureExtraction(
        V_max, V_min, V_nom, Nom_Capacity
    )

    df_unlabeled_features = feature_extraction.get_X_unlabled(
        df_unlabeled, feature_columns
    )

    # create X by merging duration and BM_Programm to the features
    X_unlabeled_features = df_unlabeled_features.merge(
        df_unlabeled.groupby("ID")[["Duration_minutes", "BM_Programm"]]
        .first()
        .reset_index(),
        on="ID",
        how="left",
    )

    # add duration quartile feature
    X_unlabeled_features["Duration_quartile"] = np.log1p(
        X_unlabeled_features["Duration_minutes"]
    )

    X_unlabeled_features["abs_Current_mean"] = X_unlabeled_features["Current_mean"].abs()

    # Coulombic duration: |I|·Δt / Nom_Capacity = fraction of nominal capacity
    # swept by the segment. abs_Current_mean is already current ÷ Nom_Capacity
    # (a C-rate), so this product is dimensionless and invariant to cell size,
    # protocol C-rate, and chemistry — the coulombic twin of Voltage_range
    # (full vs partial discharge ≈ 1.0 vs 0.5). Inert for HDBSCAN.
    X_unlabeled_features["norm_duration"] = (
        X_unlabeled_features["abs_Current_mean"]
        * X_unlabeled_features["Duration_minutes"]
        / 60.0
    )

    # Context feature for the learned classifier: predecessor end-of-charge
    # voltage. Inert for HDBSCAN (it clusters on a fixed 3-column subset).
    prev_map = prev_end_voltage_norm(dismembered_df, V_max, V_min)
    X_unlabeled_features["prev_end_voltage_norm"] = (
        X_unlabeled_features["ID"].map(prev_map).fillna(-1.0)
    )

    # True SoC swing: predecessor end-voltage to this segment's voltage rail.
    # Voltage_range (max-min within segment) understates the swing for charges
    # because PAU relaxation raises the OCV before the segment starts.
    # For charges: Voltage_max - prev_end_voltage_norm (bottom → top).
    # For discharges: prev_end_voltage_norm - Voltage_min (top → bottom).
    # Falls back to Voltage_range when no predecessor (sentinel -1).
    has_prev = X_unlabeled_features["prev_end_voltage_norm"] != -1.0
    is_charge = X_unlabeled_features["Current_mean"] > 0
    charge_swing = (
        X_unlabeled_features["Voltage_max"] - X_unlabeled_features["prev_end_voltage_norm"]
    )
    discharge_swing = (
        X_unlabeled_features["prev_end_voltage_norm"] - X_unlabeled_features["Voltage_min"]
    )
    # Vectorized in one pass (avoids the in-place masked assignment that trips
    # pandas' incompatible-dtype FutureWarning): charge swing where a charge has
    # a predecessor, discharge swing where a discharge does, else Voltage_range.
    X_unlabeled_features["true_voltage_range"] = np.where(
        has_prev & is_charge,
        charge_swing,
        np.where(
            has_prev & ~is_charge,
            discharge_swing,
            X_unlabeled_features["Voltage_range"],
        ),
    )

    # Add target column
    X_unlabeled_features["target"] = np.nan

    return X_unlabeled_features


def create_features(
    dismembered_df,
    cell,
    exception_dict,
    V_max,
    V_min,
    V_nom,
    Nom_Capacity,
    feature_columns,
):
    """
    Extract per-segment features for a single cell.

    The full segment table (with cluster labels / cluster_id) is persisted
    downstream as with_features_post_labeled, so no pre-labeled CSV cache is
    written here.

    Parameters:
        cell (str): Cell identifier
        exception_dict (dict): Dict to record any per-cell exceptions in
        V_max, V_min, V_nom, Nom_Capacity: Cell parameters
        feature_columns (list): Column names for feature extraction

    Returns:
        tuple: (X_unlabeled_features, count)
    """

    count = 0
    cell_name = cell.split(".")[0]

    print(f"Creating features for cell {cell}...")

    try:
        X_unlabeled_features = feature_extraction(
            dismembered_df, feature_columns, V_max, V_min, V_nom, Nom_Capacity
        )
        count = count + 1
        return X_unlabeled_features, count

    except Exception as e:
        print(f"Error processing {cell}: {type(e).__name__}: {e}")
        exception_dict[cell_name] = (
            np.nan
            if dismembered_df.empty
            else dismembered_df.Prozedur.unique().tolist()
        )
