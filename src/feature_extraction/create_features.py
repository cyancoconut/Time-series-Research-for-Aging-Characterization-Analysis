import os
import pandas as pd
import numpy as np
from util.bronze_column_filter import bronze_column_filter
from feature_extraction import classification
import duckdb


def prev_end_voltage_norm(dismembered_df, V_max):
    """Per-segment context feature: end-of-segment voltage of the most recent
    non-PAU predecessor, normalized by ``V_max``.

    Mirrors ``post_cluster_filter.previous_voltage`` exactly (walk back up to 4
    procedure steps, skip PAU stubs, read the predecessor's *last* row voltage)
    so a learned classifier sees the same signal the CAP rule keys on: a true
    CAP discharge is preceded by a fully charged segment, a prep discharge is
    not. Returns ``{ID: value}``; ``0.0`` when no predecessor is found.
    """
    id_summary = (
        dismembered_df.groupby("ID", sort=False)
        .agg(first_target=("target", "first"), last_voltage=("Voltage", "last"))
        .to_dict("index")
    )
    out = {}
    for seg_id in id_summary:
        value = 0.0
        try:
            group, proc = seg_id.split("_")[0], int(seg_id.split("_")[1])
            for step in range(1, 5):
                prev = id_summary.get(f"{group}_{proc - step}")
                if prev is None:
                    continue
                if str(prev["first_target"]) == "PAU":
                    continue
                value = prev["last_voltage"] / V_max
                break
        except Exception:
            value = 0.0
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

    # Context feature for the learned classifier: predecessor end-of-charge
    # voltage. Inert for HDBSCAN (it clusters on a fixed 3-column subset).
    prev_map = prev_end_voltage_norm(dismembered_df, V_max)
    X_unlabeled_features["prev_end_voltage_norm"] = (
        X_unlabeled_features["ID"].map(prev_map).fillna(0.0)
    )

    # Add target column
    X_unlabeled_features["target"] = np.nan

    return X_unlabeled_features


def create_features(
    dismembered_df,
    cell,
    working_path,
    exception_dict,
    V_max,
    V_min,
    V_nom,
    Nom_Capacity,
    feature_columns,
    overwrite=0,
):
    """
    Process a single cell file and extract features, run clustering models, and calculate results.

    Parameters:
        cell (str): Cell identifier
        loadpath_path (str): Path to load the data from
        savepath_path (str): Path to save the processed data to
        exception_dict (list): List to append any exceptions to
        df_results_cap (DataFrame): DataFrame to append capacity results to
        df_results_pulse (DataFrame): DataFrame to append pulse results to
        V_max, V_min, V_nom, Nom_Capacity: Cell parameters
        feature_columns (list): Column names for feature extraction
        hdbscan_para_layer_1, hdbscan_para_layer_2: HDBSCAN parameters
        qOCV_CRate, CAP_Rate, CAP_Type, CAP_Temp: Cell test parameters
        target_pulse_duration, pulse_type, pulse_target_unit: Pulse parameters

    Returns:
        tuple: (df_results_cap, df_results_pulse, exception_dict, count)
    """

    count = 0

    try:
        # working_path is optional: when None (pure-MinIO run), skip the
        # with_features_pre_labeled CSV cache entirely.
        savepath = (
            os.path.join(working_path, "with_features_pre_labeled", cell.split(".")[0] + ".csv")
            if working_path
            else None
        )

        if overwrite == 0 and savepath and os.path.exists(savepath):
            print(
                f"Skipping {cell} - with_features_pre_labeled file already processed"
            )
            try:
                df = pd.read_csv(savepath)
            except Exception as e:
                print(f"Error reading {savepath}: {e}. Removing file.")
                os.remove(savepath)
                return create_features(
                    dismembered_df,
                    cell,
                    working_path,
                    exception_dict,
                    V_max,
                    V_min,
                    V_nom,
                    Nom_Capacity,
                    feature_columns,
                    overwrite=overwrite,
                )
            return df, count

        print(f"Creating features for cell {cell}...")

        try:
            cell_name = cell.split(".")[0]
            # Feature extraction
            X_unlabeled_features = feature_extraction(
                dismembered_df, feature_columns, V_max, V_min, V_nom, Nom_Capacity
            )

            count = count + 1

            if savepath:
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                X_unlabeled_features.to_csv(savepath, index=False)

            return (
                X_unlabeled_features,
                count,
            )

        except Exception as e:
            print(f"Error processing {cell}: {type(e).__name__}: {e}")
            exception_dict[cell_name] = (
                np.nan
                if dismembered_df.empty
                else dismembered_df.Prozedur.unique().tolist()
            )

    except Exception as e:
        print(f"Outer error processing {cell}: {type(e).__name__}: {e}")
        exception_dict[cell_name] = (
            np.nan
            if dismembered_df.empty
            else dismembered_df.Prozedur.unique().tolist()
        )
