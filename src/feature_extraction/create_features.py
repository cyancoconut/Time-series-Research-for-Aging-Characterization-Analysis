import os
import pandas as pd
import numpy as np
from util.bronze_column_filter import bronze_column_filter
from feature_extraction import classification
import duckdb


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
        savepath = os.path.join(
            working_path, "with_features_pre_labeled", cell.split(".")[0] + ".csv"
        )

        if overwrite == 0:
            if os.path.exists(savepath):
                print(
                    f"Skipping {cell} - with_features_pre_labeled file already processed"
                )
                try:
                    df = pd.read_csv(savepath)
                except Exception as e:
                    print(f"Error reading {savepath}: {e}. Removing file.")
                    os.remove(savepath)
                    create_features(
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

            savepath = os.path.join(
                working_path, "with_features_pre_labeled", cell_name + ".csv"
            )

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
