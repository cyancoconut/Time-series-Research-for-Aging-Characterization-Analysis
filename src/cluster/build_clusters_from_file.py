import os
import pandas as pd
import numpy as np
from feature_extraction import classification
from cluster import feature_extract_HDBSCAN
from cluster import post_cluster_filter
import duckdb

from cluster.build_clusters import merge_target
from calculate import results_fetching


def feature_extraction(
    dismembered_df, feature_columns, V_max, V_min, V_nom, Nom_Capacity
):
    ## feature extraction
    df_unlabeled = dismembered_df[dismembered_df["target"] == -1]
    df_labeled = dismembered_df[dismembered_df["target"] != -1]

    feature_extraction = classification.FeatureExtraction(
        V_max, V_min, V_nom, Nom_Capacity
    )

    df_unlabeled_features, X_unlabeled = feature_extraction.get_X_unlabled(
        df_unlabeled, feature_columns
    )

    X_unlabeled_features = df_unlabeled_features.merge(
        df_unlabeled.groupby("ID")["Duration_minutes"].first().reset_index(),
        on="ID",
        how="left",
    )
    X_unlabeled_features["target"] = np.nan

    X_unlabeled_features_Duration = X_unlabeled_features[["Duration_minutes", "ID"]]

    return df_labeled, X_unlabeled_features, X_unlabeled, X_unlabeled_features_Duration


def build_clusters_from_file(
    cell,
    loadpath_path,
    savepath_path,
    exception_dict,
    V_max,
    V_min,
    V_nom,
    Nom_Capacity,
    feature_columns,
    hdbscan_para_layer_1,
    hdbscan_para_layer_2,
    qOCV_CRate,
    CAP_Rate,
    CAP_Type,
    CAP_Temp,
    target_pulse_duration,
    pulse_type,
    pulse_target_unit,
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
        savepath = os.path.join(savepath_path, cell)
        fullpath = os.path.join(loadpath_path, cell)

        if overwrite == 0:
            if os.path.exists(savepath):
                print(f"Skipping {cell} - already processed")
                return exception_dict, count

        try:
            try:
                con = duckdb.connect()
                con.execute(
                    "CREATE VIEW parquet_data AS SELECT * FROM read_parquet('{}')".format(
                        fullpath
                    )
                )
                dismembered_df = con.execute(
                    "SELECT Index, Time, Current, Voltage, Temperature,  Zustand, Prozedur, Duration_minutes, ID, BM_Programm, Ah_throughput, Label_Procedure, Capacity_py, Pulse_py FROM parquet_data"
                ).fetch_df()
                if "target" not in dismembered_df.columns:
                    dismembered_df["target"] = np.array(-1, dtype=object)

            except Exception as e:
                dismembered_df = pd.DataFrame()
                print(f"Failed to read: {cell}: {type(e).__name__}: {e}")

            cell_name = cell
            # Feature extraction
            (
                df_labeled,
                X_unlabeled_features,
                X_unlabeled,
                X_unlabeled_features_Duration,
            ) = feature_extraction(
                dismembered_df, feature_columns, V_max, V_min, V_nom, Nom_Capacity
            )

            ## Model part
            model = feature_extract_HDBSCAN.TabularAutoencoderHDBSCAN(
                input_dim=X_unlabeled_features_Duration.shape[1] - 1,
                encoding_dim=10,
                hdbscan_params=hdbscan_para_layer_1,
            )

            model, X_unlabeled_features_Duration = model.fit_cluster_only(
                X_unlabeled_features_Duration
            )

            X_unlabeled_features_updated = merge_target(
                X_unlabeled_features, X_unlabeled_features_Duration
            )
            cluster_means, cluster_std, cluster_size = model.visualize_clusters_only(
                X_unlabeled_features_updated
            )
            print(cluster_means)

            df_clustered = merge_target(X_unlabeled, X_unlabeled_features_updated)

            # Handle potential empty dataframes during concatenation
            try:
                if not df_labeled.empty and not df_clustered.empty:
                    df_final = pd.concat([df_labeled, df_clustered]).sort_index()
                elif not df_labeled.empty:
                    df_final = df_labeled.sort_index()
                elif not df_clustered.empty:
                    df_final = df_clustered.sort_index()
                else:
                    print(
                        f"Warning: Both df_labeled and df_clustered are empty for {cell}"
                    )
                    df_final = pd.DataFrame()
                    exception_dict[cell_name] = (
                        dismembered_df.Prozedur.unique().tolist()
                    )
                    return exception_dict, count
            except ValueError as e:
                if "No objects to concatenate" in str(e):
                    print(
                        f"Warning: No objects to concatenate for {cell}. Creating empty DataFrame."
                    )
                    df_final = pd.DataFrame()
                    exception_dict[cell_name] = (
                        dismembered_df.Prozedur.unique().tolist()
                    )
                    return exception_dict, count
                else:
                    raise

            ################ first layer done, try to find capacity cluster
            post_filter = post_cluster_filter.cluster_filter(
                qOCV_CRate,
                Nom_Capacity,
                V_nom,
                CAP_Rate,
                CAP_Type,
                CAP_Temp,
                target_pulse_duration,
                pulse_type,
                pulse_target_unit,
                df_final,
            )
            capacity_cluster_layer_1, counter = post_filter.find_capacity(
                cluster_means, cluster_size, 1
            )
            pulse_cluster = post_filter.find_pulses(cluster_means)
            qocv_cluster = post_filter.find_qocv(cluster_means)
            ################ if capacity cluster could not be found, use a second layer
            if counter == 1:
                second_layer_feature_columns = ["ID", "Current_mean"]
                df_potential_cap = df_final[
                    (df_final["target"].isin(capacity_cluster_layer_1))
                ]
                X_potential_cap = df_potential_cap.groupby("ID").head(1)[
                    second_layer_feature_columns
                ]

                model_layer2 = feature_extract_HDBSCAN.TabularAutoencoderHDBSCAN(
                    input_dim=X_potential_cap.shape[1] - 1,
                    encoding_dim=10,
                    hdbscan_params=hdbscan_para_layer_2,
                )

                model_layer2, X_second_cluster_for_cap = model_layer2.fit_cluster_only(
                    X_potential_cap
                )
                X_second_cluster_for_cap["target"] = (
                    "cap_layer_"
                    + X_second_cluster_for_cap["target"].astype(int).astype(str)
                )

                cluster_means, cluster_std, cluster_size = (
                    model_layer2.visualize_clusters_only(X_second_cluster_for_cap)
                )
                print(cluster_means)

                # post_filter = post_cluster_filter.cluster_filter(
                #     qOCV_CRate,
                #     Nom_Capacity,
                #     V_nom,
                #     CAP_Rate,
                #     CAP_Type,
                #     CAP_Temp,
                #     target_pulse_duration,
                #     pulse_type,
                #     pulse_target_unit,
                #     df_final,
                # )

                capacity_cluster_layer_2, counter = post_filter.find_capacity(
                    cluster_means, cluster_size, 2
                )

                capacity_cluster = "cap_layer_" + str(capacity_cluster_layer_2)
                capacity_cluster = [capacity_cluster]

                df_potential_cap_labeled = merge_target(
                    df_potential_cap, X_second_cluster_for_cap
                )
                df_potential_cap_labeled.set_index("index", inplace=True)
                df_final["target"].update(df_potential_cap_labeled["target"])

            # Capacity cluster was found, use the first layer's capacity cluster
            else:
                capacity_cluster = capacity_cluster_layer_1
                df_potential_cap = df_final[
                    (df_final["target"].isin(capacity_cluster_layer_1))
                ]

                df_potential_cap_labeled = merge_target(
                    df_potential_cap, X_unlabeled_features_Duration
                )
                df_potential_cap_labeled.set_index("index", inplace=True)
                df_final["target"].update(df_potential_cap_labeled["target"])

            ##############################
            # here the clusters are finalized
            df_result_filtered = post_filter.concat_clusters(
                capacity_cluster, pulse_cluster, qocv_cluster, counter, df_final
            )

            ## calculating
            df_final_1 = results_fetching.calculation(
                qOCV_CRate,
                Nom_Capacity,
                target_pulse_duration,
                pulse_type,
                pulse_target_unit,
                df_result_filtered,
            )
            df_result = df_final_1.update_pulse()
            df_result = df_final_1.update_capacity()

            df_after_filter = df_final.copy()
            df_after_filter.update(df_result)

            ############## storing results
            # Convert target column to string type to avoid Parquet errors
            df_after_filter["target"] = df_after_filter["target"].astype(str)

            try:
                df_after_filter.to_parquet(
                    savepath, coerce_timestamps="us", index=False
                )
                print("Saved at:", savepath)
            except Exception as e:
                print(f"Error saving to parquet: {e}")
                # Convert problematic columns to string if needed
                for col in df_after_filter.columns:
                    if df_after_filter[col].dtype == "object":
                        df_after_filter[col] = df_after_filter[col].astype(str)
                # Try saving again
                df_after_filter.to_parquet(
                    savepath, coerce_timestamps="us", index=False
                )

            count = count + 1

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

    return exception_dict, count
