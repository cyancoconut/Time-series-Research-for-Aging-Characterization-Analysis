import pandas as pd
from cluster import feature_extract_HDBSCAN
from cluster import post_cluster_filter


def merge_target(df, X_unlabeled):
    if "target" in df.columns:
        df_merged = df.merge(
            X_unlabeled[["ID", "target"]], on="ID", how="left", suffixes=("_x", "_y")
        )
        df_merged["target"] = df_merged["target_y"].fillna(df_merged["target_x"])

        df_final = df_merged.drop(["target_x", "target_y"], axis=1)

    else:
        df_final = df.merge(X_unlabeled[["ID", "target"]], on="ID", how="left")
    return df_final


def first_layer_HDBSCANModel(
    X_unlabeled_features_all,
    dismembered_df,
    cell,
    exception_dict,
    count,
    first_layer_feature_columns,
    hdbscan_para,
):
    cell_name = cell.split(".")[0]
    print(f"Running HDBSCAN Layer 1 for {cell_name}")
    X_unlabeled_features = X_unlabeled_features_all[first_layer_feature_columns]

    model = feature_extract_HDBSCAN.TabularAutoencoderHDBSCAN(
        input_dim=X_unlabeled_features.shape[1] - 1,
        encoding_dim=10,
        hdbscan_params=hdbscan_para,
    )

    model, X_unlabeled_features = model.fit_cluster_only(X_unlabeled_features)

    print(f"Layer 1 clustering completed for {cell_name}")
    X_unlabeled_features_updated = merge_target(
        X_unlabeled_features_all, X_unlabeled_features
    )

    cluster_means, cluster_std, cluster_size = model.visualize_clusters_only(
        X_unlabeled_features_updated
    )

    print(cluster_means)

    # update the target column in df and X_features

    df_clustered = merge_target(dismembered_df, X_unlabeled_features_updated)
    X_clustered = merge_target(X_unlabeled_features_all, X_unlabeled_features_updated)

    # Handle potential empty dataframes during concatenation
    try:
        if not df_clustered.empty:
            df_clustered = df_clustered.sort_index()
        else:
            print(f"Warning: Both df_labeled and df_clustered are empty for {cell}")
            df_clustered = pd.DataFrame()
            exception_dict[cell_name] = dismembered_df.Prozedur.unique().tolist()
            return exception_dict, count
    except ValueError as e:
        if "No objects to concatenate" in str(e):
            print(
                f"Warning: No objects to concatenate for {cell}. Creating empty DataFrame."
            )
            df_clustered = pd.DataFrame()
            exception_dict[cell_name] = dismembered_df.Prozedur.unique().tolist()
            return exception_dict, count
        else:
            raise

    return (
        df_clustered,
        X_clustered,
        cluster_means,
        cluster_size,
        exception_dict,
        count,
    )


def second_layer_HDBSCANModel(
    X_potential_cap,
    df_potential_cap,
    cell,
    exception_dict,
    count,
    second_layer_feature_columns,
    hdbscan_para,
    post_filter,
    df_clustered,
):
    cell_name = cell.split(".")[0]

    print(f"Running HDBSCAN Layer 2 for {cell_name}")
    X_unlabeled_features = X_potential_cap[second_layer_feature_columns]

    model = feature_extract_HDBSCAN.TabularAutoencoderHDBSCAN(
        input_dim=X_unlabeled_features.shape[1] - 1,
        encoding_dim=10,
        hdbscan_params=hdbscan_para,
    )

    model, X_unlabeled_features = model.fit_cluster_only(X_unlabeled_features)

    print(f"Layer 2 clustering completed for {cell_name}")
    X_unlabeled_features_updated = X_unlabeled_features.copy()
    X_unlabeled_features_updated["target"] = (
        "cap_layer_" + X_unlabeled_features_updated["target"].astype(int).astype(str)
    )

    cluster_means, cluster_std, cluster_size = model.visualize_clusters_only(
        X_unlabeled_features_updated
    )

    print(cluster_means)

    # update the target column in df and X_features
    capacity_cluster_layer_2, counter = post_filter.find_capacity(
        cluster_means, cluster_size, 2
    )

    capacity_cluster = "cap_layer_" + str(capacity_cluster_layer_2)
    capacity_cluster = [capacity_cluster]

    df_potential_cap_labeled = merge_target(
        df_potential_cap, X_unlabeled_features_updated
    )
    df_potential_cap_labeled.set_index("index", inplace=True)
    df_clustered["target"] = df_clustered["target"].astype(object)
    df_clustered.loc[df_potential_cap_labeled.index, "target"] = df_potential_cap_labeled["target"]

    X_clustered = merge_target(X_potential_cap, X_unlabeled_features_updated)

    return (
        df_clustered,
        X_clustered,
        capacity_cluster,
        exception_dict,
        counter,
        count,
    )


def supervised_capacity_filter(
    X_clustered, post_filter, df_clustered, cluster_means, cluster_size, layer
):

    capacity_cluster_layer_1, counter = post_filter.find_capacity(
        cluster_means, cluster_size, layer
    )

    if counter == 1:
        print("No capacity clusters found in layer 1, need to run layer 2")

        return 1, capacity_cluster_layer_1, counter

    else:
        print("All clusters found in layer 1, no need to run layer 2")
        capacity_cluster = capacity_cluster_layer_1
        df_potential_cap = df_clustered[
            (df_clustered["target"].isin(capacity_cluster_layer_1))
        ]

        df_potential_cap_labeled = merge_target(df_potential_cap, X_clustered)
        df_potential_cap_labeled.set_index("index", inplace=True)
        df_clustered["target"].update(df_potential_cap_labeled["target"])

    return 0, capacity_cluster, counter


def add_pulse_qocv_and_concat(
    post_filter, cluster_means, capacity_cluster, layer, df_clustered
):
    pulse_cluster = post_filter.find_pulses(cluster_means)
    qocv_cluster = post_filter.find_qocv(cluster_means)

    df_clustered_filtered = post_filter.concat_clusters(
        capacity_cluster, pulse_cluster, qocv_cluster, layer, df_clustered
    )
    return df_clustered_filtered
