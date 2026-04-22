import os
import glob
import json
import logging
import traceback
import pandas as pd

from dismember.dismember_raw_cell import dismember_raw_cell
from feature_extraction.create_features import create_features
from cluster import model_and_supervise, post_cluster_filter
from calculate import results_fetching

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_REQUIRED_COLS = {
    "preSILVER": {"Voltage", "Current", "Time", "Temperature", "ID", "BM_Programm", "target"},
    "features":  {"Duration_quartile", "abs_Current_mean", "Current_mean", "ID"},
    "silver":    {"Voltage", "Current", "Time", "Temperature", "ID", "BM_Programm", "target"},
}


def _validate(df, layer):
    missing = _REQUIRED_COLS[layer] - set(df.columns)
    if missing:
        raise ValueError(f"{layer} missing columns: {missing}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def run_pipeline(cfg: dict, target_specimen: list = None, overwrite: bool = False):
    working_path = cfg["working_path"]
    cells = glob.glob1(os.path.join(working_path, "BRONZE_CU"), "*.parquet")
    if target_specimen:
        cells = [c for c in cells if any(t in c for t in target_specimen)]

    exceptions = {}
    processed = 0
    for cell in cells:
        type_cell = cfg["type_cell"]
        if type_cell not in cell or "eis" in cell:
            continue
        gold_path = _build_paths(cell, working_path, type_cell)["gold"]
        if not overwrite and os.path.exists(gold_path):
            logging.info(f"Skipping {cell} — GOLD already exists")
            continue
        try:
            _process_cell(cell, working_path, cfg, exceptions)
            processed += 1
        except Exception as e:
            logging.warning(f"{cell}: {type(e).__name__}: {e}")
            traceback.print_exc()
            exceptions[cell] = str(e)

    logging.info(f"Done. {processed} cells processed, {len(exceptions)} failures.")
    if exceptions:
        logging.warning(f"Failed cells: {list(exceptions)}")
    return exceptions


def _process_cell(cell: str, working_path: str, cfg: dict, exceptions: dict):
    paths = _build_paths(cell, working_path, cfg["type_cell"])
    os.makedirs(os.path.dirname(paths["gold"]), exist_ok=True)

    # --- preSILVER ---
    logging.info(f"{cell}: dismembering")
    procedure_filter = cfg.get("procedure_filter", None)
    dismembered_df = dismember_raw_cell(
        cell,
        paths["bronze"],
        cfg["min_rows"],
        cfg["pau_duration"],
        cfg["v_max"],
        procedure_filter,
    )
    if dismembered_df is None or dismembered_df.empty:
        logging.warning(f"{cell}: empty after dismember, skipping")
        return
    _validate(dismembered_df, "preSILVER")

    n_progs = 0
    if procedure_filter is not None:
        n_progs = int(
            dismembered_df.groupby("BM_Programm")["Prozedur"]
            .apply(lambda x: x.str.contains(procedure_filter, na=False).any())
            .sum()
        )
    else:
        # If no filter, count all unique programs
        n_progs = dismembered_df["BM_Programm"].nunique()
    logging.info(f"{cell}: {n_progs} programs found")

    # --- features ---
    logging.info(f"{cell}: extracting features")
    X_features, count = create_features(
        dismembered_df,
        cell,
        working_path,
        exceptions,
        cfg["v_max"],
        cfg["v_min"],
        cfg["v_nom"],
        cfg["nom_capacity"],
        cfg["feature_columns"],
        overwrite=1,
    )
    _validate(X_features, "features")

    # --- clustering ---
    logging.info(f"{cell}: clustering")
    default_min_cluster_size = max(2, n_progs - 1)
    hdbscan_l1 = {
        "min_cluster_size": default_min_cluster_size,
        "min_samples": 1,
        **cfg["hdbscan_para_layer_1"],
    }
    post_filter = post_cluster_filter.cluster_filter(
        n_progs,
        cfg["qocv_crate"],
        cfg["nom_capacity"],
        cfg["v_nom"],
        cfg["cap_rate"],
        cfg["cap_type"],
        cfg["cap_temp"],
        cfg["target_pulse_duration"],
        cfg["pulse_type"],
        cfg["pulse_target_unit"],
    )

    df_silver, X_silver = _run_clustering(
        dismembered_df,
        X_features,
        cell,
        exceptions,
        count,
        hdbscan_l1,
        cfg["hdbscan_para_layer_2"],
        post_filter,
    )

    _validate(df_silver, "silver")

    os.makedirs(os.path.dirname(paths["X_silver"]), exist_ok=True)
    X_silver.to_csv(paths["X_silver"], index=False)

    # --- GOLD ---
    logging.info(f"{cell}: calculating results")
    calc = results_fetching.calculation(
        cfg["qocv_crate"],
        cfg["nom_capacity"],
        cfg["target_pulse_duration"],
        cfg["pulse_type"],
        cfg["pulse_target_unit"],
        df_silver,
    )
    df_gold = df_silver.copy()
    df_gold.update(calc.update_pulse())
    df_gold.update(calc.update_capacity())
    df_gold.update(calc.update_qOCV())

    # Propagate final targets back to X_silver and re-save
    target_map = df_gold.groupby("ID")["target"].first()
    X_silver["target"] = X_silver["ID"].map(target_map).fillna(X_silver["target"])
    X_silver.to_csv(paths["X_silver"], index=False)

    try:
        from visualize import add_test_schedule
        add_test_schedule.add_aging_labels(df_gold)
    except Exception as e:
        logging.warning(f"{cell}: add_aging_labels failed ({e}), skipping label step")

    for col in df_gold.columns:
        if df_gold[col].dtype == "object":
            df_gold[col] = df_gold[col].astype(str)
    df_gold.to_parquet(paths["gold"], index=False)
    logging.info(f"{cell}: GOLD exported to {paths['gold']}")


def _run_clustering(
    dismembered_df, X_features, cell, exceptions, count, hdbscan_l1, hdbscan_l2, post_filter
):
    first_layer_cols = ["Duration_quartile", "abs_Current_mean", "ID"]
    second_layer_cols = ["Current_mean", "ID"]

    df_l1, X_l1, cluster_means_l1, cluster_size_l1, exceptions, count = (
        model_and_supervise.first_layer_HDBSCANModel(
            X_features, dismembered_df, cell, exceptions, count, first_layer_cols, hdbscan_l1
        )
    )

    capacity_status, df_clustered_filtered, counter = (
        model_and_supervise.supervised_capacity_filter(
            X_l1, post_filter, df_l1, cluster_means_l1, cluster_size_l1, 1
        )
    )

    if capacity_status:
        # Layer 1 inconclusive — run layer 2 on the candidate capacity cluster
        capacity_cluster_l1 = df_clustered_filtered
        df_potential_cap = df_l1[df_l1["target"].isin(capacity_cluster_l1)]
        X_potential_cap = X_l1[X_l1["target"].isin(capacity_cluster_l1)]

        df_l2, X_l2, capacity_cluster, exceptions, counter, count = (
            model_and_supervise.second_layer_HDBSCANModel(
                X_potential_cap,
                df_potential_cap,
                cell,
                exceptions,
                count,
                second_layer_cols,
                hdbscan_l2,
                post_filter,
                df_l1,
            )
        )

        X_clustered = model_and_supervise.merge_target(X_l1, X_l2)
        df_clustered = df_l2.drop(columns=["target"]).merge(
            X_clustered[["ID", "target"]], on="ID", how="left"
        )
    else:
        X_clustered = X_l1
        capacity_cluster = df_clustered_filtered
        df_clustered = df_l1.drop(columns=["target"]).merge(
            X_clustered[["ID", "target"]], on="ID", how="left"
        )

    X_final = model_and_supervise.add_pulse_qocv_and_concat(
        post_filter, cluster_means_l1, capacity_cluster, counter, X_clustered
    )
    df_final = model_and_supervise.merge_target(df_clustered, X_final)

    # Parquet requires string object columns
    for col in df_final.columns:
        if df_final[col].dtype == "object":
            df_final[col] = df_final[col].astype(str)

    return df_final, X_final


def _build_paths(cell: str, working_path: str, type_cell: str) -> dict:
    stem = cell.split(".")[0]
    return {
        "bronze":   os.path.join(working_path, "BRONZE_CU", cell),
        "X_silver": os.path.join(working_path, "with_features_post_labeled", stem + ".csv"),
        "gold":     os.path.join(working_path, "GOLD", type_cell, cell),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="METAbatt pipeline")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cell names to process")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess cells even if GOLD already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Secrets from environment — never hardcode
    cfg["minio_access_key"] = os.environ.get("MINIO_ACCESS_KEY", "")
    cfg["minio_secret_key"] = os.environ.get("MINIO_SECRET_KEY", "")
    cfg["influx_token"] = os.environ.get("INFLUX_TOKEN", "")

    run_pipeline(cfg, target_specimen=args.cells, overwrite=args.overwrite)
