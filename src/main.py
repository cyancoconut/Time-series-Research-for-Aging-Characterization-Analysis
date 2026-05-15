import os
import glob
import json
import logging
import traceback
from contextlib import nullcontext

import pandas as pd

from dismember.dismember_raw_cell import dismember_raw_cell
from feature_extraction.create_features import create_features
from cluster import model_and_supervise, post_cluster_filter
from cluster.post_cluster_filter import ClusterNotFoundException
from calculate import results_fetching
from output.export_pulse import export_pulse
from output.export_qocv import export_qocv
from output.export_capacity import export_capacity
from util import io_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_REQUIRED_COLS = {
    "preSILVER": {
        "Voltage",
        "Current",
        "Time",
        "Temperature",
        "ID",
        "BM_Programm",
        "target",
    },
    "features": {"Duration_quartile", "abs_Current_mean", "Current_mean", "ID"},
    "silver": {
        "Voltage",
        "Current",
        "Time",
        "Temperature",
        "ID",
        "BM_Programm",
        "target",
    },
}


def _validate(df, layer):
    missing = _REQUIRED_COLS[layer] - set(df.columns)
    if missing:
        raise ValueError(f"{layer} missing columns: {missing}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def run_pipeline(cfg: dict, target_specimen: list = None, overwrite: bool = False):
    working_path = cfg.get("working_path")
    download_from = cfg.get("download_from", "local")
    upload_to = cfg.get("upload_to", "local")

    minio_client = (
        io_router.make_minio_client(cfg) if io_router.needs_minio(cfg) else None
    )

    if download_from == "minio":
        cells = io_router.list_bronze_cells(minio_client, cfg)
    else:
        if not working_path:
            raise ValueError("working_path required when download_from='local'")
        cells = glob.glob1(os.path.join(working_path, "BRONZE_CU"), "*.parquet")

    if target_specimen:
        cells = [c for c in cells if any(t in c for t in target_specimen)]

    exceptions = {}
    processed = 0
    for cell in cells:
        type_cell = cfg["type_cell"]
        if type_cell not in cell or "eis" in cell:
            continue

        # Local skip-check only applies when we'd write a local GOLD file
        if io_router.writes_local(cfg) and working_path and not overwrite:
            gold_path = _build_paths(cell, working_path)["gold"]
            if os.path.exists(gold_path):
                logging.info(f"Skipping {cell} — local GOLD already exists")
                continue

        try:
            _process_cell(cell, cfg, minio_client, exceptions)
            processed += 1
        except Exception as e:
            logging.warning(f"{cell}: {type(e).__name__}: {e}")
            traceback.print_exc()
            exceptions[cell] = str(e)

    logging.info(f"Done. {processed} cells processed, {len(exceptions)} failures.")
    if exceptions:
        logging.warning(f"Failed cells: {list(exceptions)}")
    return exceptions


def _process_cell(cell: str, cfg: dict, minio_client, exceptions: dict):
    working_path = cfg.get("working_path")
    download_from = cfg.get("download_from", "local")

    paths = _build_paths(cell, working_path) if working_path else None
    if paths and io_router.writes_local(cfg):
        os.makedirs(os.path.dirname(paths["gold"]), exist_ok=True)

    if download_from == "minio":
        bronze_ctx = io_router.fetch_bronze(minio_client, cfg, cell)
    else:
        bronze_ctx = nullcontext(paths["bronze"])

    with bronze_ctx as bronze_path:
        return _process_cell_inner(
            cell, cfg, bronze_path, paths, minio_client, exceptions
        )


def _process_cell_inner(cell, cfg, bronze_path, paths, minio_client, exceptions):
    working_path = cfg.get("working_path")
    # --- preSILVER ---
    logging.info(f"{cell}: dismembering")
    procedure_filter = cfg.get("procedure_filter", None)
    dismembered_df = dismember_raw_cell(
        cell,
        bronze_path,
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
        cfg["tolerances"]["pulse_cluster_tolerance"],
        cfg["v_max"],
    )

    try:
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
    except ClusterNotFoundException as e:
        logging.warning(
            f"{cell}: no proper checkup detected (no CAP cluster: {e}) — skipping GOLD"
        )
        return

    _validate(df_silver, "silver")

    _write_x_silver(X_silver, cell, cfg, paths, minio_client)

    # --- GOLD ---
    logging.info(f"{cell}: calculating results")
    calc = results_fetching.calculation(
        cfg["qocv_crate"],
        cfg["nom_capacity"],
        cfg["target_pulse_duration"],
        cfg["pulse_type"],
        cfg["pulse_target_unit"],
        df_silver,
        pulse_keep_per_group=cfg.get("pulse_keep_per_group"),
        pulse_group_by=cfg.get("pulse_group_by", "BM_Programm"),
        pulse_step_threshold=cfg.get("pulse_step_threshold"),
        qocv_current_tolerance=cfg.get("tolerances", {}).get(
            "qocv_current_tolerance", 0.01
        ),
        restore_current_tolerance=cfg.get("tolerances", {}).get(
            "restore_current_tolerance", 0.05
        ),
        pulse_duration_tolerance=cfg.get("tolerances", {}).get(
            "pulse_duration_tolerance", 1.08
        ),
    )
    df_gold = df_silver.copy()
    df_gold.update(calc.update_pulse())
    df_gold.update(calc.update_capacity())
    df_gold.update(calc.update_qOCV())

    # Propagate final targets back to X_silver and re-save
    target_map = df_gold.groupby("ID")["target"].first()
    X_silver["target"] = X_silver["ID"].map(target_map).fillna(X_silver["target"])
    _write_x_silver(X_silver, cell, cfg, paths, minio_client)

    # TODO: consider moving the labeling and schedule preparation steps to a separate visualization module that takes the GOLD output as input, to keep the core pipeline focused on data processing and calculation. For now, we'll include it here for simplicity.
    # try:
    #     from visualize import add_test_schedule

    #     add_test_schedule.add_aging_labels(df_gold)
    # except Exception as e:
    #     logging.warning(f"{cell}: add_aging_labels failed ({e}), skipping label step")

    for col in df_gold.columns:
        if df_gold[col].dtype == "object":
            df_gold[col] = df_gold[col].astype(str)

    _write_gold(df_gold, cell, cfg, paths, minio_client)

    df_export = df_gold[
        df_gold["target"].isin(["CAP", "PUL", "qOCV_DCH", "qOCV_CHA", "PAU"])
    ]
    soh = _build_soh_map(df_export, cfg["nom_capacity"])
    export_capacity(df_export, soh, cell, cfg, paths, minio_client)
    if cfg.get("export_pulse"):
        export_pulse(df_export, soh, cell, cfg, paths, minio_client)
    if cfg.get("export_qocv"):
        export_qocv(df_export, soh, cell, cfg, paths, minio_client)


def _build_soh_map(df_export, nom_capacity):
    cap_rows = df_export[df_export["target"] == "CAP"]
    cap_by_prog = cap_rows.groupby("BM_Programm")["Capacity_py"].first()
    soh = {}
    for bm_prog, cap in cap_by_prog.items():
        soh[bm_prog] = round(cap / nom_capacity * 100, 1) if pd.notna(cap) else "NA"
    return soh


def _write_x_silver(df, cell, cfg, paths, minio_client):
    if io_router.writes_local(cfg) and paths:
        os.makedirs(os.path.dirname(paths["X_silver"]), exist_ok=True)
        df.to_csv(paths["X_silver"], index=False)
        logging.info(f"{cell}: X_silver -> {paths['X_silver']}")
    if io_router.writes_minio(cfg):
        io_router.upload_csv(minio_client, cfg, df, io_router.x_silver_object_key(cell))


def _write_gold(df, cell, cfg, paths, minio_client):
    if io_router.writes_local(cfg) and paths:
        os.makedirs(os.path.dirname(paths["gold"]), exist_ok=True)
        df.to_parquet(paths["gold"], index=False)
        logging.info(f"{cell}: GOLD -> {paths['gold']}")
    if io_router.writes_minio(cfg):
        io_router.upload_parquet(minio_client, cfg, df, io_router.gold_object_key(cell))


def _run_clustering(
    dismembered_df,
    X_features,
    cell,
    exceptions,
    count,
    hdbscan_l1,
    hdbscan_l2,
    post_filter,
):
    first_layer_cols = ["Duration_quartile", "abs_Current_mean", "ID"]
    second_layer_cols = ["Current_mean", "ID"]

    df_l1, X_l1, cluster_means_l1, cluster_size_l1, exceptions, count = (
        model_and_supervise.first_layer_HDBSCANModel(
            X_features,
            dismembered_df,
            cell,
            exceptions,
            count,
            first_layer_cols,
            hdbscan_l1,
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
        df_clustered = model_and_supervise.merge_target(df_l2, X_clustered)
    else:
        X_clustered = X_l1
        capacity_cluster = df_clustered_filtered
        df_clustered = model_and_supervise.merge_target(df_l1, X_clustered)

    X_final = model_and_supervise.add_pulse_qocv_and_concat(
        post_filter,
        cluster_means_l1,
        capacity_cluster,
        counter,
        X_clustered,
        dismembered_df,
    )
    df_final = model_and_supervise.merge_target(df_clustered, X_final)

    # Parquet requires string object columns
    for col in df_final.columns:
        if df_final[col].dtype == "object":
            df_final[col] = df_final[col].astype(str)

    return df_final, X_final


def _build_paths(cell: str, working_path: str) -> dict:
    stem = cell.split(".")[0]
    return {
        "bronze": os.path.join(working_path, "BRONZE_CU", cell),
        "X_silver": os.path.join(
            working_path, "with_features_post_labeled", stem + ".csv"
        ),
        "gold": os.path.join(working_path, "GOLD", cell),
        "export_pulse_dir": os.path.join(working_path, "20_export_pulse", stem),
        "export_qocv_dir": os.path.join(working_path, "30_export_qocv", stem),
        "export_capacity_dir": os.path.join(working_path, "40_capacity_monitore"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="METAbatt pipeline")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "--cells", nargs="*", help="Optional subset of cell names to process"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess cells even if GOLD already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_pipeline(cfg, target_specimen=args.cells, overwrite=args.overwrite)
