"""Export per-cell capacity summary for the aging-status monitor.

One row per BM_Programm: BM_Programm, Capacity_py, SOH, CAP_start_time, CAP_end_time.

Filename: <cell_stem>_capacity.csv
Local:    <working_path>/40_capacity_monitore/
MinIO:    <minio_prefix>/40_capacity_monitore/
"""

import logging
import os

import pandas as pd

from util import io_router


def export_capacity(df_export, soh, cell, cfg, paths, minio_client):
    stem = cell.split(".")[0]
    df_cap = df_export[df_export["target"] == "CAP"]
    if df_cap.empty:
        logging.info(f"{cell}: no CAP rows to export")
        return

    rows = []
    for bm_prog, group in df_cap.groupby("BM_Programm"):
        cap = group["Capacity_py"].dropna()
        cap_val = float(cap.iloc[0]) if not cap.empty else None
        times = pd.to_datetime(group["Time"], errors="coerce")
        rows.append({
            "BM_Programm": bm_prog,
            "Capacity_py": cap_val,
            "SOH": soh.get(bm_prog, "NA"),
            "CAP_start_time": times.min(),
            "CAP_end_time": times.max(),
        })
    summary = pd.DataFrame(rows).sort_values("BM_Programm").reset_index(drop=True)

    filename = f"{stem}_capacity.csv"
    local_dir = paths["export_capacity_dir"] if paths else None

    if io_router.writes_local(cfg) and local_dir:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        summary.to_csv(local_path, index=False)
        logging.info(f"{cell}: export_capacity -> {local_path}")
    if io_router.writes_minio(cfg):
        key = io_router.export_capacity_object_key(cell, filename)
        io_router.upload_csv(minio_client, cfg, summary, key, include_tag=False)
