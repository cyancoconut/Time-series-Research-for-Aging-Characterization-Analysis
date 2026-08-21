"""Export per-cell capacity summary for the aging-status monitor.

One row per BM_Programm: BM_Programm, Capacity_py, Ah_throughput, SOH,
CAP_start_time.

Filename: <cell_stem>_capacity.csv
Local:    <working_path>/40_capacity_monitore/
MinIO:    <minio_prefix>/40_capacity_monitore/
"""

import logging
import os

import pandas as pd

from util import io_router
from util.run_context import CU, RunContext


def export_capacity(df_export, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
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
        # Ah_throughput is cumulative, so its value at the start of the CAP
        # segment is the throughput the cell had reached at this check-up.
        ah = group["Ah_throughput"].dropna() if "Ah_throughput" in group else pd.Series(dtype=float)
        ah_val = float(ah.min()) if not ah.empty else None
        rows.append({
            "BM_Programm": bm_prog,
            "Capacity_py": cap_val,
            "Ah_throughput": ah_val,
            "SOH": soh.get(bm_prog, "NA"),
            "CAP_start_time": times.min(),
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
        key = io_router.export_capacity_object_key(
            cell, filename, root=run_ctx.export_root(cell)
        )
        io_router.upload_csv(minio_client, cfg, summary, key, include_tag=False)
