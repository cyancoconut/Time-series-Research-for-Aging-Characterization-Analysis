"""Export per-BM_Programm qOCV segments as standalone parquet files.

Filenames: <cell_stem>_qocv_dch_BM<BM_Programm>_<SOH>SOH.parquet,
           <cell_stem>_qocv_cha_BM<BM_Programm>_<SOH>SOH.parquet
Local:     <working_path>/30_export_qocv/<cell_stem>/
MinIO:     <minio_prefix>/30_export_qocv/<cell_stem>/
"""

import logging
import os

from util import io_router
from util.run_context import CU, RunContext

_TARGET_TO_SUFFIX = {"qOCV_DCH": "qocv_dch", "qOCV_CHA": "qocv_cha"}


def export_qocv(df_export, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
    stem = cell.split(".")[0]
    df_qocv = df_export[df_export["target"].isin(_TARGET_TO_SUFFIX)]
    if df_qocv.empty:
        logging.info(f"{cell}: no qOCV rows to export")
        return

    local_dir = paths["export_qocv_dir"] if paths else None
    write_local = io_router.writes_local(cfg) and local_dir
    write_minio = io_router.writes_minio(cfg)

    if write_local:
        os.makedirs(local_dir, exist_ok=True)

    for (bm_prog, target), group in df_qocv.groupby(["BM_Programm", "target"]):
        soh_val = soh.get(bm_prog, "NA")
        if soh_val == "NA":
            logging.warning(
                f"{cell}: BM_Programm={bm_prog} has no CAP capacity, writing SOH=NA"
            )
        suffix = _TARGET_TO_SUFFIX[target]
        filename = f"{stem}_{suffix}_BM{bm_prog}_{soh_val}SOH.parquet"

        if write_local:
            local_path = os.path.join(local_dir, filename)
            group.to_parquet(local_path, index=False)
            logging.info(f"{cell}: export_qocv -> {local_path}")
        if write_minio:
            key = io_router.export_qocv_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
            io_router.upload_parquet(minio_client, cfg, group, key, include_tag=False)
