"""Export per-BM_Programm PUL segments as standalone parquet files.

Each pulse group also includes the adjacent PAU stubs (proc_num ±1 within the
same BM_Programm) so the relaxation voltage before/after the pulse is captured.

Filename: <cell_stem>_pulse_BM<BM_Programm>_<SOH>SOH.parquet
Local:    <working_path>/20_export_pulse/<cell_stem>/
MinIO:    <minio_prefix>/20_export_pulse/<cell_stem>/
"""

import logging
import os

import pandas as pd

from util import io_router


def _proc_num(id_str):
    try:
        return int(str(id_str).split("_")[-1])
    except (ValueError, AttributeError):
        return None


def export_pulse(df_export, soh, cell, cfg, paths, minio_client):
    stem = cell.split(".")[0]
    df_pul = df_export[df_export["target"] == "PUL"]
    if df_pul.empty:
        logging.info(f"{cell}: no PUL rows to export")
        return

    df_pau = df_export[df_export["target"] == "PAU"].copy()
    df_pau["_proc"] = df_pau["ID"].map(_proc_num)

    local_dir = paths["export_pulse_dir"] if paths else None
    write_local = io_router.writes_local(cfg) and local_dir
    write_minio = io_router.writes_minio(cfg)

    if write_local:
        os.makedirs(local_dir, exist_ok=True)

    for bm_prog, group in df_pul.groupby("BM_Programm"):
        pul_procs = {_proc_num(i) for i in group["ID"].unique()}
        pul_procs.discard(None)
        neighbor_procs = {p + d for p in pul_procs for d in (-1, 1)}
        pau_neighbors = df_pau[
            (df_pau["BM_Programm"] == bm_prog) & (df_pau["_proc"].isin(neighbor_procs))
        ].drop(columns="_proc")
        group = (
            pd.concat([group, pau_neighbors])
            .sort_values("Time")
            .reset_index(drop=True)
        )
        soh_val = soh.get(bm_prog, "NA")
        if soh_val == "NA":
            logging.warning(
                f"{cell}: BM_Programm={bm_prog} has no CAP capacity, writing SOH=NA"
            )
        filename = f"{stem}_pulse_BM{bm_prog}_{soh_val}SOH.parquet"

        if write_local:
            local_path = os.path.join(local_dir, filename)
            group.to_parquet(local_path, index=False)
            logging.info(f"{cell}: export_pulse -> {local_path}")
        if write_minio:
            key = io_router.export_pulse_object_key(cell, filename)
            io_router.upload_parquet(minio_client, cfg, group, key, include_tag=False)
