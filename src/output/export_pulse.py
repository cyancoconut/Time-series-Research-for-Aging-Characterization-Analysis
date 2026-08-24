"""Export per-BM_Programm PUL segments as standalone parquet files.

Each pulse group also includes the adjacent PAU stubs (proc_num ±1 within the
same BM_Programm) so the relaxation voltage before/after the pulse is captured.
The pauses survive in GOLD only as 2-row (first+last) stubs, so the full
relaxation curve is rehydrated here from BRONZE_CU: the stub's first/last
timestamps bracket the pause, and every BRONZE_CU sample in that window is
sliced back in (identity — ID / BM_Programm / target — is stamped from the stub,
since BRONZE_CU is unsegmented and carries no ID). Falls back to the 2-row stub
when no bronze_path is given or the window yields no rows.

Filename: <cell_stem>_pulse_BM<BM_Programm>_<SOH>SOH.parquet
Local:    <working_path>/20_export_pulse/<cell_stem>/
MinIO:    <minio_prefix>/20_export_pulse/<cell_stem>/
"""

import logging
import os

import pandas as pd
import pyarrow.parquet as pq

from util import io_router
from util.run_context import CU, RunContext

# BRONZE_CU is German-named and unsegmented; mirror dismember_raw_cell's rename.
_BRONZE_RENAME = {
    "Spannung": "Voltage",
    "Strom": "Current",
    "Zeit": "Time",
    "T1": "Temperature",
}
# Columns worth pulling from BRONZE_CU for the rehydrated rows (post-rename).
_BRONZE_WANT = ["Zeit", "Spannung", "Strom", "T1", "Prozedur", "Zustand", "Ah_throughput"]


def _proc_num(id_str):
    try:
        return int(str(id_str).split("_")[-1])
    except (ValueError, AttributeError):
        return None


def _load_bronze_renamed(bronze_path):
    """Read only the needed BRONZE_CU columns and rename German -> English.

    Returns ``None`` if the file can't be read so the caller falls back to the
    2-row stubs.
    """
    try:
        schema_names = set(pq.ParquetFile(bronze_path).schema.names)
        cols = [c for c in _BRONZE_WANT if c in schema_names]
        df = pd.read_parquet(bronze_path, columns=cols)
        return df.rename(columns=_BRONZE_RENAME)
    except Exception as exc:  # noqa: BLE001 - degrade to stubs, never fail the export
        logging.warning(f"export_pulse: could not read BRONZE_CU for rehydration ({exc})")
        return None


def _rehydrate_pau(stub_rows, bronze_df, columns):
    """Replace a 2-row PAU stub with the full BRONZE_CU window it brackets.

    Slices BRONZE_CU on the stub's [first, last] timestamps and stamps the
    stub's identity (constant within a PAU segment) onto the sliced rows.
    Returns the stub unchanged if the window finds nothing.
    """
    t0, t1 = stub_rows["Time"].min(), stub_rows["Time"].max()
    window = bronze_df[(bronze_df["Time"] >= t0) & (bronze_df["Time"] <= t1)]
    if window.empty:
        return stub_rows
    out = window.reindex(columns=columns)
    for col in ("ID", "BM_Programm", "target", "Duration_minutes"):
        if col in columns:
            out[col] = stub_rows[col].iloc[0]
    return out


def export_pulse(
    df_export, soh, cell, cfg, paths, minio_client, bronze_path=None,
    run_ctx: RunContext = CU,
):
    stem = cell.split(".")[0]
    df_pul = df_export[df_export["target"] == "PUL"]
    if df_pul.empty:
        logging.info(f"{cell}: no PUL rows to export")
        return

    df_pau = df_export[df_export["target"] == "PAU"].copy()
    df_pau["_proc"] = df_pau["ID"].map(_proc_num)

    # Lazily load BRONZE_CU once per cell to rehydrate the full relaxation curve.
    bronze_df = None
    if bronze_path and not df_pau.empty:
        bronze_df = _load_bronze_renamed(bronze_path)

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
        if bronze_df is not None and not pau_neighbors.empty:
            pau_neighbors = pd.concat(
                [
                    _rehydrate_pau(stub, bronze_df, group.columns)
                    for _, stub in pau_neighbors.groupby("ID")
                ],
                ignore_index=True,
            )
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
            key = io_router.export_pulse_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
            io_router.upload_parquet(minio_client, cfg, group, key, include_tag=False)
