"""Export per-cell capacity summary for the aging-status monitor.

One row per BM_Programm: BM_Programm, Capacity_py, Ah_throughput, SOH,
CAP_start_time. When a check-up discharges at two rates (e.g. an RPT with a C/2
and a C/5 sweep, each its own CAP segment), Capacity_slow / SOH_slow carry the
slower measurement; the reference (near cap_rate, C/2) stays in Capacity_py/SOH.

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

    nom = cfg.get("nom_capacity")
    cap_rate = cfg.get("cap_rate")

    # One row per CAP segment (ID), discharges only (mean current < 0) — SOH is a
    # discharge quantity, and a check-up may discharge at two rates.
    seg = (
        df_cap.assign(_I=pd.to_numeric(df_cap["Current"], errors="coerce"),
                      _T=pd.to_datetime(df_cap["Time"], errors="coerce"))
        .groupby(["BM_Programm", "ID"], sort=False)
        .agg(Capacity_py=("Capacity_py", "first"), I_mean=("_I", "mean"),
             Ah_throughput=("Ah_throughput", "min"), start_time=("_T", "min"))
        .reset_index()
    )
    seg = seg[seg["I_mean"] < 0]
    if seg.empty:
        logging.info(f"{cell}: no discharge CAP rows to export")
        return
    seg["crate"] = seg["I_mean"].abs() / nom if nom else float("nan")

    rows = []
    for bm_prog, g in seg.groupby("BM_Programm", sort=False):
        # reference = discharge closest to cap_rate (C/2); slow = slowest of the rest (C/5).
        # Ah_throughput is cumulative, so its value at the CAP-segment start is the
        # throughput the cell had reached at this check-up.
        idx = (g["crate"] - cap_rate).abs().idxmin() if cap_rate else g["crate"].idxmax()
        primary = g.loc[idx]
        rest = g.drop(idx)
        slow = rest.loc[rest["crate"].idxmin()] if not rest.empty else None
        cap_val = float(primary["Capacity_py"])
        slow_cap = float(slow["Capacity_py"]) if slow is not None else None
        rows.append({
            "BM_Programm": bm_prog,
            "Capacity_py": cap_val,
            "Ah_throughput": primary["Ah_throughput"],
            "SOH": round(cap_val / nom * 100, 1) if nom else "NA",
            "CAP_start_time": primary["start_time"],
            "Capacity_slow": slow_cap,
            "SOH_slow": round(slow_cap / nom * 100, 1) if (slow_cap is not None and nom) else "NA",
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
