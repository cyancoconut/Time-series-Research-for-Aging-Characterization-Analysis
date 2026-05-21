from .cluster_preparation import DismemblerFunctions
from .cluster_preparation import allocate_IDs
from util import bronze_column_filter

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc


def processing_procedure_filter(source, procedure_filter):
    """Check whether any Prozedur value in the BRONZE parquet matches the filter.

    ``source`` may be either a local file path or a seekable file-like object
    (e.g. ``io_router.open_bronze_range``). Only the ``Prozedur`` column is
    read, row group by row group, and the scan short-circuits on the first
    substring match — so on MinIO this fetches only the parquet footer plus
    one column's data, not the full file.
    """
    # If no filter is provided, allow all cells
    if procedure_filter is None:
        return True

    pf = pq.ParquetFile(source)
    if "Prozedur" not in pf.schema_arrow.names:
        return False
    for i in range(pf.num_row_groups):
        col = pf.read_row_group(i, columns=["Prozedur"]).column("Prozedur")
        if pc.any(pc.match_substring(col, procedure_filter)).as_py():
            return True
    return False


def read_and_fix_format(loadpath, V_max):

    df_cell = pd.read_parquet(loadpath)

    # Fix the format
    df_cell[df_cell.select_dtypes(np.float64).columns] = df_cell.select_dtypes(
        np.float64
    ).astype(np.float32)
    df_cell = df_cell.rename(
        columns={
            "Spannung": "Voltage",
            "Strom": "Current",
            "Zeit": "Time",
            "T1": "Temperature",
        }
    )

    # The temperature channel is exported under different names across datasets
    # (e.g. "T1", or the German "Temperatur"). Only fall back to an alias when
    # "Temperature" is not already present, so a file that has both "T1" and an
    # alias never ends up with two "Temperature" columns.
    if "Temperature" not in df_cell.columns:
        for alias in ("Temperatur", "Temp"):
            if alias in df_cell.columns:
                df_cell = df_cell.rename(columns={alias: "Temperature"})
                break

    # Convert Time to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(df_cell["Time"]):
        df_cell["Time"] = pd.to_datetime(df_cell["Time"], errors='coerce')

    df_cell[df_cell.select_dtypes(np.float64).columns] = df_cell.select_dtypes(
        np.float64
    ).astype(np.float32)
    
    # Only attempt UTC conversion if Time is actually a datetime object
    if pd.api.types.is_datetime64_any_dtype(df_cell["Time"]):
        try:
            df_cell["Time_UTC"] = df_cell.Time.dt.tz_convert("UTC")
        except Exception:
            # If it has no timezone info, tz_convert fails; try tz_localize
            try:
                df_cell["Time_UTC"] = df_cell.Time.dt.tz_localize('UTC')
            except Exception:
                df_cell["Time_UTC"] = df_cell["Time"]
    else:
        df_cell["Time_UTC"] = df_cell["Time"]
    df_cell["Zustand"] = df_cell["Zustand"].astype(object)
    df_cell["Prozedur"] = df_cell["Prozedur"].astype(object)
    df_cell["Power"] = df_cell["Current"] * df_cell["Voltage"]
    df_cell["Label_Procedure"] = np.nan
    df_cell["Capacity_py"] = np.nan
    # BM_Programm identification
    if "Ahjo_Test_ID" in df_cell.columns:
        df_cell["BM_Programm"] = df_cell.groupby("Ahjo_Test_ID").ngroup()
    else:
        # If no ID column, treat the entire file as one program (group 0)
        df_cell["BM_Programm"] = 0
    df_cell["target"] = np.array(-1, dtype=str)

    # Fix Zustand values
    if df_cell["Zustand"].dtype == "object" or df_cell["Zustand"].dtype == "category":
        zustand_values = df_cell["Zustand"].astype(str).values
        if "PAUO**" in zustand_values:
            df_cell.loc[df_cell["Zustand"].astype(str) == "PAUO**", "Zustand"] = "PAUO"
        if "PAU**" in zustand_values:
            df_cell.loc[df_cell["Zustand"].astype(str) == "PAU**", "Zustand"] = "PAU"

    # Find columns with 'Floater' and "EIS" in their name
    floater_columns = [col for col in df_cell.columns if "Floater" in col]

    eis_columns = [col for col in df_cell.columns if "EIS" in col]

    # Create a mask for rows where any of these columns have values
    has_floater = df_cell[floater_columns].notna().any(axis=1)
    has_eis = (df_cell[eis_columns].notna().any(axis=1)) & (df_cell["AhAkku"].isna())

    # Set Zustand to FLOATER and EIS for these rows
    # Ensure Zustand is a categorical type with the new categories
    if df_cell.Zustand.dtype != "O":
        new_categories = df_cell.Zustand.dtype.categories.tolist() + ["FLOATER", "EIS"]
        new_dtype = pd.CategoricalDtype(
            categories=new_categories, ordered=df_cell.Zustand.dtype.ordered
        )
        df_cell["Zustand"] = df_cell["Zustand"].astype(new_dtype)

    df_cell.loc[has_floater, ["Zustand"]] = "FLOATER"
    df_cell.loc[has_eis, ["Zustand"]] = "EIS"

    # Find columns that contain both "Floater" and "Voltage" in their names
    floater_voltage_cols = [
        col for col in df_cell.columns if "Floater" in col and "Voltage" in col
    ]

    # Create mask for rows where any of these columns exceed V_max
    # (this happens when ppl forget to turn off the floaters...)
    if floater_voltage_cols:
        rows_to_drop = False
        for col in floater_voltage_cols:
            rows_to_drop = rows_to_drop | (df_cell[col] > V_max * 1.05)

        # Keep rows where the condition is False (logical negation)
        df_cell = df_cell[rows_to_drop == False]

    return df_cell


def dismember_raw_cell(
    cell,
    loadpath,
    MIN_ROWS,
    PAU_DURATION,
    V_max,
    procedure_filter=None,
    qocv_procedure_filter=None,
):

    bronze_columns = [
        "index",
        "Time",
        "Current",
        "Voltage",
        "Temperature",
        "Zustand",
        "Prozedur",
        "Duration_minutes",
        "ID",
        "BM_Programm",
        "Ah_throughput",
        "Label_Procedure",
        "Capacity_py",
        "target",
    ]

    print(f"Processing {cell}")
    # Check if any Prozedur values contain "jri_Aging"

    result = processing_procedure_filter(loadpath, procedure_filter)

    if result:

        df_cell = read_and_fix_format(loadpath, V_max)
        # Dismember df
        globals = DismemblerFunctions(MIN_ROWS, PAU_DURATION, qocv_procedure_filter)
        prefiltered_df = globals.prefiltering(df_cell, ["SAVE", "REST"])
        prefiltered_df = globals.add_ah_throughput(prefiltered_df)
        dismembered_df = globals.dismembling(prefiltered_df)
        dismembered_df = allocate_IDs(dismembered_df)

        # Reset index, filter columns, and save
        dismembered_df = dismembered_df.reset_index()
        available_bronze_cols = [
            col for col in bronze_columns if col in dismembered_df.columns
        ]
        dismembered_df = dismembered_df[available_bronze_cols]

    else:
        print("No matching procedure found.")
        return None

    return dismembered_df
