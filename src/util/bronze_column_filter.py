import os
import duckdb
import pandas as pd
import numpy as np


def bronze_column_filter(path, cell):
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
        "Pulse_py",
        "target",
    ]

    columns_str = ", ".join(bronze_columns)

    try:
        con = duckdb.connect()
        con.execute(
            "CREATE VIEW parquet_data AS SELECT * FROM read_parquet('{}')".format(path)
        )
        dismembered_df = con.execute(
            f"SELECT {columns_str} FROM parquet_data"
        ).fetch_df()
        if "target" not in dismembered_df.columns:
            dismembered_df["target"] = np.array(-1, dtype=object)

    except Exception as e:
        dismembered_df = pd.DataFrame()
        print(f"Failed to read: {cell}: {e}. Removing file.")
        os.remove(path)

    return dismembered_df
