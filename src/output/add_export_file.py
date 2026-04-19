import duckdb


def add_export_file(fullpath, cell, df_results_cap, df_results_pulse, exception_dict):
    """
    Process capacity and pulse results from the filtered dataframe.

    Parameters:
        df_after_filter (DataFrame): The processed and filtered dataframe
        cell_name (str): Name of the cell being processed
        df_results_cap (DataFrame): Existing capacity results dataframe
        df_results_pulse (DataFrame): Existing pulse results dataframe
        exception_dict (list): List to track cells with issues
        cell (str): Cell identifier for error messages

    Returns:
        tuple: (df_results_cap, df_results_pulse, exception_dict, count)
    """
    import pandas as pd

    count = 0

    # Process capacity results
    con = duckdb.connect()

    # First query to get df_cap
    query1 = """
    WITH ranked_data AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ID ORDER BY (SELECT 0)) as row_num
        FROM read_parquet('{}')
        WHERE Capacity_py IS NOT NULL
    )
    SELECT * EXCLUDE (row_num)
    FROM ranked_data
    WHERE row_num = 1
    """.format(
        fullpath
    )

    df_cap = con.execute(query1).fetchdf()
    df_cap = df_cap.sort_values("Time")

    query2 = """
    WITH ranked_data AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ID ORDER BY (SELECT 0)) as row_num
        FROM read_parquet('{}')
        WHERE Pulse_py IS NOT NULL
    )
    SELECT * EXCLUDE (row_num)
    FROM ranked_data
    WHERE row_num = 1
    """.format(
        fullpath
    )

    df_pulse = con.execute(query2).fetchdf()
    df_pulse = df_pulse.sort_values("Time")

    # Second query to get unique Prozedur values
    query3 = """
    SELECT DISTINCT Prozedur 
    FROM read_parquet('{}')
    WHERE Prozedur IS NOT NULL
    """.format(
        fullpath
    )

    # Execute the query and fetch all results
    procedures = con.execute(query3).fetchall()

    # Convert to a Python list if needed
    prozedur_list = [row[0] for row in procedures]

    if df_cap.empty:
        exception_dict[cell] = prozedur_list
        print(f"Warning: No capacity data found for {cell}")
    else:
        df_cap = df_cap[["Time", "Ah_throughput", "Capacity_py", "Label_Procedure"]]
        df_cap["Time"] = df_cap["Time"].dt.tz_localize(None)
        df_cap["Name"] = cell
        df_cap["Procedures"] = [prozedur_list] * len(df_cap)

        # Safe concatenation
        try:
            if df_results_cap.empty:
                df_results_cap = df_cap.copy()
            else:
                df_results_cap = pd.concat([df_results_cap, df_cap])
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                print(f"Warning: Could not concatenate capacity results for {cell}")
                if not df_cap.empty:
                    df_results_cap = df_cap  # If df_results_cap was empty
            else:
                raise

    # Process pulse results
    if df_pulse.empty:
        print(f"Warning: No pulse data found for {cell}")
    else:
        df_pulse = df_pulse[
            ["Time", "Ah_throughput", "Current", "Pulse_py", "Label_Procedure"]
        ]
        df_pulse["Time"] = df_pulse["Time"].dt.tz_localize(None)
        df_pulse["Name"] = cell
        df_pulse["Procedures"] = [prozedur_list] * len(df_pulse)

        # Safe concatenation
        try:
            if df_results_pulse.empty:
                df_results_pulse = df_pulse.copy()
            else:
                df_results_pulse = pd.concat([df_results_pulse, df_pulse])
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                print(f"Warning: Could not concatenate pulse results for {cell}")
                if not df_pulse.empty:
                    df_results_pulse = df_pulse  # If df_results_pulse was empty
            else:
                raise

    # Increment count if we processed at least one result
    if not df_cap.empty or not df_pulse.empty:
        count = 1

    # return df_results_cap, df_results_pulse, exception_dict, count

    return df_results_cap, df_results_pulse, exception_dict, count


import duckdb


def add_export_file_cap(fullpath, cell, df_results_cap, exception_dict):
    """
    Process capacity and pulse results from the filtered dataframe.

    Parameters:
        df_after_filter (DataFrame): The processed and filtered dataframe
        cell_name (str): Name of the cell being processed
        df_results_cap (DataFrame): Existing capacity results dataframe
        df_results_pulse (DataFrame): Existing pulse results dataframe
        exception_dict (list): List to track cells with issues
        cell (str): Cell identifier for error messages

    Returns:
        tuple: (df_results_cap, df_results_pulse, exception_dict, count)
    """
    import pandas as pd

    count = 0

    # Process capacity results
    con = duckdb.connect()

    # First query to get df_cap
    query1 = """
    WITH ranked_data AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ID ORDER BY (SELECT 0)) as row_num
        FROM read_parquet('{}')
        WHERE Capacity_py IS NOT NULL
    )
    SELECT * EXCLUDE (row_num)
    FROM ranked_data
    WHERE row_num = 1
    """.format(
        fullpath
    )

    df_cap = con.execute(query1).fetchdf()
    df_cap = df_cap.sort_values("Time")

    # Second query to get unique Prozedur values
    query3 = """
    SELECT DISTINCT Prozedur 
    FROM read_parquet('{}')
    WHERE Prozedur IS NOT NULL
    """.format(
        fullpath
    )

    # Execute the query and fetch all results
    procedures = con.execute(query3).fetchall()

    # Convert to a Python list if needed
    prozedur_list = [row[0] for row in procedures]

    if df_cap.empty:
        exception_dict[cell] = prozedur_list
        print(f"Warning: No capacity data found for {cell}")
    else:
        df_cap = df_cap[["Time", "Ah_throughput", "Capacity_py", "Label_Procedure"]]
        df_cap["Time"] = df_cap["Time"].dt.tz_localize(None)
        df_cap["Name"] = cell
        df_cap["Procedures"] = [prozedur_list] * len(df_cap)

        # Safe concatenation
        try:
            if df_results_cap.empty:
                df_results_cap = df_cap.copy()
            else:
                df_results_cap = pd.concat([df_results_cap, df_cap])
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                print(f"Warning: Could not concatenate capacity results for {cell}")
                if not df_cap.empty:
                    df_results_cap = df_cap  # If df_results_cap was empty
            else:
                raise

    # Increment count if we processed at least one result
    if not df_cap.empty:
        count = 1

    # return df_results_cap, df_results_pulse, exception_dict, count

    return df_results_cap, exception_dict, count


def add_export_file_cap_s3(
    cell,
    df_results_cap,
    exception_dict,
    minio_endpoint,
    access_key,
    secret_key,
    minio_path,
):
    """
    Process capacity and pulse results from the filtered dataframe.

    Parameters:
        df_after_filter (DataFrame): The processed and filtered dataframe
        cell_name (str): Name of the cell being processed
        df_results_cap (DataFrame): Existing capacity results dataframe
        df_results_pulse (DataFrame): Existing pulse results dataframe
        exception_dict (list): List to track cells with issues
        cell (str): Cell identifier for error messages

    Returns:
        tuple: (df_results_cap, df_results_pulse, exception_dict, count)
    """
    import pandas as pd

    count = 0

    # Process capacity results
    con = duckdb.connect()

    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(
        f"""
        SET s3_endpoint='{minio_endpoint}';
        SET s3_access_key_id='{access_key}';
        SET s3_secret_access_key='{secret_key}';
        SET s3_use_ssl=True;
        SET s3_url_style='path';
    """
    )

    # First query to get df_cap
    query1 = """
    WITH ranked_data AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ID ORDER BY (SELECT 0)) as row_num
        FROM read_parquet(?)
        WHERE Capacity_py IS NOT NULL
    )
    SELECT * EXCLUDE (row_num)
    FROM ranked_data
    WHERE row_num = 1
    """

    df_cap = con.execute(query1, [minio_path]).fetchdf()
    df_cap = df_cap.sort_values("Time")

    # Second query to get unique Prozedur values
    query3 = """
    SELECT DISTINCT Prozedur 
    FROM read_parquet(?)
    WHERE Prozedur IS NOT NULL
    """
    # Execute the query and fetch all results
    procedures = con.execute(query3, [minio_path]).fetchall()

    # Convert to a Python list if needed
    prozedur_list = [row[0] for row in procedures]

    if df_cap.empty:
        exception_dict[cell] = prozedur_list
        print(f"Warning: No capacity data found for {cell}")
    else:
        df_cap = df_cap[["Time", "Ah_throughput", "Capacity_py", "Label_Procedure"]]
        df_cap["Time"] = df_cap["Time"].dt.tz_localize(None)
        df_cap["Name"] = cell
        df_cap["Procedures"] = [prozedur_list] * len(df_cap)

        # Safe concatenation
        try:
            if df_results_cap.empty:
                df_results_cap = df_cap.copy()
            else:
                df_results_cap = pd.concat([df_results_cap, df_cap])
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                print(f"Warning: Could not concatenate capacity results for {cell}")
                if not df_cap.empty:
                    df_results_cap = df_cap  # If df_results_cap was empty
            else:
                raise

    # Increment count if we processed at least one result
    if not df_cap.empty:
        count = 1

    # return df_results_cap, df_results_pulse, exception_dict, count

    return df_results_cap, exception_dict, count
