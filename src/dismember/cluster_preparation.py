import pandas as pd
import dask.dataframe as dd
import numpy as np
from scipy import integrate


class DismemblerFunctions:

    def __init__(self, MIN_ROWS, PAU_DURATION):

        self.MIN_ROWS = MIN_ROWS
        self.PAU_DURATION = PAU_DURATION

    def prefiltering(self, df_cell, drop_columns):
        df_cell.drop_duplicates(subset=["Time", "Zustand"], inplace=True)
        df_cell = df_cell.loc[~df_cell["Zustand"].isin(drop_columns)].copy()
        return df_cell

    def add_ah_throughput(self, df_cell):
        if hasattr(df_cell, "Ah_throughput"):
            return df_cell
        else:
            # Calculate time difference in hours
            time_h = df_cell["Time_UTC"].diff().dt.total_seconds()
            time_h = np.cumsum(time_h).values / 3600
            time_h[0] = 0

            # Calculate cumulative Ah throughput using the absolute current values
            AhThroughput = integrate.cumulative_trapezoid(
                abs(df_cell["Current"].values), x=time_h, initial=0
            )

            # Add the calculated values as a new column to your dataframe
            df_cell["Ah_throughput"] = AhThroughput
            return df_cell

    def dismembling(self, df_cell):
        processed_dfs = []
        # Define the columns that represent PAU states
        PAU_Columns = ["PAU", "PAUO", "..."]

        # Within each BM_Programm...
        for programm_name, programm_df in df_cell.groupby("BM_Programm"):
            # Check if the Programm is empty or too short
            if len(programm_df) == 1:
                programm_df["BM_Programm_procedure"] = 0
                processed_dfs.append(programm_df)

            if programm_df.empty or len(programm_df) < self.MIN_ROWS:
                print(
                    f"Skipping BM-Programm with {len(programm_df)} rows (too short or empty)"
                )
                continue
            # Check if the Programm contains only PAU states
            if (programm_df["Zustand"].isin(PAU_Columns)).all():
                print("Skipping BM-Programms with only PAUs")
                continue

            # Create a group for each different Zustand
            programm_df["ZUSTAND_group"] = (
                programm_df["Zustand"] != programm_df["Zustand"].shift()
            ).cumsum()

            # Calculate the duration of each Zustand
            zustand_informations = (
                programm_df.groupby("ZUSTAND_group")
                .agg(
                    {
                        "Zustand": "first",
                        "Time_UTC": lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds()
                        / 60,  # Duration in minutes
                    }
                )
                .reset_index()
            )
            zustand_informations.columns = [
                "ZUSTAND_group",
                "Zustand",
                "ZUSTAND_Duration_minutes",
            ]

            # Merge the durations back to the original dataframe
            programm_df = programm_df.merge(
                zustand_informations[["ZUSTAND_group", "ZUSTAND_Duration_minutes"]],
                on="ZUSTAND_group",
                how="left",
            )

            # Relabel PAU rows whose procedure contains "_EIS_" as EIS
            mask_pau = programm_df["Zustand"].isin(PAU_Columns)
            mask_eis = programm_df["Prozedur"].str.contains("_EIS_", na=False)
            programm_df.loc[mask_pau & mask_eis, ["Zustand"]] = "EIS"
            # Recompute after relabeling so EIS rows are excluded
            mask_pau = programm_df["Zustand"].isin(PAU_Columns)

            # Fire a new procedure start when exiting a long PAU into the next Zustand group
            prev_group_was_long_pau = (
                (programm_df["ZUSTAND_group"] != programm_df["ZUSTAND_group"].shift())
                & programm_df["Zustand"].shift().isin(PAU_Columns)
                & (programm_df["ZUSTAND_Duration_minutes"].shift() > self.PAU_DURATION)
            )

            # New procedure condition: PAU state + long duration + (group change OR first row)
            programm_df["new_procedure_start"] = (
                (
                    programm_df["Zustand"].isin(PAU_Columns)
                    & (programm_df["ZUSTAND_Duration_minutes"] > self.PAU_DURATION)
                    & (
                        (
                            programm_df["ZUSTAND_group"]
                            != programm_df["ZUSTAND_group"].shift()
                        )
                        | (programm_df.index == 0)
                    )
                )
                | prev_group_was_long_pau
                |
                # OR procedure change
                (programm_df["Prozedur"] != programm_df["Prozedur"].shift())
            )

            # Cumulative sum to create unique procedure identifiers
            programm_df["BM_Programm_procedure"] = programm_df[
                "new_procedure_start"
            ].cumsum()

            print(f"We are at the {programm_name} th Programm")

            # For non-EIS PAU rows: keep first+last rows of long pauses, zero everything else
            for _, pau_group in programm_df[mask_pau].groupby("ZUSTAND_group"):
                duration = pau_group["ZUSTAND_Duration_minutes"].iloc[0]
                if duration <= self.PAU_DURATION:
                    programm_df.loc[pau_group.index, "BM_Programm_procedure"] = 0
                else:
                    programm_df.loc[pau_group.index[1:-1], "BM_Programm_procedure"] = 0

            # Check if the procedure is too short (PAU stubs are exempt)
            for df_name, df_procedure in programm_df.groupby("BM_Programm_procedure"):
                if df_procedure["Zustand"].isin(PAU_Columns).all():
                    continue
                if (df_procedure.shape[0] < self.MIN_ROWS) and (
                    not (df_procedure["Zustand"] == "EIS").any()
                ):
                    print(
                        f"Skipping procedures {df_name} with {df_procedure.shape[0]} rows (too short)"
                    )
                    print(f"First index value: {df_procedure.index[0]}")

                    programm_df.loc[
                        (programm_df["BM_Programm_procedure"] == df_name),
                        "BM_Programm_procedure",
                    ] = 0

            # Drop the temporary columns used for grouping
            programm_df.drop(
                ["new_procedure_start", "ZUSTAND_group"], axis=1, inplace=True
            )
            processed_dfs.append(programm_df)

        # Concatenate all processed dataframes
        try:
            dismembered_df = pd.concat(processed_dfs, ignore_index=True)
        except ValueError as e:
            if str(e) == "No objects to concatenate.":
                # Create an empty DataFrame
                dismembered_df = pd.DataFrame()  # Or with specific columns if needed
                print("Warning: processed_dfs was empty")
            else:
                # Re-raise if it's a different ValueError
                raise

        return dismembered_df


def allocate_IDs(result_df, start_date=None, end_date=None):
    # delete the testruns between start_date and end_date
    if start_date:
        delete_invalid_mask = (result_df["Time"] > start_date) & (
            result_df["Time"] <= end_date
        )
        result_df = result_df[~delete_invalid_mask]

    # generate ID
    result_df["ID"] = (
        result_df["BM_Programm"].astype(str)
        + "_"
        + result_df["BM_Programm_procedure"].astype(str)
    )

    # Add target and prelabel the EIS-Parts
    result_df["target"] = result_df.groupby("ID")["Prozedur"].transform(
        lambda x: "EIS" if x.isna().any() else -1
    )

    # Tag PAU stubs (procedures where every row is a PAU/PAUO/... state)
    PAU_Columns = ["PAU", "PAUO", "..."]
    pau_stub_mask = result_df.groupby("ID")["Zustand"].transform(
        lambda x: x.isin(PAU_Columns).all()
    )
    result_df.loc[pau_stub_mask, "target"] = "PAU"

    # Calculate the duration of each ID
    durations = (
        result_df.groupby("ID")
        .agg(
            {
                "Time_UTC": lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds()
                / 60  # Duration in minutes
            }
        )
        .reset_index()
    )
    durations.columns = ["ID", "Duration_minutes"]

    result_df = result_df.merge(
        durations[["ID", "Duration_minutes"]], on="ID", how="left"
    )

    return result_df
