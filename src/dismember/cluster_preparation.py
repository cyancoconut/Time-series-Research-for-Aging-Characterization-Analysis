import pandas as pd
import dask.dataframe as dd

from util.add_ah_throughput import add_ah_throughput


class DismemblerFunctions:

    def __init__(self, MIN_ROWS, PAU_DURATION, QOCV_PROCEDURE_FILTER=None):

        self.MIN_ROWS = MIN_ROWS
        self.PAU_DURATION = PAU_DURATION
        # Optional substring matched against Prozedur. For procedures whose
        # discharge and charge halves share a single Prozedur (e.g. qOCV),
        # neither the Prozedur-change nor the long-PAU rule splits them, so the
        # halves collapse into one segment whose mean current cancels to ~0.
        # When set, every Zustand change inside a matching procedure also fires
        # a segment boundary so DCH and CHA become separate segments. None ->
        # behaviour is unchanged.
        self.QOCV_PROCEDURE_FILTER = QOCV_PROCEDURE_FILTER

    def prefiltering(self, df_cell, drop_columns):
        df_cell.drop_duplicates(subset=["Time", "Zustand"], inplace=True)
        df_cell = df_cell.loc[~df_cell["Zustand"].isin(drop_columns)].copy()
        return df_cell

    def add_ah_throughput(self, df_cell):
        # BRONZE_CU now carries Ah_throughput from the build step (full timeline).
        # Only recompute if it's missing (e.g. legacy parquet without the
        # column). Delegate to the shared util so the gap-masking (no phantom Ah
        # over dead time between files) is applied here too.
        return add_ah_throughput(df_cell)

    def dismembling(self, df_cell):
        processed_dfs = []
        # Define the columns that represent PAU states
        PAU_Columns = ["PAU", "PAUO", "..."]

        # Within each BM_Programm...
        for programm_name, programm_df in df_cell.groupby("BM_Programm"):
            # Cyclic-data stubs: non-CU (aging) files are stored as just their
            # first+last row (1–2 rows), carrying the aging procedure name in
            # Prozedur. Keep these tiny programms so that name reaches GOLD, but
            # skip segmentation and exclude them from clustering — route them to
            # the discard bucket and pre-label them "AGING" (target != -1).
            # Without this they hit the <MIN_ROWS drop below and the aging name
            # is lost (it only reached GOLD before when a stub incidentally
            # shared an Ahjo_Test_ID group large enough to clear MIN_ROWS).
            if len(programm_df) <= 2:
                programm_df = programm_df.copy()
                programm_df["BM_Programm_procedure"] = 0
                programm_df["pre_target"] = "AGING"
                processed_dfs.append(programm_df)
                continue

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

            # Within a qOCV procedure the discharge and charge halves share one
            # Prozedur and are only separated by a sub-threshold PAU, so neither
            # the Prozedur-change nor the long-PAU rule splits them. Fire a
            # boundary on every Zustand change for rows whose Prozedur matches
            # QOCV_PROCEDURE_FILTER. ZUSTAND_group already increments on each
            # Zustand change, so a group change == a Zustand change. Inert
            # (all-False) when the filter is unset or no Prozedur matches.
            if self.QOCV_PROCEDURE_FILTER:
                qocv_zustand_boundary = programm_df["Prozedur"].str.contains(
                    self.QOCV_PROCEDURE_FILTER, na=False
                ) & (
                    programm_df["ZUSTAND_group"]
                    != programm_df["ZUSTAND_group"].shift()
                )
            else:
                qocv_zustand_boundary = pd.Series(False, index=programm_df.index)

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
                |
                # OR Zustand change inside a qOCV procedure
                qocv_zustand_boundary
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

            # Pre-label PAU procedures NOW, before the too-short discard sweep below
            # contaminates the discard bucket (BM_Programm_procedure == 0) with non-PAU
            # rows from procedures that fail the min_rows check. At this moment bucket-0
            # holds only PAU/PAUO rows.
            pure_pau_proc = programm_df.groupby("BM_Programm_procedure")[
                "Zustand"
            ].transform(lambda x: x.isin(PAU_Columns).all())
            if "pre_target" not in programm_df.columns:
                programm_df["pre_target"] = pd.NA
            programm_df.loc[pure_pau_proc, "pre_target"] = "PAU"

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

    # Add target and prelabel the EIS-Parts.
    # Cast to object so later string assignments (e.g. "PAU") don't trip the
    # pandas FutureWarning about setting incompatible dtypes when the lambda
    # only returns ints and the column is inferred as int64.
    result_df["target"] = result_df.groupby("ID")["Prozedur"].transform(
        lambda x: "EIS" if x.isna().any() else -1
    ).astype(object)

    # Propagate dismember-time pre-labels to every row of the affected IDs.
    # `pre_target` is set in DismembererFunctions.dismember():
    #   PAU   — long-pause stubs (set after the PAU group handling, before the
    #           too-short discard sweep, so bucket-0 holds only PAU rows then)
    #   AGING — cyclic-data stubs (kept whole programms of ≤2 rows)
    # Promoting per-ID here ensures contaminating rows later dumped into the same
    # ID (typically <BM>_0) still carry the label, so the discard bucket is
    # excluded from clustering wholesale.
    if "pre_target" in result_df.columns:
        for label in ("PAU", "AGING"):
            id_mask = result_df.groupby("ID")["pre_target"].transform(
                lambda x, lbl=label: (x == lbl).any()
            )
            result_df.loc[id_mask, "target"] = label
        result_df = result_df.drop(columns=["pre_target"])

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
