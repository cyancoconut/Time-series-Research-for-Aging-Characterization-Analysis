from visualize import data_visualization
import rwth_colors
import pandas as pd
import numpy as np


def add_aging_labels(df_after_filter):
    visualizer = data_visualization.TestVisualization(rwth_colors)

    # label the procedures
    Label_CU_Capacity = (
        df_after_filter.groupby("BM_Programm", group_keys=False)
        .apply(
            lambda x: visualizer.update_cap_label(x, df_after_filter),
            include_groups=False,
        )
        .rename("Label_Procedure")
    )

    df_after_filter.update(Label_CU_Capacity)

    Label_CU_Pulse = (
        df_after_filter.groupby("BM_Programm", group_keys=False)
        .apply(
            lambda x: visualizer.update_pulse_label(x, df_after_filter),
            include_groups=False,
        )
        .rename("Label_Procedure")
    )
    df_after_filter.update(Label_CU_Pulse)
    df_after_filter.sort_values("index", inplace=True)

    # add the aging labels
    df_after_filter["Label_Procedure"] = visualizer.label_between_checkups(
        df_after_filter
    )
    df_after_filter["Label_Procedure"] = df_after_filter["Label_Procedure"].astype(str)


def preparing_schedule_overview(df_after_filter, Project_Schedule, cell):
    # get the schedule df
    df_schedule = df_after_filter[["Time", "Label_Procedure"]]
    df_schedule["Time"] = pd.to_datetime(df_schedule["Time"]).dt.tz_convert(
        "Europe/Berlin"
    )

    df_schedule["Name"] = cell
    df_schedule["Start"] = np.nan
    df_schedule["Start"] = pd.to_datetime(df_schedule["Start"]).dt.tz_localize(
        "Europe/Berlin"
    )

    df_schedule["End"] = np.nan
    df_schedule["End"] = pd.to_datetime(df_schedule["End"]).dt.tz_localize(
        "Europe/Berlin"
    )

    # First copy Time to Start where Label_Procedure is not NaN
    mask = df_schedule["Label_Procedure"].notna()
    df_schedule.loc[mask, "Start"] = df_schedule.loc[mask, "Time"]

    # Then get the last Time entry for each Label_Procedure group
    df_schedule.loc[mask, "End"] = df_schedule.groupby("Label_Procedure")[
        "Time"
    ].transform("last")

    df_schedule = df_schedule.groupby("Label_Procedure").head(1)
    Project_Schedule = pd.concat([Project_Schedule, df_schedule])

    return Project_Schedule
    # plot_gantt(Project_Schedule)
