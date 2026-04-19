import numpy as np
from scipy import integrate


def add_ah_throughput(df_cell):
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
