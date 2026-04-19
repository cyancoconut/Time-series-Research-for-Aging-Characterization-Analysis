import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TestVisualization:
    def __init__(self, rwth_colors):
        """
        Initialize the BatteryAnalyzer class.

        Parameters:S
        -----------
        rwth_colors : module
            Imported rwth_colors module containing the RWTH color palette.
        """
        # Store the rwth_colors module
        self.rwth_colors = rwth_colors.colors

        # Create color cycler using the imported rwth_colors
        self.color_cycler = cycle(
            [
                rwth_colors.colors[("blue", 100)],
                rwth_colors.colors[("black", 100)],
                rwth_colors.colors[("magenta", 100)],
                rwth_colors.colors[("yellow", 100)],
                rwth_colors.colors[("green", 100)],
                rwth_colors.colors[("bordeaux", 100)],
                rwth_colors.colors[("orange", 100)],
                rwth_colors.colors[("turqoise", 100)],
                rwth_colors.colors[("darkred", 100)],
                rwth_colors.colors[("lime", 100)],
                rwth_colors.colors[("petrol", 100)],
                rwth_colors.colors[("lavender", 100)],
                rwth_colors.colors[("red", 100)],
                rwth_colors.colors[("blue", 50)],
                rwth_colors.colors[("black", 50)],
                rwth_colors.colors[("magenta", 50)],
                rwth_colors.colors[("yellow", 50)],
                rwth_colors.colors[("green", 50)],
                rwth_colors.colors[("bordeaux", 50)],
                rwth_colors.colors[("orange", 50)],
                rwth_colors.colors[("turqoise", 50)],
                rwth_colors.colors[("darkred", 50)],
                rwth_colors.colors[("lime", 50)],
                rwth_colors.colors[("petrol", 50)],
                rwth_colors.colors[("lavender", 50)],
                rwth_colors.colors[("red", 50)],
            ]
        )

    def plot_capacity(self, df_result):
        """
        Plot capacity progress during battery aging.

        Parameters:
        -----------
        df_result : DataFrame
            DataFrame containing capacity measurements with Time and Capacity_py columns.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.grid(True)

        # Reset color cycler
        self.color_cycler = cycle(self.color_cycler)

        for pulse, group in df_result.groupby("Capacity_py"):
            ax.scatter(group.Time.head(1), pulse, color=next(self.color_cycler))

        plt.xlabel("Time as yyyy-mm")
        plt.ylabel("Capacity in Ah")
        plt.title("Capacity progress during aging")

        plt.show()

    def plot_pulse_dch(self, df_result):
        """
        Plot discharge pulse resistance progress during battery aging.

        Parameters:
        -----------
        df_result : DataFrame
            DataFrame containing pulse measurements with Time, Pulse_py, and ID columns.
        """
        discharge_resistance = df_result[df_result["Pulse_py"] < 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.grid(True)

        # Reset color cycler
        self.color_cycler = cycle(self.color_cycler)

        for pulse, group in discharge_resistance.groupby("ID"):
            ax.scatter(
                group.Time.head(1), group.Pulse_py.max(), color=next(self.color_cycler)
            )

        plt.xlabel("Time as yyyy-mm")
        plt.ylabel("Resistance in Ohm")
        plt.title("Discharge Pulseresistance progress during aging")

        plt.show()

    def plot_pulse_cha(self, df_result):
        """
        Plot charge pulse resistance progress during battery aging.

        Parameters:
        -----------
        df_result : DataFrame
            DataFrame containing pulse measurements with Time, Pulse_py, and ID columns.
        """
        charge_resistance = df_result[df_result["Pulse_py"] > 0]
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.grid(True)

        # Reset color cycler
        self.color_cycler = cycle(self.color_cycler)

        for pulse, group in charge_resistance.groupby("ID"):
            ax.scatter(
                group.Time.head(1), group.Pulse_py.max(), color=next(self.color_cycler)
            )

        plt.xlabel("Time as yyyy-mm")
        plt.ylabel("Resistance in Ohm")
        plt.title("Charge Pulseresistance progress during aging")

        plt.show()

    def update_cap_label(self, group, df_final):
        """
        Update capacity labels for checkup groups.

        Parameters:
        -----------
        group : DataFrame group
            GroupBy object containing capacity data.
        df_final : DataFrame
            Complete DataFrame with capacity measurements.

        Returns:
        --------
        Series
            Updated label series.
        """
        if group["Capacity_py"].notna().any():
            unique_programs = sorted(
                df_final[df_final["Capacity_py"].notna()]["BM_Programm"].unique()
            )
            counter = unique_programs.index(group.name)
            return pd.Series(
                ["Checkup_" + str(counter)] * len(group),
                index=group.index,
                dtype="string",
            )
        return group["Label_Procedure"]

    def update_pulse_label(self, group, df_final):
        """
        Update pulse labels for measurements.

        Parameters:
        -----------
        group : DataFrame group
            GroupBy object containing pulse data.
        df_final : DataFrame
            Complete DataFrame with pulse measurements.

        Returns:
        --------
        Series
            Updated label series.
        """
        if group["Pulse_py"].notna().any():
            filtered_df = df_final[df_final["Label_Procedure"].notna()]
            if group.name in filtered_df["BM_Programm"].unique():
                print("Pulse already in Check-Up")
            elif (group.name - 1) in filtered_df["BM_Programm"].unique():
                print(f"Pulse in program after CAP_test: {group.name}")
                mask = filtered_df["BM_Programm"] == group.name - 1
                label_value = filtered_df.loc[mask, "Label_Procedure"].iloc[0]
                print(label_value)
                return pd.Series([label_value] * len(group), index=group.index)
        return group["Label_Procedure"]

    def label_between_checkups(self, df):
        """
        Label rows between checkups as aging periods.

        Parameters:
        -----------
        df : DataFrame
            DataFrame containing Label_Procedure column.

        Returns:
        --------
        Series
            Updated Label_Procedure series.
        """
        # Create groups based on cumulative count of Checkups
        groups = df["Label_Procedure"].notna().cumsum()

        # Get aging numbers from Checkups
        aging_nums = (
            df[df["Label_Procedure"].str.contains("Checkup", na=False)][
                "Label_Procedure"
            ]
            .str.split("_")
            .str[1]
        )
        aging_map = dict(zip(groups[aging_nums.index], aging_nums))

        # Apply aging labels only to NaN rows
        mask = df["Label_Procedure"].isna()
        df.loc[mask, "Label_Procedure"] = groups[mask].map(
            lambda x: f"Aging_{aging_map.get(x)}"
        )

        return df["Label_Procedure"]


def plot_gantt(df, rwth_colors):
    """
    Create a Gantt chart to visualize battery procedures over time.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing scheduling information with Name, Start, End, and Label_Procedure columns.
    """
    print(df["Name"].unique())

    # Fill NaN values in Name
    df["Name"] = df["Name"].fillna(method="ffill")
    names = df["Name"].unique()

    # Create gantt data for all names
    df_gantt = pd.concat(
        [
            pd.DataFrame(
                {
                    "Task": name,
                    "Start": df[df["Name"] == name]["Start"],
                    "Finish": df[df["Name"] == name]["End"],
                    "Resource": df[df["Name"] == name]["Label_Procedure"],
                }
            )
            for name in names
        ]
    )
    # Create color dictionary for all procedures
    colors_dict = {
        procedure: (
            rwth_colors["orange"] if "Checkup" in procedure else rwth_colors["blue"]
        )
        for procedure in df_gantt["Resource"].unique()
    }

    fig = ff.create_gantt(
        df_gantt,
        index_col="Resource",
        colors=colors_dict,
        show_colorbar=False,
        showgrid_x=True,
        showgrid_y=True,
        group_tasks=True,
    )

    fig.show()


# Function to process data for a specific temperature with a specific color
def add_temperature_data(fig, df_temp, color, temp, row, col):
    for soc in df_temp["SOC"].unique():
        df_soc = df_temp[df_temp["SOC"] == soc]

        for name_prefix in df_soc["Name_prefix"].unique():
            df_name = df_soc[df_soc["Name_prefix"] == name_prefix]

            for c_rate in df_name["C_Rate"].unique():
                df_subset = df_name[df_name["C_Rate"] == c_rate]
                df_subset = df_subset.sort_values("Ah_throughput")

                # Determine marker symbol based on C_Rate
                if c_rate == 0.5:
                    marker_symbol = "circle"
                elif c_rate == 1.0:
                    marker_symbol = "square"
                else:
                    marker_symbol = "diamond"

                fig.add_trace(
                    go.Scatter(
                        x=df_subset["EFC"],
                        y=df_subset["Capacity_py"],
                        mode="lines+markers",
                        name=f"T={temp}°C, SOC={soc}, C={c_rate}",
                        line=dict(color=color),
                        marker=dict(color=color, symbol=marker_symbol, size=8),
                        hovertemplate="<b>Name:</b> %{text}<br>"
                        + "<b>Ah_throughput:</b> %{x:.2f}<br>"
                        + "<b>Capacity_py:</b> %{y:.2f}<br>"
                        + "<b>Temperature:</b> "
                        + str(temp)
                        + "°C<br>"
                        + "<b>SOC:</b> "
                        + str(soc)
                        + "<br>"
                        + "<b>C_Rate:</b> "
                        + str(c_rate)
                        + "<br>"
                        + "<b>Time:</b> %{customdata}<extra></extra>",
                        text=df_subset["Name_prefix"],
                        customdata=df_subset["Time"],
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )


def plot_plotly(df, rwth_colors, write_image, type_cell, C_nom):

    # Original code for Name_prefix
    df_results_cap = df.copy()
    df_results_cap["Name_prefix"] = df_results_cap["Name"].apply(
        lambda x: x.split("-")[0]
    )

    df_results_cap["Capacity_py"] = df_results_cap["Capacity_py"] / C_nom
    df_results_cap["EFC"] = df_results_cap["Ah_throughput"] / C_nom

    # Create a 3x2 subplot layout
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "DOD = 20%",
            "DOD = 40%",
            "DOD = 60%",
            "DOD = 80%",
            "DOD = 100%",
            "",
        ],
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
        specs=[[{}, {}], [{}, {}], [{}, None]],  # Last row has only one subplot
    )

    # Define subplot positions for each DOD
    subplot_positions = {20: (1, 1), 40: (1, 2), 60: (2, 1), 80: (2, 2), 100: (3, 1)}

    # Process each DOD value
    for dod in [20, 40, 60, 80, 100]:
        row, col = subplot_positions[dod]

        # Filter data for all three temperatures and this DOD
        df_filtered = df_results_cap[
            (df_results_cap["DOD"] == dod)
            & (df_results_cap["Temperature"].isin([15, 25, 35, 45]))
        ]

        # Process T=15 data (blue)
        df_15 = df_filtered[df_filtered["Temperature"] == 15].sort_values(
            by=["SOC", "EFC"]
        )
        add_temperature_data(fig, df_15, "lavender", 15, row, col)

        # Process T=35 data (red)
        df_35 = df_filtered[df_filtered["Temperature"] == 35].sort_values(
            by=["SOC", "EFC"]
        )
        add_temperature_data(fig, df_35, "lavenderblush", 35, row, col)

        # Process T=45 data (red)
        df_45 = df_filtered[df_filtered["Temperature"] == 45].sort_values(
            by=["SOC", "EFC"]
        )
        add_temperature_data(fig, df_45, "mistyrose", 45, row, col)

        # Process T=25 data with original Plotly Express approach, but with fixed colors for SOC
        df_25 = df_filtered[df_filtered["Temperature"] == 25].sort_values(
            by=["SOC", "EFC"]
        )

        # Define a colormap for SOC values
        soc_colors = {
            10: rwth_colors["blue"],
            30: rwth_colors["turqoise"],
            50: rwth_colors["orange"],
            70: rwth_colors["red"],
            90: rwth_colors["darkred"],
        }

        # Process each SOC value separately with its fixed color
        for soc in df_25["SOC"].unique():
            df_soc = df_25[df_25["SOC"] == soc]
            fig_25 = px.line(
                df_soc,
                x="EFC",
                y="Capacity_py",
                color_discrete_map={str(soc): soc_colors.get(soc, "#000000")},
                markers=True,
                line_group="Name_prefix",
                symbol="C_Rate",
                hover_data=["SOC", "DOD", "C_Rate", "Time", "Temperature"],
            )

            # Add traces to main figure
            for trace in fig_25.data:
                trace.name = f"T=25°C, SOC={soc}"
                trace.line.color = soc_colors.get(soc, "#000000")
                fig.add_trace(trace, row=row, col=col)

    # Update subplot axes labels
    for dod in [20, 40, 60, 80, 100]:
        row, col = subplot_positions[dod]
        fig.update_xaxes(title_text="Equivalent Full Cycle", row=row, col=col)
        fig.update_yaxes(title_text="Capacity", row=row, col=col)

    # Update overall layout
    fig.update_layout(
        title="Battery Capacity vs Equivalent Full Cycle at Different DOD Values",
        legend_title="SOC/C_Rate (Temperature Color Coding)",
        width=1500,
        height=1050,  # Slightly increased height to accommodate the new subplot
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Find the global min and max for both axes
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for dod in [20, 40, 60, 80, 100]:
        df_dod = df_results_cap[df_results_cap["DOD"] == dod]

        if not df_dod.empty:
            x_min = min(x_min, df_dod["EFC"].min())
            x_max = max(x_max, df_dod["EFC"].max())
            y_min = min(y_min, df_dod["Capacity_py"].min())
            y_max = max(y_max, df_dod["Capacity_py"].max())

    # Add some padding to the ranges
    x_range = [0, x_max * 1.01]
    y_range = [y_min * 0.99, y_max * 1.01]
    # Update grid lines for all subplots
    for dod in [20, 40, 60, 80, 100]:
        row, col = subplot_positions[dod]
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgrey",
            row=row,
            col=col,
            range=x_range,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgrey",
            row=row,
            col=col,
            range=y_range,
        )

    # Show plot
    fig.show()

    if write_image:
        fig.write_html(
            r"Z:\Forschung\ogP\J8005_BMWK_METABatt\Daten"
            + "/plot_"
            + type_cell
            + ".html"
        )
    return fig
