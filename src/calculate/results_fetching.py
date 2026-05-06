# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets as widgets


class calculation:
    def __init__(
        self,
        qOCV_CRate,
        Nom_Capacity,
        target_pulse_duration,
        pulse_type,
        pulse_target_unit,
        df,
        qocv_current_tolerance=0.01,
        restore_current_tolerance=0.05,
        pulse_duration_tolerance=1.08,
    ):

        self.qOCV_CRate = qOCV_CRate
        self.Nom_Capacity = Nom_Capacity
        self.target_pulse_duration = target_pulse_duration
        self.pulse_type = pulse_type  # 1 = single pulse, 2 = consecutive double pulse
        self.pulse_target_unit = pulse_target_unit
        self.df = df
        self.qocv_current_tolerance = qocv_current_tolerance
        self.restore_current_tolerance = restore_current_tolerance
        self.pulse_duration_tolerance = pulse_duration_tolerance

    def find_sign_changes(row):
        # Convert row to numpy array and remove NaN values
        arr = np.array(row.dropna())
        # Calculate the sign of each element
        signs = np.sign(arr)
        # Find where the sign changes
        sign_changes = np.diff(signs)
        # Get the indices where sign changes occur
        change_indices = np.where(sign_changes != 0)[0]
        # If the first element is negative and there's a sign change, remove the first change
        if arr[0] < 0 and len(change_indices) > 0 and change_indices[0] == 0:
            change_indices = change_indices[1:]
        # Add 1 to indices to match original DataFrame index (since we used diff)
        return change_indices

    def get_duration(self, df):
        # in seconds
        time_h = df["Time"].diff().dt.total_seconds()
        time_h = np.cumsum(time_h).values
        time_h[0] = 0
        return time_h[-1]

    def Ah_calculation(self, df):
        time_h = df["Time"].diff().dt.total_seconds()
        time_h = np.cumsum(time_h).values / 3600
        time_h[0] = 0
        AhThroughput = df["Current"].values
        AhThroughput = integrate.cumulative_trapezoid(
            abs(AhThroughput), x=time_h, initial=0
        )
        AhThroughput = AhThroughput - min(AhThroughput)
        return AhThroughput[-1]

    def R_ct_calculation(self, group_df):
        R_ct = abs(
            (group_df["Voltage"].max() - group_df["Voltage"].min())
            / group_df["Current"].mean()
        )
        return R_ct

    def R_0_calculation(self, group_df):
        pulse_idx = group_df.index[0]
        pulse_voltage = group_df["Voltage"].iloc[0]
        pulse_current = group_df[group_df["Current"] != 0]["Current"].iloc[
            0
        ]  # Get the first non-zero current
        last_voltage_before = self.df.loc[self.df.index < pulse_idx, "Voltage"].iloc[-1]

        R_0 = (last_voltage_before - pulse_voltage) / pulse_current
        return R_0

    def fetch_qOCV(self, group, ID):
        # when current mean is smaller than C/15, and the std is small, this must be a attempted qOCV measurement
        if (
            abs(group["Current"].mean()) < (self.qOCV_CRate * self.Nom_Capacity) + self.qocv_current_tolerance
        ) & (abs(group["Current"].std()) < 1 / 1000):
            calculated_capacity = self.Ah_calculation(group)
            if calculated_capacity < self.Nom_Capacity / 3:
                print("Outlier found at ID: ", ID)
                group["target"] = "-1"
            else:

                if np.sign(group["Current"].iloc[-1]) == 1:
                    if group["target"].iloc[-1] == "qOCV_CHA":
                        print("qOCV already added at ID: ", ID)
                    else:
                        group["target"] = "qOCV_CHA"
                        print("Added qOCV with Capacity: ", calculated_capacity)
                else:
                    if group["target"].iloc[-1] == "qOCV_DCH":
                        print("qOCV already added at ID: ", ID)
                    else:
                        group["target"] = "qOCV_DCH"
                        print(
                            "Added qOCV at ID ",
                            ID,
                            " with Capacity: ",
                            calculated_capacity,
                        )
        else:
            group["target"] = "-1"
        return group

    def on_submit(self, group, b):
        with self.output:
            if self.text_input.value:
                self.result = self.text_input.value
                group["target"] = self.result
                print(f"Target set to: {self.result}")
            else:
                print("Please enter a value")

    def fetch_capacity(self, group, ID):
        calculated_capacity = self.Ah_calculation(group)
        print("Calculating Capacity at ID: ", ID, calculated_capacity)
        if calculated_capacity < self.Nom_Capacity / 3:
            group["Capacity_py"] = np.nan
            print("Outlier found at ID: ", ID)
            group["target"] = "-1"

        else:
            group["Capacity_py"] = calculated_capacity
            group["target"] = "CAP"
        return group

    def fetch_pulse(self, group, ID):
        sign_change_location = calculation.find_sign_changes(group["Current"])
        duration = calculation.get_duration(self, group)
        if (
            duration < (self.pulse_type * self.target_pulse_duration) * self.pulse_duration_tolerance
        ):
            mask = group["Current"] > 0
            if mask.any():
                # now decide what unit should be calculated from the pulse
                if self.pulse_target_unit == "Power":
                    p_c = (
                        group.iloc[-1]["Current"] * group.iloc[-1]["Voltage"]
                    )  # power at chargepulse
                elif self.pulse_target_unit == "Resistance":
                    R_ct_c = self.R_ct_calculation(group)
                    R_0_c = self.R_0_calculation(group)

                list_R = [float(abs(R_ct_c)), float(abs(R_0_c))]
                group["Pulse_py"] = group["Pulse_py"].astype("object")
                for idx in group[mask].index:
                    group.at[idx, "Pulse_py"] = list_R

                print(
                    "Charge Pulse added at ID: ",
                    ID,
                    " with",
                    self.pulse_target_unit,
                    ":",
                    [abs(R_ct_c), abs(R_0_c)],
                )
                group["target"] = "PUL"

            mask = group["Current"] < 0
            if mask.any():
                if self.pulse_target_unit == "Power":
                    p_d = max(
                        abs(
                            (
                                group.iloc[sign_change_location]["Current"]
                                * group.iloc[sign_change_location]["Voltage"]
                            )
                        )
                    )  # power at dischargepulse
                elif self.pulse_target_unit == "Resistance":
                    R_ct_d = self.R_ct_calculation(group)
                    R_0_d = self.R_0_calculation(group)
                # group_sum = group.sum()
                list_R = [float(abs(R_ct_d)), float(abs(R_0_d))]
                group["Pulse_py"] = group["Pulse_py"].astype("object")
                for idx in group[mask].index:
                    group.at[idx, "Pulse_py"] = list_R

                print(
                    "Discharge Pulse added at ID: ",
                    ID,
                    " with",
                    self.pulse_target_unit,
                    ":",
                    [abs(R_ct_d), abs(R_0_d)],
                )
                group["target"] = "PUL"

        else:
            print("Outlier found at ID: ", ID, duration)
            group["target"] = "-1"
        return group

    def update_qOCV(self):
        print("Calculating qOCV")
        subset_qocv = self.df[self.df["target"] == "QOCV*"].copy()
        updated_subset = subset_qocv.groupby("ID", group_keys=False).apply(
            lambda x: calculation.fetch_qOCV(self, x, x.name), include_groups=False
        )
        self.df.update(updated_subset)
        return self.df

    def update_capacity(self):
        print("Calculating capacities")
        subset_cap = self.df[self.df["target"] == "CAP*"].copy()
        updated_subset_cap = subset_cap.groupby("ID", group_keys=False).apply(
            lambda x: calculation.fetch_capacity(self, x, x.name), include_groups=False
        )
        self.df.update(updated_subset_cap)
        return self.df

    @staticmethod
    def _filter_restore_pulses(subset_pul, restore_current_tolerance=0.05):
        """Split PUL* segments into test pulses and restore pulses.

        A restore pulse has the same |current| as the immediately preceding pulse
        in the same BM_Programm but opposite sign. All restore pulses run at C/2.

        Returns:
            test_pul: DataFrame of test pulses (to be processed)
            restore_ids: set of IDs identified as restore pulses
        """
        if subset_pul.empty:
            return subset_pul, set()

        id_stats = (
            subset_pul.groupby("ID")["Current"]
            .mean()
            .reset_index()
            .rename(columns={"Current": "Current_mean"})
        )
        id_stats["BM_Programm"] = id_stats["ID"].str.rsplit("_", n=1).str[0]
        id_stats["proc_num"] = id_stats["ID"].str.rsplit("_", n=1).str[1].astype(int)
        id_stats = id_stats.sort_values(["BM_Programm", "proc_num"])

        restore_ids = set()
        for _, grp in id_stats.groupby("BM_Programm"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                # Restores are always immediately after their test (proc_num gap = 1).
                # A gap > 1 means a non-PUL* segment sits between them (e.g. an
                # undetected 1C restore), so the pair are two tests, not test+restore.
                if grp.loc[i, "proc_num"] - grp.loc[i - 1, "proc_num"] != 1:
                    continue
                prev_I = grp.loc[i - 1, "Current_mean"]
                curr_I = grp.loc[i, "Current_mean"]
                same_magnitude = abs(abs(curr_I) - abs(prev_I)) / (abs(prev_I) + 1e-9) < restore_current_tolerance
                opposite_sign = np.sign(curr_I) != np.sign(prev_I)
                if same_magnitude and opposite_sign:
                    restore_ids.add(grp.loc[i, "ID"])

        if restore_ids:
            print(f"Filtering {len(restore_ids)} restore pulse IDs: {sorted(restore_ids)}")
        test_pul = subset_pul[~subset_pul["ID"].isin(restore_ids)]
        return test_pul, restore_ids

    def update_pulse(self):
        print("Calculating pulses")
        subset_pul = self.df[self.df["target"] == "PUL*"].copy()
        test_pul, restore_ids = calculation._filter_restore_pulses(subset_pul, self.restore_current_tolerance)

        if restore_ids:
            self.df.loc[self.df["ID"].isin(restore_ids), "target"] = "PUL*RES"

        updated_subset_pul = test_pul.groupby("ID", group_keys=False).apply(
            lambda x: calculation.fetch_pulse(self, x, x.name), include_groups=False
        )
        self.df.update(updated_subset_pul)
        return self.df
