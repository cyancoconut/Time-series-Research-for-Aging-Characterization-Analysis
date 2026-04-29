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
    ):
        self.qOCV_CRate = qOCV_CRate
        self.Nom_Capacity = Nom_Capacity
        self.target_pulse_duration = target_pulse_duration
        self.pulse_type = pulse_type  # 1 = single pulse, 2 = consecutive double pulse
        self.pulse_target_unit = pulse_target_unit
        self.df = df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def find_sign_changes(row):
        arr = np.array(row.dropna())
        signs = np.sign(arr)
        sign_changes = np.diff(signs)
        change_indices = np.where(sign_changes != 0)[0]
        if arr[0] < 0 and len(change_indices) > 0 and change_indices[0] == 0:
            change_indices = change_indices[1:]
        return change_indices

    def get_duration(self, df):
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

    def _row_at_time(self, group, t_elapsed, target_s):
        idx = (t_elapsed - target_s).abs().idxmin()
        return group.loc[idx]

    def _build_pau_voltage_lookup(self):
        """For each PAU stub, return the last-row voltage keyed by ID."""
        pau_df = self.df[self.df["target"] == "PAU"]
        return pau_df.groupby("ID")["Voltage"].last()

    def _preceding_pau_voltage(self, pulse_id, pau_last_v):
        """Return the relaxed voltage from the PAU stub immediately before pulse_id."""
        prog, proc = pulse_id.rsplit("_", 1)
        proc_num = int(proc)
        candidates = [
            (int(pid.rsplit("_", 1)[1]), v)
            for pid, v in pau_last_v.items()
            if pid.rsplit("_", 1)[0] == prog and int(pid.rsplit("_", 1)[1]) < proc_num
        ]
        if not candidates:
            return None
        _, v = max(candidates, key=lambda x: x[0])
        return v

    # ------------------------------------------------------------------
    # Pulse
    # ------------------------------------------------------------------

    def fetch_pulse(self, group, ID, v_before_override=None):
        group = group.copy()
        duration = self.get_duration(group)

        if duration >= (self.pulse_type * self.target_pulse_duration) * 1.08:
            print(f"Outlier found at ID: {ID} duration={duration:.1f}s")
            group["target"] = "-1"
            return group

        group["Time"] = pd.to_datetime(group["Time"])
        active = group[group["Current"] != 0]
        if active.empty:
            group["target"] = "-1"
            return group

        t_elapsed = (group["Time"] - active["Time"].iloc[0]).dt.total_seconds()
        t_elapsed.index = group.index

        # Use relaxed voltage from preceding PAU stub if available
        v_before = v_before_override if v_before_override is not None else active["Voltage"].iloc[0]

        row_1s  = self._row_at_time(group, t_elapsed, 1.0)
        row_10s = self._row_at_time(group, t_elapsed, 10.0)
        row_30s = self._row_at_time(group, t_elapsed, 30.0)

        def resistance(row):
            I = row["Current"]
            if I == 0:
                return np.nan
            return abs((row["Voltage"] - v_before) / I)

        R0  = resistance(row_1s)
        R10 = resistance(row_10s)
        R30 = resistance(row_30s)

        list_R = [float(R0), float(R10), float(R30)]
        group["Pulse_py"] = group["Pulse_py"].astype("object")
        for idx in active.index:
            group.at[idx, "Pulse_py"] = list_R

        zustand = "Charge" if active["Current"].iloc[0] > 0 else "Discharge"
        print(f"{zustand} Pulse added at ID: {ID}  R0={R0:.6f} Ω  R10={R10:.6f} Ω  R30={R30:.6f} Ω")
        group["target"] = "PUL"
        return group

    @staticmethod
    def _filter_restore_pulses(subset_pul):
        """Split PUL* segments into test pulses and restore pulses.

        A restore pulse has the same |current| as the immediately preceding pulse
        in the same BM_Programm but opposite sign, with proc_num gap == 1.

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
                if grp.loc[i, "proc_num"] - grp.loc[i - 1, "proc_num"] != 1:
                    continue
                prev_I = grp.loc[i - 1, "Current_mean"]
                curr_I = grp.loc[i, "Current_mean"]
                same_magnitude = abs(abs(curr_I) - abs(prev_I)) / (abs(prev_I) + 1e-9) < 0.05
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
        test_pul, restore_ids = calculation._filter_restore_pulses(subset_pul)

        if restore_ids:
            self.df.loc[self.df["ID"].isin(restore_ids), "target"] = "PUL*RES"

        pau_last_v = self._build_pau_voltage_lookup()
        updated_subset_pul = test_pul.groupby("ID", group_keys=False).apply(
            lambda x: self.fetch_pulse(
                x, x.name,
                v_before_override=self._preceding_pau_voltage(x.name, pau_last_v),
            ),
            include_groups=False,
        )
        self.df.update(updated_subset_pul)
        return self.df

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

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

    def update_capacity(self):
        print("Calculating capacities")
        subset_cap = self.df[self.df["target"] == "CAP*"].copy()
        updated_subset_cap = subset_cap.groupby("ID", group_keys=False).apply(
            lambda x: calculation.fetch_capacity(self, x, x.name), include_groups=False
        )
        self.df.update(updated_subset_cap)
        return self.df

    # ------------------------------------------------------------------
    # qOCV
    # ------------------------------------------------------------------

    def fetch_qOCV(self, group, ID):
        if (
            abs(group["Current"].mean()) < (self.qOCV_CRate * self.Nom_Capacity) + 0.01
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
                        print("Added qOCV at ID ", ID, " with Capacity: ", calculated_capacity)
        else:
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

    # ------------------------------------------------------------------
    # Interactive
    # ------------------------------------------------------------------

    def on_submit(self, group, b):
        with self.output:
            if self.text_input.value:
                self.result = self.text_input.value
                group["target"] = self.result
                print(f"Target set to: {self.result}")
            else:
                print("Please enter a value")
