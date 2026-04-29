# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class calculation:
    def __init__(self, df):
        self.df = df

    def _row_at_time(self, group, t_elapsed, target_s):
        idx = (t_elapsed - target_s).abs().idxmin()
        return group.loc[idx]

    def fetch_pulse(self, group, ID, fixed_current=None, v_before_override=None):
        group = group.copy()
        group["Time"] = pd.to_datetime(group["Time"])

        # Always anchor to the first row where current is flowing
        active = group[group["Current"] != 0]
        if active.empty:
            return pd.Series({"R0": np.nan, "R10": np.nan, "R30": np.nan})

        t_elapsed = (group["Time"] - active["Time"].iloc[0]).dt.total_seconds()
        t_elapsed.index = group.index

        # Use relaxed voltage from preceding PAU stub if available
        v_before = v_before_override if v_before_override is not None else active["Voltage"].iloc[0]

        row_1s  = self._row_at_time(group, t_elapsed, 1.0)
        row_10s = self._row_at_time(group, t_elapsed, 10.0)
        row_30s = self._row_at_time(group, t_elapsed, 30.0)

        def resistance(row):
            I = fixed_current if fixed_current is not None else row["Current"]
            if I == 0:
                return np.nan
            return abs((row["Voltage"] - v_before) / I)

        R0 = resistance(row_1s)
        R10 = resistance(row_10s)
        R30 = resistance(row_30s)

        print(f"Pulse {ID}: R0={R0:.6f} Ω, R10={R10:.6f} Ω, R30={R30:.6f} Ω")
        return pd.Series({"R0": R0, "R10": R10, "R30": R30})

    def _build_pau_voltage_lookup(self):
        """For each PAU stub, return the last-row voltage keyed by ID."""
        pau_df = self.df[self.df["target"] == "PAU"]
        return pau_df.groupby("ID")["Voltage"].last()

    def _preceding_pau_voltage(self, pulse_id, pau_last_v):
        """Return the relaxed voltage from the PAU stub immediately before pulse_id."""
        prog, proc = pulse_id.rsplit("_", 1)
        proc_num = int(proc)
        # All PAU stubs in the same BM_Programm with a lower proc_num
        candidates = [
            (int(pid.rsplit("_", 1)[1]), v)
            for pid, v in pau_last_v.items()
            if pid.rsplit("_", 1)[0] == prog and int(pid.rsplit("_", 1)[1]) < proc_num
        ]
        if not candidates:
            return None
        _, v = max(candidates, key=lambda x: x[0])
        return v

    def update_pulse(self, fixed_current=None):
        label = f"fixed I={fixed_current}A" if fixed_current is not None else "measured I"
        print(f"Calculating pulse resistances ({label})")
        pulse_df = self.df[self.df["target"].isin(["PUL", "PUL*"])]
        pau_last_v = self._build_pau_voltage_lookup()
        results = pulse_df.groupby("ID", sort=False).apply(
            lambda x: self.fetch_pulse(
                x, x.name, fixed_current=fixed_current,
                v_before_override=self._preceding_pau_voltage(x.name, pau_last_v),
            ),
            include_groups=False,
        )

        essentials = pulse_df.groupby("ID", sort=False).first()[["Temperature", "Zustand"]]
        results = essentials.join(results)

        # Assign SOC: start at 100, drop by 10 each time temperature resets from 0°C to 55°C
        bins   = [-5, 5, 15, 25, 35, 45, 52.5, 60]
        labels = [0, 10, 20, 30, 40, 50, 55]
        results["T_group"] = pd.cut(results["Temperature"], bins=bins, labels=labels).astype(int)

        # Sort by numeric ID to preserve measurement order
        results["_id_num"] = results.index.str.split("_").str[1].astype(int)
        results = results.sort_values("_id_num").drop(columns="_id_num")

        soc = 100
        soc_list = []
        prev_t = None
        for t in results["T_group"]:
            if prev_t == 0 and t == 55:
                soc -= 10
            soc_list.append(soc)
            prev_t = t
        results["SOC"] = soc_list

        return results


if __name__ == "__main__":
    INPUT_PARQUET = "GOLD/VTC/VTC_resilite_normal_pulses.parquet"
    OUTPUT_CSV    = "GOLD/VTC/VTC_resilite_pulse_resistance.csv"

    df = pd.read_parquet(INPUT_PARQUET)
    calc = calculation(df)
    results = calc.update_pulse()
    results.to_csv(OUTPUT_CSV)
    print(f"\nResults saved to {OUTPUT_CSV}")
