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
        pulse_keep_per_group=None,
        pulse_group_by="BM_Programm",
        pulse_step_threshold=None,
        qocv_current_tolerance=0.01,
        restore_current_tolerance=0.05,
        pulse_duration_tolerance=1.08,
        qocv_std_tolerance=0.002,
        features=None,
    ):

        self.qOCV_CRate = qOCV_CRate
        self.Nom_Capacity = Nom_Capacity
        self.target_pulse_duration = target_pulse_duration
        self.pulse_type = pulse_type  # 1 = single pulse, 2 = consecutive double pulse
        self.pulse_target_unit = pulse_target_unit
        self.df = df
        self.pulse_keep_per_group = pulse_keep_per_group or []
        self.pulse_group_by = pulse_group_by
        self.pulse_step_threshold = pulse_step_threshold
        self.qocv_current_tolerance = qocv_current_tolerance
        self.restore_current_tolerance = restore_current_tolerance
        self.pulse_duration_tolerance = pulse_duration_tolerance
        # Max current std for a qOCV, as a C-rate (fraction of Nom_Capacity).
        # The absolute 0.001 A cap it replaces did not scale with cell size: on
        # a ~11 Ah cell even a clean C/20 sweep has std ~0.0014 A and a harmless
        # interior low-current dip lifts it further. As a C-rate this passes a
        # constant-current sweep on any cell while still rejecting a varying
        # load (a pulse train's std is ~its mean current, far above this).
        self.qocv_std_tolerance = qocv_std_tolerance
        # Per-ID lookup of the precomputed segment features (trimmed [2:-1] +
        # normalized ÷ Nom_Capacity in create_features). The qOCV current guard
        # reads from here so it sees the same statistics as the rest of the
        # pipeline instead of recomputing on the untrimmed segment, where
        # stationary edge rows inflate the std past the cap and silently drop a
        # valid qOCV (see _qocv_current_amps).
        self.feature_lookup = None
        if features is not None and "ID" in getattr(features, "columns", []):
            cols = [
                c
                for c in ("Current_mean", "Current_std", "abs_Current_mean")
                if c in features.columns
            ]
            if cols:
                self.feature_lookup = features.set_index("ID")[cols].to_dict("index")

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

    def _qocv_current_amps(self, group, ID):
        """Return ``(|mean|, std)`` of the segment current in **Amps** for the
        qOCV guard.

        Prefers the precomputed feature table: ``create_features`` computes the
        current statistics on the trimmed ``[2:-1]`` window (dropping the
        "faulty stationary values" at the segment edges) and normalizes them by
        ``Nom_Capacity``. Multiplying back by ``Nom_Capacity`` recovers Amps, so
        the guard is fed the same trimmed statistics the rest of the pipeline
        uses — keeping qOCV DCH/CHA symmetric. Falls back to recomputing on the
        same trimmed window when no feature row is available (legacy/notebook
        callers without ``features``).
        """
        feats = self.feature_lookup.get(ID) if self.feature_lookup else None
        if feats is not None:
            abs_mean = feats.get("abs_Current_mean")
            if abs_mean is None or pd.isna(abs_mean):
                cm = feats.get("Current_mean")
                abs_mean = abs(cm) if cm is not None and pd.notna(cm) else None
            std = feats.get("Current_std")
            if (
                abs_mean is not None
                and pd.notna(abs_mean)
                and std is not None
                and pd.notna(std)
            ):
                return abs_mean * self.Nom_Capacity, std * self.Nom_Capacity

        # Fallback: recompute on the trimmed window (mirror create_features).
        current = group["Current"]
        trimmed = current.iloc[2:-1]
        if len(trimmed) == 0:
            trimmed = current
        return abs(trimmed.mean()), trimmed.std()

    def fetch_qOCV(self, group, ID):
        # A qOCV is a steady sweep at the configured rate: its mean current sits
        # in a two-sided band around qOCV_CRate x Nom_Capacity, and its std is
        # small. The band is two-sided because the upper bound alone admits
        # anything *below* the rate — including a segment at exactly 0 A. Rest
        # / EIS-dwell segments mislabelled QOCV* pass an upper-bound-only test
        # trivially (0 < i_nom, std 0 < cap) and then fall through the
        # sign(0) != 1 branch to "qOCV_DCH", producing a bundle with zero
        # current, zero integrated capacity and no sweep at all.
        i_nom = self.qOCV_CRate * self.Nom_Capacity
        abs_current_mean, current_std = self._qocv_current_amps(group, ID)
        if (
            (abs_current_mean > i_nom - self.qocv_current_tolerance)
            & (abs_current_mean < i_nom + self.qocv_current_tolerance)
        ) & (abs(current_std) < self.qocv_std_tolerance * self.Nom_Capacity):
            calculated_capacity = self.Ah_calculation(group)
            if np.sign(group["Current"].mean()) == 1:
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
        group["Capacity_py"] = calculated_capacity
        group["target"] = "CAP"
        return group

    def fetch_pulse(self, group, ID):
        print("Pulse labeled at ID: ", ID)
        group["target"] = "PUL"
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

    def _filter_pulse_group(self, subset_pul):
        """Keep only pulse positions listed in pulse_keep_per_group within each group.

        Pulses are sorted by proc_num within each BM_Programm and assigned a
        1-based position. Positions not in pulse_keep_per_group are labelled PUL*RES.

        If pulse_step_threshold is set and pulse_group_by is a numeric column,
        a new sub-group starts whenever consecutive pulses differ by more than
        the threshold (reserved for future use; ignored when pulse_group_by="BM_Programm").

        Returns (keep_df, skip_ids).
        """
        if not self.pulse_keep_per_group or subset_pul.empty:
            return subset_pul, set()

        id_rep = (
            subset_pul.groupby("ID", sort=False)
            .first()
            .reset_index()
        )
        id_rep["_proc_num"] = id_rep["ID"].str.rsplit("_", n=1).str[1].astype(int)
        id_rep["_bm"] = id_rep["ID"].str.rsplit("_", n=1).str[0]
        id_rep = id_rep.sort_values(["_bm", "_proc_num"]).reset_index(drop=True)

        if self.pulse_group_by == "BM_Programm":
            id_rep["_group"] = id_rep["_bm"]
        elif self.pulse_step_threshold is not None and self.pulse_group_by in id_rep.columns:
            id_rep["_group"] = (
                id_rep.groupby("_bm")[self.pulse_group_by]
                .transform(lambda s: s.diff().abs().gt(self.pulse_step_threshold).fillna(False).cumsum())
                .astype(str) + "_" + id_rep["_bm"]
            )
        else:
            id_rep["_group"] = id_rep["_bm"]

        id_rep["_pulse_num"] = id_rep.groupby("_group").cumcount() + 1

        keep_ids = set(id_rep.loc[id_rep["_pulse_num"].isin(self.pulse_keep_per_group), "ID"])
        skip_ids = set(id_rep["ID"]) - keep_ids

        if skip_ids:
            print(f"Filtering {len(skip_ids)} restore pulse IDs: {sorted(skip_ids)[:5]}{'...' if len(skip_ids) > 5 else ''}")

        return subset_pul[subset_pul["ID"].isin(keep_ids)], skip_ids

    def update_pulse(self):
        print("Calculating pulses")
        subset_pul = self.df[self.df["target"] == "PUL*"].copy()
        test_pul, restore_ids = self._filter_pulse_group(subset_pul)

        if restore_ids:
            self.df.loc[self.df["ID"].isin(restore_ids), "target"] = "PUL*RES"

        updated_subset_pul = test_pul.groupby("ID", group_keys=False).apply(
            lambda x: calculation.fetch_pulse(self, x, x.name), include_groups=False
        )
        self.df.update(updated_subset_pul)
        return self.df
