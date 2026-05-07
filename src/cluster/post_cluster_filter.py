import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets as widgets


class ClusterNotFoundException(Exception):
    """Raised when no suitable cluster is found."""

    pass


class cluster_filter:
    def __init__(
        self,
        number_programms,
        qOCV_CRate,
        Nom_Capacity,
        V_Nom,
        CAP_Rate,
        CAP_Type,
        CAP_Temp,
        target_pulse_duration,
        pulse_type,
        pulse_target_unit,
        pulse_cluster_tolerance,
    ):
        self.number_programms = number_programms
        self.qOCV_CRate = qOCV_CRate
        self.Nom_Capacity = Nom_Capacity
        self.V_Nom = V_Nom
        self.CAP_Rate = CAP_Rate
        self.CAP_Type = CAP_Type
        self.CAP_Temp = CAP_Temp
        self.target_pulse_duration = target_pulse_duration
        self.pulse_type = pulse_type  # 1 = single pulse, 2 = consecutive double pulse
        self.pulse_target_unit = pulse_target_unit
        self.pulse_cluster_tolerance = pulse_cluster_tolerance

    ## Filter the clusters
    def find_capacity(self, cluster_means, cluster_size, layer):
        counter = 0  # counter stays at 0 if a cluster was found
        # if the capacity is measured via constant current
        if self.CAP_Type == "CC":
            # then the standard deviation of the current should be 0 AND the current_mean should be around cap_rate
            if "Current_std" in cluster_means.columns:
                mask_std = abs(cluster_means["Current_std"]) < 0.001
            mask_CRate = (abs(cluster_means["Current_mean"]) < self.CAP_Rate * 1.1) & (
                abs(cluster_means["Current_mean"]) > self.CAP_Rate * 0.9
            )  # and the mean of current should be around cap_rate
            mask_CRate_strict = (
                abs(cluster_means["Current_mean"]) < self.CAP_Rate * 1.02
            ) & (
                abs(cluster_means["Current_mean"]) > self.CAP_Rate * 0.98
            )  # strict mask for second layer
            mask_discharge = (
                cluster_means["Current_mean"] < 0
            )  # the current should be negative (When capacity is measured at discharging)
            if "Voltage_min" in cluster_means.columns:
                mask_voltage = (
                    abs(cluster_means["Voltage_min"]) < 0.005
                )  # minimum voltage should be 0 to be completely discharged
            if "Duration_minutes" in cluster_means.columns:
                mask_duration = (
                    cluster_means["Duration_minutes"] < (1 / self.CAP_Rate) * 60 * 1.2
                ) & (
                    cluster_means["Duration_minutes"] > (1 / self.CAP_Rate) * 60 * 0.8
                )  # the duration should be round the 1/CRate in minutes

        # via constant power
        elif self.CAP_Type == "CP":
            mask_std = (
                abs(cluster_means["Power_std"]) < 0.001
            )  # the standard deviation of the current should be 0
            mask_CRate = (abs(cluster_means["Power_mean"]) < self.CAP_Rate * 1.1) & (
                abs(cluster_means["Power_mean"]) > self.CAP_Rate * 0.9
            )  # and the mean of power should be around cap_rate
            mask_discharge = (
                cluster_means["Power_mean"] < 0
            )  # the power should be negative (When capacity is measured at discharging)
            mask_voltage = (
                abs(cluster_means["Voltage_min"]) < 0.005
            )  # minimum voltage should be 0 to be completely discharged
            mask = mask_std & mask_CRate & mask_voltage
            cluster_means_CAP = cluster_means[mask]

        if layer == 1:
            mask = mask_std & mask_CRate & mask_discharge & mask_voltage & mask_duration
            cluster_means_CAP = cluster_means[mask]

            if cluster_means_CAP.shape[0] == 1:
                capacity_cluster = cluster_means_CAP.index
                print("We found the capacity cluster at Cluster: ", capacity_cluster)
                return capacity_cluster, counter

            elif cluster_means_CAP.shape[0] == 0:
                # First filter to only consider values that are smaller than 60/CAP_Rate
                valid_clusters = cluster_means
                # valid_clusters = cluster_means[
                #     cluster_means["Duration_minutes"] <= 60 / self.CAP_Rate
                # ]

                # Find index with duration closest to 60/self.CAP_Rate
                closest_idx = (
                    (valid_clusters["Duration_minutes"] - 60 / self.CAP_Rate)
                    .abs()
                    .idxmin()
                )
                capacity_cluster = [closest_idx]
                counter = 1  # counter move to 1, we need another layer of clustering
                print(
                    "Failed to find the capacity cluster. Attempting using a second layer. The closest cluster was:",
                    capacity_cluster,
                )

                return capacity_cluster, counter

            else:
                capacity_cluster = cluster_means_CAP.index.tolist()
                print(
                    "There are multiple candidates for the capacity cluster: ",
                    capacity_cluster,
                )
                return capacity_cluster, counter

        else:  # this is a second layer of cluster
            mask = mask_CRate_strict & mask_discharge

            potential_capacity_clusters = cluster_means[
                mask
            ].index.tolist()  # Convert to list to handle consistently
            print("Finished masking")
            potential_capacity_clusters = [
                idx for idx in potential_capacity_clusters if idx >= 0
            ]  # Filter out -1 cluster

            if potential_capacity_clusters:
                # Get the cluster with the minimum size
                capacity_cluster = cluster_size.loc[
                    potential_capacity_clusters
                ].idxmin()
                print(
                    "We found potential capacity clusters at Cluster: ",
                    potential_capacity_clusters,
                    " with size:",
                    cluster_size.loc[capacity_cluster],
                )
                if (
                    cluster_size.loc[capacity_cluster] > self.number_programms + 3
                ):  # Check if capacity cluster is too large, +3 as a buffer
                    print("Capacity Test and cycles might have been merged together")
                    counter = 2
                    return capacity_cluster, counter

                else:
                    print("We found capacity cluster. ")
                    return capacity_cluster, counter

            else:
                raise ClusterNotFoundException("No potential capacity clusters found")

    def find_pulses(self, cluster_means):
        possible_cluster = cluster_means[cluster_means.index >= 0]
        # Threshold: target_pulse_duration (seconds) scaled to minutes, times tolerance.
        # Catches all pulse-like clusters (e.g. 0.5C and 1C) while staying far below
        # CAP (~120 min) and qOCV (~1200 min) durations.
        pulse_threshold_min = (self.target_pulse_duration / 60) * self.pulse_cluster_tolerance
        pulse_clusters = possible_cluster[
            possible_cluster["Duration_minutes"] < pulse_threshold_min
        ].index.tolist()
        if not pulse_clusters:
            pulse_clusters = [possible_cluster["Duration_minutes"].idxmin()]
        return pulse_clusters

    def find_qocv(self, cluster_means):
        # getting the qocv through Duration Time
        possible_cluster = cluster_means[cluster_means.index >= 0]
        valid_clusters = possible_cluster[
            possible_cluster["Duration_minutes"] <= (60 / self.qOCV_CRate) + 100
        ]
        # Find index with duration closest to 60/self.qOCV_CRate
        closest_idx = (
            (valid_clusters["Duration_minutes"] - 60 / self.qOCV_CRate).abs().idxmin()
        )
        qocv_clusters = [closest_idx]

        return qocv_clusters

    def previous_voltage(self, x, original_df):
        """
        Check if the previous voltage is greater than 0.99
        Returns:
        - True if previous voltage > 0.99
        - False if previous voltage <= 0.99 or previous ID not found
        """
        try:
            current_group_cu = x.split("_")[0]
            current_group_cu_procedure = x.split("_")[1]

            previous_ID = (
                current_group_cu + "_" + str(int(current_group_cu_procedure) - 1)
            )
            # Check if previous ID exists in the dataframe
            prev_voltage_data = original_df[original_df["ID"] == previous_ID]

            if len(prev_voltage_data) == 0:
                # print(f"Previous ID {previous_ID} not found. Trying with ID-2...")
                previous_ID = (
                    current_group_cu + "_" + str(int(current_group_cu_procedure) - 2)
                )
                # Check if previous ID exists in the dataframe
                prev_voltage_data = original_df[original_df["ID"] == previous_ID]

                if len(prev_voltage_data) == 0:
                    # print(f"No previous ID found for {x}")
                    return False

                result = bool(prev_voltage_data["Voltage_max"].iloc[0] > 0.99)
                # print(f"Previous ID {previous_ID} found. Voltage check result: {result}")
                return result
            else:
                result = bool(prev_voltage_data["Voltage_max"].iloc[0] > 0.99)
                # print(f"Previous ID {previous_ID} found. Voltage check result: {result}")
                return result
        except Exception as e:
            print(f"Error processing {x}: {e}")
            return False

    def check_previous_voltage(self, df_capacity, df):
        """
        Filter the capacity dataframe to include only rows where the previous voltage was > 0.99
        """
        print("Checking previous voltage for all capacity measurements...")

        # Create an explicit filter function to ensure proper filtering
        def filter_function(row):
            id_value = row["ID"]
            has_valid_voltage = self.previous_voltage(id_value, df)
            # Explicitly log what's being kept/filtered
            if has_valid_voltage:
                print(f"Keeping row with ID {id_value}")
            else:
                print(f"Filtering out row with ID {id_value}")
            return has_valid_voltage

        # Apply the filter directly to rows
        df_capacity_filtered = df_capacity[df_capacity.apply(filter_function, axis=1)]

        print(
            f"Original rows: {len(df_capacity)}, Filtered rows: {len(df_capacity_filtered)}"
        )
        return df_capacity_filtered

    def temperature_filter(self, x):
        filter_temperature = (abs(x["Temperature_mean"]) < self.CAP_Temp + 3) & (
            abs(x["Temperature_mean"]) > self.CAP_Temp - 3
        )

        return filter_temperature

    def CRate_filter(self, x):
        if self.CAP_Type == "CC":
            filter_cap_rate = (abs(x["Current_mean"]) < self.CAP_Rate * 1.05) & (
                abs(x["Current_mean"]) > self.CAP_Rate * 0.95
            )
        elif self.CAP_Type == "CP":
            filter_cap_rate = (
                abs(x["Power_mean"]) < self.CAP_Rate * self.V_Nom * 1.05
            ) & (abs(x["Power_mean"]) > self.CAP_Rate * self.V_Nom * 0.95)
        return filter_cap_rate

    def check_if_pulse_in_same_programm(self, df_cap, df_pulse):

        # Find programs that have a "PUL" target
        programs_with_pul = df_pulse["BM_Programm"].unique()

        # Create a condition to keep rows that are in the same Programm as pulse:
        condition = df_cap["BM_Programm"].isin(programs_with_pul)

        # Apply the filter
        filtered_df = df_cap[condition]

        print("Found ", filtered_df.shape[0], " CU after programm filtered")

        return filtered_df

    def check_temperature(self, df_capacity):
        df_capacity_filtered = df_capacity[self.temperature_filter(df_capacity)]

        print("Found ", df_capacity_filtered.shape[0], " CU after temperature filtered")
        return df_capacity_filtered

    def check_CRate(self, df_capacity):
        df_capacity_filtered = df_capacity[self.CRate_filter(df_capacity)]

        print("Found ", df_capacity_filtered.shape[0], " CU after CRate filtered")
        return df_capacity_filtered

    def concat_clusters(
        self, capacity_cluster, pulse_clusters, qocv_clusters, counter, df_final
    ):
        ## filter again the capacity and then concat with pulses
        df_final.loc[df_final["target"].isin(capacity_cluster), "target"] = "CAP*"
        df_final.loc[df_final["target"].isin(pulse_clusters), "target"] = "PUL*"
        df_final.loc[df_final["target"].isin(qocv_clusters), "target"] = "QOCV*"

        df_capacity = df_final[df_final["target"] == "CAP*"].copy()
        df_pulse = df_final[df_final["target"] == "PUL*"].copy()
        df_qocv = df_final[df_final["target"] == "QOCV*"].copy()

        df_capacity_filtered = self.check_temperature(df_capacity)
        df_capacity_filtered = self.check_CRate(df_capacity_filtered)

        if counter == 2:
            # when counter is 2, the capacity check cannot be distinguished from the cycles
            # therefore, check if there is a pulse test within the same BM programm
            df_capacity_filtered = self.check_if_pulse_in_same_programm(
                df_capacity_filtered, df_pulse
            )
        df_capacity_filtered = self.check_previous_voltage(
            df_capacity_filtered, df_final
        )

        df_result_filtered = pd.concat(
            [df_capacity_filtered, df_pulse, df_qocv], axis=0
        )

        return df_result_filtered
