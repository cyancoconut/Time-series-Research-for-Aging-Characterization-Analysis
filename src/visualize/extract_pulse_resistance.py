# -*- coding: utf-8 -*-
"""Extract and calculate pulse resistance (R0, R10, R30) from a GOLD parquet file.

Pulses must be labelled PUL* in the GOLD parquet (pipeline run without update_pulse).
R0/R10/R30 are computed here at t=1s, 10s, 30s using the preceding PAU stub voltage.

Usage:
    python extract_pulse_resistance.py <input_parquet> <output_csv> [--prozedur NAME]

Example:
    python extract_pulse_resistance.py GOLD/VTC/VTC_resilite.parquet \
        GOLD/VTC/VTC_resilite_pulse_resistance.csv \
        --prozedur dng_STD_Puls_25C_1C
"""

import argparse
import os

import numpy as np
import pandas as pd


BINS   = [-5, 5, 15, 25, 35, 45, 52.5, 60]
LABELS = [0, 10, 20, 30, 40, 50, 55]


def _build_pau_voltage_lookup(df):
    pau_df = df[df["target"] == "PAU"]
    return pau_df.groupby("ID")["Voltage"].last()


def _preceding_pau_voltage(pulse_id, pau_last_v):
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


def _row_at_time(group, t_elapsed, target_s):
    idx = (t_elapsed - target_s).abs().idxmin()
    return group.loc[idx]


def _calc_resistance(group, v_before):
    group = group.copy()
    group["Time"] = pd.to_datetime(group["Time"])
    active = group[group["Current"] != 0]
    if active.empty:
        return None

    t_elapsed = (group["Time"] - active["Time"].iloc[0]).dt.total_seconds()
    t_elapsed.index = group.index

    def resistance(row):
        I = row["Current"]
        if I == 0:
            return np.nan
        return abs((row["Voltage"] - v_before) / I)

    R0  = resistance(_row_at_time(group, t_elapsed, 1.0))
    R10 = resistance(_row_at_time(group, t_elapsed, 10.0))
    R30 = resistance(_row_at_time(group, t_elapsed, 30.0))
    return R0, R10, R30


def _assign_soc(df):
    """Start at SOC=100 and decrement by 10 each time T_group resets from 0 to 55."""
    t_groups = pd.cut(df["Temperature"], bins=BINS, labels=LABELS).astype(int)
    soc, prev_t, soc_list = 100, None, []
    for t in t_groups:
        if prev_t == 0 and t == 55:
            soc -= 10
        soc_list.append(soc)
        prev_t = t
    return soc_list


def extract_pulse_resistance(input_parquet, output_csv, prozedur_filter=None):
    df = pd.read_parquet(input_parquet)
    df["Time"] = pd.to_datetime(df["Time"])

    pau_last_v = _build_pau_voltage_lookup(df)

    pul = df[df["target"].isin(["PUL", "PUL*"])].copy()
    if prozedur_filter:
        pul = pul[pul["Prozedur"] == prozedur_filter]

    records = []
    for pulse_id, group in pul.groupby("ID", sort=False):
        v_before = _preceding_pau_voltage(pulse_id, pau_last_v)
        if v_before is None:
            active = group[group["Current"] != 0]
            v_before = active["Voltage"].iloc[0] if not active.empty else np.nan

        result = _calc_resistance(group, v_before)
        if result is None:
            continue
        R0, R10, R30 = result

        meta = group.iloc[0]
        records.append({
            "ID":            pulse_id,
            "Zustand":       meta["Zustand"],
            "Temperature":   meta["Temperature"],
            "Ah_throughput": meta["Ah_throughput"],
            "R0":  R0,
            "R10": R10,
            "R30": R30,
        })

    result_df = pd.DataFrame(records)
    if result_df.empty:
        print("No pulses found.")
        return result_df

    result_df["_id_num"] = result_df["ID"].str.rsplit("_", n=1).str[1].astype(int)
    result_df = result_df.sort_values("_id_num").drop(columns="_id_num").reset_index(drop=True)
    result_df["SOC"] = _assign_soc(result_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} rows → {output_csv}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pulse resistance from GOLD parquet")
    parser.add_argument("input_parquet", help="Path to GOLD parquet")
    parser.add_argument("output_csv",    help="Path to output CSV")
    parser.add_argument("--prozedur", default=None,
                        help="Filter by Prozedur name (e.g. dng_STD_Puls_25C_1C)")
    args = parser.parse_args()

    extract_pulse_resistance(args.input_parquet, args.output_csv, args.prozedur)
