# -*- coding: utf-8 -*-

import pandas as pd

INPUT_PARQUET  = "GOLD/VTC/VTC_resilite.parquet"
OUTPUT_PARQUET = "GOLD/VTC/VTC_resilite_normal_pulses.parquet"

df = pd.read_parquet(INPUT_PARQUET)

# --- Identify normal pulse IDs (logic from filter_restoration_pulses notebook) ---

# One representative row per pulse ID (1C pulses only)
pul_df = df[(df["Prozedur"] == "dng_STD_Puls_25C_1C")]
df_all_pulse = (
    pul_df.groupby("ID", sort=False)
    .nth(2)
    .reset_index()
    .sort_values("Time")
    .reset_index(drop=True)
)

# Group pulses by temperature step (> 4°C change = new temp group)
df_all_pulse["temp_diff"] = df_all_pulse["Temperature"].diff().abs()
df_all_pulse["temp_group"] = (df_all_pulse["temp_diff"] > 4).fillna(False).cumsum()

# Number pulses within each temperature group
df_all_pulse["pulse_num"]   = df_all_pulse.groupby("temp_group").cumcount() + 1
df_all_pulse["group_count"] = df_all_pulse.groupby("temp_group")["temp_group"].transform("size")

# Keep pulse 1 & 3 normally; if group has exactly 2, keep both
normal_mask = (
    ((df_all_pulse["group_count"] == 2) & (df_all_pulse["pulse_num"].isin([1, 2]))) |
    ((df_all_pulse["group_count"] != 2) & (df_all_pulse["pulse_num"].isin([1, 3])))
)
normal_ids = set(df_all_pulse.loc[normal_mask, "ID"])

print(f"Total pulse IDs: {len(df_all_pulse)}")
print(f"Normal pulse IDs: {len(normal_ids)}")

# --- Extract full rows for those IDs from the complete dataset, including PAU stubs ---
df_out = df[(df["ID"].isin(normal_ids)) | (df["target"] == "PAU")].copy()

print(f"Output rows: {len(df_out)}")
df_out.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Saved {OUTPUT_PARQUET}")
