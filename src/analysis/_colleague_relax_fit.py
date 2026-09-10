"""Colleague's relaxation-only 2RC script, adapted to read a LOCAL parquet.

Only change vs the original: read_battery_data() reads from disk instead of S3,
and TARGET_FILE / OUTPUT_DIR point at local paths. Fitting logic is verbatim so
the comparison is honest. Standalone — does not touch the pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ====== config (local) ======
TARGET_FILE = (
    "/home/ann/Documents/Data_Metabatt/"
    "J8005_BMWK_METABatt=METABatt_Sony_Murata_18650VTC6_003=2024-10-03_142512="
    "jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS014653 _ Format01="
    "Kreis M3-034=filesize-109888838=finished.parquet"
)
OUTPUT_DIR = "/home/ann/Documents/Project_METAbatt/src/analysis/_colleague_out"
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','

PROCESS_ALL_SEGMENTS = True
MAX_SEGMENTS_TO_PROCESS = None
SAVE_SEGMENT_CSV = False
SAVE_SEGMENT_PLOT = False   # disabled for a fast comparison run
VERBOSE = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_battery_data(target_file: str):
    print(f"reading local file: {target_file}")
    return pd.read_parquet(target_file)


def extract_soc_dod_from_filename(filepath: str):
    filename = filepath.split('/')[-1] if '/' in filepath else filepath
    soc_match = re.search(r'(\d+)SOC', filename, re.IGNORECASE)
    dod_match = re.search(r'(\d+)DOD', filename, re.IGNORECASE)
    soc = soc_match.group(1) if soc_match else None
    dod = dod_match.group(1) if dod_match else None
    return soc, dod


def relaxation_model_pure(t, OCV, A1, tau1, A2, tau2):
    t_safe = np.clip(t, 0, None)
    return OCV + A1 * np.exp(-t_safe / tau1) + A2 * np.exp(-t_safe / tau2)


def identify_parameters_adaptive(t_data, V_data):
    V_start = V_data[0]
    V_end = V_data[-1]
    OCV_init = V_end
    delta_V = V_start - V_end
    abs_delta = abs(delta_V)
    A1_init = delta_V * 0.3
    A2_init = delta_V * 0.7
    tau1_init = 8.0
    tau2_init = 150.0
    initial_guess = [OCV_init, A1_init, tau1_init, A2_init, tau2_init]
    ocv_min, ocv_max = min(V_start, V_end) - 0.05, max(V_start, V_end) + 0.05
    if delta_V >= 0:
        a_min, a_max = 0.0, max(0.2, abs_delta * 1.5)
    else:
        a_min, a_max = -max(0.2, abs_delta * 1.5), 0.0
    tau1_min, tau1_max = 0.2, 80.0
    tau2_min, tau2_max = 30.0, 5000.0
    lower_bounds = [ocv_min, a_min, tau1_min, a_min, tau2_min]
    upper_bounds = [ocv_max, a_max, tau1_max, a_max, tau2_max]
    try:
        popt, pcov = curve_fit(
            relaxation_model_pure, t_data, V_data,
            p0=initial_guess, bounds=(lower_bounds, upper_bounds),
            max_nfev=150000, method='trf'
        )
        return popt, pcov, True
    except Exception:
        return None, None, False


def process_segment(df, start_idx, end_idx, rank, COL_VOLTAGE, COL_CURRENT):
    seg_df = df.loc[start_idx:end_idx].copy()
    seg_df['Relative_Time_s'] = seg_df['Time_Seconds'] - seg_df['Time_Seconds'].iloc[0]
    t_data = seg_df['Relative_Time_s'].values
    V_data = seg_df[COL_VOLTAGE].values
    try:
        lookback_idx = max(0, start_idx - 1)
        I_prev = df.loc[lookback_idx, COL_CURRENT]
        if abs(I_prev) < 0.05 and lookback_idx > 5:
            I_prev = df.loc[lookback_idx-5:lookback_idx, COL_CURRENT].mean()
    except Exception:
        I_prev = -1.4082
    try:
        V_prior = df.loc[max(0, start_idx - 1), COL_VOLTAGE]
    except Exception:
        V_prior = V_data[0]
    popt, pcov, success = identify_parameters_adaptive(t_data, V_data)
    if not success:
        return None
    OCV, A1, tau1, A2, tau2 = popt
    if abs(I_prev) > 0.01:
        R1 = abs(A1 / I_prev)
        R2 = abs(A2 / I_prev)
        V_fitted_zero = OCV + A1 + A2
        R0_raw = abs(V_prior - V_fitted_zero) / abs(I_prev)
        R0 = max(1e-6, R0_raw)
    else:
        R1, R2, R0 = 0.005, 0.01, 0.002
    C1 = tau1 / R1 if R1 > 0 else 0
    C2 = tau2 / R2 if R2 > 0 else 0
    V_fitted = relaxation_model_pure(t_data, OCV, A1, tau1, A2, tau2)
    residuals = V_data - V_fitted
    RMSE = np.sqrt(np.mean(residuals**2))
    R_squared = 1 - np.sum(residuals**2) / np.sum((V_data - np.mean(V_data))**2)
    return {
        "OCV": OCV, "A1": A1, "tau1": tau1, "A2": A2, "tau2": tau2,
        "R0": R0, "R1": R1, "R2": R2, "C1": C1, "C2": C2,
        "RMSE": RMSE, "R_squared": R_squared, "V_fitted": V_fitted,
        "I_prev": I_prev, "V_prior": V_prior
    }


def main():
    df = read_battery_data(TARGET_FILE)
    soc_value, dod_value = extract_soc_dod_from_filename(TARGET_FILE)
    print(f"SOC={soc_value} DOD={dod_value}")

    COL_TIME, COL_CURRENT, COL_VOLTAGE, COL_AH, COL_STATE = (
        'Zeit', 'Strom', 'Spannung', 'AhAkku', 'Zustand')
    df['Zustand_Clean'] = df[COL_STATE].astype(str).str.strip().str.upper()

    if pd.api.types.is_datetime64_any_dtype(df[COL_TIME]) or pd.api.types.is_timedelta64_any_dtype(df[COL_TIME]):
        df['Time_Seconds'] = (df[COL_TIME] - df[COL_TIME].iloc[0]).dt.total_seconds()
    else:
        df['Time_Seconds'] = pd.to_numeric(df[COL_TIME], errors='coerce') - pd.to_numeric(df[COL_TIME].iloc[0], errors='coerce')

    print("tracking PAU boundaries...")
    df_temp = df[['Time_Seconds', 'Zustand_Clean']].copy()
    df_temp['prev_state'] = df_temp['Zustand_Clean'].shift()
    change_points = df_temp[df_temp['Zustand_Clean'] != df_temp['prev_state']].copy().reset_index()
    print("Zustand values:", df['Zustand_Clean'].unique())

    valid_segments = []
    for k in range(len(change_points)):
        if (change_points.loc[k, 'prev_state'] == 'CHA' and change_points.loc[k, 'Zustand_Clean'] == 'PAU'):
            start_df_index = change_points.loc[k, 'index']
            end_df_index = None
            for look_ahead in range(k + 1, len(change_points)):
                next_state = change_points.loc[look_ahead, 'Zustand_Clean']
                if next_state == 'DCH':
                    end_df_index = change_points.loc[look_ahead, 'index'] - 1
                    break
                elif next_state == 'CHA':
                    break
            if end_df_index is not None:
                valid_segments.append((start_df_index, end_df_index))

    total_found = len(valid_segments)
    print(f"identified PAU(CHA->PAU->DCH) segments: {total_found}")
    if total_found == 0:
        print("no CHA->PAU->DCH closed loop found.")
        return

    num_to_extract = total_found if PROCESS_ALL_SEGMENTS else min(10, total_found)
    if MAX_SEGMENTS_TO_PROCESS is not None:
        num_to_extract = min(num_to_extract, MAX_SEGMENTS_TO_PROCESS)
    selected_indices = list(range(num_to_extract))
    print(f"processing {num_to_extract} segments")

    parameter_reports = []
    successful_count = 0
    for rank, idx in enumerate(tqdm(selected_indices, desc="fit", unit="seg")):
        start_idx, end_idx = valid_segments[idx]
        res = process_segment(df, start_idx, end_idx, rank, COL_VOLTAGE, COL_CURRENT)
        if res is None:
            continue
        successful_count += 1
        parameter_reports.append({
            "片段编号": rank + 1,
            "OCV (V)": round(res["OCV"], 6),
            "R0 (Ohm)": round(res["R0"], 6),
            "R1 (Ohm)": round(res["R1"], 6),
            "C1 (F)": round(res["C1"], 2),
            "tau1 (s)": round(res["tau1"], 2),
            "R2 (Ohm)": round(res["R2"], 6),
            "C2 (F)": round(res["C2"], 2),
            "tau2 (s)": round(res["tau2"], 2),
            "RMSE (V)": round(res["RMSE"], 6),
            "R²": round(res["R_squared"], 6),
            "I_prev (A)": round(res["I_prev"], 4),
            "V_prior (V)": round(res["V_prior"], 4),
        })

    print(f"\nsuccess {successful_count} / {num_to_extract}")
    if parameter_reports:
        summary_df = pd.DataFrame(parameter_reports)
        param_csv = os.path.join(OUTPUT_DIR, "parameter_identification.csv")
        summary_df.to_csv(param_csv, index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, encoding='utf-8-sig')
        print(f"-> {param_csv}")
        target_cols = ['OCV (V)', 'R0 (Ohm)', 'R1 (Ohm)', 'C1 (F)', 'R2 (Ohm)',
                       'C2 (F)', 'tau1 (s)', 'tau2 (s)', 'RMSE (V)', 'R²']
        print(f"\n{'param':<12}{'mean':>14}{'std':>14}{'min':>14}{'max':>14}")
        for col in target_cols:
            s = summary_df[col]
            print(f"{col:<12}{s.mean():>14.6f}{s.std():>14.6f}{s.min():>14.6f}{s.max():>14.6f}")
        # degeneracy check: how many railed tau2 (their single-start weakness)
        n_rail = int((summary_df['tau2 (s)'] >= 0.999 * 5000.0).sum())
        n_rail_lo = int((summary_df['tau2 (s)'] <= 30.1).sum())
        print(f"\ntau2 railed high (>=4995s): {n_rail} / {len(summary_df)}")
        print(f"tau2 railed low  (<=30.1s): {n_rail_lo} / {len(summary_df)}")
        print(f"R0 == 0 floor (1e-6):       {int((summary_df['R0 (Ohm)']<=1e-6).sum())}")


if __name__ == "__main__":
    main()
