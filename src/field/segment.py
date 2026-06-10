"""F2 — DRIVE / CHARGE / REST segmenter for EV field-data time series.

Operates on the canonical schema produced by ``field.io_rwth.load_field_test``.

Segmentation model (tailored to BMS-rate field logs, RWTH Aachen reference):

* **REST** = gaps in the time series. The logger sleeps when the car is parked
  with key off, so REST shows up as the *absence* of rows rather than as
  low-current samples. Any inter-row Δt above ``gap_threshold_s`` becomes a
  synthetic REST segment.
* **CHARGE** = a run of consecutive rows where ``Current > i_charge_threshold``
  lasts at least ``min_charge_duration_s``. The duration requirement
  intentionally excludes brief regen-braking bursts during driving.
* **DRIVE** = everything else inside an active session.

Sign convention (RWTH dataset): positive Current = charging (SOC rises),
negative = discharging (SOC falls). Verified at runtime by
``check_sign_convention`` against dSOC/dt.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


STATE_REST = "REST"
STATE_CHARGE = "CHARGE"
STATE_DRIVE = "DRIVE"

DEFAULT_GAP_THRESHOLD_S = 300         # log gap above this = parked, key off
DEFAULT_I_CHARGE_THRESHOLD = 0.5      # A — minimum positive current to count as charging
DEFAULT_MIN_CHARGE_DURATION_S = 60    # sustained positive-current run must exceed this


def check_sign_convention(df: pd.DataFrame) -> None:
    """Assert positive Current means charging (SOC rises).

    Uses dSOC/dt as ground truth — unlike Speed, it is unambiguous for EVs
    that regen-charge during driving (positive Current with nonzero Speed).
    """
    needed = {"Time", "SOC", "Current"}
    if not needed.issubset(df.columns):
        return
    d = df[["Time", "SOC", "Current"]].dropna().sort_values("Time")
    dt = d["Time"].diff().dt.total_seconds()
    dsoc = d["SOC"].diff()
    m = (dt > 0) & (dt < 5) & dsoc.notna()
    if m.sum() < 100:
        return
    s = d.loc[m].assign(dsoc=dsoc[m])
    pos_i = s.loc[s["dsoc"] > 0, "Current"].mean()
    neg_i = s.loc[s["dsoc"] < 0, "Current"].mean()
    if pd.isna(pos_i) or pd.isna(neg_i):
        return
    if pos_i <= 0 or neg_i >= 0:
        raise ValueError(
            f"Current sign convention looks flipped: mean Current where SOC rises "
            f"is {pos_i:.2f} A, where SOC falls is {neg_i:.2f} A. "
            f"Expected positive Current to charge the battery."
        )


def classify_states(
    df: pd.DataFrame,
    *,
    i_charge_threshold: float = DEFAULT_I_CHARGE_THRESHOLD,
    min_charge_duration_s: float = DEFAULT_MIN_CHARGE_DURATION_S,
    verify_sign: bool = True,
) -> pd.DataFrame:
    """Add a per-row ``state`` column ∈ {DRIVE, CHARGE} (REST is handled at segment level).

    A row is CHARGE if it sits inside a maximal run of consecutive rows with
    ``Current > i_charge_threshold`` whose wall-clock duration is at least
    ``min_charge_duration_s``. Everything else is DRIVE.
    """
    if verify_sign:
        check_sign_convention(df)
    df = df.sort_values("Time").reset_index(drop=True)

    is_pos = (df["Current"] > i_charge_threshold).fillna(False).to_numpy()
    # Run id changes whenever is_pos flips.
    run_id = np.cumsum(np.concatenate(([True], is_pos[1:] != is_pos[:-1])))
    t = df["Time"].to_numpy()
    run_durations = pd.Series(t, index=run_id).groupby(level=0).agg(lambda s: (s.max() - s.min()).total_seconds() if len(s) > 1 else 0.0)
    long_pos_runs = set(run_durations[(run_durations >= min_charge_duration_s)].index)
    # A run is CHARGE only if it's a positive-current run AND long enough.
    pos_run_ids = set(np.unique(run_id[is_pos]))
    charge_runs = long_pos_runs & pos_run_ids

    state = np.where(np.isin(run_id, list(charge_runs)), STATE_CHARGE, STATE_DRIVE)
    out = df.copy()
    out["state"] = state
    return out


def to_segments(
    df_with_state: pd.DataFrame,
    *,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> pd.DataFrame:
    """Collapse per-row state into one row per segment, injecting REST for log gaps.

    Within an active session: consecutive same-state rows merge into one segment.
    Between active sessions: any inter-row Δt > ``gap_threshold_s`` produces a
    synthetic REST segment spanning ``(Time[i], Time[i+1])`` with ``n_rows = 0``.
    """
    if "state" not in df_with_state.columns:
        raise ValueError("Frame is missing 'state' column — call classify_states first.")
    df = df_with_state.sort_values("Time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    dt_s = df["Time"].diff().dt.total_seconds().fillna(0.0)
    gap_break = (dt_s > gap_threshold_s).to_numpy()
    state_change = (df["state"] != df["state"].shift()).to_numpy()
    # A new active segment starts whenever the state changes OR we just crossed a gap.
    new_seg = state_change | gap_break
    seg_id = new_seg.cumsum()

    g = df.assign(_seg=seg_id).groupby("_seg", sort=True)
    # Build active-segment frame preserving tz on Time columns.
    active = pd.DataFrame({
        "state": g["state"].first().to_numpy(),
        "n_rows": g.size().to_numpy(),
        "Current_mean": g["Current"].mean().to_numpy(),
        "Voltage_mean": g["Voltage"].mean().to_numpy(),
        "SOC_start": g["SOC"].first().to_numpy(),
        "SOC_end": g["SOC"].last().to_numpy(),
    })
    active["start_time"] = g["Time"].first().reset_index(drop=True)
    active["end_time"] = g["Time"].last().reset_index(drop=True)
    if "Speed" in df.columns:
        active["Speed_max"] = g["Speed"].max().to_numpy()
    if "Odometer" in df.columns:
        active["distance_km"] = (g["Odometer"].last() - g["Odometer"].first()).to_numpy()

    # Inject synthetic REST segments for every gap.
    gap_idx = np.where(gap_break)[0]
    rests = []
    for i in gap_idx:
        rests.append({
            "state": STATE_REST,
            "start_time": df.loc[i - 1, "Time"],
            "end_time": df.loc[i, "Time"],
            "n_rows": 0,
            "Current_mean": np.nan,
            "Voltage_mean": np.nan,
            "SOC_start": df.loc[i - 1, "SOC"],
            "SOC_end": df.loc[i, "SOC"],
        })
    if rests:
        rest_df = pd.DataFrame(rests)
        for col in ("Speed_max", "distance_km"):
            if col in active.columns:
                rest_df[col] = np.nan
        out = pd.concat([active, rest_df], ignore_index=True)
    else:
        out = active

    out = out.sort_values("start_time").reset_index(drop=True)
    out.insert(0, "segment_id", out.index)
    out["duration_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds()
    return out


def validate_against_trips(
    segments: pd.DataFrame,
    trips: pd.DataFrame,
    vehicle: str,
    *,
    overlap_tolerance_s: float = 60.0,
) -> dict:
    """Compare DRIVE segments to ground-truth trips for one vehicle.

    A trip is "recovered" if at least one DRIVE segment overlaps its
    [start, start+duration] window (with ``overlap_tolerance_s`` slack on
    either end). Returns recall, precision (fraction of DRIVE segments that
    matched some trip), and median |our ΔSOC − true ΔSOC| over matched trips.
    """
    trips_v = trips[trips["vehicle"] == vehicle].copy()
    if trips_v.empty:
        return {"vehicle": vehicle, "n_trips": 0, "note": "no trips for vehicle"}
    trips_v["start"] = pd.to_datetime(trips_v["starttime"], utc=True)
    trips_v["end"] = trips_v["start"] + pd.to_timedelta(trips_v["duration_seconds"], unit="s")

    drive_segs = segments[segments["state"] == STATE_DRIVE].copy()
    if drive_segs.empty:
        return {"vehicle": vehicle, "n_trips": len(trips_v), "recall": 0.0, "precision": 0.0}

    # Ensure both sides are tz-aware UTC for safe comparison.
    drive_segs["start_time"] = pd.to_datetime(drive_segs["start_time"], utc=True)
    drive_segs["end_time"] = pd.to_datetime(drive_segs["end_time"], utc=True)

    tol = pd.to_timedelta(overlap_tolerance_s, unit="s")
    matched_trips = 0
    matched_segs = set()
    soc_errors = []
    for _, t in trips_v.iterrows():
        ovl = drive_segs[
            (drive_segs["end_time"] >= t["start"] - tol)
            & (drive_segs["start_time"] <= t["end"] + tol)
        ]
        if not ovl.empty:
            matched_trips += 1
            matched_segs.update(ovl.index.tolist())
            our_dsoc = ovl["SOC_end"].iloc[-1] - ovl["SOC_start"].iloc[0]
            true_dsoc = t["soc_stop"] - t["soc_start"]
            if pd.notna(our_dsoc) and pd.notna(true_dsoc):
                soc_errors.append(abs(our_dsoc - true_dsoc))

    return {
        "vehicle": vehicle,
        "n_trips": int(len(trips_v)),
        "n_drive_segments": int(len(drive_segs)),
        "recall": matched_trips / len(trips_v),
        "precision": len(matched_segs) / len(drive_segs),
        "median_abs_soc_delta_error": float(np.median(soc_errors)) if soc_errors else None,
    }


def _summarize(segments: pd.DataFrame) -> None:
    by_state = segments.groupby("state").agg(
        n=("segment_id", "count"),
        total_hours=("duration_s", lambda s: s.sum() / 3600),
        median_duration_s=("duration_s", "median"),
    )
    print(by_state)


if __name__ == "__main__":
    import argparse
    from field import io_rwth

    parser = argparse.ArgumentParser(description="Smoke-test the field-data segmenter")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/rwth_aachen",
    )
    parser.add_argument("--vehicle", default="Smart-1")
    parser.add_argument("--trips", default="trip_data/GeriatricCare_trip_data_datafile.parquet")
    args = parser.parse_args()

    print(f"Loading {args.vehicle}...")
    df = io_rwth.load_field_test(io_rwth.field_test_path(args.base_dir, args.vehicle))
    print(f"  {len(df):,} rows, span {df['Time'].max() - df['Time'].min()}")

    print("\nClassifying states (per-row DRIVE/CHARGE)...")
    df = classify_states(df)
    print("  per-row state counts:", df["state"].value_counts().to_dict())

    print("\nBuilding segments (REST inserted at log gaps > 5 min)...")
    segs = to_segments(df)
    print(f"  {len(segs):,} segments")
    _summarize(segs)

    trips_path = os.path.join(args.base_dir, args.trips)
    if os.path.exists(trips_path):
        print(f"\nValidating DRIVE segments against {args.trips}...")
        trips = pd.read_parquet(trips_path)
        result = validate_against_trips(segs, trips, args.vehicle)
        for k, v in result.items():
            print(f"  {k}: {v}")
