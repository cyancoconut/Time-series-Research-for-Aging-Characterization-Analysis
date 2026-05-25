"""F2 — charging-session segmentation and per-session feature extraction.

Operates on the canonical schema produced by ``field.io_shiyunliu.load_vehicle``.
Charging sessions are split where the inter-row time gap exceeds
``gap_s`` (default 10 s — matches the author's ``capacity_extract.py``).
Per-session features feed an HDBSCAN clustering downstream that picks the
"full CC-CV" cluster as the CAP equivalent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GAP_S = 10
DEFAULT_MIN_ROWS = 100           # author's threshold for a "valid" charging session
DEFAULT_CV_TAIL_FRAC = 0.10      # last 10% of rows define the CV-tail window
DEFAULT_CV_I_RATIO = 0.5         # tail median I must be < 0.5× peak I
DEFAULT_CV_V_RATIO = 0.97        # tail median V must be > 0.97× session V_max


def split_sessions(df: pd.DataFrame, *, gap_s: float = DEFAULT_GAP_S) -> pd.DataFrame:
    """Add a ``session_id`` column based on inter-row time gaps.

    A new session starts whenever the gap to the previous row exceeds ``gap_s``.
    Session ids are 0-based and monotonically increasing.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    dt_s = df["Time"].diff().dt.total_seconds()
    new_session = (dt_s > gap_s) | dt_s.isna()
    df = df.copy()
    df["session_id"] = new_session.cumsum() - 1
    return df


def _has_cv_tail(
    current: pd.Series,
    voltage: pd.Series,
    *,
    frac: float,
    i_ratio: float,
    v_ratio: float,
) -> bool:
    n = len(current)
    if n < 20:
        return False
    tail_n = max(5, int(np.ceil(n * frac)))
    tail_i = current.iloc[-tail_n:]
    tail_v = voltage.iloc[-tail_n:]
    i_peak = current.max()
    v_peak = voltage.max()
    if not np.isfinite(i_peak) or i_peak <= 0 or not np.isfinite(v_peak) or v_peak <= 0:
        return False
    return bool(
        tail_i.median() < i_ratio * i_peak
        and tail_v.median() > v_ratio * v_peak
    )


def session_features(
    df_with_session_id: pd.DataFrame,
    *,
    vehicle: str | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    cv_tail_frac: float = DEFAULT_CV_TAIL_FRAC,
    cv_i_ratio: float = DEFAULT_CV_I_RATIO,
    cv_v_ratio: float = DEFAULT_CV_V_RATIO,
) -> pd.DataFrame:
    """One row per charging session with features for downstream clustering.

    Drops sessions with fewer than ``min_rows`` rows (incomplete / noisy fragments).
    """
    if "session_id" not in df_with_session_id.columns:
        raise ValueError("Frame is missing 'session_id' — call split_sessions first.")
    df = df_with_session_id

    g = df.groupby("session_id", sort=True)
    sizes = g.size()
    keep = sizes[sizes >= min_rows].index
    df = df[df["session_id"].isin(keep)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("session_id", sort=True)

    feats = pd.DataFrame({
        "session_id": sizes.loc[keep].index,
        "n_rows": sizes.loc[keep].to_numpy(),
        "start_time": g["Time"].first().to_numpy(),
        "end_time": g["Time"].last().to_numpy(),
        "SOC_start": g["SOC"].first().to_numpy(),
        "SOC_end": g["SOC"].last().to_numpy(),
        "I_mean": g["Current"].mean().to_numpy(),
        "I_std": g["Current"].std().to_numpy(),
        "I_p99": g["Current"].quantile(0.99).to_numpy(),
        "V_start": g["Voltage"].first().to_numpy(),
        "V_end": g["Voltage"].last().to_numpy(),
        "V_max": g["Voltage"].max().to_numpy(),
        "T_mean": g["Temperature"].mean().to_numpy(),
    })
    feats["duration_s"] = (
        pd.to_datetime(feats["end_time"], utc=True) - pd.to_datetime(feats["start_time"], utc=True)
    ).dt.total_seconds()
    feats["dSOC"] = feats["SOC_end"] - feats["SOC_start"]

    # CV-tail flag — has to be computed per session, not from aggregates.
    cv_flags = g.apply(
        lambda sub: _has_cv_tail(
            sub["Current"], sub["Voltage"],
            frac=cv_tail_frac, i_ratio=cv_i_ratio, v_ratio=cv_v_ratio,
        ),
        include_groups=False,
    )
    feats["has_cv_tail"] = feats["session_id"].map(cv_flags).fillna(False).astype(bool)

    if vehicle is not None:
        feats.insert(0, "vehicle", vehicle)
    return feats.reset_index(drop=True)


def _summarize(feats: pd.DataFrame) -> None:
    n = len(feats)
    print(f"  {n:,} sessions kept")
    if n == 0:
        return
    print(f"  duration_s: median={feats['duration_s'].median():.0f}  "
          f"p90={feats['duration_s'].quantile(0.9):.0f}  "
          f"max={feats['duration_s'].max():.0f}")
    print(f"  dSOC      : median={feats['dSOC'].median():.1f}  "
          f"p90={feats['dSOC'].quantile(0.9):.1f}  "
          f"max={feats['dSOC'].max():.1f}")
    print(f"  I_mean A  : median={feats['I_mean'].median():.1f}  "
          f"I_p99 median={feats['I_p99'].median():.1f}")
    cv = int(feats["has_cv_tail"].sum())
    big_dsoc = int((feats["dSOC"].abs() >= 50).sum())
    big_dsoc_cv = int(((feats["dSOC"].abs() >= 50) & feats["has_cv_tail"]).sum())
    near_full = int((feats["dSOC"] > 70).sum())
    near_full_cv = int(((feats["dSOC"] > 70) & feats["has_cv_tail"]).sum())
    print(f"  has_cv_tail: {cv}  ({cv/n*100:.1f}%)")
    print(f"  |dSOC|≥50  : {big_dsoc}   + CV tail: {big_dsoc_cv}")
    print(f"  dSOC>70    : {near_full}   + CV tail: {near_full_cv}   ← CAP candidates")


if __name__ == "__main__":
    import argparse
    from field import io_shiyunliu

    parser = argparse.ArgumentParser(description="Smoke-test the shiyunliu session features")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev",
    )
    parser.add_argument("--vehicle", default="1")
    parser.add_argument("--all", action="store_true", help="Run on all 20 vehicles")
    args = parser.parse_args()

    vehicles = io_shiyunliu.list_vehicles(args.base_dir) if args.all else [args.vehicle]
    for v in vehicles:
        print(f"\n=== vehicle #{v} ===")
        df = io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(args.base_dir, v))
        df = split_sessions(df)
        n_raw = int(df["session_id"].max()) + 1 if len(df) else 0
        feats = session_features(df, vehicle=v)
        print(f"  raw sessions (10s gap): {n_raw}")
        _summarize(feats)
