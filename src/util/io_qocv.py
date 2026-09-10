"""Loading and coulomb-counting primitives for qOCV export bundles.

Low-level companion to :mod:`util.io_eis`: filename metadata plus the
voltage/capacity integration that turns a ``30_export_qocv`` parquet into the
``(V, Q)`` pair every qOCV consumer starts from.

These live in ``util`` rather than ``analysis.qocv_curve`` because
:mod:`util.soc_from_qocv` — the single source of SOC for the whole
characterization track — needs them, and ``util`` must not depend on
``analysis``. ``analysis.qocv_curve`` re-exports them, so the plotting CLI and
anything importing them from there keeps working unchanged.
"""

import re

import numpy as np
import pandas as pd


def _parse_soh(name):
    """Pull the SOH value out of a ``..._BM<n>_<SOH>SOH`` filename."""
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?|NA)SOH", name)
    return float(m.group(1)) if (m and m.group(1) != "NA") else np.nan


def _parse_bm(name):
    """Pull the ``BM<n>`` program number out of a filename (or ``None``)."""
    m = re.search(r"_BM(\d+)_", name)
    return int(m.group(1)) if m else None


def load_sweep(path, discharge=False):
    """Load one qOCV sweep -> ``(voltage, capacity_Ah)``, capacity from empty.

    Capacity is the trapezoidal integral of ``|Current|`` over time. For a
    discharge sweep the arrays are reversed so both branches run **empty→full**
    (voltage ascending), which makes charge/discharge directly comparable on a
    shared SOC or voltage axis.
    """
    df = pd.read_parquet(path)
    df = df.sort_values("Time").reset_index(drop=True)
    dt = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds().to_numpy()
    cur = np.abs(df["Current"].to_numpy(dtype=float))
    q = np.concatenate([[0.0], np.cumsum(cur[1:] * np.diff(dt))]) / 3600.0
    v = df["Voltage"].to_numpy(dtype=float)
    if discharge:                       # count capacity from empty, V ascending
        q = q[-1] - q
        v, q = v[::-1], q[::-1]
    return v, q


def soc_axis(q):
    """Throughput-normalised SOC (%) for a capacity array counted from empty."""
    span = q.max() - q.min()
    return (q - q.min()) / span * 100.0 if span else np.zeros_like(q)
