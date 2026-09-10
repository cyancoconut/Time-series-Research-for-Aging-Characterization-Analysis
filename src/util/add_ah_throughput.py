import numpy as np

# An interval counts as dead time (cell parked between test files) when its Δt
# exceeds this multiple of the cell's normal sampling cadence (median Δt). The
# in-test cadence is seconds; inter-file gaps are minutes-to-days, so they sit
# far above any sane multiple and the cut is not sensitive to the exact factor.
GAP_FACTOR = 50.0


def add_ah_throughput(df_cell, gap_threshold_s=None, gap_factor=GAP_FACTOR):
    """Add a cumulative ``Ah_throughput`` column (trapezoidal ∫|I| dt, in Ah).

    Rows must be time-sorted on ``Time_UTC``. The integral is the cumulative
    sum of per-interval trapezoid contributions
    ``0.5·(|I[i]| + |I[i-1]|)·Δt_h``.

    Dead time between separate measurement files is masked so it books no Ah:
    during a gap the cell is parked/disconnected and no current flows, but the
    raw trapezoid would linearly interpolate the (possibly non-zero) boundary
    currents across the gap and invent throughput — enough to make the very
    first check-up read hundreds of Ah. Genuine in-test rests carry |I|≈0, so
    masking them changes nothing.

    The gap cut is derived from the data, not a fixed seconds value: an interval
    is a gap when its Δt exceeds ``gap_factor`` × the median Δt (the cell's
    normal sampling cadence). This adapts per cell automatically. Pass an
    explicit ``gap_threshold_s`` to override with a fixed seconds value.
    """
    if "Ah_throughput" in df_cell.columns:
        return df_cell

    # Δt per interval (seconds); first row has no predecessor.
    dt_s = df_cell["Time_UTC"].diff().dt.total_seconds().to_numpy()
    dt_s[0] = 0.0

    abs_i = np.abs(df_cell["Current"].to_numpy(dtype=float))

    # Trapezoid contribution of each interval (between row i-1 and i), in Ah.
    contrib = np.zeros(len(abs_i))
    contrib[1:] = 0.5 * (abs_i[1:] + abs_i[:-1]) * (dt_s[1:] / 3600.0)

    # Resolve the gap cut: explicit override, else gap_factor × median cadence.
    if gap_threshold_s is None and len(dt_s) > 2:
        cadence = np.nanmedian(dt_s[1:])
        if cadence > 0:
            gap_threshold_s = gap_factor * cadence

    if gap_threshold_s is not None:
        contrib[dt_s > gap_threshold_s] = 0.0

    df_cell["Ah_throughput"] = np.cumsum(contrib)
    return df_cell
