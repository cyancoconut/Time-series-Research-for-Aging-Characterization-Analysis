import numpy as np
import pandas as pd

# An interval counts as dead time (cell parked between test files) when its Δt
# exceeds this multiple of the cell's normal sampling cadence (median Δt). The
# in-test cadence is seconds; inter-file gaps are minutes-to-days, so they sit
# far above any sane multiple and the cut is not sensitive to the exact factor.
GAP_FACTOR = 50.0


def gap_cut(dt_s, gap_threshold_s=None, gap_factor=GAP_FACTOR):
    """Resolve the dead-time cut in seconds: explicit value, else the cadence.

    ``dt_s`` is the per-interval spacing (the first entry is ignored, having no
    predecessor). Returns ``None`` when there is nothing to derive a cadence
    from, which means "mask nothing".
    """
    if gap_threshold_s is not None:
        return gap_threshold_s
    if len(dt_s) > 2:
        cadence = np.nanmedian(dt_s[1:])
        if cadence > 0:
            return gap_factor * cadence
    return None


def ah_contributions(time_utc, current, gap_threshold_s=None,
                     gap_factor=GAP_FACTOR):
    """Per-interval trapezoid contributions to ∫|I| dt, in Ah.

    One entry per row (the first is 0, having no predecessor), so a caller can
    place each contribution at the timestamp where the charge actually moved
    instead of only seeing the running total.

    **Must be computed within a single recording.** The trapezoid interpolates
    the current between consecutive samples, so interleaving the rows of two
    concurrent recordings makes it dip toward the other instrument's current
    between every pair of real samples. With a parallel zero-current recording
    at the same cadence that halves the integral exactly — see
    ``build_bronze_cu_with_ah._compute_ah``, which is why this is exposed
    separately from :func:`add_ah_throughput`.
    """
    # Via pandas so a tz-aware column, an object column of Timestamps and a
    # plain datetime64 all reduce to the same seconds — the callers differ.
    t = pd.to_datetime(pd.Series(time_utc), utc=True)
    dt_s = t.diff().dt.total_seconds().to_numpy()[1:]
    abs_i = np.abs(pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(dtype=float))

    contrib = np.zeros(len(abs_i))
    if len(abs_i) > 1:
        contrib[1:] = 0.5 * (abs_i[1:] + abs_i[:-1]) * (dt_s / 3600.0)
        cut = gap_cut(np.concatenate([[0.0], dt_s]), gap_threshold_s, gap_factor)
        if cut is not None:
            contrib[1:] = np.where(dt_s > cut, 0.0, contrib[1:])
    # A single unparseable current would otherwise turn the whole cumulative
    # sum downstream of it into NaN. One bad row is not a dead counter.
    return np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)


def add_ah_throughput(df_cell, gap_threshold_s=None, gap_factor=GAP_FACTOR):
    """Add a cumulative ``Ah_throughput`` column (trapezoidal ∫|I| dt, in Ah).

    Rows must be time-sorted on ``Time_UTC`` **and come from one recording** —
    see :func:`ah_contributions`.

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

    df_cell["Ah_throughput"] = np.cumsum(
        ah_contributions(
            df_cell["Time_UTC"].to_numpy(), df_cell["Current"].to_numpy(),
            gap_threshold_s=gap_threshold_s, gap_factor=gap_factor,
        )
    )
    return df_cell
