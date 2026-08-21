"""Shared charge/discharge sweep-direction detection.

A full-SOC-sweep characterization run (EIS or pulse) can go either way — start
full and empty, or start empty and fill — and a wrong guess silently inverts
the whole SOC axis (SOC is assigned by measurement *order*, not measured).
Rather than declare the direction once per cell in config, both the EIS leg
splitter (:mod:`characterize.fit_characterization`) and the pulse SOC-sweep
labeler (:func:`analysis.fit_2rc_pulse.assign_pulse_soc`) detect it from the
trend of a measured voltage (EIS: per-leg terminal ``U``; pulse: pre-pulse
``OCV_V``) using this one turning-point rule, so "rising -> charge, falling ->
discharge" reads the same way in both places.
"""

#: Turning-point sensitivity: a voltage excursion only counts as a real trend
#: (not sensor noise / a single outlier) once it exceeds max(this absolute
#: floor, this fraction of the series' total span).
U_TURN_ABS_THRESHOLD_V = 0.02
U_TURN_FRAC_OF_SPAN = 0.05


def turn_threshold(total_span: float) -> float:
    """Turning threshold (V) for a series spanning ``total_span`` volts."""
    return max(U_TURN_ABS_THRESHOLD_V, U_TURN_FRAC_OF_SPAN * total_span)


def direction_from_trend(u_start: float, u_end: float, threshold: float):
    """"charge"/"discharge" from a start->end voltage trend, or ``None`` if
    ambiguous (the excursion is below ``threshold``)."""
    diff = u_end - u_start
    if abs(diff) < threshold:
        return None
    return "charge" if diff > 0 else "discharge"


def normalize_direction(direction: str) -> str:
    """Validate + normalize a user-supplied direction override string.

    Accepts ``'discharge'``/``'dch'``/anything starting with 'dis', and
    ``'charge'``/``'cha'``/anything starting with 'cha'. Raises ``ValueError``
    on anything else — silently reading a typo as discharge (the old
    "anything not 'cha...' is discharge" behavior) is worse than failing loud.
    """
    d = str(direction).strip().lower()
    if d.startswith("dis") or d == "discharge":
        return "discharge"
    if d.startswith("cha") or d == "charge":
        return "charge"
    raise ValueError(
        f"direction={direction!r} not recognized — use 'discharge' (or "
        f"'dch'/'dis...') or 'charge' (or 'cha...')"
    )
