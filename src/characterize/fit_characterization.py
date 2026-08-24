"""Fit the characterization bundles and plot them.

Reads ``<working_path>/10_initial_characterization/<cell>/data/`` — the pulse,
EIS and qOCV parquets written by :mod:`characterize.main_para` — and writes
``<cell>_parameters.json`` plus ``plots/`` beside it. Runs standalone, so fits
can be repeated without redoing segmentation.

Models are fixed defaults:

* **pulse** — 2RC (:mod:`characterize.pulse_fit`, a fork of
  :mod:`analysis.fit_2rc_pulse` kept separate so characterization-only fixes
  — notably SOC-plateau detection — don't touch the shared paper-analysis
  module; see the fork notice at the top of ``characterize/pulse_fit.py``).
* **EIS** — 2×ZARC + series-L + generalized Warburg
  (:func:`analysis.eis_vs_soc.fit_zarc_warburg_eis`). φ is fitted; τ_d is
  **pinned** by ``DIFFUSION_TAU_BOX``, so ``R_d_z`` is the amplitude at
  ω = 1/τ_d and ``tau_d_z`` is a shape constant, not a result. Those settings
  are recorded in the params file so the numbers stay interpretable. Each
  bundle's charge/discharge sweep leg(s) are detected from its own
  per-measurement terminal voltage (config ``soc_sweep_direction`` overrides
  detection; ``soc_step_pct`` sets the SOC step, default 5.0) — see
  ``fit_eis`` / ``_split_bundle_into_legs``. The pulse fit shares the same
  ``soc_sweep_direction``/``soc_step_pct`` keys: a run sweeps SOC one way for
  both measurements, and pulses use ``fit_2rc_pulse.assign_pulse_soc`` to
  derive per-plateau ``SOC_pct`` (see ``fit_pulse``), not the module's
  90/50/10 aging-checkup schema.
* **qOCV** — no fit; the curve and its throughput-normalised capacities.

    python -m characterize.fit_characterization <battery_cfg> [--cells …]
        [--only {pulse,eis,qocv} …]

The three blocks fit independently: ``--only eis`` refits (and replots) EIS
alone and merges it into the existing ``<cell>_parameters.json``, leaving the
``pulse``/``qocv`` blocks of the previous run in place. Omitting ``--only``
fits all three, as before.
"""

import argparse
import glob
import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis import eis_vs_soc, qocv_curve
from analysis import sweep_direction as sweep_direction_mod
from characterize import pulse_fit
from main import load_config
from util import io_router
from util.run_context import CHARACTERIZATION

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PULSE_MODEL = "2rc"
EIS_MODEL = "2zarc_warburg"

#: The independently fittable blocks, in payload order. A run may do any
#: subset (``--only``); the untouched blocks are carried over from the
#: existing ``<cell>_parameters.json`` instead of being dropped.
FIT_PARTS = ("pulse", "eis", "qocv")

#: Columns lifted from the 2RC results table into the params file. No ``SOC``/
#: ``pulse_type`` here on purpose: those are the aging-checkup 90/50/10 schema
#: (``fit_2rc_pulse.SOC_ORDER``) which a full-SOC-sweep characterization run
#: does not fit — every pulse would read the same ``DEFAULT_SOC``. ``SOC_pct``
#: (from ``assign_pulse_soc``, see ``fit_pulse``) is the only SOC that appears.
PULSE_COLS = [
    "pulse_segment_id", "ID", "BM_Programm", "SOH", "direction",
    "I_A", "C_rate", "OCV_V", "R0_ohm", "R1_ohm", "tau1_s", "R2_ohm",
    "tau2_s", "rmse_mV", "SOC_pct", "pulse_amplitude_A", "pulse_C_rate",
]

#: Columns lifted from the EIS table into the params file.
EIS_COLS = [
    "eis_number", "SOC_pct", "U", "R_ohm", "R1_z", "tau1_z", "alpha1_z",
    "R2_z", "tau2_z", "alpha2_z", "R_d_z", "tau_d_z", "phi_d_z",
    "zarc_rmse", "zarc_degenerate",
]


def _jsonable(value):
    """NumPy/pandas scalars -> plain Python; NaN/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.str_, str)):
        return str(value)
    return value


def _records(table: pd.DataFrame, cols: list) -> list:
    present = [c for c in cols if c in table.columns]
    return [
        {c: _jsonable(row[c]) for c in present}
        for _, row in table[present].iterrows()
    ]


def _skipped_nasoh_bundles(files: list, tail: str = "") -> list:
    """Bundles with SOH=NA (no CAP segment found for that BM_Programm, so no
    SOH could be computed) — shared by the pulse and EIS paths so the two
    can never disagree about what counts as NA. SOH is parsed with
    ``fit_2rc_pulse._parse_soh``, the same helper ``fit_folder`` uses
    internally for the pulse path.

    Mirrors the per-BM "largest file wins" selection every caller here uses
    (``fit_folder`` for pulse, the ``best`` dict in ``fit_eis``), so this
    reports exactly the files that will actually be evaluated and skipped —
    not every NASOH-named stub that happens to sit in the folder. ``tail`` is
    an optional caller-specific suffix appended to the reason string; the
    ``skipped`` entry shape (``file``/``BM_Programm``/``reason`` keys) is
    otherwise identical for pulse and EIS, so a params-file consumer never
    needs to special-case by measurement type.
    """
    best = {}  # BM_Programm -> (size, path)
    for f in files:
        m = re.search(r"_BM(\d+)_", os.path.basename(f))
        if not m:
            continue
        bm = m.group(1)
        size = os.path.getsize(f)
        if bm not in best or size > best[bm][0]:
            best[bm] = (size, f)
    skipped = []
    for bm, (_, f) in sorted(best.items()):
        soh = pulse_fit._parse_soh(os.path.splitext(os.path.basename(f))[0])
        if soh == "NA":
            reason = "SOH=NA (no CAP segment for this BM_Programm)" + tail
            logging.info("skip %s: %s", os.path.basename(f), reason)
            skipped.append({
                "file": os.path.basename(f),
                "BM_Programm": int(bm),
                "reason": reason,
            })
    return skipped


def _pulse_cell_stem(file_name: str) -> str:
    """Strip ``_pulse_BM<n>_<soh>SOH`` off a pulse-export basename.

    Mirrors the EIS naming (``eis_2zarc_warburg_{stem}_...`` where ``stem`` is
    the *whole* source-file stem, which already embeds ``BM``/``SOH`` because
    the EIS export filename does). The pulse export filename is
    ``<cell_stem>_pulse_BM<bm>_<soh>SOH.parquet``; returning just
    ``<cell_stem>`` lets the plot name rebuild ``BM``/``SOH`` explicitly from
    the results table (not by re-parsing the filename string) while still
    carrying the same identifying prefix. Falls back to the full stem if the
    suffix doesn't match (defensive; shouldn't happen for files ``fit_folder``
    actually fit).
    """
    stem = os.path.splitext(os.path.basename(str(file_name)))[0]
    m = re.match(r"^(.*)_pulse_BM\d+_.+SOH$", stem)
    return m.group(1) if m else stem


def fit_pulse(data_dir: str, plots_dir: str, nom_capacity: float, cfg: dict = None) -> dict:
    """2RC fit of every pulse bundle in ``data_dir``.

    Uses ``characterize.pulse_fit``'s full-SOC-sweep schema
    (``assign_pulse_soc``'s plateau-derived ``SOC_pct``, always applied by
    this fork's ``fit_folder``), **not** the aging-checkup 90/50/10 schema
    (``SOC``/``pulse_type``) — a parametrization sweep has on the order of 20
    SOC plateaus, not 3 cycles, so that schema would read every pulse as the
    same default SOC. Sweep direction is detected per bundle from the trend
    of each bundle's fitted pre-pulse ``OCV_V`` (rising -> charge, falling ->
    discharge), the same rule ``fit_eis`` uses for EIS legs
    (:mod:`analysis.sweep_direction`); config ``soc_sweep_direction``
    overrides it, and ``soc_step_pct`` sets the SOC step — both shared with the
    EIS path (one sweep, one direction/step for both measurements).

    One plot per ``BM_Programm`` (mirrors the EIS per-source/leg plots, not a
    single folder-wide figure) named ``pulse_2rc_{cell_stem}_BM{bm}_{soh}SOH.png``
    — ``cell_stem`` and ``soh`` read straight off the results table (``File``/
    ``SOH`` columns), not reformatted, so e.g. ``99.5SOH`` stays ``99.5SOH``.
    Every path (or, for a group with no figure, the skip reason) is recorded
    in ``block["plots"]``, one entry per BM.
    """
    cfg = cfg or {}
    files = sorted(glob.glob(os.path.join(data_dir, "*_pulse_BM*.parquet")))
    block = {"model": PULSE_MODEL, "fits": [], "sources": [os.path.basename(f) for f in files]}
    skipped = _skipped_nasoh_bundles(files, tail=" — fit_folder skips these")
    if skipped:
        block["skipped"] = skipped
    if not files:
        logging.info("no pulse bundles in %s", data_dir)
        return block

    raw_override = cfg.get("soc_sweep_direction")
    override = _normalize_sweep_direction(raw_override) if raw_override is not None else None
    step = float(cfg["soc_step_pct"]) if cfg.get("soc_step_pct") is not None else pulse_fit.SOC_SWEEP_STEP_PCT

    try:
        results, assignment = pulse_fit.fit_folder(
            data_dir,
            nom_capacity,
            pulse_fit.REMOVE_PULSE_BEFORE_MIN,
            pulse_fit.EXCLUDE_ZUSTAND_CURRENT,
            sweep_direction=override,
            soc_step_pct=step,
        )
    except Exception as exc:                      # one bad bundle ≠ dead cell
        logging.warning("pulse fit failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block
    if results.empty:
        block["error"] = "no pulses fit"
        return block

    block["settings"] = {
        "mode": "full-SOC-sweep (assign_pulse_soc, plateau-derived SOC_pct) — "
                "not the 90/50/10 aging-checkup schema",
        "soc_step_pct": step,
        "soc_direction_override": override,
        "plateau_gap_ah": round(nom_capacity * step / 100.0, 4),
        "bundles": assignment.get("bundles", []),
    }
    block["fits"] = _records(results, PULSE_COLS)

    default_reason = (
        "no SOC_pct — assign_pulse_soc needs Ah_throughput/Zustand in the bundle"
    )
    bundle_diag_by_bm = {
        d.get("BM_Programm"): d for d in assignment.get("bundles", [])
    }

    plots = []
    for bm, group in results.groupby("BM_Programm", sort=True):
        bm_int = int(bm)
        soh = str(group["SOH"].iloc[0])
        cell_stem = (
            _pulse_cell_stem(group["File"].iloc[0]) if "File" in group.columns else "cell"
        )
        out_png = os.path.join(plots_dir, f"pulse_2rc_{cell_stem}_BM{bm_int}_{soh}SOH.png")
        entry = {"BM_Programm": bm_int, "SOH": soh}

        has_soc_pct = "SOC_pct" in group.columns and group["SOC_pct"].notna().any()
        if not has_soc_pct:
            reason = bundle_diag_by_bm.get(bm_int, {}).get("reason") or default_reason
            logging.info("BM%s: vs-SOC plot: %s", bm_int, reason)
            entry["plot_skipped"] = reason
            plots.append(entry)
            continue

        try:
            pulse_fit.plot_vs_soc(
                group, out_png, title=f"initial characterization — BM{bm_int} {soh}SOH"
            )
        except Exception as exc:
            logging.warning("BM%s: pulse plot failed: %s", bm_int, exc)
            entry["plot_skipped"] = f"{type(exc).__name__}: {exc}"
            plots.append(entry)
            continue
        # plot_vs_soc no-ops silently (logs, doesn't raise) when every fit is
        # degenerate — don't record a path to a file that wasn't written.
        if os.path.isfile(out_png) and os.path.getsize(out_png) > 0:
            entry["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
        else:
            entry["plot_skipped"] = (
                "SOC_pct present but plot_vs_soc wrote nothing (all fits degenerate?)"
            )
        plots.append(entry)

    if plots:
        block["plots"] = plots
    return block


def _normalize_sweep_direction(direction: str) -> str:
    """Validate + normalize a ``soc_sweep_direction`` config override.

    Shared by the pulse and EIS paths (one sweep direction for both
    measurements). ``build_eis_table``/``assign_pulse_soc`` only check for a
    "cha" prefix (anything else is treated as discharge), so a typo like
    "charging" happens to work but "chrage" or "" would silently be read as
    discharge. Fail loudly instead.
    """
    try:
        return sweep_direction_mod.normalize_direction(direction)
    except ValueError:
        raise ValueError(
            f"soc_sweep_direction={direction!r} not recognized — use "
            f"'discharge' (or 'dch'/'dis...') or 'charge' (or 'cha...')"
        ) from None


#: Turning-point sensitivity for leg detection: a reversal only counts as a
#: real turning point (not sensor noise) once the U excursion since the last
#: extreme exceeds max(this absolute floor, this fraction of the bundle's
#: total U span). Same rule the pulse path uses on OCV_V, via
#: :mod:`analysis.sweep_direction` (``turn_threshold``/``direction_from_trend``).
EIS_U_TURN_ABS_THRESHOLD_V = sweep_direction_mod.U_TURN_ABS_THRESHOLD_V
EIS_U_TURN_FRAC_OF_SPAN = sweep_direction_mod.U_TURN_FRAC_OF_SPAN

#: Legs shorter than this many measurements are folded into a neighbour —
#: too few points to trust as their own sweep direction.
EIS_MIN_LEG_POINTS = 3


def _zigzag_pivot_indices(u: np.ndarray, threshold: float) -> list:
    """Indices of alternating peaks/valleys in ``u`` (plus the first and last
    index), i.e. the classic "zigzag" turning-point filter: a reversal is
    only recorded once the excursion from the running extreme exceeds
    ``threshold``, so small non-monotonic wobbles don't fragment the series.
    """
    n = len(u)
    if n <= 1:
        return list(range(n))
    pivots = [0]
    trend = 0  # 0 = undetermined yet, 1 = rising, -1 = falling
    cand_idx, cand_val = 0, u[0]
    for i in range(1, n):
        v = u[i]
        if trend == 0:
            if v - cand_val > threshold:
                trend, cand_idx, cand_val = 1, i, v
            elif cand_val - v > threshold:
                trend, cand_idx, cand_val = -1, i, v
            elif abs(v - u[0]) < abs(cand_val - u[0]):
                # still flat; track the running extreme anyway for later
                pass
        elif trend == 1:
            if v >= cand_val:
                cand_idx, cand_val = i, v
            elif cand_val - v > threshold:
                pivots.append(cand_idx)
                trend, cand_idx, cand_val = -1, i, v
        else:  # trend == -1
            if v <= cand_val:
                cand_idx, cand_val = i, v
            elif v - cand_val > threshold:
                pivots.append(cand_idx)
                trend, cand_idx, cand_val = 1, i, v
    if pivots[-1] != n - 1:
        pivots.append(n - 1)
    return pivots


def _detect_sweep_legs(u: np.ndarray, threshold: float) -> list:
    """Split a U-vs-order series into monotonic legs (start, end) index pairs
    (inclusive, consecutive legs sharing their boundary point). Legs with
    fewer than :data:`EIS_MIN_LEG_POINTS` measurements are folded into a
    neighbour rather than reported on their own.
    """
    n = len(u)
    if n == 0:
        return []
    pivots = _zigzag_pivot_indices(u, threshold)
    legs = [(pivots[k], pivots[k + 1]) for k in range(len(pivots) - 1)] or [(0, n - 1)]

    changed = True
    while changed and len(legs) > 1:
        changed = False
        for i, leg in enumerate(legs):
            if leg[1] - leg[0] + 1 < EIS_MIN_LEG_POINTS:
                if i == 0:
                    legs[0:2] = [(legs[0][0], legs[1][1])]
                else:
                    legs[i - 1:i + 1] = [(legs[i - 1][0], legs[i][1])]
                changed = True
                break
    return legs


def _split_bundle_into_legs(df: pd.DataFrame, step: float, override: str,
                            source_name: str) -> tuple:
    """Detect charge/discharge legs in one EIS bundle from its per-measurement
    terminal voltage ``U`` (rising = charge, falling = discharge — the signal
    ``build_eis_table`` already computes per ``eis_number``, ordered by
    ``Time``), and build a per-leg SOC table for each with the correct
    direction. Returns ``(tables, leg_diagnostics, leg_raw_dfs)`` — the third
    element is each leg's raw measured-spectra slice (for the raw-spectra /
    fit-overlay plots, which need the actual per-frequency points, not just
    the reduced per-measurement table).
    """
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()
    if not order:
        return [], [], []
    u_means = df.groupby("eis_number")["U"].mean()
    u = np.array([float(u_means[e]) for e in order], dtype=float)
    total_span = float(np.nanmax(u) - np.nanmin(u)) if len(u) > 1 else 0.0
    threshold = max(EIS_U_TURN_ABS_THRESHOLD_V, EIS_U_TURN_FRAC_OF_SPAN * total_span)

    leg_ranges = [(0, len(u) - 1)] if override else _detect_sweep_legs(u, threshold)

    tables, leg_info, leg_dfs = [], [], []
    for leg_idx, (s, e) in enumerate(leg_ranges):
        eids = order[s:e + 1]
        leg_df = df[df["eis_number"].isin(eids)]
        assumed = False
        if override:
            direction = override
        else:
            direction = sweep_direction_mod.direction_from_trend(u[s], u[e], threshold)
            if direction is None:
                assumed = True
                direction = "discharge"
                logging.warning(
                    "EIS direction ambiguous (U span %.1f mV < %.1f mV threshold) "
                    "in %s leg %d — assuming %s",
                    abs(u[e] - u[s]) * 1000, threshold * 1000,
                    source_name, leg_idx, direction,
                )
        table = eis_vs_soc.build_eis_table(leg_df, direction=direction, step=step)
        table["direction"] = direction
        table["sweep_leg"] = leg_idx
        tables.append(table)
        leg_dfs.append(leg_df)
        leg_info.append({
            "source": source_name,
            "sweep_leg": leg_idx,
            "direction": direction,
            "assumed": assumed,
            "n_measurements": len(eids),
            "u_start_V": float(u[s]),
            "u_end_V": float(u[e]),
        })
    return tables, leg_info, leg_dfs


def fit_eis(data_dir: str, plots_dir: str, soc_direction: str = None,
            soc_step_pct: float = None) -> dict:
    """2×ZARC + generalized-Warburg fit of every spectrum in every bundle.

    Direction is **detected per sweep leg** from the bundle's own per-
    measurement terminal voltage ``U`` (rising -> charge, falling ->
    discharge) rather than declared once for the whole cell: a single
    parametrization run may contain a charge sweep, a discharge sweep, or
    both back to back, so one config value can't describe it and a wrong one
    silently inverts the SOC axis (SOC is assigned by measurement *order* in
    ``build_eis_table``, not measured). ``soc_direction`` is an optional
    override that forces every leg's direction and skips detection —
    ``None`` (default) means "detect". ``soc_step_pct`` sets the SOC step
    (default matches ``build_eis_table``'s ``SOC_SWEEP_STEP_PCT``, 5.0).
    """
    override = _normalize_sweep_direction(soc_direction) if soc_direction is not None else None
    step = float(soc_step_pct) if soc_step_pct is not None else eis_vs_soc.SOC_SWEEP_STEP_PCT
    direction_source = "config-override" if override else "detected"

    files = sorted(glob.glob(os.path.join(data_dir, "*_eis_BM*.parquet")))
    block = {
        "model": EIS_MODEL,
        "settings": {
            "element": eis_vs_soc.ZARC_DIFFUSION_ELEMENT,
            "tau_d_pinned_s": list(eis_vs_soc.DIFFUSION_TAU_BOX)[0],
            "tau_d_is_fitted": eis_vs_soc.DIFFUSION_TAU_BOX[0]
                               != eis_vs_soc.DIFFUSION_TAU_BOX[1],
            "phi_box": list(eis_vs_soc.DIFFUSION_PHI_BOX),
            "alpha_min": eis_vs_soc.ZARC_ALPHA_MIN,
            "soc_step_pct": step,
            "soc_direction_source": direction_source,
            "soc_direction_override": override,
            "soc_turn_threshold_abs_V": EIS_U_TURN_ABS_THRESHOLD_V,
            "soc_turn_threshold_frac_of_span": EIS_U_TURN_FRAC_OF_SPAN,
            "legs": [],
        },
        "fits": [],
        "sources": [os.path.basename(f) for f in files],
    }
    if not files:
        logging.info("no EIS bundles in %s", data_dir)
        return block

    # NA-SOH bundles (no CAP segment for that BM_Programm) are excluded from
    # fitting/plotting here, same as the pulse path (fit_folder skips them
    # internally) — both use the shared _skipped_nasoh_bundles/_parse_soh so
    # "NA" can never mean different things between the two measurement types.
    skipped = _skipped_nasoh_bundles(files)
    if skipped:
        block["skipped"] = skipped
    skip_names = {s["file"] for s in skipped}
    files = [f for f in files if os.path.basename(f) not in skip_names]
    if not files:
        logging.info("EIS: every bundle in %s was SOH=NA — nothing to fit", data_dir)
        return block

    tables, leg_diag = [], []
    leg_raw_by_key = {}  # (source, sweep_leg) -> raw measured-spectra df
    for path in files:
        name = os.path.basename(path)
        try:
            leg_tables, leg_info, leg_dfs = _split_bundle_into_legs(
                pd.read_parquet(path), step, override, name
            )
            for table, leg_df in zip(leg_tables, leg_dfs):
                table["source"] = name
                leg_raw_by_key[(name, int(table["sweep_leg"].iloc[0]))] = leg_df
            tables.extend(leg_tables)
            leg_diag.extend(leg_info)
        except Exception as exc:
            logging.warning("EIS fit failed for %s: %s", name, exc)
            block.setdefault("errors", []).append(
                {"source": name, "error": f"{type(exc).__name__}: {exc}"}
            )
    block["settings"]["legs"] = leg_diag
    if not tables:
        return block

    combined = pd.concat(tables, ignore_index=True)
    block["fits"] = _records(combined, EIS_COLS + ["source", "direction", "sweep_leg"])
    n_deg = int(combined.get("zarc_degenerate", pd.Series(dtype=bool)).sum())
    if n_deg:
        logging.warning("%d/%d EIS fits flagged degenerate", n_deg, len(combined))
    block["n_degenerate"] = n_deg

    # plot_zarc_vs_soc (and the raw-spectra / fit-overlay plots below) overlay
    # every row's SOC axis on one figure with no per-series separation — two
    # BM_Programms, or two legs of the same bundle, would silently overlay
    # (each leg's SOC restarts at its own sweep). Since that grouping can't be
    # expressed without editing analysis/eis_vs_soc.py, plot one figure per
    # (source, leg) instead and record every path.
    def _try_plot(kind, fn, out_png, *args, **kwargs):
        """Call a plotter and record its path only if it actually wrote a
        (non-empty) file — several of these plotters no-op silently on
        missing columns/empty input rather than raising, so a bare try/except
        would otherwise record a path to a file that doesn't exist.
        """
        try:
            fn(*args, out_png, **kwargs)
        except Exception as exc:
            logging.warning("EIS %s plot failed for %s leg %s: %s", kind, source, leg, exc)
            return
        if not (os.path.isfile(out_png) and os.path.getsize(out_png) > 0):
            logging.info("EIS %s plot: nothing written for %s leg %s", kind, source, leg)
            return
        plots.append({
            "kind": kind,
            "source": source,
            "sweep_leg": int(leg),
            "direction": direction,
            "plot": os.path.relpath(out_png, os.path.dirname(plots_dir)),
        })

    plots = []
    for (source, leg), group in combined.groupby(["source", "sweep_leg"], sort=False):
        stem = os.path.splitext(str(source))[0]
        direction = group["direction"].iloc[0]
        leg_title = f"initial characterization — {source} leg{leg} ({direction})"
        leg_df = leg_raw_by_key.get((source, int(leg)))

        out_png = os.path.join(plots_dir, f"eis_2zarc_warburg_{stem}_leg{leg}_{direction}.png")
        _try_plot("zarc_params_vs_soc", eis_vs_soc.plot_zarc_vs_soc, out_png,
                  group, title=leg_title)

        if leg_df is None or leg_df.empty:
            logging.warning(
                "no raw spectra retained for %s leg %s — skipping raw/overlay plots",
                source, leg,
            )
            continue

        raw_png = os.path.join(plots_dir, f"eis_raw_spectra_{stem}_leg{leg}_{direction}.png")
        _try_plot("raw_spectra", eis_vs_soc.plot_raw_spectra, raw_png,
                  leg_df, group, title=leg_title)

        overlay_png = os.path.join(plots_dir, f"eis_fit_overlay_{stem}_leg{leg}_{direction}.png")
        _try_plot("fit_overlay", eis_vs_soc.plot_fit_overlay, overlay_png,
                  leg_df, group, title=leg_title)
    if plots:
        block["plots"] = plots
    return block


def summarize_qocv(data_dir: str, plots_dir: str, nom_capacity: float) -> dict:
    """qOCV curve + throughput capacities. No fit — the curve is the result."""
    pairs = qocv_curve.find_pairs(data_dir)
    block = {"fits": [], "sources": []}
    if not pairs:
        logging.info("no complete qOCV cha/dch pair in %s", data_dir)
        return block

    bm, pair = qocv_curve._pick_pair(pairs)
    block["sources"] = [os.path.basename(pair["cha"]), os.path.basename(pair["dch"])]
    try:
        _, qc = qocv_curve.load_sweep(pair["cha"], discharge=False)
        _, qd = qocv_curve.load_sweep(pair["dch"], discharge=True)
        block["fits"] = [{
            "BM_Programm": int(bm),
            "SOH": _jsonable(pair.get("soh")),
            "capacity_cha_ah": round(float(np.nanmax(qc)), 4),
            "capacity_dch_ah": round(float(np.nanmax(qd)), 4),
        }]
    except Exception as exc:
        logging.warning("qOCV read failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block

    out_png = os.path.join(plots_dir, "qocv.png")
    try:
        qocv_curve.plot_qocv(pair["cha"], pair["dch"], out_png, nom_capacity=nom_capacity)
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("qOCV plot failed: %s", exc)
    return block


def normalize_parts(parts) -> list:
    """Validate/order a ``--only`` selection; ``None`` (or empty) → all parts."""
    if not parts:
        return list(FIT_PARTS)
    wanted = {str(p).strip().lower() for p in parts}
    unknown = sorted(wanted - set(FIT_PARTS))
    if unknown:
        raise ValueError(
            f"unknown fit part(s) {unknown} — pick from {list(FIT_PARTS)}"
        )
    return [p for p in FIT_PARTS if p in wanted]


def _merge_parameters(out_json: str, payload: dict) -> dict:
    """Overlay `payload` on the existing params file, keeping unfitted blocks."""
    if not os.path.isfile(out_json):
        return payload
    try:
        with open(out_json) as f:
            previous = json.load(f)
    except (OSError, ValueError) as exc:
        logging.warning("%s unreadable, writing a fresh one: %s", out_json, exc)
        return payload
    if not isinstance(previous, dict):
        logging.warning("%s is not a JSON object, writing a fresh one", out_json)
        return payload
    return {**previous, **payload}


def fit_cell(cell_dir: str, nom_capacity: float, cfg: dict = None,
             parts: list = None) -> dict:
    """Fit one `10_initial_characterization/<cell>/` folder; return the payload.

    `parts` selects which blocks to (re)fit — any subset of `FIT_PARTS`;
    ``None`` means all of them. Blocks left out are simply absent from the
    returned payload (`run` merges them into the existing parameters.json).
    """
    cfg = cfg or {}
    parts = normalize_parts(parts)
    stem = os.path.basename(os.path.normpath(cell_dir))
    data_dir = os.path.join(cell_dir, "data")
    plots_dir = os.path.join(cell_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    payload = {
        "cell": stem,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nom_capacity": nom_capacity,
    }
    if "pulse" in parts:
        payload["pulse"] = fit_pulse(data_dir, plots_dir, nom_capacity, cfg=cfg)
    if "eis" in parts:
        payload["eis"] = fit_eis(
            data_dir, plots_dir,
            soc_direction=cfg.get("soc_sweep_direction"),
            soc_step_pct=cfg.get("soc_step_pct"),
        )
    if "qocv" in parts:
        payload["qocv"] = summarize_qocv(data_dir, plots_dir, nom_capacity)
    return payload


def run(cfg: dict, target_cells: list = None, parts: list = None) -> None:
    parts = normalize_parts(parts)
    working_path = cfg.get("working_path")
    if not working_path:
        raise ValueError("working_path required in the battery config")
    if not io_router.writes_local(cfg):
        upload_to = cfg.get("upload_to", "local")
        raise ValueError(
            "characterize.fit_characterization reads bundles from local disk "
            f"only, but this config's upload_to={upload_to!r} does not write "
            "locally. Set upload_to to 'local' or 'both' (MinIO-only bundles "
            "produced by main_para are not read by this stage yet)."
        )
    root = os.path.join(working_path, CHARACTERIZATION.export_root_prefix)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found — run `python -m characterize.main_para <cfg>` first"
        )

    cells = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if target_cells:
        cells = [c for c in cells if any(t in c for t in target_cells)]
    if not cells:
        logging.warning("no characterization cell folders under %s", root)
        return

    n_ok, n_failed = 0, 0
    for stem in cells:
        cell_dir = os.path.join(root, stem)
        logging.info("fitting %s (%s)", stem, ", ".join(parts))
        try:
            payload = fit_cell(
                cell_dir, float(cfg["nom_capacity"]), cfg=cfg, parts=parts,
            )
            out_json = os.path.join(cell_dir, f"{stem}_parameters.json")
            with open(out_json, "w") as f:
                json.dump(_merge_parameters(out_json, payload), f, indent=2)
            logging.info("%s: parameters -> %s", stem, out_json)
            n_ok += 1
        except Exception as exc:           # one bad cell must not abort the run
            logging.warning("%s: fit_cell failed, skipping cell: %s", stem, exc)
            n_failed += 1
    logging.info("done: %d cell(s) fit, %d failed", n_ok, n_failed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit + plot the initial-characterization bundles"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cell stems")
    parser.add_argument(
        "--only", nargs="+", choices=list(FIT_PARTS), metavar="PART",
        help=(
            "Fit only these blocks (%s). Omit for all three. The blocks left "
            "out keep their previous results in <cell>_parameters.json."
            % "/".join(FIT_PARTS)
        ),
    )
    args = parser.parse_args()
    run(load_config(args.config), target_cells=args.cells, parts=args.only)


if __name__ == "__main__":
    main()
