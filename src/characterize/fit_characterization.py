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
* **EIS** — N×ZARC + series-L + generalized Warburg
  (:func:`analysis.eis_vs_soc.fit_nzarc_warburg_eis`). **N is not fixed**: it
  is read off the bundle's own DRT
  (:func:`analysis.eis_drt.select_model_order`), 2 on the parametrization
  cells and 1 where only one arc is resolved, so a second branch is never
  fitted against structure that isn't there. φ is fitted; τ_d is
  **pinned** by ``DIFFUSION_TAU_BOX``, so ``R_d_z`` is the amplitude at
  ω = 1/τ_d and ``tau_d_z`` is a shape constant, not a result. Those settings
  are recorded in the params file so the numbers stay interpretable. Each
  bundle's charge/discharge sweep leg(s) are detected from its own
  per-measurement terminal voltage (config ``soc_sweep_direction`` overrides
  detection; ``soc_step_pct`` sets the SOC step, default 5.0) — see
  ``fit_eis`` / ``_bundle_direction``. The pulse fit shares the same
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

from analysis import eis_drt, eis_vs_soc, qocv_curve
from util import soc_from_qocv, soc_from_steps
from analysis import sweep_direction as sweep_direction_mod
from characterize import pulse_fit
from main import load_config
from util import io_qocv, io_router
from util.run_context import CHARACTERIZATION

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PULSE_MODEL = "2rc"
#: The EIS branch count is no longer part of the model name: it is chosen per
#: bundle from that bundle's DRT (:func:`analysis.eis_drt.select_model_order`),
#: so a fixed "2zarc_warburg" would be a claim the run may not honour. The
#: number actually used is in ``settings.bundles[].n_zarc`` and in the
#: ``n_zarc`` column of every fit row.
EIS_MODEL = "nzarc_warburg"

#: Fixed DRT regularisation used by the pipeline. See the note in ``fit_eis``
#: for why this is not the L-curve corner.
DRT_LAMBDA_DEFAULT = 1e-3

#: The independently fittable blocks, in payload order. A run may do any
#: subset (``--only``); the untouched blocks are carried over from the
#: existing ``<cell>_parameters.json`` instead of being dropped.
FIT_PARTS = ("pulse", "eis", "qocv")

#: Columns lifted from the 2RC results table into the params file. No ``SOC``/
#: ``pulse_type`` here on purpose: those are the aging-checkup 90/50/10 schema
#: (``fit_2rc_pulse.SOC_ORDER``) which a full-SOC-sweep characterization run
#: does not fit — every pulse would read the same ``DEFAULT_SOC``. ``SOC_pct``
#: (from ``assign_pulse_soc``, see ``fit_pulse``) is the only SOC that appears.
#: ``direction`` is the pulse's own CHA/DCH polarity; ``sweep_direction`` is
#: the direction of the SOC sweep it sits in (which qOCV hysteresis branch it
#: maps against). Two different things — hence two columns.
PULSE_COLS = [
    "pulse_segment_id", "ID", "BM_Programm", "SOH", "direction",
    "sweep_direction", "sweep_direction_source",
    "I_A", "C_rate", "OCV_V", "T_degC", "R0_ohm", "R1_ohm", "tau1_s", "R2_ohm",
    "tau2_s", "rmse_mV", "SOC_pct",
    "pulse_amplitude_A", "pulse_C_rate",
]

#: Columns lifted from the EIS table into the params file.
EIS_COLS = [
    "eis_number", "SOC_pct", "U",
    # Borrowed from the same-BM pulse bundle (see _pulse_temperatures);
    # temperature_source in settings.bundles[] says whether it is that or the
    # cell mean.
    "T_degC",
    "sweep_direction", "sweep_direction_source",
    # R_cross is the fit-free Z_imag=0 crossing; R0_z is the fitted series
    # term. They are not the same number and R_cross is NOT the ohmic
    # resistance (it was called R_ohm until #70, which invited exactly that
    # reading): on an inductive cell the crossing sits at a finite frequency
    # — f_cross_Hz, 255-355 Hz on the NFPP sweep — where the arcs still
    # contribute real part, so R_cross runs 12-16 % above R0_z, by an
    # SOC-dependent margin. f_cross_Hz is exported next to it so that is
    # visible from the CSV alone rather than having to be inferred.
    "R_cross", "f_cross_Hz", "R0_z", "L_z",
    "R0_hf", "R0_hf_sigma", "hf_rmse", "hf_n", "r0_pinned",
    # Branch slots are fixed (eis_vs_soc.ZARC_COLUMN_SLOTS) even when the DRT
    # asked for one arc, so one cell fitted with one branch and another fitted
    # with two still share a CSV schema. `n_zarc` says how many were fitted and
    # `drt_n_arcs` how many this spectrum's own DRT supported — they differ when
    # the bundle-level consensus overrode a single spectrum.
    "n_zarc", "drt_n_arcs",
    "R1_z", "tau1_z", "alpha1_z",
    "R2_z", "tau2_z", "alpha2_z", "R_d_z", "tau_d_z", "phi_d_z",
    "zarc_rmse", "zarc_degenerate",
]

#: Columns of the per-bundle DRT peak table (``<cell>_eis_drt_peaks.csv``).
#: ``width_decades`` is the discriminator the ECM cannot report: a *broad*
#: peak is one dispersed process, several *narrow* ones are separate branches
#: the 2×ZARC model may have no element for.
DRT_PEAK_COLS = [
    "eis_number", "SOC_pct", "tau_peak", "gamma_peak", "R_peak",
    "width_decades", "source",
]


def _temp_tag(temp) -> str:
    """Filename tag for a bundle temperature: ``26degreeC`` / ``-10degreeC``
    / ``NAdegreeC``.

    Rounded to whole °C: this is the *measured* mean, so a 25 °C-setpoint
    chamber reads e.g. ``26degreeC``. The unrounded value is in the params
    file — the tag exists to tell two otherwise identically-named plots apart,
    not to be read as a measurement.
    """
    if temp is None or pd.isna(temp):
        return "NAdegreeC"
    return f"{float(temp):.0f}degreeC"


def _parquet_temperature(path: str):
    """Mean ``Temperature`` of one bundle parquet, or NaN.

    Reads only that column. Missing column / unreadable file / all-NaN ->
    NaN, so a bundle exported before temperature was carried still fits.
    """
    try:
        col = pd.read_parquet(path, columns=["Temperature"])["Temperature"]
    except Exception as exc:
        logging.info("%s: no temperature (%s)", os.path.basename(path), exc)
        return np.nan
    col = pd.to_numeric(col, errors="coerce")
    return round(float(col.mean()), 2) if col.notna().any() else np.nan


def _pulse_temperatures(data_dir: str) -> tuple:
    """``({BM_Programm: T_degC}, cell_mean)`` from the pulse bundles.

    The EIS spectra carry no temperature of their own — the EIS device file
    has no thermocouple channel, only the cycler does — so an EIS bundle takes
    the temperature of the **pulse bundle of the same BM_Programm**, which is
    the same cell in the same chamber during the same programme. ``cell_mean``
    is the fallback for an EIS programme with no pulse counterpart; it is
    recorded as such (``temperature_source``) rather than passed off as a
    same-programme measurement.
    """
    by_bm = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_pulse_BM*.parquet"))):
        m = re.search(r"_BM(\d+)_", os.path.basename(path))
        if not m:
            continue
        temp = _parquet_temperature(path)
        if not pd.isna(temp):
            by_bm[int(m.group(1))] = temp
    values = list(by_bm.values())
    cell_mean = round(float(np.mean(values)), 2) if values else np.nan
    return by_bm, cell_mean


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


def fit_pulse(data_dir: str, plots_dir: str, nom_capacity: float,
              cfg: dict = None, steps: pd.DataFrame = None) -> dict:
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
    single folder-wide figure) named
    ``pulse_2rc_{cell_stem}_BM{bm}_{soh}SOH_{direction}_{temp}.png`` —
    ``cell_stem`` and ``soh`` read straight off the results table (``File``/
    ``SOH`` columns), not reformatted, so e.g. ``99.5SOH`` stays ``99.5SOH``;
    ``temp`` is the bundle's mean measured cell temperature (``26degreeC``, or
    ``NAdegreeC`` if the bundle carries none).
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

    bundle_diags = assignment.get("bundles", [])
    # Step charge first (a direct measurement of what moved between plateaus),
    # the qOCV curve for whatever it could not resolve.
    results, stepped_bms = _map_pulse_soc_from_steps(
        results, steps, bundle_diags, files, nom_capacity
    )
    results = _map_pulse_soc_from_qocv(
        results, data_dir, files, bundle_diags, ir_ohm=cfg.get("qocv_ir_ohm"),
        skip_bms=stepped_bms,
    )

    block["settings"] = {
        "mode": "full-SOC-sweep — SOC_pct counted from the charge of the "
                "SOC-adjust step before each EIS block and held over the "
                "pulses that follow it, falling back to the same-direction "
                "qOCV curve (the order-based ladder was removed); not the "
                "90/50/10 aging-checkup schema",
        "soc_step_pct": step,
        "soc_direction_override": override,
        "plateau_gap_ah": round(nom_capacity * step / 100.0, 4),
        "bundles": bundle_diags,
    }
    block["fits"] = _records(results, PULSE_COLS)

    default_reason = (
        "no SOC_pct — needs a same-direction qOCV sweep to map against, and "
        "Ah_throughput/Zustand in the bundle for plateau detection"
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
        # Sweep direction in the filename, matching the EIS plots.
        direction = (
            str(group["sweep_direction"].iloc[0])
            if "sweep_direction" in group.columns
               and pd.notna(group["sweep_direction"].iloc[0])
            else bundle_diag_by_bm.get(bm_int, {}).get("direction", "unknown")
        )
        # Mean cell temperature of the bundle, from the per-pulse T_degC that
        # fit_2rc measured. In the filename because two runs of the same
        # programme at different chamber temperatures are different
        # measurements, and the fitted resistances are not comparable between
        # them — the name has to say which one a plot is.
        temp = (
            float(group["T_degC"].mean())
            if "T_degC" in group.columns and group["T_degC"].notna().any()
            else np.nan
        )
        out_png = os.path.join(
            plots_dir,
            f"pulse_2rc_{cell_stem}_BM{bm_int}_{soh}SOH_{direction}_{_temp_tag(temp)}.png",
        )
        entry = {
            "BM_Programm": bm_int, "SOH": soh, "sweep_direction": direction,
            "temperature_degC": _jsonable(None if pd.isna(temp) else round(temp, 2)),
        }

        has_soc_pct = "SOC_pct" in group.columns and group["SOC_pct"].notna().any()
        if not has_soc_pct:
            reason = bundle_diag_by_bm.get(bm_int, {}).get("reason") or default_reason
            logging.info("BM%s: vs-SOC plot: %s", bm_int, reason)
            entry["plot_skipped"] = reason
            plots.append(entry)
            continue

        try:
            title = f"initial characterization — BM{bm_int} {soh}SOH"
            if not pd.isna(temp):
                title += f" @ {temp:.1f} °C"
            pulse_fit.plot_vs_soc(group, out_png, title=title)
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


def load_step_table(cell_dir: str) -> pd.DataFrame:
    """Per-segment step charge from the run's ``GOLD.parquet``, or empty.

    GOLD sits one level above ``data/`` in the characterization export root, so
    the fit stage can read the segment IDs and ``Ah_throughput`` it needs for
    the step-counted SOC without re-running the pipeline. An absent or unusable
    GOLD is not an error — the SOC falls back to the qOCV mapping, which works
    from the bundle parquets alone.
    """
    path = os.path.join(cell_dir, "GOLD.parquet")
    if not os.path.exists(path):
        logging.info(
            "no GOLD.parquet in %s — SOC falls back to the qOCV mapping", cell_dir
        )
        return pd.DataFrame()
    try:
        cols = ["ID", "Current", "Ah_throughput", "target"]
        gold = pd.read_parquet(path, columns=cols)
    except Exception as exc:
        logging.warning(
            "GOLD.parquet in %s unreadable (%s) — SOC falls back to the qOCV "
            "mapping", cell_dir, exc,
        )
        return pd.DataFrame()
    return soc_from_steps.segment_steps(gold)


def _bundle_capacity(name: str, nom_capacity: float) -> float:
    """The bundle's own measured capacity, from the ``<SOH>SOH`` filename token.

    The step count needs the capacity the SOC percentages are *of*, which is the
    measured capacity at that check-up, not the nominal one. It is already in
    the filename (``export_pulse``/``export_eis`` write
    ``SOH = Capacity_py / nom * 100``), so no extra file has to be read back.
    SOH is stored to 1 dp, which on a 28 Ah cell is 0.028 Ah — a 0.1 % SOC
    rounding, well below the step sizes this resolves. Falls back to nominal
    when the token is NA.
    """
    # _parse_soh returns the token as a string ("99.0", or "NA" when the export
    # found no CAP segment for that program).
    soh = pd.to_numeric(
        pulse_fit._parse_soh(os.path.splitext(os.path.basename(name))[0]),
        errors="coerce",
    )
    if pd.isna(soh) or soh <= 0:
        return nom_capacity
    return nom_capacity * float(soh) / 100.0


def _bundle_start_times(files: list) -> dict:
    """``BM_Programm -> earliest Time`` for each pulse bundle file.

    The pulse results table carries no timestamp, but matching a bundle to the
    nearest qOCV sweep needs one, so read just the ``Time`` column back from
    each bundle parquet.
    """
    times = {}
    for path in files:
        try:
            df = pd.read_parquet(path, columns=["Time", "BM_Programm"])
        except Exception as exc:
            logging.warning("%s: cannot read Time/BM_Programm: %s", os.path.basename(path), exc)
            continue
        for bm, group in df.groupby("BM_Programm"):
            t = pd.to_datetime(group["Time"], errors="coerce").min()
            if pd.notna(t):
                times[int(bm)] = min(t, times[int(bm)]) if int(bm) in times else t
    return times


#: What ``soc_source`` reads when the step count supplied the SOC.
STEP_SOC_SOURCE = "coulomb count of the SOC-adjust step before each EIS block"


def _bundle_step_soc(steps, bm: int, direction: str, capacity: float,
                     label: str):
    """The program's EIS blocks with a SOC on each, plus diagnostics."""
    blocks = soc_from_steps.eis_blocks(steps, bm)
    return soc_from_steps.block_soc(blocks, direction, capacity, label=label)


def _map_pulse_soc_from_steps(results, steps, bundle_diags: list,
                              files: list, nom_capacity: float):
    """Per BM_Programm, hold each EIS block's step-counted SOC over its pulses.

    The pulses between one EIS block and the next were all measured at the SOC
    that block sits at — the steps happen in front of the blocks, not between
    the pulses. So the SOC comes from the block ladder and each pulse simply
    takes the last block at or before its own procedure number. Pulses ahead of
    the first block are on the setup ramp, not the sweep, and stay NaN.

    Returns ``(results, filled_bms)`` — the BM_Programms whose SOC was set, so
    the caller can send only the rest to the qOCV mapping.
    """
    filled = set()
    if steps is None or steps.empty or "ID" not in results.columns:
        return results, filled

    capacity_by_bm = {
        io_qocv._parse_bm(os.path.basename(f)): _bundle_capacity(f, nom_capacity)
        for f in files
        if io_qocv._parse_bm(os.path.basename(f)) is not None
    }
    diag_by_bm = {d.get("BM_Programm"): d for d in bundle_diags}

    out = []
    for bm, group in results.groupby("BM_Programm", sort=False):
        group = group.copy()
        direction = str(group["sweep_direction"].iloc[0])
        label = f"pulse BM{int(bm)}"
        blocks, info = _bundle_step_soc(
            steps, int(bm), direction,
            capacity_by_bm.get(int(bm), nom_capacity), label,
        )
        n = soc_from_steps.assign_pulse_soc(group, blocks)
        if n:
            info["soc_source"] = STEP_SOC_SOURCE
            info["sweep_direction"] = direction
            info["n_pulses_from_steps"] = n
            diag_by_bm.get(int(bm), {}).update(info)
            filled.add(int(bm))
            logging.info(
                "%s: SOC held from %d EIS block(s) over %d of %d pulses "
                "(mean step %.3f %%)",
                label, info.get("n_eis_blocks", 0), n, len(group),
                info.get("step_pct_mean") or float("nan"),
            )
        else:
            logging.info(
                "%s: no step-counted SOC (%s) — falling back to the qOCV curve",
                label, info.get("reason", "no EIS block matched a pulse"),
            )
        out.append(group)
    return (pd.concat(out, ignore_index=True) if out else results), filled


def _map_pulse_soc_from_qocv(results, data_dir: str, files: list,
                             bundle_diags: list, ir_ohm: float = None,
                             skip_bms: set = None):
    """Per BM_Programm, remap ``SOC_pct`` onto the same-direction qOCV curve.

    Each bundle's own detected ``sweep_direction`` picks the branch, and its
    start time picks the nearest qOCV sweep. Diagnostics are merged into the
    matching entry of ``bundle_diags`` so ``parameters.json`` records which
    qOCV file every SOC came from.

    ``skip_bms`` are the programs whose SOC the step count already measured;
    they are left alone so a fallback never overwrites a measurement.
    """
    skip_bms = skip_bms or set()
    if "sweep_direction" not in results.columns:
        return results
    if all(int(bm) in skip_bms for bm in results["BM_Programm"].dropna().unique()):
        return results

    sweeps = soc_from_qocv.find_sweeps(data_dir)
    if not sweeps:
        logging.warning(
            "no qOCV export in %s — pulse SOC_pct stays NaN", data_dir
        )
        return results

    start_times = _bundle_start_times(files)
    diag_by_bm = {d.get("BM_Programm"): d for d in bundle_diags}

    out = []
    for bm, group in results.groupby("BM_Programm", sort=False):
        group = group.copy()
        if int(bm) in skip_bms:
            out.append(group)
            continue
        direction = str(group["sweep_direction"].iloc[0])
        diag = soc_from_qocv.map_table(
            group, "OCV_V", direction, sweeps,
            t_ref=start_times.get(int(bm)), label=f"pulse BM{int(bm)}",
            ir_ohm=ir_ohm,
        )
        diag_by_bm.get(int(bm), {}).update(diag)
        out.append(group)
    return pd.concat(out, ignore_index=True) if out else results


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


#: Turning-point sensitivity for direction detection: a voltage excursion only
#: counts as a real trend (not sensor noise) once it exceeds max(this absolute
#: floor, this fraction of the bundle's total U span). Same rule the pulse path
#: uses on OCV_V, via :mod:`analysis.sweep_direction`
#: (``turn_threshold``/``direction_from_trend``).
EIS_U_TURN_ABS_THRESHOLD_V = sweep_direction_mod.U_TURN_ABS_THRESHOLD_V
EIS_U_TURN_FRAC_OF_SPAN = sweep_direction_mod.U_TURN_FRAC_OF_SPAN


def _bundle_direction(df: pd.DataFrame, override: str, source_name: str) -> tuple:
    """Sweep direction of one EIS bundle, from its per-measurement terminal
    voltage ``U`` (rising = charge, falling = discharge — the signal
    ``build_eis_table`` already computes per ``eis_number``, ordered by
    ``Time``).

    **One bundle carries one direction**: a bundle is a single
    ``BM_Programm``, and the parametrization procedure puts a charge sweep and
    a discharge sweep in separate programmes. So the direction is the plain
    first->last trend, matching how the pulse path reads ``OCV_V``. A bundle
    that nevertheless reverses mid-sweep is reported in ``reversal_mV`` and
    warned about rather than silently split — its SOC axis (assigned by
    measurement *order* in ``build_eis_table``) would be wrong either way, and
    a warning is more use than a quietly mislabelled half.

    Returns ``(direction, diagnostics)``; ``(None, {})`` for an empty bundle.
    """
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()
    if not order:
        return None, {}
    u_means = df.groupby("eis_number")["U"].mean()
    u = np.array([float(u_means[e]) for e in order], dtype=float)
    total_span = float(np.nanmax(u) - np.nanmin(u)) if len(u) > 1 else 0.0
    threshold = max(EIS_U_TURN_ABS_THRESHOLD_V, EIS_U_TURN_FRAC_OF_SPAN * total_span)

    if override:
        direction, source = override, "config-override"
    else:
        direction = sweep_direction_mod.direction_from_trend(u[0], u[-1], threshold)
        source = "detected"
        if direction is None:
            direction, source = "discharge", "assumed (ambiguous U trend)"
            logging.warning(
                "EIS direction ambiguous (U span %.1f mV < %.1f mV threshold) "
                "in %s — assuming %s",
                abs(u[-1] - u[0]) * 1000, threshold * 1000, source_name, direction,
            )

    # Largest excursion against the overall trend: 0 for a clean monotonic
    # sweep, large only if the bundle really does turn around.
    signed = u - u[0] if direction == "charge" else u[0] - u
    reversal = (
        max(0.0, float(np.max(np.maximum.accumulate(signed) - signed)))
        if len(u) > 1 else 0.0
    )
    if reversal > threshold:
        logging.warning(
            "%s: U reverses by %.1f mV against the %s trend (threshold %.1f mV) — "
            "this bundle may contain more than one sweep; its SOC axis assumes one",
            source_name, reversal * 1000, direction, threshold * 1000,
        )

    return direction, {
        "source": source_name,
        "sweep_direction": direction,
        "sweep_direction_source": source,
        "n_measurements": len(order),
        "u_start_V": float(u[0]),
        "u_end_V": float(u[-1]),
        "reversal_mV": round(reversal * 1000, 1),
    }


def _eis_bundle_temperature(name: str, bundle_df: pd.DataFrame,
                            pulse_temps: dict, cell_temp) -> tuple:
    """``(T_degC, source)`` for one EIS bundle.

    In order of preference: the bundle's own ``Temperature`` column (nothing
    writes one today, but an EIS export that gains a thermocouple channel
    should win); the pulse bundle of the same ``BM_Programm``; the mean over
    the cell's pulse bundles. ``source`` records which, so a borrowed
    cell-level number is never mistaken for a same-programme measurement.
    """
    if "Temperature" in bundle_df.columns:
        own = pd.to_numeric(bundle_df["Temperature"], errors="coerce")
        if own.notna().any():
            return round(float(own.mean()), 2), "eis-bundle"

    bm = None
    m = re.search(r"_BM(\d+)_", name)
    if m:
        bm = int(m.group(1))
    elif "BM_Programm" in bundle_df.columns and not bundle_df.empty:
        bm = int(bundle_df["BM_Programm"].iloc[0])

    if bm is not None and bm in pulse_temps:
        return pulse_temps[bm], f"pulse-BM{bm}"
    if not pd.isna(cell_temp):
        logging.info(
            "%s: no pulse bundle for BM%s — using the cell mean %.2f °C",
            name, bm, cell_temp,
        )
        return cell_temp, "cell-mean"
    return np.nan, "unavailable"


def _bundle_model_order(bundle_df: pd.DataFrame, name: str, drt: bool = True,
                        drt_lambda: float = None):
    """How many ZARC branches to fit **each spectrum** with, from its own DRT.

    Returns ``(n_by_eid, tau_seeds_by_eid, diag, solved)``. ``solved`` is the
    :func:`analysis.eis_drt.solve_bundle` result, handed back so the DRT plots
    later in :func:`fit_eis` reuse it instead of solving a second time.

    The count is **per spectrum**: a SOC whose DRT resolves two or more arcs is
    fitted with two ZARC branches, one that resolves a single arc with one. A
    spectrum with no discrete second peak has nothing for a second branch to
    attach to, and fitting one anyway is how α pins to a bound and two branches
    start swapping roles between adjacent SOC points.

    **This makes the branch slots mean different things at different SOC.**
    Slots are ordered τ-ascending, so on a two-arc spectrum slot 1 is the *fast*
    arc while on a one-arc spectrum the single (dominant, slow) branch also
    lands in slot 1 — ``tau1_z`` then jumps by decades at the transition. That
    is a property of the sweep, not an artefact, but it does mean the
    ``R1_z``/``tau1_z`` vs-SOC curves must be read together with ``n_zarc``.
    :func:`analysis.eis_vs_soc.plot_zarc_vs_soc` marks the one-arc points
    separately for that reason.

    With the DRT off, every spectrum falls back to the two-branch default: two
    is what the parametrization spectra support, and dropping to one on no
    evidence would be a worse guess than the model we already trust.
    """
    default_n = eis_drt.MAX_ZARC_BRANCHES
    eids = bundle_df["eis_number"].unique().tolist()
    if not drt:
        return ({e: default_n for e in eids}, {},
                {"n_zarc_source": "default (DRT off)"}, None)

    solved, _ = eis_drt.solve_bundle(bundle_df, lam=drt_lambda)
    if not solved:
        return ({e: default_n for e in eids}, {},
                {"n_zarc_source": "default (DRT found nothing)"}, solved)

    seeds, n_by_eid = {}, {}
    for sol in solved:
        # Ceiling = the fit's own ZARC τ box, so the DRT never asks for a
        # branch the ECM has no room to place (which is how a slow branch ends
        # up sitting on τ_d absorbing the Warburg).
        n_i, tau_i, _ = eis_drt.select_model_order(
            sol["peaks"], sol["f_min"], sol["f_max"],
            tau_max=eis_vs_soc.zarc_tau_box(sol["f_min"], sol["f_max"])[1],
        )
        seeds[sol["eis_number"]] = tau_i
        n_by_eid[sol["eis_number"]] = n_i

    counts = list(n_by_eid.values())
    diag = {
        "n_zarc_source": "drt-per-spectrum",
        "n_zarc_by_measurement": dict(n_by_eid),
        "n_zarc_counts": {str(k): counts.count(k) for k in sorted(set(counts))},
        "drt_arc_min_r_fraction": eis_drt.ARC_MIN_R_FRACTION,
        "drt_tau_diffusion_floor_s": eis_vs_soc.DIFFUSION_TAU_BOX[0],
        "zarc_tau_margin_decades": eis_vs_soc.ZARC_DIFFUSION_MARGIN_DECADES,
    }
    if len(set(counts)) > 1:
        logging.info(
            "EIS %s: DRT arc count varies across the sweep (%s) — each spectrum "
            "fitted with its own branch count; tau1_z/R1_z are not comparable "
            "across the transition, read them with n_zarc",
            name, diag["n_zarc_counts"],
        )
    else:
        logging.info("EIS %s: DRT resolves %d arc(s) on every spectrum",
                     name, counts[0])
    return n_by_eid, seeds, diag, solved


def fit_eis(data_dir: str, plots_dir: str, soc_direction: str = None,
            soc_step_pct: float = None, ir_ohm: float = None,
            two_stage_r0: bool = True, hf_f_min: float = None,
            drt: bool = True, drt_lambda: float = None,
            steps: pd.DataFrame = None, nom_capacity: float = None) -> dict:
    """2×ZARC + generalized-Warburg fit of every spectrum in every bundle.

    Direction is **detected per bundle** from its own per-measurement terminal
    voltage ``U`` (rising -> charge, falling -> discharge) rather than declared
    once for the whole cell: a parametrization run holds a charge sweep and a
    discharge sweep in separate ``BM_Programm``s, so one config value can't
    describe both and a wrong one silently inverts the SOC axis (SOC is
    assigned by measurement *order* in ``build_eis_table``, not measured).
    One bundle = one direction (see :func:`_bundle_direction`).
    ``soc_direction`` is an optional override that forces every bundle's
    direction and skips detection — ``None`` (default) means "detect".
    ``soc_step_pct`` sets the SOC step (default matches ``build_eis_table``'s
    ``SOC_SWEEP_STEP_PCT``, 5.0).

    ``two_stage_r0`` (config ``eis_two_stage_r0``) measures R0 on the
    high-frequency window and pins it, instead of fitting it against the
    correlated mid-frequency arc — see :func:`analysis.eis_vs_soc.fit_hf_r0`.

    ``steps`` is the GOLD step table from :func:`load_step_table`; with it, SOC
    is counted from the charge each SOC-adjust step moved, and the qOCV mapping
    is the fallback. Pass ``None`` to use the qOCV mapping alone.
    """
    override = _normalize_sweep_direction(soc_direction) if soc_direction is not None else None
    step = float(soc_step_pct) if soc_step_pct is not None else eis_vs_soc.SOC_SWEEP_STEP_PCT
    direction_source = "config-override" if override else "detected"
    two_stage_r0 = bool(two_stage_r0)
    hf_f_min = float(hf_f_min) if hf_f_min is not None else eis_vs_soc.HF_R0_MIN_FREQ_HZ
    # DRT runs at a **fixed** λ, not the L-curve corner. The corner is the
    # better choice for a one-off investigation but it is not reproducible
    # enough to bake into the pipeline: across one NFPP sweep it picked λ from
    # 2.5e-6 to 1.6e-2 and the peak count swung 2–10, so consecutive SOC steps
    # would be smoothed differently and the vs-SOC structure would be an
    # artefact of λ. It is also ~11x slower (4.7 s vs 0.4 s per bundle).
    drt = bool(drt)
    drt_lambda = float(drt_lambda) if drt_lambda is not None else DRT_LAMBDA_DEFAULT

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
            "two_stage_r0": two_stage_r0,
            "hf_r0_f_min_hz": hf_f_min if two_stage_r0 else None,
            "drt": drt,
            "drt_lambda": drt_lambda if drt else None,
            # The DRT is not only a companion diagnostic any more: it picks the
            # ZARC branch count, **per spectrum**. These are the thresholds that
            # turn its peaks into that count — see eis_drt.select_model_order.
            # settings.bundles[].n_zarc_by_measurement records what each SOC got.
            "zarc_branches_from_drt": drt,
            "zarc_branches_per_spectrum": drt,
            "zarc_branches_default": eis_drt.MAX_ZARC_BRANCHES,
            "zarc_branches_max": eis_drt.MAX_ZARC_BRANCHES,
            "drt_arc_min_r_fraction": eis_drt.ARC_MIN_R_FRACTION,
            "zarc_column_slots": eis_vs_soc.ZARC_COLUMN_SLOTS,
            "soc_step_pct": step,
            "soc_direction_source": direction_source,
            "soc_direction_override": override,
            "soc_turn_threshold_abs_V": EIS_U_TURN_ABS_THRESHOLD_V,
            "soc_turn_threshold_frac_of_span": EIS_U_TURN_FRAC_OF_SPAN,
            "bundles": [],
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

    qocv_sweeps = soc_from_qocv.find_sweeps(data_dir)
    if not qocv_sweeps:
        logging.warning(
            "no qOCV export in %s — EIS SOC_pct stays NaN", data_dir
        )
    # EIS spectra carry no temperature of their own (the EIS device file has no
    # thermocouple channel) — borrow it from the pulse bundle of the same
    # BM_Programm, which is the same cell in the same chamber during the same
    # programme.
    pulse_temps, cell_temp = _pulse_temperatures(data_dir)

    tables, bundle_diag = [], []
    raw_by_source = {}   # source -> raw measured-spectra df
    solved_by_source = {}  # source -> eis_drt.solve_bundle result, reused for plots
    for path in files:
        name = os.path.basename(path)
        try:
            bundle_df = pd.read_parquet(path)
            direction, diag = _bundle_direction(bundle_df, override, name)
            if direction is None:
                logging.warning("%s: no EIS measurements — skipping", name)
                continue
            # Model order comes from the bundle's own DRT, before the ECM fit
            # runs — it is the only thing here that can say how many relaxation
            # processes a spectrum contains at all, which is exactly the
            # question "how many ZARC branches" asks. The solve is kept and
            # reused for the DRT plots further down, so this costs nothing
            # extra; only the SOC labelling waits for the fit.
            n_by_eid, tau_seeds, order_diag, solved = _bundle_model_order(
                bundle_df, name, drt=drt, drt_lambda=drt_lambda)
            diag.update(order_diag)
            solved_by_source[name] = solved
            table = eis_vs_soc.build_eis_table(
                bundle_df, direction=direction, step=step,
                two_stage_r0=two_stage_r0, hf_f_min=hf_f_min,
                n_zarc=n_by_eid, tau_seeds_by_eid=tau_seeds)
            # `n_zarc` (from the fit) is what was fitted; `drt_n_arcs` is what
            # the DRT resolved. They are the same number on this path — the
            # branch count is no longer overridden at bundle level — but the
            # column is kept so a run that *does* override (DRT off) still says
            # what the DRT would have asked for.
            table["drt_n_arcs"] = table["eis_number"].map(n_by_eid)
            table["sweep_direction"] = direction
            table["sweep_direction_source"] = diag["sweep_direction_source"]
            table["source"] = name
            temp, temp_source = _eis_bundle_temperature(
                name, bundle_df, pulse_temps, cell_temp
            )
            table["T_degC"] = temp
            diag["temperature_degC"] = _jsonable(temp)
            diag["temperature_source"] = temp_source
            # SOC_pct is NaN until one of these fills it. Count the charge of
            # the SOC-adjust step in front of each measurement first — a direct
            # measurement of what moved — and fall back to interpolating the
            # rest voltage onto the same-direction qOCV curve when the step
            # chain is unavailable (no GOLD, no segment_ID, a broken chain).
            bm = io_qocv._parse_bm(name)
            n_stepped = 0
            if steps is not None and not steps.empty and bm is not None:
                blocks, sinfo = _bundle_step_soc(
                    steps, bm, direction,
                    _bundle_capacity(name, nom_capacity), f"EIS {name}",
                )
                n_stepped = soc_from_steps.assign_eis_soc(table, blocks)
                if n_stepped:
                    sinfo["soc_source"] = STEP_SOC_SOURCE
                    sinfo["n_measurements_from_steps"] = n_stepped
                    diag.update(sinfo)
                    logging.info(
                        "EIS %s: SOC from step charge — %d of %d measurements, "
                        "%.1f–%.1f%% (mean step %.3f %%)",
                        name, n_stepped, len(table),
                        float(table["SOC_pct"].min()), float(table["SOC_pct"].max()),
                        sinfo.get("step_pct_mean") or float("nan"),
                    )
                else:
                    logging.info(
                        "EIS %s: no step-counted SOC (%s) — falling back to the "
                        "qOCV curve", name, sinfo.get("reason", "no block matched"),
                    )
            if not n_stepped:
                diag.update(soc_from_qocv.map_table(
                    table, "U", direction, qocv_sweeps,
                    t_ref=table["Time"].min() if "Time" in table.columns else None,
                    label=f"EIS {name}", ir_ohm=ir_ohm,
                ))
            raw_by_source[name] = bundle_df
            tables.append(table)
            bundle_diag.append(diag)
        except Exception as exc:
            logging.warning("EIS fit failed for %s: %s", name, exc)
            block.setdefault("errors", []).append(
                {"source": name, "error": f"{type(exc).__name__}: {exc}"}
            )
    block["settings"]["bundles"] = bundle_diag
    if not tables:
        return block

    combined = pd.concat(tables, ignore_index=True)
    block["fits"] = _records(combined, EIS_COLS + ["source"])
    n_deg = int(combined.get("zarc_degenerate", pd.Series(dtype=bool)).sum())
    if n_deg:
        logging.warning("%d/%d EIS fits flagged degenerate", n_deg, len(combined))
    block["n_degenerate"] = n_deg

    # plot_zarc_vs_soc (and the raw-spectra / fit-overlay plots below) overlay
    # every row's SOC axis on one figure with no per-series separation, so two
    # BM_Programms would silently overlay (each bundle's SOC restarts at its
    # own sweep). Since that grouping can't be expressed without editing
    # analysis/eis_vs_soc.py, plot one figure per source and record every path.
    # The sweep direction goes in the filename, matching the pulse plots.
    def _try_plot(kind, fn, out_png, *args, **kwargs):
        """Call a plotter and record its path only if it actually wrote a
        (non-empty) file — several of these plotters no-op silently on
        missing columns/empty input rather than raising, so a bare try/except
        would otherwise record a path to a file that doesn't exist.
        """
        try:
            fn(*args, out_png, **kwargs)
        except Exception as exc:
            logging.warning("EIS %s plot failed for %s: %s", kind, source, exc)
            return
        if not (os.path.isfile(out_png) and os.path.getsize(out_png) > 0):
            logging.info("EIS %s plot: nothing written for %s", kind, source)
            return
        plots.append({
            "kind": kind,
            "source": source,
            "sweep_direction": direction,
            "temperature_degC": _jsonable(temp),
            "plot": os.path.relpath(out_png, os.path.dirname(plots_dir)),
        })

    plots = []
    drt_peaks = []
    # `_bundle_direction` keys each bundle's diagnostics by its file name, which
    # is what `combined` groups on — so the DRT can be told which SOC source the
    # fits used rather than guessing.
    diag_by_source = {d.get("source"): d for d in bundle_diag}
    for source, group in combined.groupby("source", sort=False):
        stem = os.path.splitext(str(source))[0]
        direction = group["sweep_direction"].iloc[0]
        # Temperature goes in every filename beside the direction: impedance is
        # as temperature-dependent as it is SOC-dependent, so a plot of one
        # sweep is only readable next to another at the same temperature.
        temp = (
            float(group["T_degC"].iloc[0])
            if "T_degC" in group.columns and pd.notna(group["T_degC"].iloc[0])
            else np.nan
        )
        tag = f"{direction}_{_temp_tag(temp)}"
        # Every vs-SOC plotter sorts on SOC_pct; with no qOCV to map against it
        # is all-NaN and they would draw an empty axis instead of failing.
        if group["SOC_pct"].isna().all():
            logging.warning(
                "%s: no SOC (neither the step charge nor a qOCV mapping "
                "resolved one) — skipping the vs-SOC plots", source
            )
            plots.append({
                "source": source, "sweep_direction": direction,
                "temperature_degC": _jsonable(temp),
                "plot_skipped": "no SOC_pct — no qOCV sweep to map against",
            })
            continue
        title = f"initial characterization — {source} ({direction})"
        if not pd.isna(temp):
            title += f" @ {temp:.1f} °C"
        raw_df = raw_by_source.get(source)

        # Stem dropped the "2": the branch count is per bundle now, so baking a
        # 2 into every filename would mislabel a one-arc fit.
        out_png = os.path.join(plots_dir, f"eis_zarc_warburg_{stem}_{tag}.png")
        _try_plot("zarc_params_vs_soc", eis_vs_soc.plot_zarc_vs_soc, out_png,
                  group, title=title)

        if raw_df is None or raw_df.empty:
            logging.warning(
                "no raw spectra retained for %s — skipping raw/overlay plots", source,
            )
            continue

        raw_png = os.path.join(plots_dir, f"eis_raw_spectra_{stem}_{tag}.png")
        _try_plot("raw_spectra", eis_vs_soc.plot_raw_spectra, raw_png,
                  raw_df, group, title=title)

        overlay_png = os.path.join(plots_dir, f"eis_fit_overlay_{stem}_{tag}.png")
        _try_plot("fit_overlay", eis_vs_soc.plot_fit_overlay, overlay_png,
                  raw_df, group, title=title)

        # The Nyquist plane on its own, with the R0-region / MF-arc zoom
        # insets. Its only other caller is eis_vs_soc's standalone CLI, which
        # doesn't map SOC from the qOCV curve — so without this the figure has
        # no working producer.
        nyq_png = os.path.join(plots_dir, f"eis_nyquist_{stem}_{tag}.png")
        _try_plot("nyquist_by_soc", eis_vs_soc.plot_nyquist_by_soc, nyq_png,
                  raw_df, group, title=title)

        # Model-free DRT of the same bundle. It answers a question the ECM
        # cannot ask of itself — how many relaxation processes are in the
        # spectrum at all — so it is worth having beside every fit rather than
        # only when someone remembers to run the standalone CLI. It is handed
        # the SOC this bundle's fits are already using — whether that came from
        # the step charge or the qOCV curve — because a DRT panel and the
        # eis_fits.csv row beside it must never disagree about which SOC a
        # spectrum was measured at; letting it re-derive its own is exactly how
        # they would. `group` supplies the fitted τ so the two are overlaid on
        # one axis: an ECM τ sitting in a DRT *valley* (rather than on a peak)
        # is the signature of one element blanketing a region with more
        # structure than it has parameters for.
        if not drt:
            continue
        solved = solved_by_source.get(source)
        if not solved:
            logging.info("DRT: no solve retained for %s — skipping its plots", source)
            continue
        try:
            # Label the solve that already chose this bundle's model order,
            # rather than solving again: same γ(τ), and the SOC on the DRT
            # panel is by construction the SOC the fits used.
            soc_by_eid = {k: v for k, v in
                          zip(group["eis_number"], group["SOC_pct"])
                          if pd.notna(v)}
            curves, peaks, meta = eis_drt.label_bundle(
                solved, soc_by_eid=soc_by_eid, direction=direction, step=step,
                soc_source=(diag_by_source.get(source, {}).get("soc_source")
                            or "supplied by caller"),
            )
        except Exception as exc:
            logging.warning("DRT failed for %s: %s", source, exc)
            block.setdefault("errors", []).append(
                {"source": source, "error": f"DRT {type(exc).__name__}: {exc}"}
            )
        else:
            if not peaks.empty:
                drt_peaks.append(peaks.assign(source=source))
            gam_png = os.path.join(plots_dir, f"eis_drt_gamma_{stem}_{tag}.png")
            _try_plot("drt_gamma", eis_drt.plot_drt, gam_png, curves, meta, group)
            map_png = os.path.join(plots_dir, f"eis_drt_map_{stem}_{tag}.png")
            _try_plot("drt_map", eis_drt.plot_drt_map, map_png, curves)

    if plots:
        block["plots"] = plots
    if drt_peaks:
        block["drt"] = {
            "lambda": drt_lambda,
            "peaks": _records(pd.concat(drt_peaks, ignore_index=True), DRT_PEAK_COLS),
        }
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
    # Mean over both branches — the sweep is 20 h long, so the temperature it
    # was measured at is part of the curve's identity (it sets the hysteresis
    # the IR correction removes).
    temps = [_parquet_temperature(pair[k]) for k in ("cha", "dch")]
    temps = [t for t in temps if not pd.isna(t)]
    temp = round(float(np.mean(temps)), 2) if temps else np.nan
    try:
        _, qc = qocv_curve.load_sweep(pair["cha"], discharge=False)
        _, qd = qocv_curve.load_sweep(pair["dch"], discharge=True)
        block["fits"] = [{
            "BM_Programm": int(bm),
            "SOH": _jsonable(pair.get("soh")),
            "T_degC": _jsonable(temp),
            "capacity_cha_ah": round(float(np.nanmax(qc)), 4),
            "capacity_dch_ah": round(float(np.nanmax(qd)), 4),
        }]
    except Exception as exc:
        logging.warning("qOCV read failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block

    out_png = os.path.join(plots_dir, f"qocv_{_temp_tag(temp)}.png")
    try:
        qocv_curve.plot_qocv(
            pair["cha"], pair["dch"], out_png, nom_capacity=nom_capacity,
            temp_degC=None if pd.isna(temp) else temp,
        )
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
    # Segment step charges, read once from GOLD and shared by both blocks —
    # pulse and EIS measure the same sweep and must agree about its SOC axis.
    steps = load_step_table(cell_dir)

    if "pulse" in parts:
        payload["pulse"] = fit_pulse(
            data_dir, plots_dir, nom_capacity, cfg=cfg, steps=steps
        )
    if "eis" in parts:
        payload["eis"] = fit_eis(
            data_dir, plots_dir,
            soc_direction=cfg.get("soc_sweep_direction"),
            soc_step_pct=cfg.get("soc_step_pct"),
            ir_ohm=cfg.get("qocv_ir_ohm"),
            two_stage_r0=cfg.get("eis_two_stage_r0", True),
            hf_f_min=cfg.get("eis_hf_r0_f_min_hz"),
            drt=cfg.get("eis_drt", True),
            drt_lambda=cfg.get("eis_drt_lambda"),
            steps=steps, nom_capacity=nom_capacity,
        )
    if "qocv" in parts:
        payload["qocv"] = summarize_qocv(data_dir, plots_dir, nom_capacity)
    return payload


def write_fit_csvs(cell_dir: str, stem: str, payload: dict) -> list:
    """Write one flat CSV per fitted block beside ``<stem>_parameters.json``.

    The JSON stays the authoritative output — these are the same ``fits``
    records as a table, because reading a SOC sweep out of nested JSON is
    needless work. Only blocks present in ``payload`` are written, so a
    ``--only`` run refreshes just its own CSVs and leaves the others in place
    (mirroring how ``run`` merges the JSON).
    """
    written = []
    for part in FIT_PARTS:
        block = payload.get(part)
        if not isinstance(block, dict):
            continue
        fits = block.get("fits") or []
        if not fits:
            logging.info("%s: no %s fits to write as CSV", stem, part)
            continue
        out_csv = os.path.join(cell_dir, f"{stem}_{part}_fits.csv")
        try:
            pd.DataFrame(fits).to_csv(out_csv, index=False)
        except Exception as exc:          # a CSV failure must not lose the JSON
            logging.warning("%s: %s CSV failed: %s", stem, part, exc)
            continue
        logging.info("%s: %s fits -> %s", stem, part, out_csv)
        written.append(out_csv)

    # The DRT peak table rides alongside the EIS fits rather than in FIT_PARTS:
    # it is not a fit, and it has its own row granularity (one row per peak,
    # not per spectrum), so it cannot share the `fits` shape above.
    peaks = ((payload.get("eis") or {}).get("drt") or {}).get("peaks")
    if peaks:
        out_csv = os.path.join(cell_dir, f"{stem}_eis_drt_peaks.csv")
        try:
            pd.DataFrame(peaks).to_csv(out_csv, index=False)
        except Exception as exc:
            logging.warning("%s: DRT peaks CSV failed: %s", stem, exc)
        else:
            logging.info("%s: DRT peaks -> %s", stem, out_csv)
            written.append(out_csv)
    return written


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
            # Merge *before* opening for write: "w" truncates, so reading the
            # previous file inside the with-block would always see it empty
            # and a --only run would drop the blocks it didn't refit.
            merged = _merge_parameters(out_json, payload)
            with open(out_json, "w") as f:
                json.dump(merged, f, indent=2)
            logging.info("%s: parameters -> %s", stem, out_json)
            write_fit_csvs(cell_dir, stem, payload)
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
