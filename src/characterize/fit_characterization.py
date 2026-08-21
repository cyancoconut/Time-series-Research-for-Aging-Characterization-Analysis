"""Fit the characterization bundles and plot them.

Reads ``<working_path>/10_initial_characterization/<cell>/data/`` — the pulse,
EIS and qOCV parquets written by :mod:`characterize.main_para` — and writes
``<cell>_parameters.json`` plus ``plots/`` beside it. Runs standalone, so fits
can be repeated without redoing segmentation.

Models are fixed defaults:

* **pulse** — 2RC (:mod:`analysis.fit_2rc_pulse`).
* **EIS** — 2×ZARC + series-L + generalized Warburg
  (:func:`analysis.eis_vs_soc.fit_zarc_warburg_eis`). φ is fitted; τ_d is
  **pinned** by ``DIFFUSION_TAU_BOX``, so ``R_d_z`` is the amplitude at
  ω = 1/τ_d and ``tau_d_z`` is a shape constant, not a result. Those settings
  are recorded in the params file so the numbers stay interpretable. Each
  bundle's charge/discharge sweep leg(s) are detected from its own
  per-measurement terminal voltage (config ``eis_soc_direction`` overrides
  detection; ``eis_soc_step_pct`` sets the SOC step, default 5.0) — see
  ``fit_eis`` / ``_split_bundle_into_legs``.
* **qOCV** — no fit; the curve and its throughput-normalised capacities.

    python -m characterize.fit_characterization <battery_cfg> [--cells …]
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
from analysis import fit_2rc_pulse as pulse_fit
from main import load_config
from util import io_router
from util.run_context import CHARACTERIZATION

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PULSE_MODEL = "2rc"
EIS_MODEL = "2zarc_warburg"

#: Columns lifted from the 2RC results table into the params file.
PULSE_COLS = [
    "pulse_segment_id", "ID", "BM_Programm", "SOH", "SOC", "direction",
    "I_A", "C_rate", "OCV_V", "R0_ohm", "R1_ohm", "tau1_s", "R2_ohm",
    "tau2_s", "rmse_mV",
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


def _skipped_nasoh_bundles(files: list) -> list:
    """Bundles ``fit_2rc_pulse.fit_folder`` will silently skip (SOH=NA), i.e.
    no CAP segment was found for that BM_Programm so no SOH could be computed.
    Mirrors ``fit_folder``'s per-BM "largest file wins" selection, so this
    reports exactly the files it will actually evaluate and skip — not every
    NASOH-named stub that happens to sit in the folder.
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
            skipped.append({
                "file": os.path.basename(f),
                "BM_Programm": int(bm),
                "reason": "SOH=NA (no CAP segment for this BM_Programm) — fit_folder skips these",
            })
    return skipped


def fit_pulse(data_dir: str, plots_dir: str, nom_capacity: float) -> dict:
    """2RC fit of every pulse bundle in ``data_dir``."""
    files = sorted(glob.glob(os.path.join(data_dir, "*_pulse_BM*.parquet")))
    block = {"model": PULSE_MODEL, "fits": [], "sources": [os.path.basename(f) for f in files]}
    skipped = _skipped_nasoh_bundles(files)
    if skipped:
        block["skipped"] = skipped
    if not files:
        logging.info("no pulse bundles in %s", data_dir)
        return block
    try:
        results = pulse_fit.fit_folder(
            data_dir,
            nom_capacity,
            pulse_fit.REMOVE_PULSE_BEFORE_MIN,
            pulse_fit.EXCLUDE_ZUSTAND_CURRENT,
        )
    except Exception as exc:                      # one bad bundle ≠ dead cell
        logging.warning("pulse fit failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block
    if results.empty:
        block["error"] = "no pulses fit"
        return block

    block["fits"] = _records(results, PULSE_COLS)
    out_png = os.path.join(plots_dir, "pulse_2rc.png")
    try:
        pulse_fit.plot_vs_soc(results, out_png, title="initial characterization")
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("pulse plot failed: %s", exc)
    return block


def _normalize_eis_direction(direction: str) -> str:
    """Validate + normalize an ``eis_soc_direction`` config override.

    ``build_eis_table`` only checks for a "cha" prefix (anything else is
    treated as discharge), so a typo like "charging" happens to work but
    "chrage" or "" would silently be read as discharge. Fail loudly instead.
    """
    d = str(direction).strip().lower()
    if d.startswith("dis") or d == "discharge":
        return "discharge"
    if d.startswith("cha") or d == "charge":
        return "charge"
    raise ValueError(
        f"eis_soc_direction={direction!r} not recognized — use "
        f"'discharge' (or 'dch'/'dis...') or 'charge' (or 'cha...')"
    )


#: Turning-point sensitivity for leg detection: a reversal only counts as a
#: real turning point (not sensor noise) once the U excursion since the last
#: extreme exceeds max(this absolute floor, this fraction of the bundle's
#: total U span).
EIS_U_TURN_ABS_THRESHOLD_V = 0.02
EIS_U_TURN_FRAC_OF_SPAN = 0.05

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


def _leg_direction(u_start: float, u_end: float, threshold: float):
    """"charge"/"discharge" from a leg's endpoint U, or None if ambiguous
    (the U excursion across the whole leg is below the turning threshold)."""
    diff = u_end - u_start
    if abs(diff) < threshold:
        return None
    return "charge" if diff > 0 else "discharge"


def _split_bundle_into_legs(df: pd.DataFrame, step: float, override: str,
                            source_name: str) -> tuple:
    """Detect charge/discharge legs in one EIS bundle from its per-measurement
    terminal voltage ``U`` (rising = charge, falling = discharge — the signal
    ``build_eis_table`` already computes per ``eis_number``, ordered by
    ``Time``), and build a per-leg SOC table for each with the correct
    direction. Returns ``(tables, leg_diagnostics)``.
    """
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()
    if not order:
        return [], []
    u_means = df.groupby("eis_number")["U"].mean()
    u = np.array([float(u_means[e]) for e in order], dtype=float)
    total_span = float(np.nanmax(u) - np.nanmin(u)) if len(u) > 1 else 0.0
    threshold = max(EIS_U_TURN_ABS_THRESHOLD_V, EIS_U_TURN_FRAC_OF_SPAN * total_span)

    leg_ranges = [(0, len(u) - 1)] if override else _detect_sweep_legs(u, threshold)

    tables, leg_info = [], []
    for leg_idx, (s, e) in enumerate(leg_ranges):
        eids = order[s:e + 1]
        leg_df = df[df["eis_number"].isin(eids)]
        assumed = False
        if override:
            direction = override
        else:
            direction = _leg_direction(u[s], u[e], threshold)
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
        leg_info.append({
            "source": source_name,
            "sweep_leg": leg_idx,
            "direction": direction,
            "assumed": assumed,
            "n_measurements": len(eids),
            "u_start_V": float(u[s]),
            "u_end_V": float(u[e]),
        })
    return tables, leg_info


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
    override = _normalize_eis_direction(soc_direction) if soc_direction is not None else None
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

    tables, leg_diag = [], []
    for path in files:
        name = os.path.basename(path)
        try:
            leg_tables, leg_info = _split_bundle_into_legs(
                pd.read_parquet(path), step, override, name
            )
            for table in leg_tables:
                table["source"] = name
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

    # plot_zarc_vs_soc overlays every row's SOC axis on one figure with no
    # per-series separation — two BM_Programms, or two legs of the same
    # bundle, would silently overlay (each leg's SOC restarts at its own
    # sweep). Since that grouping can't be expressed without editing
    # analysis/eis_vs_soc.py, plot one figure per (source, leg) instead and
    # record every path.
    plots = []
    for (source, leg), group in combined.groupby(["source", "sweep_leg"], sort=False):
        stem = os.path.splitext(str(source))[0]
        direction = group["direction"].iloc[0]
        out_png = os.path.join(plots_dir, f"eis_2zarc_warburg_{stem}_leg{leg}_{direction}.png")
        try:
            eis_vs_soc.plot_zarc_vs_soc(
                group, out_png,
                title=f"initial characterization — {source} leg{leg} ({direction})",
            )
            plots.append({
                "source": source,
                "sweep_leg": int(leg),
                "direction": direction,
                "plot": os.path.relpath(out_png, os.path.dirname(plots_dir)),
            })
        except Exception as exc:
            logging.warning("EIS plot failed for %s leg %s: %s", source, leg, exc)
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


def fit_cell(cell_dir: str, nom_capacity: float, cfg: dict = None) -> dict:
    """Fit one `10_initial_characterization/<cell>/` folder; return the payload."""
    cfg = cfg or {}
    stem = os.path.basename(os.path.normpath(cell_dir))
    data_dir = os.path.join(cell_dir, "data")
    plots_dir = os.path.join(cell_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "cell": stem,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nom_capacity": nom_capacity,
        "pulse": fit_pulse(data_dir, plots_dir, nom_capacity),
        "eis": fit_eis(
            data_dir, plots_dir,
            soc_direction=cfg.get("eis_soc_direction"),
            soc_step_pct=cfg.get("eis_soc_step_pct"),
        ),
        "qocv": summarize_qocv(data_dir, plots_dir, nom_capacity),
    }


def run(cfg: dict, target_cells: list = None) -> None:
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
        logging.info("fitting %s", stem)
        try:
            payload = fit_cell(cell_dir, float(cfg["nom_capacity"]), cfg=cfg)
            out_json = os.path.join(cell_dir, f"{stem}_parameters.json")
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
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
    args = parser.parse_args()
    run(load_config(args.config), target_cells=args.cells)


if __name__ == "__main__":
    main()
