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
  are recorded in the params file so the numbers stay interpretable.
* **qOCV** — no fit; the curve and its throughput-normalised capacities.

    python -m characterize.fit_characterization <battery_cfg> [--cells …]
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis import eis_vs_soc, qocv_curve
from analysis import fit_2rc_pulse as pulse_fit
from main import load_config
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


def fit_pulse(data_dir: str, plots_dir: str, nom_capacity: float) -> dict:
    """2RC fit of every pulse bundle in ``data_dir``."""
    files = sorted(glob.glob(os.path.join(data_dir, "*_pulse_BM*.parquet")))
    block = {"model": PULSE_MODEL, "fits": [], "sources": [os.path.basename(f) for f in files]}
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


def fit_eis(data_dir: str, plots_dir: str) -> dict:
    """2×ZARC + generalized-Warburg fit of every spectrum in every bundle."""
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
        },
        "fits": [],
        "sources": [os.path.basename(f) for f in files],
    }
    if not files:
        logging.info("no EIS bundles in %s", data_dir)
        return block

    tables = []
    for path in files:
        try:
            table = eis_vs_soc.build_eis_table(pd.read_parquet(path))
            table["source"] = os.path.basename(path)
            tables.append(table)
        except Exception as exc:
            logging.warning("EIS fit failed for %s: %s", os.path.basename(path), exc)
            block.setdefault("errors", []).append(
                {"source": os.path.basename(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    if not tables:
        return block

    combined = pd.concat(tables, ignore_index=True)
    block["fits"] = _records(combined, EIS_COLS + ["source"])
    n_deg = int(combined.get("zarc_degenerate", pd.Series(dtype=bool)).sum())
    if n_deg:
        logging.warning("%d/%d EIS fits flagged degenerate", n_deg, len(combined))
    block["n_degenerate"] = n_deg

    out_png = os.path.join(plots_dir, "eis_2zarc_warburg.png")
    try:
        eis_vs_soc.plot_zarc_vs_soc(combined, out_png, title="initial characterization")
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("EIS plot failed: %s", exc)
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


def fit_cell(cell_dir: str, nom_capacity: float) -> dict:
    """Fit one `10_initial_characterization/<cell>/` folder; return the payload."""
    stem = os.path.basename(os.path.normpath(cell_dir))
    data_dir = os.path.join(cell_dir, "data")
    plots_dir = os.path.join(cell_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "cell": stem,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nom_capacity": nom_capacity,
        "pulse": fit_pulse(data_dir, plots_dir, nom_capacity),
        "eis": fit_eis(data_dir, plots_dir),
        "qocv": summarize_qocv(data_dir, plots_dir, nom_capacity),
    }


def run(cfg: dict, target_cells: list = None) -> None:
    working_path = cfg.get("working_path")
    if not working_path:
        raise ValueError("working_path required in the battery config")
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

    for stem in cells:
        cell_dir = os.path.join(root, stem)
        logging.info("fitting %s", stem)
        payload = fit_cell(cell_dir, float(cfg["nom_capacity"]))
        out_json = os.path.join(cell_dir, f"{stem}_parameters.json")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logging.info("%s: parameters -> %s", stem, out_json)


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
