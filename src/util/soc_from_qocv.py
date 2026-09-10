"""Map a measured rest voltage to SOC using the run's own qOCV curve.

This is the **only** source of SOC in the characterization track. It replaces
an order-based ladder (``100 - step * index``) that has been removed from
``build_eis_table`` and ``assign_pulse_soc``, which now leave ``SOC_pct`` NaN
for this module to fill. The ladder assumed every step moved the same charge
and was wrong twice over: the measured voltages contradict it (the first NFPP
EIS step drops 155 mV, the next ones ~15 mV, all labelled "5 %"), and the run
puts a CHA *and* a DCH pulse on each SOC step, so the pulse index advanced
twice per real step and the ladder ran to **-75 % SOC**.

A qOCV sweep is a slow (~C/20) full traverse, so its own coulomb count gives
SOC directly:

    SOC(V) = 100 * (Q(V) - Q_min) / (Q_max - Q_min)

Two constraints on picking *which* qOCV sweep to map against:

* **Same direction.** Charge and discharge branches are separated by the qOCV
  hysteresis, so a discharge measurement must map on the discharge branch. The
  direction comes from the ``_qocv_cha_`` / ``_qocv_dch_`` filename token,
  which is what the export writes.
* **Nearest in time.** The qOCV sweep usually sits in a different
  ``BM_Programm`` than the pulse/EIS bundle it is mapped for, so the two are
  matched by start time.

**IR correction.** The qOCV curve is measured *under load* at ~C/20, while the
values mapped onto it (EIS ``U``, pulse ``OCV_V``) are **rest** voltages, so
the branch must have its overpotential removed first. The polarization is
measured from the pair itself:

    eta(SOC) = [V_cha(SOC) - V_dch(SOC)] / 2

and each branch is corrected toward the middle (``V_dch + eta``,
``V_cha - eta``). Why not an independently measured resistance? Because none
of them are valid at this timescale: on the NFPP cell eta implies ~11.4 mOhm
mid-SOC, while the 2RC fit of a 30 s pulse gives R0+R1+R2 = 5.6 mOhm (R0 alone
3.8 mOhm). Over a 20 h sweep slow diffusion contributes resistance a short
pulse never sees, so a pulse- or EIS-derived R removes only about half the
offset. Correcting both branches by their own ``eta`` makes them coincide, so
this is equivalent to mapping on the cha/dch mean — but expressed per SOC,
recorded in the diagnostics, and degrading gracefully to a scalar
``ir_ohm`` (``V -/+ I*R``) when only one branch was exported.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

from util import io_qocv

#: Filename tokens marking a qOCV export's sweep direction.
DIRECTION_TOKENS = {"_qocv_cha_": "charge", "_qocv_dch_": "discharge"}


def find_sweeps(data_dir: str) -> list:
    """Every qOCV export in ``data_dir`` as a dict of its metadata.

    Unlike :func:`analysis.qocv_curve.find_pairs`, sweeps are listed
    **individually** — the mapping needs one branch of a given direction, not a
    complete cha/dch pair, and a run may well export only one of the two.
    """
    sweeps = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_qocv_*_BM*.parquet"))):
        base = os.path.basename(path)
        direction = next(
            (d for token, d in DIRECTION_TOKENS.items() if token in base), None
        )
        if direction is None:
            continue
        try:
            times = pd.read_parquet(path, columns=["Time"])["Time"]
            t_start = pd.to_datetime(times.min())
        except Exception as exc:            # unreadable file ≠ dead run
            logging.warning("qOCV %s: cannot read Time, skipping: %s", base, exc)
            continue
        sweeps.append({
            "path": path,
            "file": base,
            "direction": direction,
            "BM_Programm": io_qocv._parse_bm(base),
            "SOH": io_qocv._parse_soh(base),
            "t_start": t_start,
        })
    return sweeps


def pick_sweep(sweeps: list, direction: str, t_ref) -> dict:
    """The same-direction sweep starting nearest ``t_ref``; ``None`` if none.

    Direction is a hard filter, not a preference — mapping a discharge
    measurement onto a charge branch would bias it by the full qOCV
    hysteresis, which is exactly what this matching exists to avoid.
    """
    candidates = [s for s in sweeps if s["direction"] == direction]
    if not candidates:
        return None
    t_ref = pd.to_datetime(t_ref)
    if pd.isna(t_ref):
        return candidates[0]

    def gap(s):
        dt = s["t_start"] - t_ref
        return abs(dt.total_seconds()) if pd.notna(dt) else float("inf")

    return min(candidates, key=gap)


#: SOC grid (%) the pair-derived polarization is evaluated on.
ETA_SOC_GRID = np.linspace(0.0, 100.0, 201)


def polarization_from_pair(cha_path: str, dch_path: str) -> dict:
    """Half the cha/dch voltage gap vs SOC — the C/20 polarization ``eta``.

    Both branches come out of ``load_sweep`` oriented empty->full, so their
    coulomb-counted SOC axes are directly comparable. ``eta`` is returned on a
    fixed SOC grid; a caller interpolates it at whatever SOC it needs.
    """
    vc, qc = io_qocv.load_sweep(cha_path, discharge=False)
    vd, qd = io_qocv.load_sweep(dch_path, discharge=True)
    sc, sd = io_qocv.soc_axis(qc), io_qocv.soc_axis(qd)
    v_cha = np.interp(ETA_SOC_GRID, sc, vc)
    v_dch = np.interp(ETA_SOC_GRID, sd, vd)
    eta = (v_cha - v_dch) / 2.0
    mid = (ETA_SOC_GRID >= 20) & (ETA_SOC_GRID <= 80)
    return {
        "soc": ETA_SOC_GRID,
        "eta": eta,
        "eta_mid_mV": round(float(eta[mid].mean()) * 1000, 2),
        "source": os.path.basename(cha_path) + " / " + os.path.basename(dch_path),
    }


def _apply_ir(v: np.ndarray, soc: np.ndarray, direction: str,
              eta: dict = None, ir_ohm: float = None,
              current_a: float = None) -> tuple:
    """Shift a branch's voltage toward the rest (OCV) curve.

    A discharge branch sits **below** OCV, a charge branch above, so the
    correction is ``+eta`` and ``-eta`` respectively. Returns
    ``(v_corrected, description)``; ``(v, "none")`` when nothing was supplied.
    """
    if eta is not None:
        shift = np.interp(soc, eta["soc"], eta["eta"])
        desc = f"pair eta(SOC), {eta['eta_mid_mV']} mV mid-SOC"
    elif ir_ohm and current_a:
        shift = np.full_like(v, float(ir_ohm) * float(current_a))
        desc = f"scalar I*R = {float(current_a):.4f} A x {float(ir_ohm)*1000:.2f} mOhm"
    else:
        return v, "none"
    return (v + shift if direction == "discharge" else v - shift), desc


def build_lookup(path: str, direction: str, eta: dict = None,
                 ir_ohm: float = None) -> dict:
    """Monotonic voltage -> SOC(%) lookup from one qOCV sweep.

    ``load_sweep`` already orients both branches empty->full (voltage
    ascending) and counts capacity from empty, so SOC is just
    :func:`io_qocv.soc_axis`. The raw C/20 voltage is quantised and locally
    non-monotonic, which ``np.interp`` cannot use, so duplicate voltages are
    averaged and the series is reduced to a strictly increasing one.
    """
    v, q = io_qocv.load_sweep(path, discharge=(direction == "discharge"))
    soc = io_qocv.soc_axis(q)

    # Remove the C/20 overpotential *before* building the lookup: the values
    # mapped onto it are rest voltages, this branch is under load.
    current_a = None
    if ir_ohm and eta is None:
        try:
            cur = pd.read_parquet(path, columns=["Current"])["Current"]
            current_a = float(np.abs(pd.to_numeric(cur, errors="coerce")).mean())
        except Exception as exc:
            logging.warning("%s: cannot read Current for IR: %s", os.path.basename(path), exc)
    v, ir_desc = _apply_ir(v, soc, direction, eta=eta, ir_ohm=ir_ohm, current_a=current_a)

    ok = np.isfinite(v) & np.isfinite(soc)
    v, soc = v[ok], soc[ok]
    if len(v) < 2:
        raise ValueError(f"{os.path.basename(path)}: fewer than 2 usable points")

    # Average SOC over repeated voltages (quantisation), then keep a strictly
    # increasing V so np.interp is well defined.
    frame = pd.DataFrame({"v": v, "soc": soc}).groupby("v", sort=True)["soc"].mean()
    v_u = frame.index.to_numpy(dtype=float)
    soc_u = frame.to_numpy(dtype=float)
    keep = np.concatenate([[True], np.diff(v_u) > 0])
    v_u, soc_u = v_u[keep], soc_u[keep]
    if len(v_u) < 2:
        raise ValueError(f"{os.path.basename(path)}: voltage is not monotonic")

    return {
        "v": v_u,
        "soc": soc_u,
        "v_min": float(v_u[0]),
        "v_max": float(v_u[-1]),
        "file": os.path.basename(path),
        "direction": direction,
        "ir_correction": ir_desc,
    }


def map_soc(voltages, lookup: dict) -> tuple:
    """Interpolate voltages onto ``lookup`` -> ``(soc_array, n_clipped)``.

    Voltages outside the sweep's range are clipped to its ends (``np.interp``
    does this) and counted, so a caller can report how many measurements sat
    beyond what the qOCV actually traversed rather than trusting a silently
    saturated value.
    """
    v = pd.to_numeric(pd.Series(voltages), errors="coerce").to_numpy(dtype=float)
    soc = np.interp(v, lookup["v"], lookup["soc"])
    finite = np.isfinite(v)
    n_clipped = int(
        ((v < lookup["v_min"]) | (v > lookup["v_max"])).sum()
    )
    soc = np.where(finite, soc, np.nan)
    return soc, n_clipped


def _pair_eta_for(sweep: dict, sweeps: list, label: str) -> dict:
    """``eta(SOC)`` from the opposite-direction sweep of the same BM_Programm.

    The counterpart must be the *same* check-up — comparing branches measured
    at different ages would fold capacity fade into the polarization.
    """
    opposite = "charge" if sweep["direction"] == "discharge" else "discharge"
    mate = next(
        (s for s in sweeps
         if s["direction"] == opposite and s["BM_Programm"] == sweep["BM_Programm"]),
        None,
    )
    if mate is None:
        logging.info(
            "%s: no %s counterpart for BM%s — no pair-derived IR correction",
            label, opposite, sweep["BM_Programm"],
        )
        return None
    cha = sweep if sweep["direction"] == "charge" else mate
    dch = mate if sweep["direction"] == "charge" else sweep
    try:
        return polarization_from_pair(cha["path"], dch["path"])
    except Exception as exc:
        logging.warning("%s: pair polarization failed: %s", label, exc)
        return None


def map_table(table: pd.DataFrame, voltage_col: str, direction: str,
              sweeps: list, t_ref, label: str, ir_ohm: float = None) -> dict:
    """Add qOCV-derived ``SOC_pct`` to ``table`` in place; return diagnostics.

    This is the **only** source of SOC — the order-based ladder was removed
    (see ``eis_vs_soc.build_eis_table`` / ``pulse_fit.assign_pulse_soc``), so
    when no same-direction qOCV exists ``SOC_pct`` stays NaN and the vs-SOC
    plots skip. A missing SOC is honest; a fabricated one is not.
    """
    diag = {"soc_source": "none (SOC_pct NaN)", "sweep_direction": direction}
    if voltage_col not in table.columns:
        diag["reason"] = f"no {voltage_col} column to map"
        return diag

    sweep = pick_sweep(sweeps, direction, t_ref)
    if sweep is None:
        diag["reason"] = f"no {direction} qOCV sweep in the bundle folder"
        logging.warning("%s: %s — SOC_pct stays NaN", label, diag["reason"])
        return diag

    eta = _pair_eta_for(sweep, sweeps, label)
    try:
        lookup = build_lookup(sweep["path"], direction, eta=eta, ir_ohm=ir_ohm)
    except Exception as exc:
        diag["reason"] = f"{type(exc).__name__}: {exc}"
        logging.warning("%s: qOCV lookup failed (%s) — keeping the ladder", label, exc)
        return diag

    soc, n_clipped = map_soc(table[voltage_col], lookup)
    table["SOC_pct"] = np.round(soc, 2)

    dt_h = None
    if pd.notna(t_ref) and pd.notna(sweep["t_start"]):
        dt_h = round(
            (sweep["t_start"] - pd.to_datetime(t_ref)).total_seconds() / 3600.0, 2
        )
    diag.update({
        "soc_source": f"qOCV {direction} branch ({voltage_col})",
        "soc_source_file": sweep["file"],
        "soc_source_bm": sweep["BM_Programm"],
        "soc_source_soh": sweep["SOH"],
        "soc_dt_hours": dt_h,
        "qocv_v_min": round(lookup["v_min"], 4),
        "qocv_v_max": round(lookup["v_max"], 4),
        "n_clipped": n_clipped,
        "ir_correction": lookup["ir_correction"],
    })
    if eta is not None:
        diag["ir_eta_mid_mV"] = eta["eta_mid_mV"]
    if n_clipped:
        logging.warning(
            "%s: %d measurement(s) outside the qOCV range %.3f–%.3f V — SOC clipped",
            label, n_clipped, lookup["v_min"], lookup["v_max"],
        )
    logging.info(
        "%s: SOC from %s (%s branch, %s h away)",
        label, sweep["file"], direction, dt_h,
    )
    return diag


def assign_soc(table: pd.DataFrame, voltage_col: str, direction: str,
               data_dir: str, t_ref=None, label: str = "", ir_ohm: float = None,
               sweeps: list = None) -> dict:
    """Find the qOCV sweeps in ``data_dir`` and map ``table`` onto them.

    One-call form of :func:`find_sweeps` + :func:`map_table`, which is the pair
    every consumer needs. Use this rather than re-deriving SOC: an order-based
    ladder (``100 - step * index``) looks reasonable and is wrong — see the
    module docstring.

    ``table`` gains ``SOC_pct`` in place (NaN where no same-direction sweep
    exists — an absent SOC is honest, a fabricated one is not). Returns the
    diagnostics dict for ``settings.bundles[]``.

    ``t_ref`` is the measurement time used to pick the nearest sweep; defaults
    to ``table["Time"].min()`` when the column is present. Pass ``sweeps`` to
    reuse a listing already built for another bundle in the same folder.
    """
    if sweeps is None:
        sweeps = find_sweeps(data_dir)
    if not sweeps:
        logging.warning(
            "%s: no qOCV export in %s — SOC_pct stays NaN", label, data_dir
        )
        return {"soc_source": "none (SOC_pct NaN)", "sweep_direction": direction,
                "reason": f"no qOCV export in {data_dir}"}
    if t_ref is None and "Time" in table.columns and len(table):
        t_ref = table["Time"].min()
    return map_table(table, voltage_col, direction, sweeps, t_ref, label,
                     ir_ohm=ir_ohm)
