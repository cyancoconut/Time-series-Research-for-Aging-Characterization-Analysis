"""Validate the EIS-fitted ECM on the pulses of the same sweep.

Nothing is fitted here. Each pulse is predicted from the ECM of the EIS
measurement taken at its SOC step (the last spectrum finished before the pulse
starts), and the prediction is compared with the measured voltage — during the
pulse and through the relaxation behind it.

Time-domain response
--------------------
The fitted impedance ``Z(s)`` (``eis_vs_soc``'s own element functions, so every
diffusion element the fitter writes is evaluated exactly as it was fitted) is
turned into a step response ``h(t) = L⁻¹{Z(s)/s}`` by fixed-Talbot numerical
inversion (Abate & Valkó 2004), and superposed over the piecewise-constant
current history::

    V(t) = V_pre + Σ_k ΔI_k · h(t − t_k)

Steps sit at each segment's first row (where the cycler switched), with the
segment's median current. ``R0`` is added exactly (``R0·step``) rather than
through the inversion; ``L`` is dropped — at the logger's ≥ 10 ms cadence it is
invisible. History reaches back ``HISTORY_S`` but stops at the last data gap
longer than ``GAP_S``, since the current through a gap is unknown.

Two variants are reported:

``ecm``
    the EIS model alone.
``ocv``
    ``ecm + dOCV/dQ · ΔQ``, the coulomb-counted OCV drift on the slope of the
    mean qOCV branch. Needed with a **transmissive** diffusion element, which
    stores no charge; with the **blocking** element it double-counts whatever
    storage the capacitive tail already carries.

Relaxation is scored up to the next sample that carries current (or the next
data gap), so a following restore step never leaks into it.

Measured on ``J8049_Namey_NFPP_28Ah_02`` BM22: on the 30 s 0.35C pulses
R_DC(1 s) is 3.65–3.70 measured vs 3.68–3.69 mΩ predicted, relaxation RMSE
1.0–1.2 mV, with either diffusion element. On the 200 s 0.1C pulses the
blocking element (τ_d pinned 5 s, its capacitor ``τ_d/R_d`` 6–18× too small)
puts end-of-pulse R_DC 10–25 % high; the transmissive one + OCV −7…+6 %. 1C
charge pulses below 12 % SOC are over-predicted by tens to hundreds of mV by
both — a large-signal (Butler–Volmer) effect a small-signal impedance cannot
carry.

Usage::

    python -m analysis.validate_ecm_on_pulse <cell_dir> [--fits CSV] [--bm N]
        [-o OUT_DIR] [--nom-capacity AH] [--ocv-variant]
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import eis_vs_soc as ev  # noqa: E402
from util.io_qocv import load_sweep  # noqa: E402

HISTORY_S = 1800.0      # current history carried into each pulse window
GAP_S = 5.0             # a sample gap above this ends the known history
PRE_S = 5.0             # pre-pulse rest window for V_pre
LIVE_A = 0.05           # |I| above this ends a relaxation window
TALBOT_M = 24
MOHM = 1e-3             # eis_fits.csv is in mΩ
RDC_DT_S = (1.0, 10.0, 20.0)
#: diffusion elements ``eis_vs_soc._z_diffusion_from_row`` evaluates
KNOWN_ELEMENTS = {"generalized", "generalized_transmissive", "coth", "tanh"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_ECM, C_OCV = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------- impedance
def z_ecm(s, p):
    """Fitted ECM without R0 and L, at Laplace variable ``s`` (Ω).

    The element functions take ω and form ``jω`` internally, so they are
    called with ``ω = s/j``; a branch that is NaN (a one-ZARC fit) is skipped.
    """
    w = s / 1j
    z = np.zeros_like(s)
    for i in (1, 2):
        r = p.get(f"R{i}_z", np.nan)
        if np.isfinite(r):
            z = z + ev._z_zarc(r, p[f"tau{i}_z"], p[f"alpha{i}_z"], w)
    if np.isfinite(p.get("R_d_z", np.nan)):
        z = z + ev._z_diffusion_from_row(p, w)
    return z * MOHM


def talbot(F, t, M=TALBOT_M):
    """Fixed-Talbot inverse Laplace transform, vectorized in ``t`` (0 at t ≤ 0)."""
    t = np.atleast_1d(np.asarray(t, float))
    out = np.zeros_like(t)
    pos = t > 0
    if not pos.any():
        return out
    tt = t[pos][:, None]
    r = 2 * M / (5 * tt)
    th = np.arange(1, M)[None, :] * np.pi / M
    cot = 1 / np.tan(th)
    s = r * th * (cot + 1j)
    sig = th + (th * cot - 1) * cot
    term0 = 0.5 * np.exp(r * tt) * F(r + 0j)
    terms = np.exp(tt * s) * F(s) * (1 + 1j * sig)
    out[pos] = (r / M * (term0.real + terms.real.sum(axis=1, keepdims=True)))[:, 0]
    return out


def step_response(p, t):
    """Voltage response (V per A) to a unit current step at t = 0."""
    return talbot(lambda s: z_ecm(s, p) / s, t) + p["R0_z"] * MOHM * (np.asarray(t) > 0)


def _self_test():
    """Talbot against the exact RC step response; fails loudly if broken."""
    p = dict(R1_z=1.0, tau1_z=0.5, alpha1_z=1.0, R2_z=np.nan, R_d_z=np.nan, R0_z=0.0)
    t = np.array([0.05, 0.5, 2.0, 10.0])
    exact = MOHM * (1 - np.exp(-t / 0.5))
    err = np.max(np.abs(talbot(lambda s: z_ecm(s, p) / s, t) - exact))
    assert err < 1e-9, f"Talbot self-test failed: {err:.2e}"


# ---------------------------------------------------------------- OCV slope
def ocv_slope_fn(cha_path: Path, dch_path: Path):
    """``dOCV/dQ`` (V/Ah) vs SOC on the mean of the qOCV cha/dch branches.

    The mean removes the branches' ~C/20 overpotential to first order. Smoothed
    over ±1 % SOC before differentiating. Returns ``(slope(soc), capacity_Ah)``.
    """
    vc, qc = load_sweep(cha_path)
    vd, qd = load_sweep(dch_path, discharge=True)
    grid = np.linspace(0, 100, 1001)
    cap = 0.5 * (qc.max() + qd.max())
    v = 0.5 * (np.interp(grid, qc / qc.max() * 100, vc)
               + np.interp(grid, qd / qd.max() * 100, vd))
    w = 21
    vs = np.convolve(np.pad(v, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    dvdq = np.gradient(vs, grid * cap / 100)
    return (lambda soc: float(np.interp(soc, grid, dvdq))), cap


# ---------------------------------------------------------------- data
def _bm_of(name: str):
    m = re.search(r"_BM(\d+)_", str(name))
    return int(m.group(1)) if m else None


def _one(paths, what):
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no {what} found")
    if len(paths) > 1:
        logging.info("several %s, using %s", what, paths[0].name)
    return paths[0]


def load_inputs(cell_dir: Path, fits_csv: Path, bm: int | None):
    """Fits of one EIS bundle (time-ordered) + that programme's pulse bundle."""
    data = cell_dir / "data"
    fits = pd.read_csv(fits_csv)
    if bm is None:
        bms = sorted({b for b in fits.get("source", pd.Series(dtype=str)).map(_bm_of) if b is not None})
        if len(bms) != 1:
            raise SystemExit(f"fits cover BM {bms or '?'}; pick one with --bm")
        bm = bms[0]
    if "source" in fits:
        fits = fits[fits["source"].map(_bm_of) == bm]
    # _z_diffusion_from_row falls back to the blocking form for any name it does
    # not know, so a table written by a newer fitter would be evaluated with the
    # wrong element without a word — refuse it instead.
    unknown = set(fits.get("diff_element", pd.Series(dtype=str)).dropna()) - KNOWN_ELEMENTS
    if unknown:
        raise SystemExit(f"unknown diff_element {sorted(unknown)}; "
                         "eis_vs_soc._z_diffusion_from_row cannot evaluate it")
    eis = pd.read_parquet(_one(data.glob(f"*_eis_BM{bm}_*.parquet"), f"EIS bundle BM{bm}"))
    fits = fits.assign(t_end=fits["eis_number"].map(eis.groupby("eis_number")["Time"].max()))
    fits = fits.dropna(subset=["t_end"]).sort_values("t_end").reset_index(drop=True)

    pulse = pd.read_parquet(_one(data.glob(f"*_pulse_BM{bm}_*.parquet"), f"pulse bundle BM{bm}"))
    pulse = pulse.sort_values("Time").reset_index(drop=True)
    pulse["t"] = (pulse["Time"] - pulse["Time"].iloc[0]).dt.total_seconds()
    return fits, pulse, bm


def segments(pulse):
    """One row per segment with idealised (median) current; rests are 0 A."""
    seg = pulse.groupby("ID", sort=False).agg(
        t0=("t", "min"), t1=("t", "max"), Z=("Zustand", "first"),
        tgt=("target", "first"), I=("Current", "median"), T0=("Time", "min"))
    seg.loc[seg["Z"] == "PAU", "I"] = 0.0
    return seg.sort_values("t0")


def current_steps(pulse, seg, t_start, t_stop):
    """``(t_k, ΔI_k, history_start)`` of the idealised current over known history."""
    win = pulse[(pulse["t"] >= t_start - HISTORY_S) & (pulse["t"] <= t_stop)]
    gaps = (win["t"].diff() > GAP_S) & (win["t"] <= t_start)
    if gaps.any():                     # current before the last gap is unknown
        win = win.loc[win.index[gaps][-1]:]
    h0 = win["t"].iloc[0]
    s = seg[(seg["t1"] >= h0) & (seg["t0"] <= t_stop)]
    times, dI, prev = [], [], 0.0
    for _, r in s.iterrows():
        if r["I"] != prev:
            times.append(max(r["t0"], h0))
            dI.append(r["I"] - prev)
            prev = r["I"]
    return np.array(times), np.array(dI), h0


# ---------------------------------------------------------------- validation
def validate(fits, pulse, slope_at, nom_capacity):
    """Per-pulse scores + the curves behind them."""
    seg = segments(pulse)
    rows, curves = [], {}
    for pid, ps in seg[seg["tgt"] == "PUL"].iterrows():
        prior = fits[fits["t_end"] <= ps["T0"]]
        if prior.empty:
            continue
        p = prior.iloc[-1].to_dict()          # the spectrum of this SOC step
        nxt = seg[seg["t0"] > ps["t1"]]
        t_stop = nxt["t1"].iloc[0] if len(nxt) and nxt["Z"].iloc[0] == "PAU" else ps["t1"]
        tk, dI, h0 = current_steps(pulse, seg, ps["t0"], t_stop)

        w = pulse[(pulse["t"] >= ps["t0"] - PRE_S) & (pulse["t"] <= t_stop)]
        live = (w["t"] > ps["t1"]) & (w["Current"].abs() > LIVE_A)
        if live.any():
            w = w[w["t"] < w.loc[live, "t"].iloc[0]]
        gap = (w["t"] > ps["t1"]) & (w["t"].diff() > GAP_S)
        if gap.any():
            w = w[w["t"] < w.loc[gap, "t"].iloc[0]]
        t = w["t"].to_numpy()
        v = w["Voltage"].to_numpy()
        pre = t < ps["t0"]
        if not pre.any():
            continue
        vpre = v[pre].mean()
        t_ref = t[pre].mean()

        def resp(tq):
            return sum(d * step_response(p, tq - k) for k, d in zip(tk, dI))

        v_ecm = vpre + resp(t) - resp(np.array([t_ref]))[0]
        dq = sum(d * (np.clip(t - k, 0, None) - max(t_ref - k, 0))
                 for k, d in zip(tk, dI)) / 3600
        v_ocv = v_ecm + slope_at(p["SOC_pct"]) * dq

        on = (t >= ps["t0"]) & (t <= ps["t1"])
        off = t > ps["t1"]
        I = ps["I"]
        rec = dict(ID=pid, dir="CHA" if I > 0 else "DCH", I_A=round(I, 2),
                   C_rate=round(I / nom_capacity, 3), dur_s=round(ps["t1"] - ps["t0"], 1),
                   SOC_pct=p["SOC_pct"], eis_number=p["eis_number"],
                   diff_element=p.get("diff_element", "generalized"),
                   eis_degenerate=p.get("zarc_degenerate"),
                   history_s=round(ps["t0"] - h0, 0), relax_s=round(t[-1] - ps["t1"], 0))
        for name, vm in (("ecm", v_ecm), ("ocv", v_ocv)):
            rec[f"rmse_pulse_mV_{name}"] = 1e3 * np.sqrt(np.mean((vm[on] - v[on]) ** 2))
            rec[f"rmse_relax_mV_{name}"] = (1e3 * np.sqrt(np.mean((vm[off] - v[off]) ** 2))
                                            if off.any() else np.nan)
            rec[f"err_end_mV_{name}"] = 1e3 * (vm[on][-1] - v[on][-1])
        for dt_ in RDC_DT_S:
            j = np.searchsorted(t, ps["t0"] + dt_ - 1e-6)
            if j < len(t) and t[j] <= ps["t1"] + 1e-6:
                rec[f"Rdc_{dt_:g}s_meas_mOhm"] = 1e3 * (v[j] - vpre) / I
                rec[f"Rdc_{dt_:g}s_ecm_mOhm"] = 1e3 * (v_ecm[j] - vpre) / I
        rec["Rdc_end_meas_mOhm"] = 1e3 * (v[on][-1] - vpre) / I
        rec["Rdc_end_ecm_mOhm"] = 1e3 * (v_ecm[on][-1] - vpre) / I
        rows.append(rec)
        curves[pid] = (t - ps["t0"], v, v_ecm, v_ocv, ps["t1"] - ps["t0"], rec)
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------- plots
def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(res, curves, out_dir: Path, title: str, show_ocv: bool):
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED})
    pick = []                                  # each amplitude at low/mid/high SOC
    for _, g in res.groupby(["dir", "I_A"]):
        g = g.sort_values("SOC_pct")
        pick += list(dict.fromkeys(g["ID"].iloc[[0, len(g) // 2, -1]]))
    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False)
    for ax, pid in zip(axes.flat, pick):
        tr, v, ve, vo, dur, rec = curves[pid]
        m = tr <= dur + 300
        ax.plot(tr[m], v[m], color=INK, lw=2, label="measured")
        ax.plot(tr[m], ve[m], color=C_ECM, lw=1.5, label="EIS ECM")
        if show_ocv:
            ax.plot(tr[m], vo[m], color=C_OCV, lw=1.5, ls="--", label="EIS ECM + OCV slope")
        ax.set_title(f"{pid}  {rec['dir']} {abs(rec['I_A']):.1f} A  SOC {rec['SOC_pct']:.0f} %"
                     f"   rmse {rec['rmse_pulse_mV_ecm']:.1f} / {rec['rmse_relax_mV_ecm']:.1f} mV",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("time since pulse start (s)")
        ax.set_ylabel("V")
        _style(ax)
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}: EIS-fitted ECM predicting pulses (no pulse fitting) — "
                 "title rmse = pulse / relaxation (ECM alone)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "pulse_overlays.png", dpi=130)
    plt.close(fig)

    groups = list(res.groupby(["dir", "I_A"]))
    fig, axes = plt.subplots(2, len(groups), figsize=(3.6 * len(groups), 6),
                             squeeze=False, sharey="row")
    for j, ((d, I), g) in enumerate(groups):
        g = g.sort_values("SOC_pct")
        for i, dt_ in enumerate(("1s", "end")):
            ax = axes[i, j]
            mc, ec = f"Rdc_{dt_}_meas_mOhm", f"Rdc_{dt_}_ecm_mOhm"
            if mc in g:
                ax.plot(g["SOC_pct"], g[mc], "o-", color=INK, lw=2, ms=4, label="pulse")
                ax.plot(g["SOC_pct"], g[ec], "s-", color=C_ECM, lw=1.5, ms=4, label="EIS ECM")
            ax.set_title(f"{d} {abs(I):.1f} A  R_DC({dt_})", fontsize=9, loc="left")
            ax.set_xlabel("SOC (%)")
            ax.set_ylabel("mΩ")
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "rdc_vs_soc.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate an EIS-fitted ECM on the pulses.")
    ap.add_argument("cell_dir", type=Path,
                    help="10_initial_characterization/<cell_stem> folder (holds data/)")
    ap.add_argument("--fits", type=Path, default=None,
                    help="EIS fits CSV (default <cell_dir>/<cell_stem>_eis_fits.csv)")
    ap.add_argument("--bm", type=int, default=None,
                    help="BM_Programm to validate (default: the one the fits cover)")
    ap.add_argument("-o", "--out-dir", type=Path, default=None,
                    help="output folder (default <cell_dir>/ecm_pulse_validation)")
    ap.add_argument("--nom-capacity", type=float, default=None,
                    help="Ah for the C-rate column (default: qOCV capacity)")
    ap.add_argument("--no-ocv-variant", action="store_true",
                    help="skip the OCV-slope variant (and the qOCV it needs)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _self_test()
    cell_dir = args.cell_dir
    stem = cell_dir.name
    fits_csv = args.fits or cell_dir / f"{stem}_eis_fits.csv"
    out_dir = args.out_dir or cell_dir / "ecm_pulse_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    fits, pulse, bm = load_inputs(cell_dir, fits_csv, args.bm)
    if args.no_ocv_variant:
        slope_at, cap = (lambda soc: 0.0), np.nan
    else:
        data = cell_dir / "data"
        slope_at, cap = ocv_slope_fn(_one(data.glob("*_qocv_cha_*.parquet"), "qOCV cha"),
                                     _one(data.glob("*_qocv_dch_*.parquet"), "qOCV dch"))
    nom = args.nom_capacity or cap
    if not np.isfinite(nom):
        raise SystemExit("--nom-capacity is required with --no-ocv-variant")

    res, curves = validate(fits, pulse, slope_at, nom)
    if res.empty:
        raise SystemExit("no pulse has an EIS measurement before it")
    out_csv = out_dir / f"{stem}_BM{bm}_ecm_pulse_validation.csv"
    res.to_csv(out_csv, index=False)
    cols = [c for c in ("rmse_pulse_mV_ecm", "rmse_relax_mV_ecm", "err_end_mV_ecm",
                        "rmse_pulse_mV_ocv", "rmse_relax_mV_ocv", "err_end_mV_ocv",
                        "Rdc_1s_meas_mOhm", "Rdc_1s_ecm_mOhm",
                        "Rdc_end_meas_mOhm", "Rdc_end_ecm_mOhm")
            if c in res and not (args.no_ocv_variant and c.endswith("_ocv"))]
    pd.set_option("display.width", 250)
    print(res.groupby(["dir", "I_A"])[cols].median().round(3).to_string())
    plot(res, curves, out_dir, f"{stem} BM{bm}", show_ocv=not args.no_ocv_variant)
    logging.info("%d pulses -> %s", len(res), out_csv)


if __name__ == "__main__":
    main()
