"""Extract EIS impedance parameters from an export bundle and plot vs SOC.

Companion to the pulse ``plot_vs_soc`` / ``plot_warburg_vs_soc`` family, but for
the EIS spectra produced by ``output/export_eis.py``. A bundled export
(``25_export_eis/<cell>/<cell>_eis_BM<bm>_<SOH>SOH.parquet``) holds every EIS
measurement of one check-up (BM_Programm) stacked, each a settled per-frequency
spectrum tagged by ``eis_number`` / ``Time`` / ``U``. A full-SOC-sweep
check-up records one EIS per SOC plateau, so — exactly like the pulse sweep —
SOC is assigned by time order:

    discharge sweep -> SOC = 100 - soc_step_pct * i   (i = 0 at the top)
    charge sweep    -> SOC =   0 + soc_step_pct * i

reusing the same ``SOC_SWEEP_DIRECTION`` / ``SOC_SWEEP_STEP_PCT`` defaults as
``fit_2rc_pulse`` so EIS and pulse SOC axes line up.

Per measurement, a small set of scale-free, fit-free readouts is extracted from
the Nyquist curve (no ECM fit, so nothing to converge):

    R_ohm   ohmic/series resistance — Z_real at the high-frequency Z_imag=0
            crossing (inductive -> capacitive), interpolated.
    R_tot   Z_real at the lowest frequency (DC-ish total resistance).
    R_pol   polarisation = R_tot - R_ohm (charge-transfer + diffusion).
    arc_pk  semicircle height = max(-Z_imag) over the capacitive branch.
    f_pk    characteristic frequency at that arc peak (~ 1/2πτ).

Impedance values are in the raw unit of the export (for these cells that is
mΩ — a ~28 Ah cell has ohmic R of order 1 mΩ, which is what the data shows).

Usage:
    cd src
    python -m analysis.eis_vs_soc <eis_export.parquet> [-o out_stem]
                                  [--direction discharge|charge] [--step 5.0]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

try:  # keep SOC parameterisation identical to the pulse sweep
    from analysis.fit_2rc_pulse import SOC_SWEEP_DIRECTION, SOC_SWEEP_STEP_PCT
except Exception:  # standalone fallback if imported outside the package
    SOC_SWEEP_DIRECTION = "discharge"
    SOC_SWEEP_STEP_PCT = 5.0


def eis_features(spec: pd.DataFrame) -> dict:
    """Fit-free impedance readouts from one settled spectrum.

    ``spec`` carries ``frequency, Z_real, Z_imag`` (canonical io_eis columns).
    Returns ``R_ohm, R_tot, R_pol, arc_pk, f_pk`` (NaN where undefined).
    """
    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zr = s["Z_real"].to_numpy(float)
    zi = s["Z_imag"].to_numpy(float)

    # R_ohm: Z_real where Z_imag crosses zero at the high-frequency end (the
    # inductive branch is positive, the capacitive branch negative). Take the
    # highest-frequency sign change and interpolate Z_real to Z_imag = 0.
    sign = np.sign(zi)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings):
        k = crossings[-1]
        z0, z1 = zi[k], zi[k + 1]
        r_ohm = zr[k] if z1 == z0 else zr[k] + (zr[k + 1] - zr[k]) * (0 - z0) / (z1 - z0)
    else:
        r_ohm = zr[int(np.argmin(np.abs(zi)))]

    r_tot = zr[int(np.argmin(f))]  # lowest frequency
    cap = zi < 0
    if cap.any():
        neg = -zi[cap]
        j = int(np.argmax(neg))
        arc_pk = float(neg[j])
        f_pk = float(f[cap][j])
    else:
        arc_pk, f_pk = np.nan, np.nan

    return {
        "R_ohm": float(r_ohm),
        "R_tot": float(r_tot),
        "R_pol": float(r_tot - r_ohm),
        "arc_pk": arc_pk,
        "f_pk": f_pk,
    }


def fit_2rc_eis(spec: pd.DataFrame, r_ohm0=None, r_tot0=None) -> dict:
    """Complex least-squares 2RC + series-inductance fit of one spectrum.

    Model:

        Z(ω) = R0 + jωL + R1/(1 + jω τ1) + R2/(1 + jω τ2)

    The series inductance ``L`` captures the high-frequency inductive tail
    (``Z_imag > 0``), so the fit runs over **all** frequencies (no branch is
    dropped). Residuals are normalised by ``|Z|`` so every decade weighs in; the
    slow branch (larger τ) absorbs the low-frequency diffusion tail. Branches are
    returned ordered ``τ1 < τ2``; ``rmse`` is the unweighted complex residual in
    the raw unit (mΩ). ``L`` is in mΩ·s (= raw-Z unit × s/rad).

    Returns ``{R0, L, R1, tau1, R2, tau2, C1, C2, rmse}`` (NaN on failure).
    """
    from scipy.optimize import least_squares

    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zr = s["Z_real"].to_numpy(float)
    zi = s["Z_imag"].to_numpy(float)
    w = 2 * np.pi * f
    zd = zr + 1j * zi
    absz = np.abs(zd)

    r0 = r_ohm0 if r_ohm0 is not None else zr[int(np.argmax(w))]
    rtot = r_tot0 if r_tot0 is not None else zr[int(np.argmin(w))]
    rp = max(rtot - r0, 1e-3)
    # L guess from the highest-frequency inductive point (Z_imag/ω), else tiny.
    khi = int(np.argmax(w))
    l0 = zi[khi] / w[khi] if zi[khi] > 0 else 1e-6
    x0 = [max(r0, 1e-3), max(l0, 1e-9), 0.4 * rp, 0.6 * rp, np.log(0.02), np.log(2.0)]
    lb = [0.0, 0.0, 0.0, 0.0, np.log(1e-4), np.log(1e-4)]
    ub = [np.inf, np.inf, np.inf, np.inf, np.log(1e4), np.log(1e4)]

    def model(x, wv):
        R0_, L_, R1_, R2_, lt1, lt2 = x
        return (R0_ + 1j * wv * L_
                + R1_ / (1 + 1j * wv * np.exp(lt1))
                + R2_ / (1 + 1j * wv * np.exp(lt2)))

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    try:
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=5000)
        R0_, L_, R1_, R2_, lt1, lt2 = res.x
        t1, t2 = np.exp(lt1), np.exp(lt2)
        if t1 > t2:
            R1_, R2_, t1, t2 = R2_, R1_, t2, t1
        Zf = model([R0_, L_, R1_, R2_, np.log(t1), np.log(t2)], w)
        rmse = float(np.sqrt(np.mean(np.abs(Zf - zd) ** 2)))
        return {
            "R0": float(R0_), "L": float(L_),
            "R1": float(R1_), "tau1": float(t1),
            "R2": float(R2_), "tau2": float(t2),
            "C1": float(t1 / R1_) if R1_ else np.nan,
            "C2": float(t2 / R2_) if R2_ else np.nan,
            "rmse": rmse,
        }
    except Exception as exc:  # noqa: BLE001 — a bad spectrum shouldn't kill the sweep
        logging.warning("fit_2rc_eis failed: %s", exc)
        return {k: np.nan for k in ("R0", "L", "R1", "tau1", "R2", "tau2", "C1", "C2", "rmse")}


def _z_warburg(rd, td, w):
    """Finite-length (short) Warburg impedance R_d·tanh(√(jωτ_d))/√(jωτ_d)."""
    x = np.sqrt(1j * w * td)
    return rd * np.tanh(x) / x


def fit_warburg_eis(spec: pd.DataFrame, seed: dict = None, r_tot0=None) -> dict:
    """2RC + series-L + finite-length Warburg fit — the EIS twin of the pulse model.

        Z(ω) = R0 + jωL + R1/(1+jω τ1) + R2/(1+jω τ2)
               + R_d·tanh(√(jω τ_d)) / √(jω τ_d)

    The Warburg branch (``R_d``, ``τ_d``) captures the low-frequency diffusion
    tail explicitly, so the two RC branches are freed to describe the semicircle.
    Seeded from a plain 2RC+L fit (``seed``) for identifiability. Params are
    namespaced ``*_w`` to sit beside the 2RC-only columns, mirroring the pulse
    ``R1_w`` / ``tau1_w`` / ``R_d_ohm`` / ``tau_d_s`` naming.

    Returns ``{R0_w, L_w, R1_w, tau1_w, R2_w, tau2_w, R_d, tau_d, warburg_rmse,
    warburg_degenerate}`` (NaN/True on failure).
    """
    from scipy.optimize import least_squares

    keys = ("R0_w", "L_w", "R1_w", "tau1_w", "R2_w", "tau2_w", "R_d", "tau_d", "warburg_rmse")
    fail = {k: np.nan for k in keys}
    fail["warburg_degenerate"] = True

    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
    w = 2 * np.pi * f
    absz = np.abs(zd)
    seed = seed or {}

    r0 = seed.get("R0", s["Z_real"].to_numpy(float)[int(np.argmax(w))])
    l0 = seed.get("L", 1e-6)
    r1 = seed.get("R1", 1.0)
    r2 = seed.get("R2", 1.0)
    t1 = seed.get("tau1", 0.02)
    t2 = seed.get("tau2", 2.0)
    rtot = r_tot0 if r_tot0 is not None else s["Z_real"].to_numpy(float)[int(np.argmin(w))]
    rd0 = max(rtot - (r0 + r1 + r2), 0.5)
    td0 = 20.0

    x0 = [max(r0, 1e-3), max(l0, 1e-9), max(r1, 1e-3), max(r2, 1e-3),
          np.log(max(t1, 1e-3)), np.log(max(t2, 1e-3)), rd0, np.log(td0)]
    lb = [0.0, 0.0, 0.0, 0.0, np.log(1e-4), np.log(1e-4), 0.0, np.log(1e-2)]
    ub = [np.inf, np.inf, np.inf, np.inf, np.log(1e4), np.log(1e4), np.inf, np.log(1e5)]

    def model(x, wv):
        R0_, L_, R1_, R2_, lt1, lt2, Rd_, ltd = x
        return (R0_ + 1j * wv * L_
                + R1_ / (1 + 1j * wv * np.exp(lt1))
                + R2_ / (1 + 1j * wv * np.exp(lt2))
                + _z_warburg(Rd_, np.exp(ltd), wv))

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    try:
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=8000)
        R0_, L_, R1_, R2_, lt1, lt2, Rd_, ltd = res.x
        ta, tb = np.exp(lt1), np.exp(lt2)
        if ta > tb:
            R1_, R2_, ta, tb = R2_, R1_, tb, ta
        Zf = model([R0_, L_, R1_, R2_, np.log(ta), np.log(tb), Rd_, ltd], w)
        rmse = float(np.sqrt(np.mean(np.abs(Zf - zd) ** 2)))
        td = float(np.exp(ltd))
        # Degenerate when the Warburg branch collapses or merges into the slow RC.
        degenerate = bool(Rd_ < 1e-2 or (0.5 < td / tb < 2.0 and abs(Rd_ - R2_) / max(R2_, 1e-6) < 0.1))
        return {
            "R0_w": float(R0_), "L_w": float(L_),
            "R1_w": float(R1_), "tau1_w": float(ta),
            "R2_w": float(R2_), "tau2_w": float(tb),
            "R_d": float(Rd_), "tau_d": td,
            "warburg_rmse": rmse, "warburg_degenerate": degenerate,
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning("fit_warburg_eis failed: %s", exc)
        return fail


def _z_zarc(r, tau, alpha, w):
    """ZARC (R ∥ CPE) impedance R / (1 + (jωτ)^α); α=1 is an ideal RC."""
    return r / (1 + (1j * w * tau) ** alpha)


def fit_zarc_warburg_eis(spec: pd.DataFrame, seed: dict = None, r_tot0=None) -> dict:
    """2×ZARC + series-L + finite-length Warburg fit.

        Z(ω) = R0 + jωL + R1/(1+(jω τ1)^α1) + R2/(1+(jω τ2)^α2)
               + R_d·tanh(√(jω τ_d)) / √(jω τ_d)

    Replaces the two ideal RC branches of :func:`fit_warburg_eis` with **ZARC**
    (depressed-arc) branches — each gains a CPE exponent ``α ∈ (0,1]`` so a real,
    flattened Nyquist semicircle is captured without inflating R/τ. Seeded from
    the 2RC+L+Warburg fit (``seed``) with α=0.85. Params namespaced ``*_z``.

    Returns ``{R0_z, L_z, R1_z, tau1_z, alpha1_z, R2_z, tau2_z, alpha2_z, R_d_z,
    tau_d_z, zarc_rmse, zarc_degenerate}`` (NaN/True on failure).
    """
    from scipy.optimize import least_squares

    keys = ("R0_z", "L_z", "R1_z", "tau1_z", "alpha1_z", "R2_z", "tau2_z",
            "alpha2_z", "R_d_z", "tau_d_z", "zarc_rmse")
    fail = {k: np.nan for k in keys}
    fail["zarc_degenerate"] = True

    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
    w = 2 * np.pi * f
    absz = np.abs(zd)
    seed = seed or {}

    r0 = seed.get("R0_w", seed.get("R0", s["Z_real"].to_numpy(float)[int(np.argmax(w))]))
    l0 = seed.get("L_w", seed.get("L", 1e-6))
    r1 = seed.get("R1_w", 1.0)
    r2 = seed.get("R2_w", 1.0)
    t1 = seed.get("tau1_w", 0.02)
    t2 = seed.get("tau2_w", 2.0)
    rd0 = seed.get("R_d", 1.0)
    td0 = seed.get("tau_d", 20.0)

    # x = [R0, L, R1, R2, lt1, lt2, a1, a2, Rd, ltd]
    x0 = [max(r0, 1e-3), max(l0, 1e-9), max(r1, 1e-3), max(r2, 1e-3),
          np.log(max(t1, 1e-3)), np.log(max(t2, 1e-3)), 0.85, 0.85,
          max(rd0, 1e-2), np.log(max(td0, 1e-2))]
    lb = [0.0, 0.0, 0.0, 0.0, np.log(1e-4), np.log(1e-4), 0.3, 0.3, 0.0, np.log(1e-2)]
    ub = [np.inf, np.inf, np.inf, np.inf, np.log(1e4), np.log(1e4), 1.0, 1.0, np.inf, np.log(1e5)]

    def model(x, wv):
        R0_, L_, R1_, R2_, lt1, lt2, a1, a2, Rd_, ltd = x
        return (R0_ + 1j * wv * L_
                + _z_zarc(R1_, np.exp(lt1), a1, wv)
                + _z_zarc(R2_, np.exp(lt2), a2, wv)
                + _z_warburg(Rd_, np.exp(ltd), wv))

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    try:
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=10000)
        R0_, L_, R1_, R2_, lt1, lt2, a1, a2, Rd_, ltd = res.x
        ta, tb, aa, ab = np.exp(lt1), np.exp(lt2), a1, a2
        if ta > tb:
            R1_, R2_, ta, tb, aa, ab = R2_, R1_, tb, ta, ab, aa
        Zf = model([R0_, L_, R1_, R2_, np.log(ta), np.log(tb), aa, ab, Rd_, ltd], w)
        rmse = float(np.sqrt(np.mean(np.abs(Zf - zd) ** 2)))
        td = float(np.exp(ltd))
        degenerate = bool(Rd_ < 1e-2 or (0.5 < td / tb < 2.0 and abs(Rd_ - R2_) / max(R2_, 1e-6) < 0.1))
        return {
            "R0_z": float(R0_), "L_z": float(L_),
            "R1_z": float(R1_), "tau1_z": float(ta), "alpha1_z": float(aa),
            "R2_z": float(R2_), "tau2_z": float(tb), "alpha2_z": float(ab),
            "R_d_z": float(Rd_), "tau_d_z": td,
            "zarc_rmse": rmse, "zarc_degenerate": degenerate,
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning("fit_zarc_warburg_eis failed: %s", exc)
        return fail


def build_eis_table(df: pd.DataFrame, direction=None, step=None,
                    fit_2rc=True, fit_warburg=True, fit_zarc=True) -> pd.DataFrame:
    """Per-measurement feature table with a time-ordered sweep SOC.

    One row per ``eis_number`` (measurement), ordered by ``Time``, with SOC
    assigned by plateau index — mirroring ``assign_pulse_soc``.
    """
    direction = direction if direction is not None else SOC_SWEEP_DIRECTION
    step = step if step is not None else SOC_SWEEP_STEP_PCT

    order = (
        df.groupby("eis_number")["Time"].min().sort_values().index.tolist()
    )
    rows = []
    for i, eid in enumerate(order):
        spec = df[df["eis_number"] == eid]
        feat = eis_features(spec)
        if fit_2rc:
            fit = fit_2rc_eis(spec, r_ohm0=feat["R_ohm"], r_tot0=feat["R_tot"])
            feat.update(fit)
            if fit_warburg:
                wfit = fit_warburg_eis(spec, seed=fit, r_tot0=feat["R_tot"])
                feat.update(wfit)
                if fit_zarc:
                    feat.update(fit_zarc_warburg_eis(spec, seed=wfit, r_tot0=feat["R_tot"]))
        if str(direction).lower().startswith("cha"):
            soc = 0.0 + step * i
        else:
            soc = 100.0 - step * i
        feat.update(
            {
                "eis_number": eid,
                "SOC_pct": soc,
                "U": float(spec["U"].mean()),
                "Time": spec["Time"].min(),
            }
        )
        rows.append(feat)
    out = pd.DataFrame(rows)
    logging.info(
        "eis_vs_soc: %d measurements -> SOC %.0f..%.0f%% (%s sweep, %.1f%% step)",
        len(out), out["SOC_pct"].iloc[0], out["SOC_pct"].iloc[-1], direction, step,
    )
    return out


def plot_eis_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of EIS readouts vs SOC — the EIS analogue of ``plot_vs_soc``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("R_ohm", "R_ohm — series (mΩ)"),
        ("R_pol", "R_pol — total polarisation (mΩ)"),
        ("R_tot", "R_tot — DC (mΩ)"),
        ("arc_pk", "arc height max(-Z_imag) (mΩ)"),
        ("f_pk", "char. frequency at arc (Hz)"),
        ("U", "rest voltage / OCV (V)"),
    ]
    t = table.sort_values("SOC_pct")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#2f6fdb")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col == "f_pk":
            ax.set_yscale("log")
    fig.suptitle(f"EIS parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS vs-SOC plot -> %s", out_png)


def plot_2rc_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of the fitted 2RC parameters vs SOC — EIS analogue of the pulse 2RC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R0" not in table.columns:
        logging.info("2RC vs-SOC plot: no fit columns, skipping")
        return
    metrics = [
        ("R0", "R0 — series (mΩ)"),
        ("L", "L — series inductance (mΩ·s)"),
        ("R1", "R1 — fast branch (mΩ)"),
        ("tau1", "τ1 (s)"),
        ("R2", "R2 — slow branch (mΩ)"),
        ("tau2", "τ2 (s)"),
        ("rmse", "fit rmse (mΩ)"),
    ]
    metrics = [m for m in metrics if m[0] in table.columns]
    t = table.sort_values("SOC_pct")
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#c0392b")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col in ("tau1", "tau2"):
            ax.set_yscale("log")
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f"EIS 2RC+L fit parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS 2RC vs-SOC plot -> %s", out_png)


def plot_warburg_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of the 2RC+L+Warburg fit parameters vs SOC (hides degenerate fits)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R_d" not in table.columns:
        logging.info("Warburg vs-SOC plot: no Warburg columns, skipping")
        return
    t = table.sort_values("SOC_pct").copy()
    if "warburg_degenerate" in t.columns:
        n = int((t["warburg_degenerate"] == True).sum())  # noqa: E712
        if n:
            logging.info("Warburg vs-SOC plot: hiding %d degenerate fit(s)", n)
        t = t[t["warburg_degenerate"] != True]  # noqa: E712
    if t.empty:
        logging.info("Warburg vs-SOC plot: nothing to plot")
        return
    metrics = [
        ("R0_w", "R0 — series (mΩ)"),
        ("R1_w", "R1 (mΩ)"), ("tau1_w", "τ1 (s)"),
        ("R2_w", "R2 (mΩ)"), ("tau2_w", "τ2 (s)"),
        ("R_d", "R_d — diffusion (mΩ)"), ("tau_d", "τ_d — diffusion (s)"),
        ("warburg_rmse", "2RC+W rmse (mΩ)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#8e44ad")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col in ("tau1_w", "tau2_w", "tau_d"):
            ax.set_yscale("log")
    fig.suptitle(f"EIS 2RC + Warburg fit parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS Warburg vs-SOC plot -> %s", out_png)


def plot_zarc_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of the 2×ZARC+L+Warburg fit parameters vs SOC (hides degenerate)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R_d_z" not in table.columns:
        logging.info("ZARC vs-SOC plot: no ZARC columns, skipping")
        return
    t = table.sort_values("SOC_pct").copy()
    if "zarc_degenerate" in t.columns:
        n = int((t["zarc_degenerate"] == True).sum())  # noqa: E712
        if n:
            logging.info("ZARC vs-SOC plot: hiding %d degenerate fit(s)", n)
        t = t[t["zarc_degenerate"] != True]  # noqa: E712
    if t.empty:
        logging.info("ZARC vs-SOC plot: nothing to plot")
        return
    metrics = [
        ("R0_z", "R0 — series (mΩ)"),
        ("R1_z", "R1 (mΩ)"), ("tau1_z", "τ1 (s)"), ("alpha1_z", "α1"),
        ("R2_z", "R2 (mΩ)"), ("tau2_z", "τ2 (s)"), ("alpha2_z", "α2"),
        ("R_d_z", "R_d — diffusion (mΩ)"), ("tau_d_z", "τ_d (s)"),
        ("zarc_rmse", "2ZARC+W rmse (mΩ)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#16a085")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col in ("tau1_z", "tau2_z", "tau_d_z"):
            ax.set_yscale("log")
        if col in ("alpha1_z", "alpha2_z"):
            ax.set_ylim(0.25, 1.05)
    fig.suptitle(f"EIS 2×ZARC + Warburg fit parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS ZARC vs-SOC plot -> %s", out_png)


def plot_fit_overlay(df: pd.DataFrame, table: pd.DataFrame, out_png: str,
                     title: str = "", n_show: int = 6):
    """Measured vs fitted Nyquist (2RC, 2RC+Warburg, 2ZARC+Warburg) per SOC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R0" not in table.columns:
        return
    t = table.dropna(subset=["R0"]).sort_values("SOC_pct")
    if t.empty:
        return
    pick = t.iloc[np.linspace(0, len(t) - 1, min(n_show, len(t))).astype(int)]
    has_w = "R_d" in table.columns

    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, pick.iterrows()):
        spec = df[df["eis_number"] == row["eis_number"]].sort_values("frequency")
        w = 2 * np.pi * spec["frequency"].to_numpy(float)
        z2 = (row["R0"] + 1j * w * row.get("L", 0.0)
              + row["R1"] / (1 + 1j * w * row["tau1"])
              + row["R2"] / (1 + 1j * w * row["tau2"]))
        ax.plot(spec["Z_real"], -spec["Z_imag"], "o", ms=3, color="#2f6fdb", label="measured")
        ax.plot(z2.real, -z2.imag, "-", color="#c0392b", label=f"2RC ({row['rmse']:.3f})")
        if has_w and np.isfinite(row.get("R_d", np.nan)):
            zw = (row["R0_w"] + 1j * w * row["L_w"]
                  + row["R1_w"] / (1 + 1j * w * row["tau1_w"])
                  + row["R2_w"] / (1 + 1j * w * row["tau2_w"])
                  + _z_warburg(row["R_d"], row["tau_d"], w))
            ax.plot(zw.real, -zw.imag, "-", color="#8e44ad", label=f"2RC+W ({row['warburg_rmse']:.3f})")
        if "R_d_z" in table.columns and np.isfinite(row.get("R_d_z", np.nan)):
            zz = (row["R0_z"] + 1j * w * row["L_z"]
                  + _z_zarc(row["R1_z"], row["tau1_z"], row["alpha1_z"], w)
                  + _z_zarc(row["R2_z"], row["tau2_z"], row["alpha2_z"], w)
                  + _z_warburg(row["R_d_z"], row["tau_d_z"], w))
            ax.plot(zz.real, -zz.imag, "-", color="#16a085", label=f"2ZARC+W ({row['zarc_rmse']:.3f})")
        ax.set_title(f"SOC {row['SOC_pct']:.0f}%", fontsize=9)
        ax.set_xlabel("Z_real (mΩ)")
        ax.set_ylabel("-Z_imag (mΩ)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[len(pick):]:
        ax.set_visible(False)
    fig.suptitle(f"EIS fit overlay — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS fit overlay -> %s", out_png)


def plot_nyquist_by_soc(df: pd.DataFrame, table: pd.DataFrame, out_png: str, title: str = ""):
    """All spectra on one Nyquist plane, coloured by SOC (context companion)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    soc = dict(zip(table["eis_number"], table["SOC_pct"]))
    norm = colors.Normalize(vmin=0, vmax=100)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for eid, g in df.groupby("eis_number"):
        g = g.sort_values("frequency")
        ax.plot(g["Z_real"], -g["Z_imag"], "-", lw=1,
                color=cmap(norm(soc.get(eid, np.nan))))
    ax.set_xlabel("Z_real (mΩ)")
    ax.set_ylabel("-Z_imag (mΩ)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="SOC (%)")
    fig.suptitle(f"EIS Nyquist by SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS Nyquist plot -> %s", out_png)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="EIS parameters vs SOC from an export bundle")
    ap.add_argument("export", help="path to a *_eis_BM*.parquet export bundle")
    ap.add_argument("-o", "--out-stem", default=None,
                    help="output PNG stem (default: alongside the export)")
    ap.add_argument("--direction", default=None, choices=["discharge", "charge"],
                    help="sweep direction (default: SOC_SWEEP_DIRECTION)")
    ap.add_argument("--step", type=float, default=None,
                    help="SOC step per measurement in %% (default: SOC_SWEEP_STEP_PCT)")
    args = ap.parse_args()

    df = pd.read_parquet(args.export)
    table = build_eis_table(df, direction=args.direction, step=args.step)

    stem = args.out_stem or os.path.splitext(args.export)[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    title = os.path.basename(stem)
    table.to_csv(f"{stem}_eis_params.csv", index=False)
    logging.info("EIS param table -> %s", f"{stem}_eis_params.csv")
    plot_eis_vs_soc(table, f"{stem}_eis_vs_SOC.png", title=title)
    plot_2rc_vs_soc(table, f"{stem}_eis_2RC_vs_SOC.png", title=title)
    plot_warburg_vs_soc(table, f"{stem}_eis_2RC_warburg_vs_SOC.png", title=title)
    plot_zarc_vs_soc(table, f"{stem}_eis_2ZARC_warburg_vs_SOC.png", title=title)
    plot_fit_overlay(df, table, f"{stem}_eis_fit_overlay.png", title=title)
    plot_nyquist_by_soc(df, table, f"{stem}_eis_nyquist_by_SOC.png", title=title)


if __name__ == "__main__":
    main()
