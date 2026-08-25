"""Benchmark the in-house EIS fitters against the ``impedance.py`` library.

Fits every spectrum of an EIS export bundle twice — once with
``analysis.eis_vs_soc`` (hand-rolled ``scipy.least_squares``) and once with
``impedance.py``'s ``CustomCircuit`` — on matched circuit topologies, then
compares fit quality, parameter agreement, SOC-smoothness and runtime.

Read-only: nothing here is wired into the pipeline.

Element parity (verified against ``impedance.models.circuits.elements``):

    impedance.py            eis_vs_soc                       relation
    ------------------------------------------------------------------------
    p(R, C)                 R/(1 + jωτ)                      τ = R·C
    p(R, CPE)               _z_zarc                          τ = (R·Q)^(1/α)
    Ws  Z0·tanh(√(jωτ))/√() _z_warburg                       identical
    Wo  Z0·coth(√(jωτ))/√() _z_warburg_generalized(φ=0.5)    identical
    L                       jωL                              identical

so both stacks are reduced to one canonical parameter set
(``R0, L, R1, tau1, alpha1, R2, tau2, alpha2, R_d, tau_d, phi_d``) and scored
with the *same* unweighted complex RMSE, computed here from each stack's own
returned parameters. Impedances stay in the export's raw unit (mΩ).

Variants
--------
``A``  2RC + L                    vs  ``L0-R0-p(R1,C1)-p(R2,C2)``
``B``  2RC + L + tanh-Warburg     vs  ``L0-R0-p(R1,C1)-p(R2,C2)-Ws0``
``C``  production 2ZARC + L + generalized coth-Warburg (φ fitted, τ_d pinned)
       vs ``L0-R0-p(R1,CPE1)-p(R2,CPE2)-Wo0``
``C0`` same as C but with our φ pinned to 0.5 — the *exact* topology match for
       impedance.py's ``Wo``, which separates "different optimizer" from
       "different model". Run with τ_d free on the library side and, as ``C0p``,
       with τ_d held at our pinned value via ``constants``.

Usage:
    cd src
    python -m analysis.eis_compare_impedancepy <eis_export.parquet> [-o outdir]
"""

import argparse
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd

from analysis.eis_vs_soc import (
    DIFFUSION_TAU_BOX,
    build_eis_table,
    eis_features,
    fit_2rc_eis,
    fit_warburg_eis,
    fit_zarc_warburg_eis,
    _z_warburg,
    _z_warburg_generalized,
    _z_zarc,
)

#: Canonical comparison parameters. Anything a variant does not have is NaN.
CANON = ["R0", "L", "R1", "tau1", "alpha1", "R2", "tau2", "alpha2",
         "R_d", "tau_d", "phi_d"]


# --------------------------------------------------------------------------
# canonical model + score
# --------------------------------------------------------------------------

def z_canon(p: dict, w: np.ndarray) -> np.ndarray:
    """Impedance of a canonical parameter set at angular frequencies ``w``."""
    z = p["R0"] + 1j * w * p.get("L", 0.0)
    for k in (1, 2):
        r, tau, a = p.get(f"R{k}"), p.get(f"tau{k}"), p.get(f"alpha{k}", 1.0)
        if r is not None and np.isfinite(r) and np.isfinite(tau):
            z = z + _z_zarc(r, tau, 1.0 if not np.isfinite(a) else a, w)
    rd, td, phi = p.get("R_d"), p.get("tau_d"), p.get("phi_d")
    if rd is not None and np.isfinite(rd) and np.isfinite(td):
        if p.get("diffusion") == "tanh":
            z = z + _z_warburg(rd, td, w)
        else:
            z = z + _z_warburg_generalized(rd, td, phi if np.isfinite(phi) else 0.5, w)
    return z


def score(p: dict, w: np.ndarray, zd: np.ndarray) -> dict:
    """Unweighted and modulus-weighted complex RMSE of a canonical fit."""
    if not np.isfinite(p.get("R0", np.nan)):
        return {"rmse": np.nan, "rmse_w": np.nan}
    zf = z_canon(p, w)
    d = zf - zd
    return {
        "rmse": float(np.sqrt(np.mean(np.abs(d) ** 2))),
        "rmse_w": float(np.sqrt(np.mean(np.abs(d / np.abs(zd)) ** 2))),
    }


def _nan_canon(**kw) -> dict:
    p = {k: np.nan for k in CANON}
    p.update(kw)
    return p


# --------------------------------------------------------------------------
# our fitters -> canonical
# --------------------------------------------------------------------------

def ours_A(spec, feat):
    f = fit_2rc_eis(spec, r_ohm0=feat["R_cross"], r_tot0=feat["R_tot"])
    return _nan_canon(R0=f["R0"], L=f["L"], R1=f["R1"], tau1=f["tau1"], alpha1=1.0,
                      R2=f["R2"], tau2=f["tau2"], alpha2=1.0), f


def ours_B(spec, feat, seed):
    f = fit_warburg_eis(spec, seed=seed, r_tot0=feat["R_tot"])
    return _nan_canon(R0=f["R0_w"], L=f["L_w"], R1=f["R1_w"], tau1=f["tau1_w"], alpha1=1.0,
                      R2=f["R2_w"], tau2=f["tau2_w"], alpha2=1.0,
                      R_d=f["R_d"], tau_d=f["tau_d"], phi_d=0.5,
                      diffusion="tanh", degenerate=f["warburg_degenerate"]), f


def ours_C(spec, feat, seed, phi_fixed=False):
    """Production ZARC fit. ``phi_fixed`` pins φ=0.5 (exact ``Wo`` topology)."""
    if phi_fixed:
        import analysis.eis_vs_soc as ev
        old = ev.DIFFUSION_PHI_BOX
        # scipy needs lb < ub, so "pinned" is a hairline box around 0.5. The
        # production degeneracy check then always flags φ as edge-pinned — read
        # ``degenerate`` for this variant with that in mind.
        ev.DIFFUSION_PHI_BOX = (0.4999, 0.5001)
        try:
            f = fit_zarc_warburg_eis(spec, seed=seed, r_tot0=feat["R_tot"])
        finally:
            ev.DIFFUSION_PHI_BOX = old
    else:
        f = fit_zarc_warburg_eis(spec, seed=seed, r_tot0=feat["R_tot"])
    return _nan_canon(R0=f["R0_z"], L=f["L_z"],
                      R1=f["R1_z"], tau1=f["tau1_z"], alpha1=f["alpha1_z"],
                      R2=f["R2_z"], tau2=f["tau2_z"], alpha2=f["alpha2_z"],
                      R_d=f["R_d_z"], tau_d=f["tau_d_z"], phi_d=f["phi_d_z"],
                      degenerate=f["zarc_degenerate"]), f


# --------------------------------------------------------------------------
# impedance.py -> canonical
# --------------------------------------------------------------------------

def _lib_fit(f, zd, circuit, guess, constants=None, weight=True, bounds=None):
    """Run ``CustomCircuit.fit`` and return ``(named params, seconds)``."""
    from impedance.models.circuits import CustomCircuit

    cc = CustomCircuit(circuit=circuit, initial_guess=guess,
                       constants=constants or {})
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc.fit(np.asarray(f, float), np.asarray(zd, complex),
               weight_by_modulus=weight, bounds=bounds)
    dt = time.perf_counter() - t0
    vals = dict(zip(cc.get_param_names()[0], cc.parameters_))
    vals.update(constants or {})
    return vals, dt


def _tau_cpe(r, q, a):
    """ZARC τ from a ``p(R, CPE)`` pair: τ = (R·Q)^(1/α)."""
    if not (r > 0 and q > 0 and a > 0):
        return np.nan
    lt = np.log(r * q) / a  # log-space: α→0 otherwise overflows the power
    return float(np.exp(lt)) if abs(lt) < 700 else np.nan


def lib_A(f, zd, feat, weight=True):
    r0, rp = feat["R_cross"], max(feat["R_tot"] - feat["R_cross"], 1e-3)
    guess = [1e-6, max(r0, 1e-3), 0.4 * rp, 0.02 / (0.4 * rp), 0.6 * rp, 2.0 / (0.6 * rp)]
    v, dt = _lib_fit(f, zd, "L0-R0-p(R1,C1)-p(R2,C2)", guess, weight=weight)
    p = _nan_canon(R0=v["R0"], L=v["L0"], R1=v["R1"], tau1=v["R1"] * v["C1"], alpha1=1.0,
                   R2=v["R2"], tau2=v["R2"] * v["C2"], alpha2=1.0)
    return _order(p), dt, v


def lib_B(f, zd, feat, weight=True):
    r0, rp = feat["R_cross"], max(feat["R_tot"] - feat["R_cross"], 1e-3)
    guess = [1e-6, max(r0, 1e-3), 0.3 * rp, 0.02 / (0.3 * rp), 0.4 * rp, 2.0 / (0.4 * rp),
             0.3 * rp, 20.0]
    v, dt = _lib_fit(f, zd, "L0-R0-p(R1,C1)-p(R2,C2)-Ws0", guess, weight=weight)
    p = _nan_canon(R0=v["R0"], L=v["L0"], R1=v["R1"], tau1=v["R1"] * v["C1"], alpha1=1.0,
                   R2=v["R2"], tau2=v["R2"] * v["C2"], alpha2=1.0,
                   R_d=v["Ws0_0"], tau_d=v["Ws0_1"], phi_d=0.5, diffusion="tanh")
    return _order(p), dt, v


def lib_C(f, zd, feat, weight=True, pin_tau_d=False):
    r0, rp = feat["R_cross"], max(feat["R_tot"] - feat["R_cross"], 1e-3)
    # CPE Q from a target τ: Q = τ^α / R  (α seeded at 0.85)
    a0 = 0.85
    r1, r2, rd = 0.3 * rp, 0.5 * rp, 0.3 * rp
    guess = [1e-6, max(r0, 1e-3),
             r1, (1e-3 ** a0) / r1, a0,
             r2, (1e-2 ** a0) / r2, a0,
             rd]
    const = {}
    if pin_tau_d:
        const["Wo0_1"] = float(DIFFUSION_TAU_BOX[0])
    else:
        guess.append(30.0)
    v, dt = _lib_fit(f, zd, "L0-R0-p(R1,CPE1)-p(R2,CPE2)-Wo0", guess,
                     constants=const, weight=weight)
    p = _nan_canon(R0=v["R0"], L=v["L0"],
                   R1=v["R1"], tau1=_tau_cpe(v["R1"], v["CPE1_0"], v["CPE1_1"]),
                   alpha1=v["CPE1_1"],
                   R2=v["R2"], tau2=_tau_cpe(v["R2"], v["CPE2_0"], v["CPE2_1"]),
                   alpha2=v["CPE2_1"],
                   R_d=v["Wo0_0"], tau_d=v["Wo0_1"], phi_d=0.5)
    return _order(p), dt, v


def _order(p: dict) -> dict:
    """Order the two RC/ZARC branches by τ so parameters are comparable."""
    t1, t2 = p.get("tau1", np.nan), p.get("tau2", np.nan)
    if np.isfinite(t1) and np.isfinite(t2) and t1 > t2:
        for a, b in (("R1", "R2"), ("tau1", "tau2"), ("alpha1", "alpha2")):
            p[a], p[b] = p[b], p[a]
    return p


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def compare(df: pd.DataFrame, direction=None, step=None) -> pd.DataFrame:
    """One row per (spectrum, variant) with canonical params + scores."""
    socs = build_eis_table(df, direction=direction, step=step,
                           fit_2rc=False, fit_warburg=False, fit_zarc=False)
    soc_of = dict(zip(socs["eis_number"], socs["SOC_pct"]))

    rows = []
    for eid in socs["eis_number"]:
        spec = df[df["eis_number"] == eid].sort_values("frequency")
        f = spec["frequency"].to_numpy(float)
        zd = spec["Z_real"].to_numpy(float) + 1j * spec["Z_imag"].to_numpy(float)
        w = 2 * np.pi * f
        feat = eis_features(spec)

        def add(variant, stack, p, dt):
            rows.append({"eis_number": eid, "SOC_pct": soc_of[eid],
                         "variant": variant, "stack": stack,
                         **{k: p.get(k, np.nan) for k in CANON},
                         "degenerate": bool(p.get("degenerate", False)),
                         **score(p, w, zd), "seconds": dt})

        t0 = time.perf_counter(); pA, rawA = ours_A(spec, feat); add("A_2RC", "ours", pA, time.perf_counter() - t0)
        t0 = time.perf_counter(); pB, rawB = ours_B(spec, feat, rawA); add("B_2RC_tanhW", "ours", pB, time.perf_counter() - t0)
        t0 = time.perf_counter(); pC, _ = ours_C(spec, feat, rawB); add("C_2ZARC_genW", "ours", pC, time.perf_counter() - t0)
        t0 = time.perf_counter(); pC0, _ = ours_C(spec, feat, rawB, phi_fixed=True); add("C0_2ZARC_cothW", "ours", pC0, time.perf_counter() - t0)

        for weight, tag in ((True, ""), (False, "_unw")):
            for name, fn in (("A_2RC", lib_A), ("B_2RC_tanhW", lib_B)):
                try:
                    p, dt, _ = fn(f, zd, feat, weight=weight)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("impedance.py %s%s failed on %s: %s", name, tag, eid, exc)
                    p, dt = _nan_canon(), np.nan
                add(name, f"impedance.py{tag}", p, dt)
            for name, pin in (("C0_2ZARC_cothW", False), ("C0p_tau_d_pinned", True)):
                try:
                    p, dt, _ = lib_C(f, zd, feat, weight=weight, pin_tau_d=pin)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("impedance.py %s%s failed on %s: %s", name, tag, eid, exc)
                    p, dt = _nan_canon(), np.nan
                add(name, f"impedance.py{tag}", p, dt)

    return pd.DataFrame(rows)


def roughness(res: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter SOC-smoothness: mean |Δ| between adjacent SOC steps,
    normalised by the parameter's median — the metric that drove the τ-pinning
    and multistart choices in the production fitter."""
    out = []
    for (variant, stack), g in res.groupby(["variant", "stack"]):
        g = g.sort_values("SOC_pct")
        row = {"variant": variant, "stack": stack}
        for c in CANON:
            v = g[c].to_numpy(float)
            v = v[np.isfinite(v)]
            med = np.median(np.abs(v)) if len(v) else np.nan
            row[c] = float(np.mean(np.abs(np.diff(v))) / med) if len(v) > 1 and med else np.nan
        out.append(row)
    return pd.DataFrame(out)


#: Matched (ours, impedance.py) pairs that fit the *same* model, so their
#: parameters are directly comparable. ``C0`` differs only in τ_d (ours pinned,
#: library free); ``C0p`` pins the library too — that one is exact parity.
AGREEMENT_PAIRS = [
    ("A_2RC", "A_2RC", "2RC+L"),
    ("B_2RC_tanhW", "B_2RC_tanhW", "2RC+L+Ws"),
    ("C0_2ZARC_cothW", "C0_2ZARC_cothW", "2ZARC+L+Wo (lib τ_d free)"),
    ("C0_2ZARC_cothW", "C0p_tau_d_pinned", "2ZARC+L+Wo (τ_d pinned both)"),
]


def agreement(res: pd.DataFrame) -> pd.DataFrame:
    """Median relative parameter disagreement between the two stacks.

    ``|a - b| / max(|a|, |b|)`` per spectrum, median over the SOC sweep — 0
    means the two optimizers land on the same physical answer, ~1 means they
    describe the spectrum with entirely different element values.
    """
    out = []
    for ours_v, lib_v, label in AGREEMENT_PAIRS:
        a = res[(res["variant"] == ours_v) & (res["stack"] == "ours")].set_index("eis_number")
        b = res[(res["variant"] == lib_v) & (res["stack"] == "impedance.py")].set_index("eis_number")
        if a.empty or b.empty:
            continue
        row = {"model": label, "n": len(a.index.intersection(b.index))}
        for c in CANON:
            x, y = a[c].reindex(a.index).to_numpy(float), b[c].reindex(a.index).to_numpy(float)
            den = np.maximum(np.abs(x), np.abs(y))
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(x - y) / den
            rel = rel[np.isfinite(rel)]
            row[c] = float(np.median(rel)) if len(rel) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["variant", "stack"])
    s = g.agg(n=("rmse", "size"),
              n_fail=("rmse", lambda x: int(x.isna().sum())),
              rmse_mean=("rmse", "mean"), rmse_med=("rmse", "median"),
              rmse_max=("rmse", "max"), rmse_w_mean=("rmse_w", "mean"),
              sec_mean=("seconds", "mean"),
              n_degenerate=("degenerate", "sum")).reset_index()
    return s.sort_values(["variant", "stack"])


def plot_overlay(df, res, out_png, n_show=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [("C_2ZARC_genW", "ours", "#16a085"),
            ("C0_2ZARC_cothW", "ours", "#c0392b"),
            ("C0_2ZARC_cothW", "impedance.py", "#8e44ad")]
    ids = res.drop_duplicates("eis_number").sort_values("SOC_pct")
    pick = ids.iloc[np.linspace(0, len(ids) - 1, min(n_show, len(ids))).astype(int)]

    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r0) in zip(axes, pick.iterrows()):
        spec = df[df["eis_number"] == r0["eis_number"]].sort_values("frequency")
        w = 2 * np.pi * spec["frequency"].to_numpy(float)
        ax.plot(spec["Z_real"], -spec["Z_imag"], "o", ms=3, color="#2f6fdb", label="measured")
        for variant, stack, colr in show:
            m = res[(res["eis_number"] == r0["eis_number"]) & (res["variant"] == variant)
                    & (res["stack"] == stack)]
            if m.empty or not np.isfinite(m.iloc[0]["R0"]):
                continue
            p = m.iloc[0].to_dict()
            z = z_canon(p, w)
            ax.plot(z.real, -z.imag, "-", color=colr,
                    label=f"{stack}:{variant.split('_')[0]} ({p['rmse']:.3f})")
        ax.set_title(f"SOC {r0['SOC_pct']:.0f}%", fontsize=9)
        ax.set_xlabel("Z_real (mΩ)"); ax.set_ylabel("-Z_imag (mΩ)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[len(pick):]:
        ax.set_visible(False)
    fig.suptitle("ours vs impedance.py — Nyquist overlay", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("overlay -> %s", out_png)


#: Model-C series: production vs the two ways of forcing φ = 0.5. Holding the
#: model fixed and swapping only the optimizer (rows 2 vs 3) separates "our
#: fitter is better" from "our diffusion element is better".
#: ``(variant, stack, colour, figure legend, short in-panel tag)``
MODEL_C_SERIES = [
    ("C_2ZARC_genW", "ours", "#16a085", "ours — genW, φ fitted (production)", "ours genW"),
    ("C0_2ZARC_cothW", "ours", "#c0392b", "ours — Wo, φ = 0.5", "ours Wo"),
    ("C0_2ZARC_cothW", "impedance.py", "#8e44ad", "impedance.py — Wo, φ = 0.5", "imp.py Wo"),
]


def plot_model_c(df, res, out_png, n_show=4):
    """Focused Model-C comparison: Nyquist, residual spectrum, params vs SOC.

    Row 1 — measured vs the three fits at evenly spaced SOC.
    Row 2 — per-frequency residual |Z_fit − Z_meas| for those same SOC, which
            localises *where* the φ = 0.5 forms fail (the low-frequency tail).
    Row 3 — RMSE and the diffusion parameters over the whole SOC sweep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = res.drop_duplicates("eis_number").sort_values("SOC_pct")
    pick = ids.iloc[np.linspace(len(ids) - 1, 0, min(n_show, len(ids))).astype(int)]

    fig, axes = plt.subplots(3, n_show, figsize=(4.6 * n_show, 12.5))
    handles = []

    for col, (_, r0) in enumerate(pick.iterrows()):
        spec = df[df["eis_number"] == r0["eis_number"]].sort_values("frequency")
        fr = spec["frequency"].to_numpy(float)
        w = 2 * np.pi * fr
        zd = spec["Z_real"].to_numpy(float) + 1j * spec["Z_imag"].to_numpy(float)

        ax_n, ax_r = axes[0, col], axes[1, col]
        (ln,) = ax_n.plot(zd.real, -zd.imag, "o", ms=3.5, color="#2f6fdb",
                          label="measured", zorder=3)
        if col == 0:
            handles.append(ln)
        for variant, stack, colr, lab, tag in MODEL_C_SERIES:
            m = res[(res["eis_number"] == r0["eis_number"])
                    & (res["variant"] == variant) & (res["stack"] == stack)]
            if m.empty or not np.isfinite(m.iloc[0]["R0"]):
                continue
            p = m.iloc[0].to_dict()
            z = z_canon(p, w)
            (ln,) = ax_n.plot(z.real, -z.imag, "-", lw=1.6, color=colr,
                              label=f"{tag} ({p['rmse']:.3f})")
            if col == 0:
                handles.append(plt.Line2D([], [], color=colr, lw=1.6, label=lab))
            ax_r.plot(fr, np.abs(z - zd), "-", lw=1.4, color=colr)
        ax_n.set_title(f"SOC {r0['SOC_pct']:.0f}%", fontsize=10)
        ax_n.set_xlabel("Z_real (mΩ)"); ax_n.set_ylabel("−Z_imag (mΩ)")
        ax_n.grid(alpha=0.3); ax_n.legend(fontsize=7, loc="upper left")
        ax_r.set_xscale("log"); ax_r.set_yscale("log")
        ax_r.set_xlabel("frequency (Hz)"); ax_r.set_ylabel("|Z_fit − Z_meas| (mΩ)")
        ax_r.grid(alpha=0.3, which="both")
        ax_r.set_title(f"residual — SOC {r0['SOC_pct']:.0f}%", fontsize=9)

    sweep = [("rmse", "complex RMSE (mΩ)", True),
             ("R_d", "R_d — diffusion (mΩ)", True),
             ("phi_d", "φ_d — diffusion exponent", False),
             ("alpha1", "α1 — fast ZARC CPE exponent", False)]
    for col, (c, label, logy) in enumerate(sweep[:n_show]):
        ax = axes[2, col]
        for variant, stack, colr, _lab, _tag in MODEL_C_SERIES:
            g = res[(res["variant"] == variant) & (res["stack"] == stack)].sort_values("SOC_pct")
            if g.empty:
                continue
            ax.plot(g["SOC_pct"], g[c], "o-", ms=4, color=colr)
        ax.set_xlabel("SOC (%)"); ax.set_ylabel(label); ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        if c in ("phi_d", "alpha1"):
            ax.set_ylim(0.25, 1.05)
            ax.axhline(0.5 if c == "phi_d" else 1.0, ls=":", lw=1, color="0.5")
    for ax in axes[2, len(sweep):]:
        ax.set_visible(False)

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Model C — 2×ZARC + L + finite-length Warburg: production vs impedance.py",
        fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("model-C comparison -> %s", out_png)


def plot_params(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("C_2ZARC_genW", "ours", "#16a085", "ours 2ZARC+genW (production)"),
              ("C0_2ZARC_cothW", "ours", "#c0392b", "ours 2ZARC+cothW (φ=0.5)"),
              ("C0_2ZARC_cothW", "impedance.py", "#8e44ad", "impedance.py Wo (τ_d free)"),
              ("C0p_tau_d_pinned", "impedance.py", "#e67e22", "impedance.py Wo (τ_d pinned)")]
    cols = ["R0", "R1", "tau1", "alpha1", "R2", "tau2", "alpha2", "R_d", "tau_d", "rmse"]
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    axes = axes.ravel()
    handles = []
    for ax, c in zip(axes, cols):
        for variant, stack, colr, lab in series:
            g = res[(res["variant"] == variant) & (res["stack"] == stack)].sort_values("SOC_pct")
            if g.empty:
                continue
            (ln,) = ax.plot(g["SOC_pct"], g[c], "o-", ms=4, color=colr, label=lab)
            if ax is axes[0]:
                handles.append(ln)
        ax.set_xlabel("SOC (%)"); ax.set_ylabel(c); ax.grid(alpha=0.3)
        if c in ("tau1", "tau2", "tau_d"):
            ax.set_yscale("log")
    if handles:
        axes[0].legend(handles=handles, fontsize=7)
    fig.suptitle("Fitted parameters vs SOC — ours vs impedance.py", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("params -> %s", out_png)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="compare eis_vs_soc fits against impedance.py")
    ap.add_argument("export", help="path to a *_eis_BM*.parquet export bundle")
    ap.add_argument("-o", "--out-dir", default=".", help="output directory")
    ap.add_argument("--direction", default=None, choices=["discharge", "charge"])
    ap.add_argument("--step", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.export)
    res = compare(df, direction=args.direction, step=args.step)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "eis_impedancepy_compare")
    res.to_csv(f"{stem}_segments.csv", index=False)
    summ = summarise(res)
    summ.to_csv(f"{stem}_summary.csv", index=False)
    rough = roughness(res)
    rough.to_csv(f"{stem}_roughness.csv", index=False)
    agree = agreement(res)
    agree.to_csv(f"{stem}_agreement.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== fit quality (complex RMSE, mΩ; same scorer both stacks) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== SOC-roughness (mean |Δ| between adjacent SOC / median) ===")
    print(rough.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\n=== parameter agreement, ours vs impedance.py (median rel. diff) ===")
    print(agree.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    plot_overlay(df, res, f"{stem}_overlay.png")
    plot_model_c(df, res, f"{stem}_model_C.png")
    plot_params(res, f"{stem}_params_vs_SOC.png")


if __name__ == "__main__":
    main()
