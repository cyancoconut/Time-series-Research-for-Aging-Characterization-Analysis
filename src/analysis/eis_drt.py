"""Distribution of Relaxation Times (DRT) for the EIS export bundles.

Model-free companion to the ECM fits in :mod:`analysis.eis_vs_soc`. Where an
ECM asks "how well does *this* circuit describe the spectrum", the DRT asks
"how many distinct relaxation processes are in the spectrum at all, and how
broad is each one" — which is the discriminator we need for the diffusion
branch: a genuinely **dispersed** transport process appears as one *broad*
low-frequency peak, whereas a **missing ECM branch** appears as a separate
*discrete* peak the 2×ZARC model has no element for.

Model
-----

    Z(ω) = R_inf + jωL + 1/(jωC_blk) + ∫ γ(lnτ) / (1 + jωτ) dlnτ

The series ``C_blk`` matters: a blocking (capacitive) low-frequency tail is
**not** representable by any non-negative sum of RC elements — forced to
describe it, the DRT piles up spurious mass at the largest τ in the grid. With
``C_blk`` free, the tail is absorbed by the element that actually produces it
and γ is left describing the relaxation processes only.

Solution
--------
γ is discretised on a log-τ grid and recovered by non-negative Tikhonov
regularisation with a first-difference penalty, residuals weighted by 1/|Z|
(matching the ECM fits). ``R_inf``, ``L`` and ``1/C_blk`` are free non-negative
parameters, excluded from the penalty.

λ is the one subjective knob, so it is chosen by the **L-curve corner**
(maximum curvature of the log-residual / log-penalty trade-off) and every plot
also carries a λ-sensitivity panel: a conclusion that survives a decade of λ
either side of the corner is a conclusion about the data, not about λ.

Usage:
    cd src
    python -m analysis.eis_drt <eis_export.parquet> [-o outdir] [--lam ...]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

#: Log-τ grid: points per decade, and how far past the measured band to extend.
#: Keep the padding small. τ beyond ~1/(2π·f_min) is not constrained by the
#: sweep at all, and a generous pad lets the solver park unidentifiable mass out
#: there — which then inflates any width/span statistic measured on γ. Report
#: spans only over ``0.1 s … 1/(2π·f_min)``.
TAU_PPD = 12
TAU_PAD_DECADES = 0.5

#: λ grid scanned for the L-curve corner.
LAMBDA_GRID = np.logspace(-6, 1, 36)


def tau_grid(f: np.ndarray) -> np.ndarray:
    """Log-spaced τ grid spanning the measured band plus padding."""
    t_lo = np.log10(1.0 / (2 * np.pi * f.max())) - TAU_PAD_DECADES
    t_hi = np.log10(1.0 / (2 * np.pi * f.min())) + TAU_PAD_DECADES
    n = int(np.ceil((t_hi - t_lo) * TAU_PPD)) + 1
    return np.logspace(t_lo, t_hi, n)


def design(f: np.ndarray, tau: np.ndarray):
    """Real/imag design matrix for ``[γ_1..γ_M, R_inf, L, 1/C_blk]``."""
    w = 2 * np.pi * f
    M = len(tau)
    dln = float(np.mean(np.diff(np.log(tau))))

    wt = np.outer(w, tau)
    den = 1.0 + wt ** 2
    a_re = (1.0 / den) * dln
    a_im = (-wt / den) * dln

    n = len(f)
    A_re = np.zeros((n, M + 3))
    A_im = np.zeros((n, M + 3))
    A_re[:, :M] = a_re
    A_im[:, :M] = a_im
    A_re[:, M] = 1.0            # R_inf -> real only
    A_im[:, M + 1] = w          # L     -> +jωL
    A_im[:, M + 2] = -1.0 / w   # 1/C   -> -j/(ωC)
    return A_re, A_im, dln


def _penalty(M: int, n_extra: int = 3) -> np.ndarray:
    """First-difference operator on γ only; R_inf / L / C are unpenalised."""
    D = np.zeros((M - 1, M + n_extra))
    for i in range(M - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


def solve_drt(f, zd, tau, lam):
    """Non-negative Tikhonov solve. Returns ``(x, resid_norm, pen_norm)``."""
    from scipy.optimize import lsq_linear

    A_re, A_im, _ = design(f, tau)
    M = len(tau)
    wgt = 1.0 / np.abs(zd)
    A = np.vstack([A_re * wgt[:, None], A_im * wgt[:, None]])
    b = np.concatenate([zd.real * wgt, zd.imag * wgt])

    D = _penalty(M)
    Aa = np.vstack([A, np.sqrt(lam) * D])
    ba = np.concatenate([b, np.zeros(D.shape[0])])

    res = lsq_linear(Aa, ba, bounds=(0.0, np.inf), max_iter=500)
    x = res.x
    return x, float(np.linalg.norm(A @ x - b)), float(np.linalg.norm(D @ x))


def lcurve_lambda(f, zd, tau, lam_grid=None):
    """λ at the L-curve corner (max curvature in log-log residual/penalty)."""
    lam_grid = LAMBDA_GRID if lam_grid is None else lam_grid
    rows = []
    for lam in lam_grid:
        _, r, p = solve_drt(f, zd, tau, lam)
        rows.append((lam, r, p))
    lam_g = np.array([r[0] for r in rows])
    rr = np.log(np.array([r[1] for r in rows]) + 1e-300)
    pp = np.log(np.array([r[2] for r in rows]) + 1e-300)

    # discrete curvature of (rr, pp)
    d1r, d1p = np.gradient(rr), np.gradient(pp)
    d2r, d2p = np.gradient(d1r), np.gradient(d1p)
    curv = (d1r * d2p - d1p * d2r) / (d1r ** 2 + d1p ** 2 + 1e-300) ** 1.5
    # ignore the ends, where the gradient stencil is one-sided
    k = int(np.nanargmax(curv[2:-2])) + 2
    return float(lam_g[k]), pd.DataFrame(rows, columns=["lam", "resid", "penalty"])


def reconstruct(f, tau, x):
    A_re, A_im, _ = design(f, tau)
    return (A_re @ x) + 1j * (A_im @ x)


def drt_peaks(tau, gamma, rel_height=0.05):
    """Peaks of γ with their integrated area (= resistance) and log-width."""
    from scipy.signal import find_peaks, peak_widths

    if gamma.max() <= 0:
        return pd.DataFrame(columns=["tau_peak", "gamma_peak", "R_peak", "width_decades"])
    idx, _ = find_peaks(gamma, height=rel_height * gamma.max())
    if not len(idx):
        return pd.DataFrame(columns=["tau_peak", "gamma_peak", "R_peak", "width_decades"])
    dln = float(np.mean(np.diff(np.log(tau))))
    widths = peak_widths(gamma, idx, rel_height=0.5)[0] * dln / np.log(10)
    rows = []
    for i, wd in zip(idx, widths):
        lo = max(0, i - int(round(2 * wd * np.log(10) / dln)))
        hi = min(len(tau), i + int(round(2 * wd * np.log(10) / dln)) + 1)
        rows.append({"tau_peak": float(tau[i]), "gamma_peak": float(gamma[i]),
                     "R_peak": float(np.sum(gamma[lo:hi]) * dln),
                     "width_decades": float(wd)})
    return pd.DataFrame(rows)


#: Fraction of the in-band DRT peak area a peak must carry to count as a
#: **separate ZARC branch** in :func:`select_model_order`. The NFPP sweep
#: carries one dominant arc (R ≈ 2.3 mΩ at τ ≈ 0.01 s) plus two small ones
#: (≈ 0.3–0.4 mΩ) and one runt (≈ 0.1 mΩ at τ ≈ 0.2 s); 5 % keeps the three
#: and drops the runt. Raise it if the fit starts chasing noise.
ARC_MIN_R_FRACTION = 0.05

#: Hard cap on the number of ZARC branches the model order may request. Two
#: is what the parametrization spectra support: a third has no discrete peak
#: to attach to below 20 % SOC (α pins to 1.0 on 19/21 spectra and branches
#: 2/3 swap roles between adjacent SOC points).
MAX_ZARC_BRANCHES = 2


def select_model_order(peaks: pd.DataFrame, f_min: float, f_max: float,
                       tau_diffusion_floor: float = None,
                       min_r_fraction: float = ARC_MIN_R_FRACTION,
                       max_branches: int = MAX_ZARC_BRANCHES) -> tuple:
    """How many ZARC branches this spectrum's DRT actually supports.

    Returns ``(n_zarc, tau_seeds, diag)``. ``n_zarc`` is at least 1 — a
    spectrum with no usable peak still gets one arc, because the alternative
    is an ECM with no kinetics at all — and at most ``max_branches``.
    ``tau_seeds`` are the τ of the retained peaks, largest ``R_peak`` first, so
    the fit is seeded on the structure the DRT found rather than on a fixed
    spread.

    A DRT peak counts as an arc when all three hold:

    * **inside the resolvable band** ``1/(2π f_max) … 1/(2π f_min)`` — the τ
      grid is padded half a decade past the sweep on both sides (see
      :data:`TAU_PAD_DECADES`) and the solver parks unidentifiable mass out
      there, so a peak in the pad is a property of the grid, not the cell;
    * **faster than the diffusion floor** — anything at or beyond
      ``tau_diffusion_floor`` (the ECM's pinned τ_d) belongs to the Warburg
      branch, which is already in the model, so counting it would buy a second
      element for one process;
    * **not a runt** — ``R_peak`` at least ``min_r_fraction`` of the total
      in-band peak area.

    Peak counts are λ-dependent (the same NFPP_02 data reads differently at
    4e-3 and 1e-3), so this is a statement about (data, λ), like every other
    number the DRT produces. ``diag`` carries the retained/rejected split so
    that is auditable from ``parameters.json`` rather than only from the plot.
    """
    tau_lo = 1.0 / (2 * np.pi * float(f_max))
    tau_hi = 1.0 / (2 * np.pi * float(f_min))
    if tau_diffusion_floor is not None and np.isfinite(tau_diffusion_floor):
        tau_hi = min(tau_hi, float(tau_diffusion_floor))

    diag = {
        "tau_band_s": [float(tau_lo), float(tau_hi)],
        "min_r_fraction": float(min_r_fraction),
        "n_peaks_total": int(len(peaks)),
    }
    if peaks is None or peaks.empty:
        diag.update({"n_peaks_in_band": 0, "n_arcs": 1,
                     "reason": "no DRT peaks — defaulting to 1 arc"})
        return 1, [], diag

    in_band = peaks[(peaks["tau_peak"] >= tau_lo) & (peaks["tau_peak"] <= tau_hi)]
    diag["n_peaks_in_band"] = int(len(in_band))
    if in_band.empty:
        diag.update({"n_arcs": 1, "reason": "no DRT peak inside the resolvable band"})
        return 1, [], diag

    total = float(in_band["R_peak"].sum())
    keep = in_band[in_band["R_peak"] >= min_r_fraction * total] if total > 0 else in_band
    keep = keep.sort_values("R_peak", ascending=False)
    diag["n_peaks_significant"] = int(len(keep))
    diag["rejected_tau_s"] = [
        float(t) for t in in_band.loc[~in_band.index.isin(keep.index), "tau_peak"]
    ]

    n = int(min(max(len(keep), 1), max_branches))
    seeds = [float(t) for t in keep["tau_peak"].head(n)]
    diag.update({"n_arcs": n, "tau_seeds_s": seeds})
    return n, seeds, diag


def solve_bundle(df: pd.DataFrame, lam=None):
    """DRT solve for every spectrum in a bundle — **no SOC**.

    Split out of :func:`run_bundle` because γ(τ) does not depend on SOC (SOC
    is only a label on the result) while the two consumers need it at
    different times: :func:`select_model_order` has to see the peaks *before*
    the ECM fit runs, and the plots need the SOC that only exists *after* it.
    Solving once here and labelling afterwards keeps that from costing a second
    solve — and keeps the DRT panel and the ``eis_fits.csv`` row beside it from
    ever disagreeing about which SOC a spectrum was measured at.

    Returns ``(solved, lcurves)`` where ``solved`` is a list of per-measurement
    dicts in bundle time order, each carrying ``eis_number``, the ``tau``/
    ``gamma`` arrays, the ``peaks`` frame, the frequency band and the fit
    diagnostics. Pass it to :func:`label_bundle` to get the plot-ready frames.
    """
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()
    solved, lcurves = [], {}
    for eid in order:
        s = df[df["eis_number"] == eid].sort_values("frequency")
        f = s["frequency"].to_numpy(float)
        zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
        tau = tau_grid(f)
        M = len(tau)

        if lam is None:
            lam_i, lc = lcurve_lambda(f, zd, tau)
            lcurves[eid] = lc
        else:
            lam_i = lam
        x, _, _ = solve_drt(f, zd, tau, lam_i)
        gamma, r_inf, L, invC = x[:M], x[M], x[M + 1], x[M + 2]

        zf = reconstruct(f, tau, x)
        rmse = float(np.sqrt(np.mean(np.abs(zf - zd) ** 2)))
        solved.append({
            "eis_number": eid,
            "tau": tau,
            "gamma": gamma,
            "peaks": drt_peaks(tau, gamma),
            "f_min": float(f.min()),
            "f_max": float(f.max()),
            "lam": lam_i,
            "rmse": rmse,
            "R_inf": r_inf,
            "L": L,
            "C_blk": (1.0 / invC) if invC > 0 else np.inf,
            "R_drt_total": float(np.sum(gamma) * np.mean(np.diff(np.log(tau)))),
        })
    return solved, lcurves


def label_bundle(solved, soc_by_eid=None, soc_source="unknown", direction=None,
                 step=None):
    """Attach SOC to a :func:`solve_bundle` result → ``(curves, peaks, meta)``.

    ``soc_by_eid`` is the SOC the caller already assigned. Measurements it
    does not cover fall back to the order-based ladder ``100 - step * i``,
    which is **wrong whenever the steps moved unequal charge** — it exists only
    so a CLI run still labels its panels with something. ``meta["soc_source"]``
    records which was used; check it before quoting a SOC off a DRT plot.
    """
    from analysis.eis_vs_soc import SOC_SWEEP_DIRECTION, SOC_SWEEP_STEP_PCT

    direction = direction or SOC_SWEEP_DIRECTION
    step = SOC_SWEEP_STEP_PCT if step is None else step
    soc_by_eid = soc_by_eid or {}
    charging = str(direction).lower().startswith("cha")

    curves, peaks, meta = [], [], []
    for i, sol in enumerate(solved):
        eid = sol["eis_number"]
        ladder = (0.0 + step * i) if charging else (100.0 - step * i)
        soc = soc_by_eid.get(eid, ladder)
        tau, gamma = sol["tau"], sol["gamma"]

        for t, g in zip(tau, gamma):
            curves.append({"eis_number": eid, "SOC_pct": soc, "tau": t, "gamma": g})
        pk = sol["peaks"].copy()
        pk.insert(0, "SOC_pct", soc)
        pk.insert(0, "eis_number", eid)
        peaks.append(pk)
        meta.append({"eis_number": eid, "SOC_pct": soc, "lam": sol["lam"],
                     "rmse": sol["rmse"], "R_inf": sol["R_inf"], "L": sol["L"],
                     "C_blk": sol["C_blk"], "R_drt_total": sol["R_drt_total"],
                     "n_peaks": len(pk), "soc_source": soc_source})
        logging.info("DRT %s (SOC %3.0f%%): lam=%.3g rmse=%.4f peaks=%d",
                     eid, soc, sol["lam"], sol["rmse"], len(pk))

    return (pd.DataFrame(curves), pd.concat(peaks, ignore_index=True),
            pd.DataFrame(meta))


def run_bundle(df: pd.DataFrame, lam=None, direction=None, step=None,
               data_dir=None, ir_ohm=None, soc_by_eid=None, soc_source=None):
    """DRT for every spectrum in a bundle. Returns ``(curves, peaks, meta)``.

    ``soc_by_eid`` is the SOC the caller **already** assigned, keyed by
    ``eis_number``. Pass it whenever one exists: a DRT panel and the
    ``eis_fits.csv`` row beside it must never disagree about which SOC a
    spectrum was measured at, and re-deriving SOC here is how they come to
    disagree — ``fit_eis`` now measures it from the step charge
    (:mod:`util.soc_from_steps`) while this function only knows how to ask the
    qOCV curve.

    Failing that, and when ``data_dir`` is given, SOC is mapped onto the run's
    own qOCV curve via :func:`util.soc_from_qocv.assign_soc`.

    Failing both, SOC falls back to the order-based ladder ``100 - step * i``,
    which is **wrong whenever the steps moved unequal charge** — it is kept only
    so the CLI still labels its panels with something. ``meta["soc_source"]``
    records which of the three was used; check it before quoting a SOC off a DRT
    plot.
    """
    from analysis.eis_vs_soc import SOC_SWEEP_DIRECTION, SOC_SWEEP_STEP_PCT
    from util import soc_from_qocv

    direction = direction or SOC_SWEEP_DIRECTION
    step = SOC_SWEEP_STEP_PCT if step is None else step
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()

    if soc_by_eid:
        soc_by_eid = {k: v for k, v in soc_by_eid.items() if pd.notna(v)}
    if soc_by_eid:
        soc_source = soc_source or "supplied by caller"
    else:
        # One rest voltage per measurement -> qOCV SOC, in bundle time order.
        soc_by_eid, soc_source = {}, "ladder (100 - step*i) — NOT measured"
    if not soc_by_eid and data_dir:
        u = (df.groupby("eis_number")
               .agg(U=("U", "mean"), Time=("Time", "min"))
               .reindex(order).reset_index())
        diag = soc_from_qocv.assign_soc(
            u, "U", direction, data_dir, t_ref=u["Time"].min(),
            label="DRT bundle", ir_ohm=ir_ohm,
        )
        if "SOC_pct" in u.columns and u["SOC_pct"].notna().any():
            soc_by_eid = dict(zip(u["eis_number"], u["SOC_pct"]))
            soc_source = diag.get("soc_source", "qOCV")
        else:
            logging.warning("DRT: qOCV SOC unavailable (%s) — falling back to "
                            "the order-based ladder", diag.get("reason", "?"))

    solved, lcurves = solve_bundle(df, lam=lam)
    curves, peaks, meta = label_bundle(
        solved, soc_by_eid=soc_by_eid, soc_source=soc_source,
        direction=direction, step=step,
    )
    return curves, peaks, meta, lcurves


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_drt(curves, meta, ecm, out_png, n_show=4):
    """γ(τ) at selected SOC, with the ECM time constants marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    socs = meta.sort_values("SOC_pct", ascending=False)
    pick = socs.iloc[np.linspace(0, len(socs) - 1, min(n_show, len(socs))).astype(int)]

    fig, axes = plt.subplots(1, len(pick), figsize=(4.8 * len(pick), 4.4), squeeze=False)
    for ax, (_, m) in zip(axes[0], pick.iterrows()):
        g = curves[curves["eis_number"] == m["eis_number"]]
        ax.semilogx(g["tau"], g["gamma"], "-", lw=1.8, color="#16a085")
        ax.fill_between(g["tau"], 0, g["gamma"], color="#16a085", alpha=0.15)
        # `ecm` is an empty DataFrame when --ecm was not passed, and an empty
        # one has no columns at all — so test for the column before indexing.
        e = (ecm[ecm["eis_number"] == m["eis_number"]]
             if "eis_number" in getattr(ecm, "columns", []) else ecm)
        if not e.empty:
            e = e.iloc[0]
            for col, colr, lab in (("tau1_z", "#c0392b", "ZARC τ1"),
                                   ("tau2_z", "#8e44ad", "ZARC τ2"),
                                   ("tau_d_z", "#e67e22", "τ_d")):
                if col in e.index and np.isfinite(e[col]):
                    ax.axvline(e[col], ls="--", lw=1.2, color=colr, label=lab)
        ax.set_title(f"SOC {m['SOC_pct']:.0f}%  (λ={m['lam']:.2g}, rmse={m['rmse']:.3f})",
                     fontsize=9)
        ax.set_xlabel("τ (s)"); ax.set_ylabel("γ(ln τ)  (mΩ)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.suptitle("DRT — how many processes, and how broad", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("DRT plot -> %s", out_png)


def plot_lambda_sensitivity(df, eis_number, out_png, lams=(1e-4, 1e-3, 1e-2, 1e-1)):
    """Same spectrum at several λ — separates data features from smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = df[df["eis_number"] == eis_number].sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
    tau = tau_grid(f)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(lams):
        x, _, _ = solve_drt(f, zd, tau, lam)
        zf = reconstruct(f, tau, x)
        rmse = np.sqrt(np.mean(np.abs(zf - zd) ** 2))
        ax.semilogx(tau, x[:len(tau)], "-", lw=1.7, color=cmap(i / max(len(lams) - 1, 1)),
                    label=f"λ={lam:g}  (rmse {rmse:.3f})")
    ax.set_xlabel("τ (s)"); ax.set_ylabel("γ(ln τ)  (mΩ)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"λ sensitivity — {eis_number}", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("λ-sensitivity plot -> %s", out_png)


def plot_drt_map(curves, out_png):
    """γ(τ) over the whole SOC sweep as a heat map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    piv = curves.pivot_table(index="SOC_pct", columns="tau", values="gamma")
    z = piv.to_numpy()
    # Log colour scale: γ at SOC 0 % is ~30x the mid-sweep level, so a linear
    # scale renders every other row as black.
    pos = z[z > 0]
    norm = LogNorm(vmin=max(pos.min(), pos.max() / 1e3), vmax=pos.max()) if pos.size else None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    m = ax.pcolormesh(piv.columns.to_numpy(), piv.index.to_numpy(), z,
                      shading="auto", cmap="magma", norm=norm)
    ax.set_xscale("log")
    ax.set_xlabel("τ (s)"); ax.set_ylabel("SOC (%)")
    fig.colorbar(m, ax=ax, label="γ(ln τ)  (mΩ)")
    ax.set_title("DRT across the SOC sweep", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logging.info("DRT map -> %s", out_png)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="DRT of an EIS export bundle")
    ap.add_argument("export")
    ap.add_argument("-o", "--out-dir", default=".")
    ap.add_argument("--lam", type=float, default=None,
                    help="fixed regularisation (default: per-spectrum L-curve corner)")
    ap.add_argument("--ecm", default=None,
                    help="optional eis_params.csv to mark ECM time constants")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding the qOCV exports for the qOCV-derived "
                         "SOC (default: the export's own folder). Without a "
                         "qOCV there, SOC falls back to the order-based ladder")
    ap.add_argument("--sweep-direction", default=None,
                    help="override the sweep direction used to pick the qOCV branch")
    args = ap.parse_args()

    df = pd.read_parquet(args.export)
    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.export))
    curves, peaks, meta, _ = run_bundle(df, lam=args.lam, data_dir=data_dir,
                                        direction=args.sweep_direction)

    ecm = pd.read_csv(args.ecm) if args.ecm and os.path.exists(args.ecm) else pd.DataFrame()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "eis_drt")
    curves.to_csv(f"{stem}_curves.csv", index=False)
    peaks.to_csv(f"{stem}_peaks.csv", index=False)
    meta.to_csv(f"{stem}_meta.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== per-spectrum DRT ===")
    print(meta.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== peaks ===")
    print(peaks.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot_drt(curves, meta, ecm, f"{stem}_gamma.png")
    plot_drt_map(curves, f"{stem}_map.png")
    mid = meta.iloc[len(meta) // 2]["eis_number"]
    plot_lambda_sensitivity(df, mid, f"{stem}_lambda.png")


if __name__ == "__main__":
    main()
