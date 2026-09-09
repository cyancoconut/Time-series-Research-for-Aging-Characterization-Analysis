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


def run_bundle(df: pd.DataFrame, lam=None, direction=None, step=None,
               data_dir=None, ir_ohm=None):
    """DRT for every spectrum in a bundle. Returns ``(curves, peaks, meta)``.

    ``SOC_pct`` comes from the run's own qOCV curve via
    :func:`util.soc_from_qocv.assign_soc` when ``data_dir`` is given — the same
    single source the ECM fits use, so a DRT panel and an ``eis_fits.csv`` row
    can never disagree about which SOC a spectrum was measured at.

    Without ``data_dir`` (or with no qOCV export in it) SOC falls back to the
    order-based ladder ``100 - step * i``, which is **wrong whenever the steps
    moved unequal charge** — it is kept only so the CLI still labels its panels
    with something. ``meta["soc_source"]`` records which of the two was used;
    check it before quoting a SOC off a DRT plot.
    """
    from analysis.eis_vs_soc import SOC_SWEEP_DIRECTION, SOC_SWEEP_STEP_PCT
    from util import soc_from_qocv

    direction = direction or SOC_SWEEP_DIRECTION
    step = SOC_SWEEP_STEP_PCT if step is None else step
    order = df.groupby("eis_number")["Time"].min().sort_values().index.tolist()

    # One rest voltage per measurement -> qOCV SOC, in bundle time order.
    soc_by_eid, soc_source = {}, "ladder (100 - step*i) — NOT measured"
    if data_dir:
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

    curves, peaks, meta, lcurves = [], [], [], {}
    for i, eid in enumerate(order):
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
        soc = soc_by_eid.get(
            eid,
            (0.0 + step * i) if str(direction).lower().startswith("cha")
            else (100.0 - step * i),
        )

        for t, g in zip(tau, gamma):
            curves.append({"eis_number": eid, "SOC_pct": soc, "tau": t, "gamma": g})
        pk = drt_peaks(tau, gamma)
        pk.insert(0, "SOC_pct", soc)
        pk.insert(0, "eis_number", eid)
        peaks.append(pk)
        meta.append({"eis_number": eid, "SOC_pct": soc, "lam": lam_i, "rmse": rmse,
                     "R_inf": r_inf, "L": L,
                     "C_blk": (1.0 / invC) if invC > 0 else np.inf,
                     "R_drt_total": float(np.sum(gamma) * np.mean(np.diff(np.log(tau)))),
                     "n_peaks": len(pk), "soc_source": soc_source})
        logging.info("DRT %s (SOC %3.0f%%): lam=%.3g rmse=%.4f peaks=%d",
                     eid, soc, lam_i, rmse, len(pk))

    return (pd.DataFrame(curves), pd.concat(peaks, ignore_index=True),
            pd.DataFrame(meta), lcurves)


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


def plot_drt_overlay(curves, meta, peaks=None, out_png=None, ecm=None):
    """Every spectrum's γ(τ) on **one** axis, coloured by SOC.

    :func:`plot_drt` shows a handful of SOC side by side and
    :func:`plot_drt_map` shows all of them as a heat map; neither lets you read
    *how far* a peak walks along τ between two SOC, because in one the curves
    sit on separate axes and in the other a peak is a smear of colour. Here
    they share an axis.

    Three panels, because one is not enough:

    ``absolute``
        γ in mΩ. Honest about amplitude — which is most of the story at the
        empty end, where γ grows by more than an order of magnitude — but for
        that same reason the mid-sweep curves are pressed flat against zero.
    ``self-normalised``
        each curve divided by its own maximum, so **position** is comparable
        across the whole sweep. Read peak *movement* here and peak *size* on
        the left; a shift visible only after normalisation is still a real
        shift.
    ``peak τ vs SOC``
        the peak table plotted directly: τ on x, SOC on y, marker area ∝
        ``R_peak``. A process that walks shows as a slanted track, one that
        merely grows as a vertical one, and a branch appearing or vanishing
        mid-sweep as a track that starts or stops.

    ``ecm`` optionally marks the fitted ZARC/diffusion τ (the same frame
    :func:`plot_drt` takes) on the γ panels. Each τ is itself SOC-dependent, so
    it is drawn as the **band** it spans over the sweep with its median as a
    line, never as one vertical line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    m = meta.sort_values("SOC_pct")
    socs = pd.to_numeric(m["SOC_pct"], errors="coerce").to_numpy(float)
    finite = socs[np.isfinite(socs)]
    # No SOC at all (no qOCV, < 2 EIS blocks) -> colour by measurement order and
    # say so in the title, rather than draw a colorbar over fabricated numbers.
    have_soc = finite.size > 0
    norm = Normalize(vmin=finite.min(), vmax=finite.max()) if have_soc else None
    cmap = plt.get_cmap("viridis")

    has_peaks = peaks is not None and not peaks.empty
    ncol = 3 if has_peaks else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 4.8), squeeze=False)
    ax_abs, ax_nrm = axes[0][0], axes[0][1]

    for i, (_, row) in enumerate(m.iterrows()):
        g = curves[curves["eis_number"] == row["eis_number"]].sort_values("tau")
        if g.empty:
            continue
        tau = g["tau"].to_numpy(float)
        gam = g["gamma"].to_numpy(float)
        soc = row["SOC_pct"]
        colour = (cmap(norm(soc)) if have_soc and np.isfinite(soc)
                  else cmap(i / max(len(m) - 1, 1)))
        ax_abs.semilogx(tau, gam, "-", lw=1.3, color=colour, alpha=0.85)
        peak = gam.max()
        if peak > 0:
            ax_nrm.semilogx(tau, gam / peak, "-", lw=1.3, color=colour, alpha=0.85)

    for ax, ylab, title in (
        (ax_abs, "γ(ln τ)  (mΩ)", "absolute — amplitude"),
        (ax_nrm, "γ / max(γ)", "self-normalised — position"),
    ):
        ax.set_xlabel("τ (s)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")

    if ecm is not None and not getattr(ecm, "empty", True):
        for col, colr, lab in (("tau1_z", "#c0392b", "ZARC τ1"),
                               ("tau2_z", "#8e44ad", "ZARC τ2"),
                               ("tau_d_z", "#e67e22", "τ_d")):
            if col not in getattr(ecm, "columns", []):
                continue
            v = pd.to_numeric(ecm[col], errors="coerce").dropna()
            v = v[v > 0]
            if v.empty:
                continue
            for ax in (ax_abs, ax_nrm):
                ax.axvspan(v.min(), v.max(), color=colr, alpha=0.10)
                ax.axvline(v.median(), ls="--", lw=1.1, color=colr,
                           label=f"{lab} (median)")
        ax_abs.legend(fontsize=7)

    if has_peaks:
        ax_pk = axes[0][2]
        p = peaks.copy()
        for c in ("tau_peak", "SOC_pct", "R_peak"):
            p[c] = pd.to_numeric(p.get(c), errors="coerce")
        p = p[p["tau_peak"] > 0]
        r = p["R_peak"].to_numpy(float)
        rmax = np.nanmax(r) if r.size and np.isfinite(r).any() else 1.0
        size = 20 + 260 * np.nan_to_num(r / (rmax or 1.0), nan=0.0)
        ax_pk.scatter(p["tau_peak"], p["SOC_pct"], s=size,
                      c=(p["SOC_pct"] if have_soc else "#16a085"),
                      cmap="viridis", norm=norm, edgecolor="k",
                      linewidth=0.4, alpha=0.9)
        ax_pk.set_xscale("log")
        ax_pk.set_xlabel("τ of peak (s)")
        ax_pk.set_ylabel("SOC (%)")
        ax_pk.set_title("peak track — marker area ∝ R_peak", fontsize=10)
        ax_pk.grid(alpha=0.3, which="both")

    # The τ grid runs TAU_PAD_DECADES past the measured band at both ends, and
    # γ out there is not constrained by any measured point — the solver simply
    # parks unidentifiable mass in it. On a single-spectrum panel that is easy
    # to remember; with 21 curves overlaid the resulting edge ramp reads as a
    # peak walking off the axis, which is exactly the misreading this figure is
    # for. Grey it out on every panel instead of relying on the docstring.
    all_tau = pd.to_numeric(curves["tau"], errors="coerce").dropna()
    if not all_tau.empty:
        pad = 10 ** TAU_PAD_DECADES
        band = (all_tau.min() * pad, all_tau.max() / pad)
        for ax in axes[0]:
            ax.axvspan(all_tau.min(), band[0], color="0.5", alpha=0.13, lw=0)
            ax.axvspan(band[1], all_tau.max(), color="0.5", alpha=0.13, lw=0)
            ax.set_xlim(all_tau.min(), all_tau.max())

    if have_soc:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0].tolist(), label="SOC (%)", pad=0.02)

    src = ""
    if "soc_source" in m.columns and m["soc_source"].notna().any():
        src = f"   SOC: {m['soc_source'].dropna().iloc[0]}"
    lam = float(m["lam"].iloc[0]) if "lam" in m.columns and len(m) else float("nan")
    fig.suptitle(
        f"DRT across the SOC sweep — {len(m)} spectra, λ={lam:.3g}{src}"
        + ("" if have_soc else "   [no SOC — coloured by measurement order]"),
        fontsize=12,
    )
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logging.info("DRT overlay -> %s", out_png)


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
    plot_drt_overlay(curves, meta, peaks, f"{stem}_overlay.png", ecm=ecm)
    plot_drt_map(curves, f"{stem}_map.png")
    mid = meta.iloc[len(meta) // 2]["eis_number"]
    plot_lambda_sensitivity(df, mid, f"{stem}_lambda.png")


if __name__ == "__main__":
    main()
