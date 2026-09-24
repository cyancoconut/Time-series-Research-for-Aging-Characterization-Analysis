"""Lin-KK validity test (Schönleber 2014) and pre-fit outlier removal.

An EIS spectrum is only worth fitting an ECM to if it is **Kramers-Kronig
consistent**: causal, linear, stable, i.e. its real and imaginary parts are
transforms of one another. A point that violates that cannot be a property of
the cell — it is a disturbance during the sweep (a current spike, a contact
wobble, the cell drifting between two low-frequency points, mains pickup) —
and the ECM fit has no way to reject it: least squares pulls the arc toward a
bad point like any other measurement.

The test fits a circuit that is KK-compliant **by construction**

    Ẑ(ω) = R_ohm + jωL + Σ_{k=1..M} R_k / (1 + jω τ_k)

with the τ_k **fixed**, log-spaced across the band, so the only free
parameters enter linearly: one linear solve, no starting values, nothing to
converge. Whatever the data cannot be represented by is what is not
KK-consistent, and it shows up as a residual at the offending frequency.

**Built on impedance.py** (``impedance.validation``, a pinned dependency):
:func:`_design` uses its ``get_tc_distribution`` for the τ grid and its ``K``
element for the RC columns, and μ is its ``calc_mu``. Three things in its
``linKK`` wrapper are deliberately not used, each for a measured reason:

* **The solve.** ``fit_linKK(fit_type="complex")`` solves the normal equations
  as ``inv(AᵀA)·Aᵀb``, which squares an already badly conditioned matrix. Past
  M ≈ 40 it does not merely lose accuracy, it diverges — on a 48-point NFPP-
  shaped spectrum the residual runs 6e-3 at M=40, 0.68 at M=48 and 7e4 at
  M=60. This module solves the same design matrix with ``lstsq`` (SVD), which
  is monotone in M out to M=120. The ``"real"``/``"imag"`` fit types do use a
  pseudo-inverse but recover L from a second stage that leaves 10–26 %
  residuals on a spectrum with a diffusion tail.
* **The M loop / μ criterion.** μ counts negative-resistance mass as the
  signature of overfitting. Under the diverging solve above it was counting
  *numerical* noise, and it stopped at M=9–12 with the fit still 4–10 % off.
  With the τ range extended (below) μ collapses for a second, unrelated
  reason: elements outside the measured band are unconstrained, so their signs
  are arbitrary. M is therefore set by :func:`n_elements` from the data, and μ
  is **reported, not used** — ``kk_mu`` is in the diagnostics.
* **``eval_linKK`` / ``residuals_linKK``.** They evaluate the fitted circuit by
  ``eval()``-ing a string built with ``repr`` of the parameters, which on
  NumPy 2 renders a float as ``np.float64(0.1)`` and raises ``NameError``
  inside the eval. The circuit is evaluated in :func:`_eval_model` instead —
  three lines, and exactly the circuit above.

**Two settings, both forced by measurement, neither tuned per cell:**

*τ range extended by* :data:`TAU_EXTEND_DECADES` *decades either side.* A basis
truncated at ``1/(2πf_min)`` cannot reproduce a dispersive diffusion tail,
whose τ content continues past the last measured frequency — the model then
misfits the low-frequency *edge* of every spectrum that has one, by 20–40 %,
at any M. That is an artefact of the basis, and an outlier rule reading it
would amputate the diffusion branch the Warburg is fitted to. With one decade
of extension the fit of a clean synthetic NFPP spectrum is exact to ~1e-5.

*M from* :func:`n_elements`: :data:`ELEMENTS_PER_DECADE` per decade of measured
band, **capped at half the point count**. The cap is the load-bearing half.
With M above ~N/2 the model has enough freedom to interpolate the data, which
is precisely the freedom to absorb a bad point: on a 48-point sweep a 5 %
spike at the low-frequency end stands 17× above its neighbours' residual at
M=25 and 0.4× — invisible — at M=60. Below the cap the RC basis is smooth
across decades and cannot chase a single point.

**Outlier rule.** A point is dropped only when its normalized residual
``|ΔZ|/|Z|`` clears *both* bars:

* an absolute floor (:data:`RESID_FLOOR`, 1 %), and
* ``median + k·σ`` of this spectrum's own residuals, σ robust (1.4826·MAD,
  :data:`SIGMA_K`).

Each alone fails in a way that matters here. The floor alone condemns a
spectrum that is *uniformly* off — a cell drifting through a slow sweep raises
every low-frequency residual together, which is non-stationarity, not an
outlier, and there is nothing to be gained by deleting the evidence. The MAD
term alone condemns points on a very clean spectrum, where σ ≈ 1e-4 makes an
ordinary 0.3 % point a 20σ event. Together they mean: *far from the KK model
in absolute terms, and far from what the rest of this spectrum manages.*

Removal is iterated (refitting without the flagged points sharpens σ) and
**capped** at :data:`MAX_REMOVED_FRAC` of the sweep. When the cap binds that is
reported rather than obeyed quietly: a spectrum with 20 % KK-violating points
is not a spectrum with outliers, it is a bad measurement, and trimming it into
shape would hide exactly that.

Points are flagged, never deleted: :func:`annotate_bundle` writes ``kk_outlier``
/ ``kk_resid`` columns, the fits drop the flagged rows, and the raw-spectra and
fit-overlay plots draw them as red crosses. A point removed from a fit but
absent from the figure is indistinguishable from one that was never measured.
"""

import logging

import numpy as np
import pandas as pd

#: RC elements per decade of measured band. Five resolves a ZARC's own width
#: several times over; the value matters far less than the cap below, because
#: the fit is already at the noise floor well before it.
ELEMENTS_PER_DECADE = 5.0

#: Hard cap on M as a fraction of the number of measured points. Above ~0.5 the
#: model can interpolate the data, and an outlier stops being a residual — see
#: the module docstring for the measured collapse in spike visibility.
MAX_POINT_FRACTION = 0.5

#: Decades the τ grid extends past the measured band at each end, so the basis
#: can represent relaxations whose τ lie just outside it. Without this the
#: low-frequency edge of any spectrum with a diffusion tail misfits by tens of
#: percent whatever M is.
TAU_EXTEND_DECADES = 1.0

#: Absolute floor on ``|ΔZ|/|Z|`` for a point to be an outlier candidate at
#: all. 1 % is well above a settled dwell's noise (~0.3 %) and well below the
#: 5–20 % a real disturbance shows.
RESID_FLOOR = 0.01

#: How many robust σ above the spectrum's own median residual a point must sit.
SIGMA_K = 5.0

#: Cap on the fraction of a spectrum that may be removed. Above this the
#: spectrum is reported as KK-suspect instead of being trimmed into shape.
MAX_REMOVED_FRAC = 0.10

#: Fewer points than this and the test is skipped: M would be capped into
#: uselessness and there is nothing to be robust against.
MIN_POINTS = 12

#: Outlier passes. Each refits the KK model without what is already flagged.
MAX_ITERATIONS = 3

#: Columns :func:`annotate_bundle` adds to the bundle frame.
POINT_COLS = ("kk_resid", "kk_outlier")


def n_elements(f: np.ndarray) -> int:
    """How many RC elements to fit this spectrum with.

    :data:`ELEMENTS_PER_DECADE` per decade of measured band, capped at
    :data:`MAX_POINT_FRACTION` of the point count — the cap is what keeps the
    model from interpolating the data and absorbing the outlier it is there to
    expose. At least 3, or there is no distribution to speak of.
    """
    f = np.asarray(f, float)
    decades = np.log10(f.max() / f.min())
    by_band = ELEMENTS_PER_DECADE * max(decades, 1.0)
    by_points = MAX_POINT_FRACTION * f.size
    return int(max(3, min(by_band, by_points)))


def _tau_grid(f: np.ndarray, m: int) -> np.ndarray:
    """impedance.py's log-spaced τ grid, over the band widened by
    :data:`TAU_EXTEND_DECADES` at each end (``get_tc_distribution`` reads only
    the extremes of the frequency array it is given)."""
    from impedance.validation import get_tc_distribution

    f = np.asarray(f, float)
    ext = 10.0 ** TAU_EXTEND_DECADES
    return get_tc_distribution(np.array([f.min() / ext, f.max() * ext]), int(m))


def _design(f: np.ndarray, z: np.ndarray, taus: np.ndarray):
    """``(A, b)`` for the weighted linear KK problem.

    The columns are impedance.py's: ``1/|Z|`` for R_ohm, ``K([1, τ], f)/|Z|``
    (its own RC element) for each branch, and ``ω/|Z|`` for L, real and
    imaginary parts stacked — i.e. ``fit_linKK``'s matrix. Only the *solve*
    differs, see the module docstring.
    """
    from impedance.models.circuits.elements import K

    w = 2 * np.pi * f
    az = np.abs(z)
    m = len(taus)
    a_re = np.zeros((f.size, m + 2))
    a_im = np.zeros((f.size, m + 2))
    a_re[:, 0] = 1.0 / az
    a_im[:, -1] = w / az
    for i, tau in enumerate(taus):
        k = K([1, tau], f)
        a_re[:, i + 1] = k.real / az
        a_im[:, i + 1] = k.imag / az
    return (np.vstack([a_re, a_im]),
            np.concatenate([z.real / az, z.imag / az]))


def _fit(f: np.ndarray, z: np.ndarray, m: int = None):
    """``(elements, taus, M, mu)`` — the linear KK fit, solved by SVD.

    ``elements`` is ``[R_ohm, R_1..R_M, L]``, μ is impedance.py's ``calc_mu``
    (reported, not used to choose M — see the module docstring).
    """
    from impedance.validation import calc_mu

    m = int(m) if m else n_elements(f)
    taus = _tau_grid(f, m)
    a, b = _design(f, z, taus)
    elements = np.linalg.lstsq(a, b, rcond=None)[0]
    return elements, taus, m, float(calc_mu(elements[1:-1]))


def _eval_model(elements: np.ndarray, taus: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``R_ohm + jωL + Σ R_k/(1+jωτ_k)`` at ``f`` — the circuit
    :func:`_design` built its matrix from, evaluated here rather than through
    the library's broken ``eval_linKK`` (module docstring). Evaluating it
    ourselves also lets the model be read at frequencies it was *not* fitted
    on, so a point excluded in an earlier pass still gets a residual.
    """
    w = 2 * np.pi * np.asarray(f, float)
    z = np.full(w.shape, complex(elements[0])) + 1j * w * float(elements[-1])
    for r_k, tau_k in zip(elements[1:-1], taus):
        z = z + float(r_k) / (1.0 + 1j * w * float(tau_k))
    return z


def lin_kk(f, z, m: int = None) -> dict:
    """Lin-KK fit of one spectrum.

    Returns ``{M, mu, Z_fit, resid_real, resid_imag, resid, resid_rms}``, the
    residuals normalized by ``|Z|`` point by point, ``resid`` their magnitude.
    ``m`` overrides :func:`n_elements`; ``mu`` is reported, never a selector.
    """
    f = np.asarray(f, float)
    z = np.asarray(z, complex)
    elements, taus, m, mu = _fit(f, z, m)
    z_fit = _eval_model(elements, taus, f)
    err = (z_fit - z) / np.abs(z)
    return {
        "M": int(m), "mu": mu, "Z_fit": z_fit,
        "resid_real": err.real, "resid_imag": err.imag,
        "resid": np.abs(err),
        "resid_rms": float(np.sqrt(np.mean(np.abs(err) ** 2))),
        "elements": elements, "taus": taus,
    }


def _robust_threshold(resid: np.ndarray, sigma_k: float) -> float:
    """``median + k·σ`` with σ from the MAD (1.4826·MAD ≈ σ for a normal).

    Returns ``inf`` when σ is degenerate (every residual identical), which
    disables the relative bar rather than flagging the whole spectrum.
    """
    finite = resid[np.isfinite(resid)]
    if finite.size == 0:
        return np.inf
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        return np.inf
    return med + sigma_k * sigma


def screen_spectrum(f, z, resid_floor: float = RESID_FLOOR, sigma_k: float = SIGMA_K,
                    max_removed_frac: float = MAX_REMOVED_FRAC,
                    min_points: int = MIN_POINTS,
                    max_iter: int = MAX_ITERATIONS, label: str = "") -> dict:
    """KK-screen one spectrum; flag the points to exclude from the ECM fit.

    Returns ``{outlier, resid, M, mu, resid_max, resid_rms, n_points,
    n_outliers, capped, skipped}``. ``outlier`` is a boolean array over the
    input points, ``resid`` the normalized residual of every point against the
    **final** KK model (including the excluded ones, which is what makes the
    number on a rejected point meaningful).

    ``skipped`` is a reason string when the test could not run (too few points,
    or the fit failed) — in that case nothing is flagged. A KK test that cannot
    run is not evidence that the data is good, so the caller keeps every point
    and the reason travels with the row.
    """
    f = np.asarray(f, float)
    z = np.asarray(z, complex)
    n = f.size
    out = {
        "outlier": np.zeros(n, bool), "resid": np.full(n, np.nan),
        "M": np.nan, "mu": np.nan, "resid_max": np.nan, "resid_rms": np.nan,
        "n_points": int(n), "n_outliers": 0, "capped": False, "skipped": "",
    }
    if n < max(4, int(min_points)):
        out["skipped"] = f"only {n} points (< {min_points})"
        return out

    budget = int(np.floor(max_removed_frac * n))
    keep = np.ones(n, bool)
    for _ in range(max(1, int(max_iter))):
        try:
            # M is set from the points still in play, so a spectrum that
            # loses points does not keep a basis sized for the full sweep.
            res = lin_kk(f[keep], z[keep])
        except Exception as exc:  # noqa: BLE001 — a failed test must not stop a fit
            logging.warning("Lin-KK fit failed%s: %s",
                            f" for {label}" if label else "", exc)
            out["skipped"] = f"{type(exc).__name__}: {exc}"
            out["outlier"][:] = False
            return out
        # Residual of *every* point against this model, so a point excluded in
        # an earlier pass still reports how far out it was.
        resid = np.abs((_eval_model(res["elements"], res["taus"], f) - z) / np.abs(z))
        out.update(M=res["M"], mu=res["mu"], resid=resid,
                   resid_rms=float(np.sqrt(np.mean(resid[keep] ** 2))),
                   resid_max=float(np.nanmax(resid[keep])) if keep.any() else np.nan)

        thresh = max(float(resid_floor), _robust_threshold(resid[keep], sigma_k))
        cand = keep & (resid > thresh)
        if not cand.any():
            break
        idx = np.flatnonzero(cand)
        room = budget - int((~keep).sum())
        if room <= 0:
            out["capped"] = True
            break
        if idx.size > room:
            # Take the worst that fit in the budget and say the cap bound: a
            # spectrum this far from KK-consistent is a bad measurement, not
            # one with a few outliers, and trimming it to fit would hide that.
            idx = idx[np.argsort(resid[idx])[::-1][:room]]
            out["capped"] = True
        keep[idx] = False
        if out["capped"]:
            break

    out["outlier"] = ~keep
    out["n_outliers"] = int((~keep).sum())
    return out


def annotate_bundle(df: pd.DataFrame, resid_floor: float = RESID_FLOOR,
                    sigma_k: float = SIGMA_K,
                    max_removed_frac: float = MAX_REMOVED_FRAC,
                    min_points: int = MIN_POINTS, label: str = "") -> tuple:
    """KK-screen every measurement of a bundle.

    Returns ``(annotated_df, diag)``: the frame with ``kk_resid`` /
    ``kk_outlier`` per row, and a per-measurement diagnostic frame
    (``eis_number, kk_M, kk_mu, kk_resid_max, kk_resid_rms, kk_n_points,
    kk_n_outliers, kk_capped, kk_skipped``) for the fits table.

    Idempotent by inspection: a frame that already carries ``kk_outlier`` is
    returned untouched with an empty diagnostic frame, so the screen runs once
    per bundle however many consumers ask for it.
    """
    if "kk_outlier" in df.columns:
        return df, pd.DataFrame()
    out = df.copy()
    out["kk_resid"] = np.nan
    out["kk_outlier"] = False
    rows = []
    for eid, spec in out.groupby("eis_number", sort=False):
        s = spec.sort_values("frequency")
        res = screen_spectrum(
            s["frequency"].to_numpy(float),
            s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float),
            resid_floor=resid_floor,
            sigma_k=sigma_k, max_removed_frac=max_removed_frac,
            min_points=min_points, label=f"{label} eis {eid}".strip(),
        )
        out.loc[s.index, "kk_resid"] = res["resid"]
        out.loc[s.index, "kk_outlier"] = res["outlier"]
        rows.append({
            "eis_number": eid, "kk_M": res["M"], "kk_mu": res["mu"],
            "kk_resid_max": res["resid_max"], "kk_resid_rms": res["resid_rms"],
            "kk_n_points": res["n_points"], "kk_n_outliers": res["n_outliers"],
            "kk_capped": res["capped"], "kk_skipped": res["skipped"],
        })
        if res["n_outliers"]:
            logging.info(
                "Lin-KK %s eis %s: %d/%d point(s) excluded (M=%s, mu=%.3f, "
                "max resid %.2f%%)%s",
                label or "bundle", eid, res["n_outliers"], res["n_points"],
                res["M"], res["mu"], 100 * res["resid_max"],
                " — CAP REACHED, spectrum is KK-suspect" if res["capped"] else "",
            )
        elif res["skipped"]:
            logging.info("Lin-KK %s eis %s: skipped (%s)",
                         label or "bundle", eid, res["skipped"])
    return out, pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """The rows an annotated bundle should be fitted on (flagged ones gone)."""
    if "kk_outlier" not in df.columns:
        return df
    return df[~df["kk_outlier"].astype(bool)]
