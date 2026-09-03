"""Extract EIS impedance parameters from an export bundle and plot vs SOC.

Companion to the pulse ``plot_vs_soc`` / ``plot_warburg_vs_soc`` family, but for
the EIS spectra produced by ``output/export_eis.py``. A bundled export
(``25_export_eis/<cell>/<cell>_eis_BM<bm>_<SOH>SOH.parquet``) holds every EIS
measurement of one check-up (BM_Programm) stacked, each a settled per-frequency
spectrum tagged by ``eis_number`` / ``Time`` / ``U``. A full-SOC-sweep
check-up records one EIS per SOC plateau.

``build_eis_table`` leaves ``SOC_pct`` **NaN**: SOC is measured, not assigned
by order. The caller fills it via :mod:`util.soc_from_qocv`, which
interpolates each measurement's terminal voltage ``U`` onto the run's own qOCV
curve — see :func:`characterize.fit_characterization.fit_eis`. The old ladder
(``100 - step * i``) was removed because it assumes every step moved the same
charge, which the measured voltages contradict.

**The standalone CLI below does not do that mapping**, so its vs-SOC plots are
empty and its curves grey. Use ``characterize.fit_characterization``, which is
also the only producer of the Nyquist figure with SOC colouring.

Per measurement, a small set of scale-free, fit-free readouts is extracted from
the Nyquist curve (no ECM fit, so nothing to converge):

    R_cross Z_real at the Z_imag=0 crossing (inductive -> capacitive),
            interpolated. **Not** the ohmic resistance, despite being widely
            used as one: the crossing sits where ωL cancels the arcs' negative
            reactance, which on an inductive cell is a *finite* frequency
            (255-355 Hz on the NFPP sweep, L ~ 158 nH), and there the arcs
            still contribute real part. It therefore runs above the true
            series R by an SOC-dependent margin — 12-16 %, 0.135-0.211 mΩ,
            on that sweep. For the series resistance use the fitted ``R0_z``,
            or ``R0_hf`` from the two-stage fit. See :func:`fit_hf_r0`.
    f_cross_Hz  the frequency at that crossing — records where R_cross was
            actually measured, since "high-frequency limit" it is not.
    R_tot   Z_real at the lowest frequency (DC-ish total resistance).
    R_pol   R_tot - R_cross. Inherits R_cross's bias with the opposite sign,
            so it *understates* true polarisation by the same margin.
    arc_pk  semicircle height = max(-Z_imag) over the capacitive branch.
    f_pk    characteristic frequency at that arc peak (~ 1/2πτ).

Impedance values are in the raw unit of the export (for these cells that is
mΩ — a ~28 Ah cell has ohmic R of order 1 mΩ, which is what the data shows).

**The fitted model is N×ZARC + series-L + generalized Warburg**
(:func:`fit_nzarc_warburg_eis`). N is chosen per bundle from that bundle's DRT
by :func:`analysis.eis_drt.select_model_order` — see
:func:`characterize.fit_characterization._bundle_model_order`. The plain 2RC
and 2RC+Warburg fits that used to run ahead of it as seeds are **retired**: see
the comment in :func:`build_eis_table`. Their functions (:func:`fit_2rc_eis`,
:func:`fit_warburg_eis`) are still defined for
:mod:`analysis.eis_compare_impedancepy`, but nothing in the pipeline calls
them, and the ``R0``/``R1``/``R0_w``/``R_d`` column families they produced are
gone from the table. Use the ``*_z`` columns.

Usage:
    cd src
    python -m analysis.eis_vs_soc <eis_export.parquet> [-o out_stem]
                                  [--direction discharge|charge] [--step 5.0]
"""

import argparse
import logging
import os
import textwrap

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
    Returns ``R_cross, f_cross_Hz, R_tot, R_pol, arc_pk, f_pk`` (NaN where
    undefined). ``R_cross`` is the Z_imag=0 crossing and is **not** the ohmic
    resistance — see the module docstring; use ``R0_z``/``R0_hf`` for that.
    """
    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zr = s["Z_real"].to_numpy(float)
    zi = s["Z_imag"].to_numpy(float)

    # R_cross: Z_real where Z_imag crosses zero (the inductive branch is
    # positive, the capacitive branch negative). Take the highest-frequency
    # sign change and interpolate both Z_real and the frequency to Z_imag = 0.
    # f_cross is reported because the crossing is *not* at the top of the
    # sweep: it is wherever ωL happens to cancel the arcs' reactance.
    sign = np.sign(zi)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings):
        k = crossings[-1]
        z0, z1 = zi[k], zi[k + 1]
        if z1 == z0:
            r_cross, f_cross = zr[k], f[k]
        else:
            t = (0 - z0) / (z1 - z0)
            r_cross = zr[k] + (zr[k + 1] - zr[k]) * t
            f_cross = f[k] + (f[k + 1] - f[k]) * t
    else:
        # No sign change in the band — fall back to the point closest to the
        # real axis, and say so by reporting its actual frequency.
        j0 = int(np.argmin(np.abs(zi)))
        r_cross, f_cross = zr[j0], f[j0]

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
        "R_cross": float(r_cross),
        "f_cross_Hz": float(f_cross),
        "R_tot": float(r_tot),
        "R_pol": float(r_tot - r_cross),
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


#: Below this |Re(x)|, evaluate ``np.tanh(x)`` directly; at/above it, use the
#: ±1 asymptote instead. ``tanh``/``coth`` reach the double-precision limit
#: (``tanh(x) == ±1`` to the ULP) by |Re(x)| ≈ 20 — e.g. |tanh(20+0.3j) - 1|
#: ≈ 5e-18, far below eps ≈ 2.2e-16 — so this cuts in ~17x before that, with
#: huge margin left before the ``exp(2·Re(x))`` term inside numpy's
#: sinh/cosh-based ``tanh`` starts losing precision (~Re(x) > 350) or the
#: complex overflow/underflow warnings numpy raises past there. The branch is
#: exact, not approximate: below the threshold nothing changes; at/above it
#: the "wrong" formula would already have rounded to exactly ±1 anyway.
_TANH_SAFE_RE = 20.0


def _safe_tanh(x):
    """``np.tanh(x)`` without overflow/underflow warnings for large |Re(x)|.

    Scalar- and array-safe, preserves shape and complex dtype. For
    ``|Re(x)| >= _TANH_SAFE_RE`` the result is the exact double-precision
    limit ``tanh(x) -> sign(Re(x))`` (``+1`` as ``Re(x) -> +inf``, ``-1`` as
    ``Re(x) -> -inf``) — see :data:`_TANH_SAFE_RE` for why that's already
    what ``np.tanh`` itself would round to, just without the intermediate
    overflow.
    """
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    # The overflow guard only fires for a genuinely large *finite* Re(x); NaN
    # and +-inf (real or imaginary part) must fall through to np.tanh itself
    # so non-finite input propagates exactly like np.tanh(x) -- e.g. NaN in
    # -> NaN out, not a fabricated +-1. Do not "optimize" this finite check
    # away. np.isfinite(nan) is False, so NaN never reaches the asymptote
    # branch below.
    large = np.isfinite(x.real) & (np.abs(x.real) >= _TANH_SAFE_RE)
    safe = ~large
    if np.any(safe):
        out[safe] = np.tanh(x[safe])
    if np.any(large):
        out[large] = np.where(x.real[large] >= 0, 1.0, -1.0)
    return out[()] if out.shape == () else out


def _z_warburg(rd, td, w):
    """Finite-length (short) Warburg impedance R_d·tanh(√(jωτ_d))/√(jωτ_d)."""
    x = np.sqrt(1j * w * td)
    return rd * _safe_tanh(x) / x


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


def _z_warburg_reflective(a_w, td, w):
    """Reflective (blocking) finite-length Warburg A·coth(π√(jωτ))/(π√(jωτ)).

    Counterpart to :func:`_z_warburg` (transmissive, ``tanh``): that one
    saturates to a finite resistance plateau at low frequency, this one
    diverges as 1/(jωC) — a rising capacitive tail. Pick the reflective form
    when the measured spectrum is still climbing at the lowest frequency; a
    tanh element can only mimic that by pushing τ_d far outside the measured
    band, which is what makes the diffusion branch swap with the slow ZARC.

    Form follows ISEA/RWTH ``FittingGUI`` (``ESBe_ReflectiveWarburgTauCLim``,
    MIT); reparametrized from (τ, C_lim) to the amplitude ``A = τ·π²/C_lim``
    so it carries the same units as ``R_d`` and stays comparable across fits.
    """
    x = np.pi * np.sqrt(1j * w * td)
    return a_w / _safe_tanh(x) / x


def _z_warburg_generalized(rd, td, phi, w):
    """Generalized blocking Warburg R·coth((jωτ)^φ)/((jωτ)^φ).

    ``φ`` is the diffusion exponent — the Warburg counterpart of the ZARC's CPE
    exponent ``α``. It sets both asymptotes: the sloped branch (ωτ ≫ 1) leaves
    the real axis at 90·φ°, and the low-frequency tail (ωτ ≪ 1) at 180·φ°.

    φ = 0.5 recovers ideal Fickian diffusion — the 45° line turning into a pure
    blocking capacitor, i.e. exactly :func:`_z_warburg_reflective`. φ < 0.5 is
    anomalous / sub-diffusive transport: a distribution of particle sizes, a
    tortuous porous geometry or a rough interface, all of which disperse the
    single diffusion time constant Fick's law assumes.

    Descriptive, not mechanistic — it says the transport is dispersed, not why.
    """
    x = (1j * w * td) ** phi
    return rd / _safe_tanh(x) / x


def _z_zarc(r, tau, alpha, w):
    """ZARC (R ∥ CPE) impedance R / (1 + (jωτ)^α); α=1 is an ideal RC."""
    return r / (1 + (1j * w * tau) ** alpha)


#: CPE exponent floor. ISEA's FittingGUI boxes φ into [0.8, 1]; 0.6 is looser
#: (real NFPP arcs here sit at 0.65–0.9) but still rejects the α→0.3 fits that
#: only ever appeared on spectra the model cannot describe.
ZARC_ALPHA_MIN = 0.6

#: Diffusion τ box, seconds. Below ~1 s the branch competes with the slow ZARC;
#: the upper end is far outside any practical sweep. Only the amplitude ``A``
#: is identifiable when ωτ_d ≫ 1 over the whole band, so τ_d is a shape
#: parameter here — read ``R_d_z`` (the amplitude), not ``tau_d_z``.
#: Equal bounds **pin** τ_d rather than fitting it, which is what we want once φ
#: is free: the branch then has three parameters but the data constrains only
#: two, and a free τ_d bifurcates — some spectra land near 1 s, others run out
#: past 100 s, and ``R_d`` is not comparable between the two families (its
#: roughness went 0.12 → 0.35 across this sweep). Pinned, ``R_d_z`` is the
#: amplitude at ω = 1/τ_d and ``phi_d_z`` the slope, both identifiable.
#:
#: 5 s measured best here — it beat 1 s and 2 s on every parameter's smoothness
#: and on the SOC-100 fit, for +0.002 mΩ mean RMSE against an unpinned box.
#: Retune per dataset if the sweep's frequency range changes materially.
DIFFUSION_TAU_BOX = (5.0, 5.0)

#: Diffusion exponent box for the ``generalized`` element. 0.5 is ideal Fickian
#: diffusion; the fits here land at 0.32–0.42 (sub-diffusive), so the box is
#: wide enough to be informative without letting φ absorb arbitrary residual.
DIFFUSION_PHI_BOX = (0.2, 0.9)

#: Clearance, in decades, between the slowest a ZARC branch may be and the
#: pinned diffusion τ. The ZARC and diffusion τ boxes are supposed to be
#: **disjoint** so the two elements cannot trade roles — but capping the ZARC
#: box at ``DIFFUSION_TAU_BOX[0]`` exactly makes them *touch*, which is not the
#: same thing. A slow branch then walks to the top of its box and sits on τ_d
#: absorbing the Warburg: on an NFPP sweep that showed up as ``tau2_z`` = 5.000
#: s with ``alpha2_z`` pinned to 1.0 (an ideal RC, i.e. no arc at all) on every
#: two-branch spectrum, all flagged degenerate. 0.7 decades (factor ~5, so
#: τ_d = 5 s → a 1 s ceiling) separates them properly.
#:
ZARC_DIFFUSION_MARGIN_DECADES = 0.7


def zarc_tau_box(f_min: float, f_max: float) -> tuple:
    """``(lo, hi)`` bounds for a ZARC τ: the resolvable band, clear of τ_d.

    ``lo``/``hi`` extend half a decade past the measured band — outside it the
    data constrains nothing — and ``hi`` is additionally held
    :data:`ZARC_DIFFUSION_MARGIN_DECADES` below the pinned diffusion τ so a
    kinetic branch cannot merge with the Warburg.
    """
    lo = 1.0 / (2 * np.pi * float(f_max)) / 3.0
    hi = min(1.0 / (2 * np.pi * float(f_min)) * 3.0,
             DIFFUSION_TAU_BOX[0] / 10 ** ZARC_DIFFUSION_MARGIN_DECADES)
    return lo, hi

#: Diffusion element for the 2×ZARC fit. Read by both the fit and the overlay
#: plot so the drawn curve always matches the fitted one.
#:
#: * ``"generalized"`` — R·coth((jωτ)^φ)/((jωτ)^φ) with φ **fitted**. Only form
#:   whose low-frequency slope adapts, so the only one that describes both the
#:   steep tail at full charge and the flat ones mid-sweep.
#: * ``"coth"`` — as above with φ pinned to 0.5 (ISEA FittingGUI's element).
#: * ``"tanh"`` — transmissive finite-length Warburg; plateaus at low frequency.
ZARC_DIFFUSION_ELEMENT = "generalized"

#: High-frequency window for the two-stage R0 estimate, in Hz. Above this the
#: spectrum is described by ``R0 + jωL + (the fast arc)`` alone — the slow arc
#: and the diffusion branch are flat there and cannot trade against R0. 100 Hz
#: keeps ~1/3 of the NFPP sweep's points (6 kHz top) while staying clear of the
#: mid-frequency arc's peak (τ ≈ 6 ms → ~26 Hz).
HF_R0_MIN_FREQ_HZ = 100.0

#: Minimum points in the HF window for the stage-1 fit to be attempted. Five
#: free parameters need more than five points to be identifiable.
HF_R0_MIN_POINTS = 10


def fit_hf_r0(spec: pd.DataFrame, f_min: float = None) -> dict:
    """Stage-1 series-resistance fit on the high-frequency window only.

        Z(ω) = R0 + jωL + R1/(1 + (jω τ1)^α1)      for f >= ``f_min``

    Why a separate stage: in the full-band fit R0 and the mid-frequency ZARC
    are correlated, because a **depressed** arc (small α) has a broad
    high-frequency foot that reaches back to the real axis and can absorb part
    of the series resistance. That correlation is harmless while the arc stays
    round, but on this NFPP sweep α falls to 0.71 below 20 % SOC as the arc
    grows ~8×, and R0 then gives way — it turns over and *falls* by 0.06 mΩ
    towards the empty end while the measured spectrum says it is still rising.
    Restricted to the HF window the slow branches are flat, the trade-off is
    gone, and R0 is well posed (1σ ≈ 0.01 mΩ across this sweep, no degeneracy).

    Note this is **not** the ``R_cross`` of :func:`eis_features`. That one is the
    Z_imag = 0 crossing, which on an inductive cell sits at a *finite*
    frequency (255–355 Hz here, with L ≈ 158 nH) where the arcs still add
    0.11–0.19 mΩ of real part — so it overestimates R0 by an SOC-dependent
    bias. This fit removes ωL and the arc foot explicitly.

    Returns ``{R0_hf, R0_hf_sigma, L_hf, R1_hf, tau1_hf, alpha1_hf, hf_rmse,
    hf_n, hf_f_min}`` (NaN where the window is too thin or the fit fails).
    """
    from scipy.optimize import least_squares

    f_min = HF_R0_MIN_FREQ_HZ if f_min is None else float(f_min)
    keys = ("R0_hf", "R0_hf_sigma", "L_hf", "R1_hf", "tau1_hf", "alpha1_hf",
            "hf_rmse")
    fail = {k: np.nan for k in keys}
    fail.update({"hf_n": 0, "hf_f_min": f_min})

    s = spec[spec["frequency"] >= f_min].sort_values("frequency")
    if len(s) < HF_R0_MIN_POINTS:
        logging.warning(
            "fit_hf_r0: only %d point(s) at f >= %g Hz (need %d) — no HF R0",
            len(s), f_min, HF_R0_MIN_POINTS,
        )
        return fail

    f = s["frequency"].to_numpy(float)
    zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
    w = 2 * np.pi * f
    absz = np.abs(zd)

    # τ1 box: the fast arc must stay inside the window the HF points resolve,
    # else it flattens into a constant and merges with R0 — the very
    # degeneracy this stage exists to avoid.
    t_lo = 1.0 / (2 * np.pi * f.max()) / 10.0
    t_hi = 1.0 / (2 * np.pi * f_min) * 10.0

    # R0 seed: the real part at the top of the sweep, with its inductive part
    # removed via the highest-frequency Z_imag (L ≈ Z_imag/ω there).
    khi = int(np.argmax(w))
    l0 = zd.imag[khi] / w[khi] if zd.imag[khi] > 0 else 1e-6
    r0 = max(zd.real[khi] - 0.0, 1e-3)
    rspan = max(float(zd.real.max() - zd.real.min()), 1e-2)

    def model(x, wv):
        R0_, L_, R1_, lt1, a1 = x
        return R0_ + 1j * wv * L_ + _z_zarc(R1_, np.exp(lt1), a1, wv)

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    lb = [0.0, 0.0, 0.0, np.log(t_lo), ZARC_ALPHA_MIN]
    ub = [np.inf, np.inf, np.inf, np.log(t_hi), 1.0]
    best = None
    for t1s, a1s in ((1e-3, 0.85), (1e-4, 0.9), (5e-3, 0.75)):
        x0 = np.clip([r0, max(l0, 1e-9), 0.5 * rspan, np.log(t1s), a1s], lb, ub)
        try:
            res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
        except Exception as exc:  # noqa: BLE001
            logging.warning("fit_hf_r0 start failed: %s", exc)
            continue
        err = float(np.sqrt(np.mean(np.abs(model(res.x, w) - zd) ** 2)))
        if best is None or err < best[0]:
            best = (err, res)
    if best is None:
        return fail

    rmse, res = best
    R0_, L_, R1_, lt1, a1 = res.x
    # 1σ on R0 from the Gauss-Newton covariance, scaled by the residual
    # variance. Reported so a caller can see when the HF window is too thin
    # to pin R0 any better than the full-band fit does.
    sigma = np.nan
    try:
        dof = max(len(res.fun) - len(res.x), 1)
        cov = np.linalg.inv(res.jac.T @ res.jac) * (2 * res.cost / dof)
        sigma = float(np.sqrt(np.diag(cov))[0])
    except np.linalg.LinAlgError:
        pass

    return {
        "R0_hf": float(R0_), "R0_hf_sigma": sigma, "L_hf": float(L_),
        "R1_hf": float(R1_), "tau1_hf": float(np.exp(lt1)),
        "alpha1_hf": float(a1), "hf_rmse": rmse,
        "hf_n": int(len(s)), "hf_f_min": f_min,
    }


#: Branch slots the ``*_z`` column set always carries, whatever ``n_zarc`` the
#: DRT asked for. A 1-ZARC fit leaves slot 2 NaN rather than dropping the
#: columns, so one cell fitted with one arc and another fitted with two still
#: concatenate into one table and one CSV schema.
ZARC_COLUMN_SLOTS = 2


def _tau_starts(n: int, tau_seeds, lo: float, hi: float) -> list:
    """Multistart τ vectors for ``n`` ZARC branches, clipped into ``[lo, hi]``.

    The first start is the DRT's own peak τ when it supplied any — the whole
    point of asking the DRT for the model order is that it also knows *where*
    the processes are, so the fit should not then start from a generic spread.
    The remaining starts are fixed log-spaced spreads across the resolvable
    band: a single start lands in a different local minimum from spectrum to
    spectrum, which is half of why the vs-SOC curves used to jump.
    """
    starts = []
    if tau_seeds:
        s = sorted(float(t) for t in tau_seeds if np.isfinite(t) and t > 0)
        if s:
            while len(s) < n:      # fewer peaks than branches: spread off the slowest
                s.append(s[-1] * 3.0)
            starts.append(s[:n])
    span = np.log10(hi / lo)
    for f_lo, f_hi in ((0.05, 0.45), (0.02, 0.70), (0.15, 0.30)):
        starts.append(list(np.logspace(np.log10(lo) + f_lo * span,
                                       np.log10(lo) + f_hi * span, n)))
    return [list(np.clip(s, lo, hi)) for s in starts]


def fit_nzarc_warburg_eis(spec: pd.DataFrame, n_zarc: int = 2,
                          tau_seeds=None, seed: dict = None, r_tot0=None,
                          element: str = None, pin_r0=None) -> dict:
    """``n_zarc`` × ZARC + series-L + finite-length Warburg fit.

        Z(ω) = R0 + jωL + Σᵢ Rᵢ/(1+(jω τᵢ)^αᵢ) + Z_diff(ω)

    Each branch is a **ZARC** (depressed arc): a CPE exponent ``α ∈ (0,1]`` lets
    a real, flattened Nyquist semicircle be captured without inflating R/τ.

    ``n_zarc`` is not fixed by the caller in normal use: :func:`build_eis_table`
    starts every spectrum at :data:`ZARC_COLUMN_SLOTS` and
    :func:`fit_zarc_with_parsimony` drops a branch that comes back degenerate,
    so a cell whose spectrum is a single merged arc (LFP) lands on one branch
    and one with two resolved processes (NFPP) keeps two.

    ``tau_seeds`` optionally supplies the first multistart's τ. Measured to make
    no difference on either chemistry — the fixed spreads find the same minima —
    so nothing in the pipeline passes it; it is kept for one-off experiments.

    ``element`` picks the diffusion branch — see :data:`ZARC_DIFFUSION_ELEMENT`
    for the three forms; ``None`` takes that module default. Only
    ``"generalized"`` fits the diffusion exponent φ; the fixed-exponent forms
    can each reproduce one low-frequency slope and misfit every other one.

    **Identifiability.** The ZARC τ are boxed into the measured band and the
    diffusion τ into :data:`DIFFUSION_TAU_BOX`, disjoint from it, so the slowest
    ZARC and the diffusion branch cannot trade roles — without this the two
    swap between spectra and the vs-SOC curves jump decades at the swap points.
    Branches are returned ordered ``τ1 < τ2 < …`` and the fit is multistarted.
    τ_d normally lands on a box edge and carries no information; use ``R_d_z``.

    ``pin_r0`` fixes R0 at a value measured beforehand (see :func:`fit_hf_r0`)
    instead of fitting it — the second stage of the two-stage R0. R0 leaves the
    free-parameter vector entirely rather than being boxed to zero width, so
    the remaining branches are fitted against a *known* series term and cannot
    borrow from it. ``None`` (default) fits R0 as before.

    Returns ``{R0_z, L_z, R1_z, tau1_z, alpha1_z, R2_z, tau2_z, alpha2_z, R_d_z,
    tau_d_z, phi_d_z, n_zarc, zarc_rmse, zarc_degenerate, r0_pinned}`` (NaN/True
    on failure). Slots past ``n_zarc`` are NaN — see :data:`ZARC_COLUMN_SLOTS`.
    ``phi_d_z`` is the fitted φ, or the fixed exponent of the chosen element.
    """
    from scipy.optimize import least_squares

    n_zarc = max(1, int(n_zarc))
    n_slots = max(n_zarc, ZARC_COLUMN_SLOTS)
    keys = ["R0_z", "L_z", "R_d_z", "tau_d_z", "phi_d_z", "zarc_rmse"]
    for i in range(1, n_slots + 1):
        keys += [f"R{i}_z", f"tau{i}_z", f"alpha{i}_z"]
    fail = {k: np.nan for k in keys}
    fail["zarc_degenerate"] = True
    fail["n_zarc"] = n_zarc
    fail["r0_pinned"] = pin_r0 is not None and np.isfinite(pin_r0)

    # A non-finite pin (the HF stage failed on this spectrum) falls back to
    # fitting R0 — better a correlated R0 than none at all.
    pin_r0 = float(pin_r0) if pin_r0 is not None and np.isfinite(pin_r0) else None

    element = element or ZARC_DIFFUSION_ELEMENT
    if element not in ("tanh", "coth", "generalized"):
        raise ValueError(f"unknown diffusion element {element!r}")
    free_phi = element == "generalized"

    s = spec.sort_values("frequency")
    f = s["frequency"].to_numpy(float)
    zd = s["Z_real"].to_numpy(float) + 1j * s["Z_imag"].to_numpy(float)
    w = 2 * np.pi * f
    absz = np.abs(zd)
    seed = seed or {}
    # All three forms share the (R, τ, φ, ω) signature; the fixed-exponent ones
    # ignore φ, which is then reported as their implied 0.5.
    z_diff = {
        "tanh": lambda rd, td, phi, wv: _z_warburg(rd, td, wv),
        "coth": lambda rd, td, phi, wv: _z_warburg_reflective(rd, td, wv),
        "generalized": _z_warburg_generalized,
    }[element]

    # ZARC τ box = the band the sweep can actually resolve, half a decade of
    # slack either side. Anything outside is unconstrained by the data.
    # Capped at the diffusion box floor so the two boxes stay **disjoint** —
    # overlap is what lets the slow ZARC and the diffusion branch trade roles.
    td_lo, td_hi = DIFFUSION_TAU_BOX
    tz_lo, tz_hi = zarc_tau_box(f.min(), f.max())

    # Seeds. ``seed`` is only populated when a caller still runs the old 2RC /
    # 2RC+W ladder (the impedance.py comparison script); the pipeline no longer
    # does, so both fall back to reading the spectrum itself rather than to a
    # constant. L especially: a fixed 1e-6 mΩ·s is ~100x below the ~1.6e-4
    # mΩ·s (158 nH) these fixtures actually show, and the inductive tail is
    # steep enough that starting there wastes the first decade of the solve.
    khi = int(np.argmax(w))
    r0 = seed.get("R0_w", seed.get("R0", zd.real[khi]))
    if pin_r0 is not None:  # keep rspan consistent with the series term in use
        r0 = pin_r0
    l0 = seed.get("L_w", seed.get("L"))
    if l0 is None or not np.isfinite(l0) or l0 <= 0:
        l0 = zd.imag[khi] / w[khi] if zd.imag[khi] > 0 else 1e-6
    rtot = r_tot0 if r_tot0 is not None else s["Z_real"].to_numpy(float)[int(np.argmin(w))]
    rspan = max(rtot - r0, 0.5)

    # Free-parameter layout, in order:
    #   [R0 unless pinned], L, R_1..R_n, lt_1..lt_n, a_1..a_n, Rd,
    #   [ltd only when DIFFUSION_TAU_BOX has width — equal bounds pin it],
    #   [phi only when it is fitted].
    # A pinned R0 is dropped from the vector rather than boxed to zero width —
    # least_squares needs lb < ub.
    n = n_zarc
    ph_lo, ph_hi = DIFFUSION_PHI_BOX
    fit_tau_d = td_hi > td_lo
    fit_r0 = pin_r0 is None
    lb = [0.0] + [0.0] * n + [np.log(tz_lo)] * n + [ZARC_ALPHA_MIN] * n + [0.0]
    ub = [np.inf] + [np.inf] * n + [np.log(tz_hi)] * n + [1.0] * n + [np.inf]
    if fit_r0:
        lb, ub = [0.0] + lb, [np.inf] + ub
    if fit_tau_d:
        lb, ub = lb + [np.log(td_lo)], ub + [np.log(td_hi)]
    if free_phi:
        lb, ub = lb + [ph_lo], ub + [ph_hi]
    # core = [R0?] L, R*n, lt*n, a*n, Rd
    n_core = (1 if fit_r0 else 0) + 2 + 3 * n
    i_ltd = n_core if fit_tau_d else None
    i_phi = (n_core + 1 if fit_tau_d else n_core) if free_phi else None
    # Offsets into the core block, after R0 has been put back for a pinned fit.
    o_r, o_lt, o_a, o_rd = 2, 2 + n, 2 + 2 * n, 2 + 3 * n

    def _unpack(x):
        """``(R0, L, [R], [lnτ], [α], Rd, lnτ_d, φ)`` from the free vector."""
        ltd = x[i_ltd] if fit_tau_d else np.log(td_lo)
        phi = x[i_phi] if free_phi else 0.5
        core = list(x[:n_core]) if fit_r0 else [pin_r0] + list(x[:n_core])
        return (core[0], core[1], core[o_r:o_lt], core[o_lt:o_a],
                core[o_a:o_rd], core[o_rd], ltd, phi)

    def model(x, wv):
        R0_, L_, rs, lts, als, Rd_, ltd, phi = _unpack(x)
        z = R0_ + 1j * wv * L_ + z_diff(Rd_, np.exp(ltd), phi, wv)
        for r_i, lt_i, a_i in zip(rs, lts, als):
            z = z + _z_zarc(r_i, np.exp(lt_i), a_i, wv)
        return z

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    # Multistart: the DRT's own peak τ first (when it supplied them), then
    # fixed spreads across the band. τ_d and φ are spread alongside.
    tau_starts = _tau_starts(n, tau_seeds, tz_lo, tz_hi)
    td_starts = [seed.get("tau_d", 30.0), 3.0, 100.0, 30.0]
    phi_starts = [0.5, 0.4, 0.3, 0.45]
    best = None
    for k, taus in enumerate(tau_starts):
        x0 = [max(l0, 1e-9)]
        x0 += [rspan / max(n, 1)] * n                 # R_i
        x0 += [np.log(t) for t in taus]               # lnτ_i
        x0 += [0.85] * n                              # α_i
        x0 += [0.3 * rspan]                           # R_d
        if fit_r0:
            x0 = [max(r0, 1e-3)] + x0
        if fit_tau_d:
            x0.append(np.log(td_starts[k % len(td_starts)]))
        if free_phi:
            x0.append(phi_starts[k % len(phi_starts)])
        x0 = np.clip(x0, lb, ub)
        try:
            res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
        except Exception as exc:  # noqa: BLE001
            logging.warning("fit_nzarc_warburg_eis start failed: %s", exc)
            continue
        err = float(np.sqrt(np.mean(np.abs(model(res.x, w) - zd) ** 2)))
        if best is None or err < best[0]:
            best = (err, res.x)
    if best is None:
        return fail

    rmse, x = best
    R0_, L_, rs, lts, als, Rd_, ltd, phi = _unpack(x)
    phi = float(phi)
    td = float(np.exp(ltd))
    # Branches ordered fast -> slow, so R1_z/tau1_z means the same thing in
    # every row of the table regardless of which start won.
    branches = sorted(
        ((float(np.exp(lt)), float(r), float(a)) for r, lt, a in zip(rs, lts, als)),
        key=lambda b: b[0],
    )

    # Degenerate when the diffusion branch collapses, a ZARC τ sits on the edge
    # of the resolvable band, or a CPE / diffusion exponent pins to a bound —
    # each means the reported parameters are not constrained by the data.
    def _on_edge(v, lo, hi):
        return bool(v <= lo * 1.05 or v >= hi * 0.95)

    degenerate = bool(
        Rd_ < 1e-2
        or any(_on_edge(t, tz_lo, tz_hi) for t, _, _ in branches)
        or any(a <= ZARC_ALPHA_MIN * 1.02 or a >= 0.999 for _, _, a in branches)
        or (free_phi and _on_edge(phi, ph_lo, ph_hi))
    )
    out = {
        "R0_z": float(R0_), "L_z": float(L_),
        "R_d_z": float(Rd_), "tau_d_z": td, "phi_d_z": phi,
        "n_zarc": n, "zarc_rmse": rmse, "zarc_degenerate": degenerate,
        "r0_pinned": pin_r0 is not None,
    }
    for i in range(1, n_slots + 1):
        t_i, r_i, a_i = branches[i - 1] if i <= len(branches) else (np.nan,) * 3
        out[f"R{i}_z"], out[f"tau{i}_z"], out[f"alpha{i}_z"] = r_i, t_i, a_i
    return out


def fit_zarc_warburg_eis(spec: pd.DataFrame, seed: dict = None, r_tot0=None,
                         element: str = None, pin_r0=None) -> dict:
    """Backwards-compatible 2×ZARC alias of :func:`fit_nzarc_warburg_eis`.

    Kept because :mod:`analysis.eis_compare_impedancepy` exists specifically to
    benchmark the fixed 2RC → 2RC+W → 2×ZARC ladder against ``impedance.py``,
    so it must keep getting a hard-coded two-branch fit. The pipeline calls
    ``fit_nzarc_warburg_eis`` with the DRT's model order instead.
    """
    return fit_nzarc_warburg_eis(spec, n_zarc=2, seed=seed, r_tot0=r_tot0,
                                 element=element, pin_r0=pin_r0)


#: How much worse (fractionally) a one-branch-simpler fit's RMSE may be and
#: still be preferred, when the richer fit came back **degenerate**. A
#: degenerate branch is by definition one the data does not constrain — it has
#: pinned α to a bound, run its τ to the edge of the box, or split one arc into
#: two near-identical ones — so a small RMSE gain bought that way is fitting
#: noise, not structure. 25 % is loose enough that a branch which genuinely
#: helps still survives.
ZARC_PARSIMONY_RMSE_TOLERANCE = 0.25


def fit_zarc_with_parsimony(spec: pd.DataFrame, n_zarc: int = 2,
                            tau_seeds=None, **kw) -> dict:
    """:func:`fit_nzarc_warburg_eis`, dropping a branch that earns nothing.

    **This is the model-order selector.** Fit at ``n_zarc``; if that comes back
    degenerate and ``n_zarc > 1``, fit again one branch simpler and keep the
    simpler result unless the richer one is materially better on RMSE
    (:data:`ZARC_PARSIMONY_RMSE_TOLERANCE`). Adds one solve only on the spectra
    that actually hit the degenerate path.

    Counting the DRT's peaks was tried first and dropped. It cannot do the job:
    a dispersive Warburg (φ < 0.5) deposits γ mass *inside* the ZARC τ box that
    is both large (41 % of the in-band peak area on a synthetic single-arc
    spectrum, 13–36 % on the LFP sweep) and **narrower** than the real arc, so
    neither an area threshold nor a width test separates it from a genuine
    second process. Measured against this guard on real data it added nothing
    on the LFP cell (identical fit) and made the NFPP cell *worse*, wrongly
    dropping one spectrum to a single branch (mean rmse 0.0233 → 0.0320).

    What does work is asking the ECM itself: given a branch it does not need,
    the fit comes back **degenerate** — α pinned to a bound, τ on the edge of
    its box, or one arc split into two near-identical ones. That test is a
    property of the fit rather than of a regularisation parameter, so unlike a
    DRT peak count it carries no dependence on λ.

    ``n_zarc_requested`` records what was asked for, ``n_zarc`` what was kept.
    """
    out = fit_nzarc_warburg_eis(spec, n_zarc=n_zarc, tau_seeds=tau_seeds, **kw)
    out["n_zarc_requested"] = int(n_zarc)
    if n_zarc <= 1 or not out.get("zarc_degenerate"):
        return out

    simpler = fit_nzarc_warburg_eis(spec, n_zarc=n_zarc - 1,
                                    tau_seeds=(tau_seeds or [])[: n_zarc - 1],
                                    **kw)
    simpler["n_zarc_requested"] = int(n_zarc)
    r_full, r_simple = out.get("zarc_rmse"), simpler.get("zarc_rmse")
    if not np.isfinite(r_simple):
        return out
    # Keep the richer fit only if the extra branch bought a real improvement;
    # it is degenerate either way, so it has to earn its place on residual.
    if np.isfinite(r_full) and r_full < r_simple / (1 + ZARC_PARSIMONY_RMSE_TOLERANCE):
        return out
    logging.info(
        "ZARC fit: %d branches came back degenerate (rmse %.4g) — keeping %d "
        "(rmse %.4g, degenerate=%s)",
        n_zarc, r_full if np.isfinite(r_full) else float("nan"),
        n_zarc - 1, r_simple, simpler.get("zarc_degenerate"),
    )
    return simpler


def build_eis_table(df: pd.DataFrame, direction=None, step=None,
                    fit_zarc=True, two_stage_r0=True, hf_f_min=None,
                    n_zarc: int = ZARC_COLUMN_SLOTS) -> pd.DataFrame:
    """Per-measurement feature table with a time-ordered sweep SOC.

    ``two_stage_r0`` measures R0 on the high-frequency window first
    (:func:`fit_hf_r0`) and pins it in the ZARC fit, instead of letting the
    full-band fit trade R0 against the mid-frequency arc. **On by default** —
    it is the standard path; pass ``False`` to reproduce a pre-#70 fit.
    ``hf_f_min`` overrides :data:`HF_R0_MIN_FREQ_HZ` for that first stage.

    ``n_zarc`` is the number of ZARC branches to *start* from; each spectrum
    then keeps that many or fewer, because :func:`fit_zarc_with_parsimony`
    drops a branch that comes back degenerate. The default of two is the right
    starting point for both chemistries measured so far — a cell whose spectrum
    is a single merged arc (LFP) lands on one branch on its own.

    One row per ``eis_number`` (measurement), ordered by ``Time``.
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
        pin = None
        if two_stage_r0:
            hf = fit_hf_r0(spec, f_min=hf_f_min)
            feat.update(hf)
            pin = hf["R0_hf"]
        # The plain 2RC and the 2RC+Warburg stages are **out of the chain**.
        # They existed to seed the ZARC fit, which now seeds itself from the
        # DRT peaks (a better guess than an ideal-RC fit of a depressed arc)
        # and from the spectrum's own high-frequency point. Their outputs were
        # never a result — the ZARC fit superseded both — so running them cost
        # two extra least-squares solves per spectrum to produce columns
        # nothing downstream read. The functions themselves are kept for
        # `analysis.eis_compare_impedancepy`.
        #
        #     fit = fit_2rc_eis(spec, r_ohm0=feat["R_cross"], r_tot0=feat["R_tot"])
        #     feat.update(fit)
        #     wfit = fit_warburg_eis(spec, seed=fit, r_tot0=feat["R_tot"])
        #     feat.update(wfit)
        if fit_zarc:
            feat.update(fit_zarc_with_parsimony(
                spec, n_zarc=n_zarc, r_tot0=feat["R_tot"], pin_r0=pin))
        # No order-based SOC ladder: `100 - step * i` assumes every step moved
        # the same charge, which the measured voltages contradict (the first
        # NFPP step drops 155 mV, the next ones ~15 mV, all labelled "5 %").
        # SOC is measured downstream — by counting the charge of the SOC-adjust
        # step in front of each measurement (`util/soc_from_steps.py`), falling
        # back to the run's own qOCV curve (`util/soc_from_qocv.py`) and this
        # measurement's own U.
        feat.update(
            {
                "eis_number": eid,
                "SOC_pct": np.nan,
                "U": float(spec["U"].mean()),
                "Time": spec["Time"].min(),
            }
        )
        if "segment_ID" in spec.columns:
            # The GOLD segment this spectrum was measured in, written by
            # `output.export_eis`. `util.soc_from_steps` reads the segment one
            # procedure number earlier to get the step that set this SOC.
            feat["segment_ID"] = str(spec["segment_ID"].iloc[0])
        rows.append(feat)
    out = pd.DataFrame(rows)
    logging.info(
        "eis_vs_soc: %d measurements (%s sweep) — SOC_pct left NaN, filled "
        "from the step charge (or the qOCV curve)",
        len(out), direction,
    )
    return out


#: Default padding around a plotted data range, as a fraction of its span.
#: Enough that markers on the extremes are not clipped by the frame.
AXIS_PAD_FRACTION = 0.05


def _padded_limits(*values, frac: float = AXIS_PAD_FRACTION, floor: float = None):
    """``(lo, hi)`` spanning every finite value in ``values``, padded by ``frac``.

    Returns ``None`` when there is nothing finite to frame, so the caller can
    leave the axis on autoscale rather than setting a degenerate limit. A
    constant series is given a symmetric window around its value instead of a
    zero-width one. ``floor`` clamps the lower bound (e.g. an α axis must not
    run below :data:`ZARC_ALPHA_MIN`, the bound α was fitted against).
    """
    vals = []
    for v in values:
        arr = pd.to_numeric(pd.Series(np.asarray(v).ravel()), errors="coerce")
        arr = arr[np.isfinite(arr)]
        if len(arr):
            vals.append(arr)
    if not vals:
        return None
    allv = pd.concat(vals)
    lo, hi = float(allv.min()), float(allv.max())
    span = hi - lo
    if span <= 0:
        span = abs(lo) if lo else 1.0
        return lo - 0.5 * span, hi + 0.5 * span
    lo, hi = lo - frac * span, hi + frac * span
    if floor is not None:
        lo = max(lo, floor)
    return lo, hi


def _autoscale(ax, x=None, y=None, **kw):
    """Set padded x/y limits on ``ax`` from the given data, skipping empties."""
    if x is not None:
        lim = _padded_limits(x, **kw)
        if lim:
            ax.set_xlim(*lim)
    if y is not None:
        lim = _padded_limits(y, **kw)
        if lim:
            ax.set_ylim(*lim)


def _frame_nyquist(ax, *frames):
    """Frame a Nyquist axis on its **full** measured extent, padded.

    Two things are going on. The limits are set explicitly rather than left to
    autoscale so the frame comes from the data rather than from matplotlib's
    tick rounding — which on these spectra padded the diffusion tail out by
    most of a decade. And the equal aspect is made ``adjustable="box"``: with
    the default ``"datalim"`` matplotlib holds the axes box fixed and *expands
    the data limits* to make the scales equal, which silently undoes any limit
    set here and stretches the plotted range. Adjusting the box instead honours
    the limits and reshapes the frame around them.

    ``frames`` are ``(Z_real, -Z_imag)`` pairs — pass the fitted curve as well
    as the measured points so an overlay that overshoots stays in view.
    """
    ax.set_aspect("equal", adjustable="box")
    xlim = _padded_limits(*[f[0] for f in frames])
    ylim = _padded_limits(*[f[1] for f in frames])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def plot_eis_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of EIS readouts vs SOC — the EIS analogue of ``plot_vs_soc``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("R_cross", "R_cross — Z_imag=0 crossing (mΩ)"),
        ("R_pol", "R_pol — R_tot − R_cross (mΩ)"),
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
        _autoscale(ax, x=t["SOC_pct"])
        if col == "f_pk":
            ax.set_yscale("log")
        else:
            # A log axis padded in linear space would clip the decades; leave
            # those on autoscale, which is already sensible in log.
            _autoscale(ax, y=t[col])
    fig.suptitle(f"EIS parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS vs-SOC plot -> %s", out_png)


# ---------------------------------------------------------------------------
# Retired plotters: the 2RC and the 2RC+Warburg fits are no longer in
# `build_eis_table`'s chain (see the comment there), so the tables these two
# drew from carry none of their columns and both would no-op on every call.
# Commented out rather than deleted: they are the reference for what the
# retired fits reported, and `analysis.eis_compare_impedancepy` still fits
# those models. The ZARC grid below covers the live model.
# ---------------------------------------------------------------------------
# def plot_2rc_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
#     """Grid of the fitted 2RC parameters vs SOC — EIS analogue of the pulse 2RC."""
#     import matplotlib
#
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#
#     if "R0" not in table.columns:
#         logging.info("2RC vs-SOC plot: no fit columns, skipping")
#         return
#     metrics = [
#         ("R0", "R0 — series (mΩ)"),
#         ("L", "L — series inductance (mΩ·s)"),
#         ("R1", "R1 — fast branch (mΩ)"),
#         ("tau1", "τ1 (s)"),
#         ("R2", "R2 — slow branch (mΩ)"),
#         ("tau2", "τ2 (s)"),
#         ("rmse", "fit rmse (mΩ)"),
#     ]
#     metrics = [m for m in metrics if m[0] in table.columns]
#     t = table.sort_values("SOC_pct")
#     fig, axes = plt.subplots(2, 4, figsize=(19, 8))
#     for ax, (col, label) in zip(axes.ravel(), metrics):
#         ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#c0392b")
#         ax.set_xlabel("SOC (%)")
#         ax.set_ylabel(label)
#         ax.grid(alpha=0.3)
#         if col in ("tau1", "tau2"):
#             ax.set_yscale("log")
#     for ax in axes.ravel()[len(metrics):]:
#         ax.set_visible(False)
#     fig.suptitle(f"EIS 2RC+L fit parameters vs SOC — {title}", fontsize=11)
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=120)
#     plt.close(fig)
#     logging.info("EIS 2RC vs-SOC plot -> %s", out_png)
#
#
# def plot_warburg_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
#     """Grid of the 2RC+L+Warburg fit parameters vs SOC (hides degenerate fits)."""
#     import matplotlib
#
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#
#     if "R_d" not in table.columns:
#         logging.info("Warburg vs-SOC plot: no Warburg columns, skipping")
#         return
#     t = table.sort_values("SOC_pct").copy()
#     if "warburg_degenerate" in t.columns:
#         n = int((t["warburg_degenerate"] == True).sum())  # noqa: E712
#         if n:
#             logging.info("Warburg vs-SOC plot: hiding %d degenerate fit(s)", n)
#         t = t[t["warburg_degenerate"] != True]  # noqa: E712
#     if t.empty:
#         logging.info("Warburg vs-SOC plot: nothing to plot")
#         return
#     metrics = [
#         ("R0_w", "R0 — series (mΩ)"),
#         ("R1_w", "R1 (mΩ)"), ("tau1_w", "τ1 (s)"),
#         ("R2_w", "R2 (mΩ)"), ("tau2_w", "τ2 (s)"),
#         ("R_d", "R_d — diffusion (mΩ)"), ("tau_d", "τ_d — diffusion (s)"),
#         ("warburg_rmse", "2RC+W rmse (mΩ)"),
#     ]
#     fig, axes = plt.subplots(2, 4, figsize=(19, 8))
#     for ax, (col, label) in zip(axes.ravel(), metrics):
#         ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#8e44ad")
#         ax.set_xlabel("SOC (%)")
#         ax.set_ylabel(label)
#         ax.grid(alpha=0.3)
#         if col in ("tau1_w", "tau2_w", "tau_d"):
#             ax.set_yscale("log")
#     fig.suptitle(f"EIS 2RC + Warburg fit parameters vs SOC — {title}", fontsize=11)
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=120)
#     plt.close(fig)
#     logging.info("EIS Warburg vs-SOC plot -> %s", out_png)


def _z_diffusion_from_row(row, w):
    """Diffusion branch of a fitted table row, matching the element used.

    Prefers the row's own ``phi_d_z`` so a table fitted under one element still
    replots correctly after :data:`ZARC_DIFFUSION_ELEMENT` is changed.
    """
    if ZARC_DIFFUSION_ELEMENT == "tanh":
        return _z_warburg(row["R_d_z"], row["tau_d_z"], w)
    phi = row.get("phi_d_z", 0.5)
    if not np.isfinite(phi):
        phi = 0.5
    return _z_warburg_generalized(row["R_d_z"], row["tau_d_z"], phi, w)


def plot_zarc_vs_soc(table: pd.DataFrame, out_png: str, title: str = ""):
    """Grid of the N×ZARC+L+Warburg fit parameters vs SOC (hides degenerate).

    ``N`` comes from the bundle's DRT, so the branch panels are built from the
    columns that actually carry data: an all-NaN ``R2_z`` (a one-arc fit) drops
    its three panels instead of drawing three empty axes.
    """
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
    metrics = [("R0_z", "R0 — series (mΩ)")]
    for i in range(1, ZARC_COLUMN_SLOTS + 1):
        metrics += [(f"R{i}_z", f"R{i} (mΩ)"), (f"tau{i}_z", f"τ{i} (s)"),
                    (f"alpha{i}_z", f"α{i}")]
    metrics += [
        ("R_d_z", "R_d — diffusion (mΩ)"), ("tau_d_z", "τ_d (s) — shape only"),
        ("phi_d_z", "φ_d — diffusion exponent"),
        ("zarc_rmse", "ZARC+W rmse (mΩ)"),
    ]
    # Drop both absent and all-NaN columns: with n_zarc=1 the slot-2 columns
    # exist (the schema is fixed) but hold nothing to draw.
    metrics = [m for m in metrics if m[0] in t.columns and t[m[0]].notna().any()]
    # The branch count is per spectrum, so one sweep can hold both one- and
    # two-arc fits. Slots are ordered τ-ascending: on a two-arc spectrum slot 1
    # is the *fast* arc, while a one-arc spectrum's single (dominant, slow)
    # branch also lands in slot 1 — so `tau1_z` genuinely jumps by decades at
    # the transition. Mark those points instead of letting the line imply a
    # continuous trend through them.
    n_col = t["n_zarc"] if "n_zarc" in t.columns else pd.Series(index=t.index, dtype=float)
    counts = sorted(int(v) for v in n_col.dropna().unique())
    one_arc = n_col == 1
    mixed = len(counts) > 1
    ncol = 6
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    for ax, (col, label) in zip(axes.ravel(), metrics):
        ax.plot(t["SOC_pct"], t[col], "o-", ms=5, color="#16a085",
                label="2 ZARC" if mixed else None)
        if mixed and one_arc.any():
            ax.plot(t.loc[one_arc, "SOC_pct"], t.loc[one_arc, col], "s",
                    ms=6, mfc="none", mec="#d35400", mew=1.4,
                    label="1 ZARC (DRT: one arc)")
            if col in ("R1_z", "tau1_z", "alpha1_z"):
                # On these three the slot changes meaning, not just the value.
                ax.set_facecolor("#fdf6f0")
        # Overlay the stage-1 HF estimate on the R0 panel: when R0 was pinned
        # the two coincide by construction, and when it was not, the gap is
        # exactly the series resistance the full-band fit lost to the arc.
        if col == "R0_z" and "R0_hf" in t.columns and t["R0_hf"].notna().any():
            ax.plot(t["SOC_pct"], t["R0_hf"], "--", lw=1.2, color="#c0392b",
                    label="HF-window R0")
            if "R0_hf_sigma" in t.columns and t["R0_hf_sigma"].notna().any():
                ax.fill_between(t["SOC_pct"],
                                t["R0_hf"] - t["R0_hf_sigma"],
                                t["R0_hf"] + t["R0_hf_sigma"],
                                color="#c0392b", alpha=0.15, lw=0)
            ax.legend(fontsize=7, loc="best")
        elif mixed and one_arc.any():
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        _autoscale(ax, x=t["SOC_pct"])
        if col.startswith("tau"):
            ax.set_yscale("log")   # padding a log axis linearly clips decades
        elif col.startswith("alpha"):
            # Was a hard-coded (0.25, 1.05). α is fitted inside
            # [ZARC_ALPHA_MIN, 1], so a fixed window both wasted the band below
            # the bound and hid how close a fit sat to it. Frame the data, then
            # clamp to the box α was actually fitted against.
            _autoscale(ax, y=t[col], floor=ZARC_ALPHA_MIN)
            ax.axhline(ZARC_ALPHA_MIN, color="0.6", ls=":", lw=0.9)
        elif col == "R0_z" and "R0_hf" in t.columns:
            # Include the HF overlay and its ±1σ band in the frame.
            sig = t["R0_hf_sigma"] if "R0_hf_sigma" in t.columns else 0.0
            _autoscale(ax, y=[t[col], t["R0_hf"] - sig, t["R0_hf"] + sig])
        else:
            _autoscale(ax, y=t[col])
    if not counts:
        model_name = "N×ZARC"
    elif mixed:
        n_one = int(one_arc.sum())
        model_name = (f"{'/'.join(str(c) for c in counts)}×ZARC per SOC "
                      f"({n_one} of {len(t)} spectra fitted with one arc — "
                      f"slot 1 is the fast arc on the 2-arc points and the "
                      f"only arc on the 1-arc ones, so R1/τ1/α1 are not one "
                      f"continuous trend)")
    else:
        model_name = f"{counts[0]}×ZARC"
    fig.suptitle(f"EIS {model_name} + Warburg fit parameters vs SOC — {title}",
                 fontsize=11 if not mixed else 9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS ZARC vs-SOC plot -> %s", out_png)


def plot_fit_overlay(df: pd.DataFrame, table: pd.DataFrame, out_png: str,
                     title: str = "", n_show: int = 6):
    """Measured vs fitted Nyquist (N×ZARC + Warburg) per SOC.

    The 2RC and 2RC+Warburg traces are gone with the fits that produced them
    (see :func:`build_eis_table`) — this now gates on the ZARC columns.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R_d_z" not in table.columns:
        logging.info("EIS fit overlay: no ZARC columns, skipping")
        return
    t = table.dropna(subset=["R_d_z"]).sort_values("SOC_pct")
    if t.empty:
        logging.info("EIS fit overlay: no converged fit to draw")
        return
    pick = t.iloc[np.linspace(0, len(t) - 1, min(n_show, len(t))).astype(int)]

    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, pick.iterrows()):
        spec = df[df["eis_number"] == row["eis_number"]].sort_values("frequency")
        w = 2 * np.pi * spec["frequency"].to_numpy(float)
        zz = row["R0_z"] + 1j * w * row["L_z"] + _z_diffusion_from_row(row, w)
        n_branch = 0
        for i in range(1, ZARC_COLUMN_SLOTS + 1):
            r_i, tau_i, a_i = (row.get(f"R{i}_z"), row.get(f"tau{i}_z"),
                               row.get(f"alpha{i}_z"))
            # Slots past this row's n_zarc are NaN — skip them rather than
            # letting a NaN branch wipe out the whole curve.
            if not all(np.isfinite(v) for v in (r_i, tau_i, a_i)):
                continue
            zz = zz + _z_zarc(r_i, tau_i, a_i, w)
            n_branch += 1
        ax.plot(spec["Z_real"], -spec["Z_imag"], "o", ms=3, color="#2f6fdb",
                label="measured")
        ax.plot(zz.real, -zz.imag, "-", color="#16a085",
                label=f"{n_branch}ZARC+W ({row['zarc_rmse']:.3f})")
        title_bits = [f"SOC {row['SOC_pct']:.0f}%"]
        if bool(row.get("zarc_degenerate", False)):
            title_bits.append("DEGENERATE")
        ax.set_title(" | ".join(title_bits), fontsize=9)
        ax.set_xlabel("Z_real (mΩ)")
        ax.set_ylabel("-Z_imag (mΩ)")
        ax.grid(alpha=0.3)
        # Frame on measured *and* fitted, so a fit that overshoots is visible
        # as an overshoot rather than by silently rescaling the measured data.
        _frame_nyquist(ax, (spec["Z_real"], -spec["Z_imag"]), (zz.real, -zz.imag))
        ax.legend(fontsize=7)
    for ax in axes[len(pick):]:
        ax.set_visible(False)
    fig.suptitle(f"EIS fit overlay — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS fit overlay -> %s", out_png)


#: Inset geometry on the parent Nyquist axes: [left, bottom, width, height] in
#: axes fractions. Two side by side along the bottom-right, clear of the arcs
#: which rise to the left: the narrow R0 zoom then the wider MF-arc zoom.
R0_INSET_RECT = (0.34, 0.08, 0.29, 0.29)
MF_INSET_RECT = (0.68, 0.08, 0.29, 0.29)

#: MF zoom width, as a multiple of the fitted arc diameter. The mid-frequency
#: arc is *both* ZARCs — on the NFPP bundle ``R1_z`` alone is 0.41 mOhm while
#: the arc plainly runs to ~3 mOhm because ``R2_z`` (1.62 mOhm) is the bulk.
MF_INSET_SPAN_ARCS = 1.6

#: R0 zoom width, as a multiple of the first ZARC diameter. Much tighter than
#: MF: this view is only for the region where the spectra meet the real axis,
#: so the arc itself is allowed to run out of frame.
R0_INSET_SPAN_ARCS = 1.2


def _arc_diameter(table: pd.DataFrame, cols: tuple) -> float:
    """Summed median of the given fitted ZARC diameters; 0 if unavailable."""
    arc = 0.0
    for col in cols:
        if col in table.columns:
            vals = pd.to_numeric(table[col], errors="coerce")
            if vals.notna().any():
                med = float(vals.median())
                if np.isfinite(med) and med > 0:
                    arc += med
    return arc


def _hf_zoom_window(df: pd.DataFrame, table: pd.DataFrame,
                    arc_cols: tuple = ("R1_z", "R2_z"),
                    span: float = MF_INSET_SPAN_ARCS,
                    fallback_frac: float = 0.25) -> tuple:
    """``(x0, x1, y0, y1)`` framing a high-frequency region and its real-axis
    crossing, or ``None`` if the data can't support a sensible window.

    The window is anchored on the **measured** high-frequency intercept
    (min ``Z_real``) rather than a fitted value, so it still frames correctly
    when a fit is degenerate. Its width comes from the fitted ZARC diameters in
    ``arc_cols`` (the physical scale of the region) and falls back to a
    fraction of the total Nyquist span. ``y0`` is pushed below zero on purpose:
    the point of the zoom is to see the curve *cross* -Z_imag = 0.
    """
    zr = pd.to_numeric(df["Z_real"], errors="coerce")
    zi = -pd.to_numeric(df["Z_imag"], errors="coerce")
    if not np.isfinite(zr).any():
        return None
    x0 = float(np.nanmin(zr))
    total_span = float(np.nanmax(zr) - x0)
    if not np.isfinite(total_span) or total_span <= 0:
        return None

    arc = _arc_diameter(table, arc_cols)
    width = arc * span if arc > 0 else total_span * fallback_frac
    width = min(width, total_span)          # never zoom out past the full plot

    pad = width * 0.06
    x_lo, x_hi = x0 - pad, x0 + width
    in_win = (zr >= x_lo) & (zr <= x_hi)
    y_vals = zi[in_win].dropna()

    # Height comes from the arc, not from the data minimum: above the arc the
    # cell turns inductive and -Z_imag dives (to -5.9 mΩ on the NFPP bundle,
    # ~14x the 0.41 mΩ arc). Framing that tail would flatten the semicircle
    # into a line — the opposite of what the zoom is for. Show only enough
    # below zero to make the real-axis crossing legible.
    positive = y_vals[y_vals > 0]
    y_top = float(positive.max()) if len(positive) else width * 0.3
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = width * 0.3
    y_lo, y_hi = -0.6 * y_top, 1.15 * y_top
    if not (np.isfinite(y_lo) and np.isfinite(y_hi)) or y_hi <= y_lo:
        return None
    return x_lo, x_hi, y_lo, y_hi


def _draw_zoom_inset(ax, df, soc, cmap, norm, rect, window, title, marker):
    """One zoomed copy of the spectra on ``ax``, framed by ``window``."""
    x_lo, x_hi, y_lo, y_hi = window
    axin = ax.inset_axes(rect)
    for eid, g in _spectra_by_soc(df, soc):
        g = g.sort_values("frequency")
        axin.plot(g["Z_real"], -g["Z_imag"], marker, ms=2.5, lw=0.9,
                  color=_soc_color(soc.get(eid), cmap, norm))
    axin.axhline(0.0, color="0.35", lw=0.8, zorder=1)
    axin.set_xlim(x_lo, x_hi)
    axin.set_ylim(y_lo, y_hi)
    axin.set_aspect("equal", adjustable="box")
    axin.tick_params(labelsize=6.5, length=2)
    axin.grid(alpha=0.25)
    axin.set_title(title, fontsize=8, pad=2)
    for spine in axin.spines.values():
        spine.set_edgecolor("0.4")
    try:
        ax.indicate_inset_zoom(axin, edgecolor="0.4", alpha=0.45)
    except Exception:            # older matplotlib without the connector API
        pass
    return axin


def add_hf_inset(ax, df: pd.DataFrame, table: pd.DataFrame, soc: dict,
                 cmap, norm, marker: str = "o-"):
    """Draw two zoomed copies of the high-frequency corner as insets on ``ax``.

    The full Nyquist view is dominated by the low-frequency diffusion tail, so
    everything that carries the kinetics collapses into a few pixels near the
    origin. Two scales are useful and one can't serve both:

    * **R0 region** — tight on the real-axis intercept, where the series
      resistance is read off. Narrow enough that the individual SOC curves
      separate at their crossing of -Z_imag = 0.
    * **MF arc** — the whole mid-frequency semicircle (both ZARCs), wide
      enough that the arc closes instead of running off the top.

    Returns ``(ax_r0, ax_mf)``; either may be ``None`` if its window couldn't
    be built.
    """
    out = []
    for rect, arc_cols, span, frac, title in (
        (R0_INSET_RECT, ("R1_z",), R0_INSET_SPAN_ARCS, 0.08, "R0 region"),
        (MF_INSET_RECT, ("R1_z", "R2_z"), MF_INSET_SPAN_ARCS, 0.25, "MF arc"),
    ):
        window = _hf_zoom_window(df, table, arc_cols=arc_cols, span=span,
                                 fallback_frac=frac)
        if window is None:
            logging.info("Nyquist %s inset: no usable window, skipping", title)
            out.append(None)
            continue
        out.append(_draw_zoom_inset(ax, df, soc, cmap, norm, rect, window,
                                    title, marker))
    return tuple(out)


def _spectra_by_soc(df: pd.DataFrame, soc: dict):
    """``(eis_number, spectrum)`` pairs ordered **low SOC first**.

    Drawing order is z-order: whatever is plotted last sits on top. Iterating
    the raw groupby draws in ``eis_number`` order, which for a discharge sweep
    is high SOC first — so the SOC-0 spectrum ends up painted over SOC 100,
    hiding the very curve the high-frequency zooms exist to show. Sorting
    ascending puts the top-SOC spectrum last, on top. Spectra with no SOC
    (no qOCV mapping) sort first, underneath everything.
    """
    def key(item):
        value = soc.get(item[0])
        if value is None or not np.isfinite(value):
            return (0, 0.0)
        return (1, float(value))

    return sorted(df.groupby("eis_number"), key=key)


def _soc_color(value, cmap, norm):
    """Colour for a SOC value; grey when SOC is missing (no qOCV mapping)."""
    if value is None or not np.isfinite(value):
        return "0.6"
    return cmap(norm(value))


def plot_nyquist_by_soc(df: pd.DataFrame, table: pd.DataFrame, out_png: str, title: str = ""):
    """All spectra on one Nyquist plane, coloured by SOC (context companion).

    Carries a bottom-right inset zoomed on the high-frequency arc — see
    :func:`add_hf_inset`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    soc = dict(zip(table["eis_number"], table["SOC_pct"]))
    norm = colors.Normalize(vmin=0, vmax=100)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for eid, g in _spectra_by_soc(df, soc):
        g = g.sort_values("frequency")
        ax.plot(g["Z_real"], -g["Z_imag"], "-", lw=1,
                color=_soc_color(soc.get(eid), cmap, norm))
    ax.set_xlabel("Z_real (mΩ)")
    ax.set_ylabel("-Z_imag (mΩ)")
    ax.grid(alpha=0.3)
    _frame_nyquist(ax, (df["Z_real"], -df["Z_imag"]))
    add_hf_inset(ax, df, table, soc, cmap, norm, marker="-")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="SOC (%)")
    # Bundle filenames are long and this figure is narrow, so wrap rather than
    # letting the title run off both edges.
    fig.suptitle("\n".join(textwrap.wrap(f"EIS Nyquist by SOC — {title}", 62)),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("EIS Nyquist plot -> %s", out_png)


def plot_raw_spectra(df: pd.DataFrame, table: pd.DataFrame, out_png: str, title: str = ""):
    """Raw measured EIS spectra: Nyquist + Bode (|Z|, phase) vs frequency.

    No fit — this is the as-measured data, drawn before any model is trusted
    (companion to :func:`plot_nyquist_by_soc`, which draws the Nyquist plane
    alone; this adds the Bode pair). One series per ``eis_number``, coloured
    by ``SOC_pct`` (from ``table``) so the sweep is readable at a glance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    if df.empty:
        logging.info("raw spectra plot: empty bundle, skipping")
        return
    soc = dict(zip(table["eis_number"], table["SOC_pct"]))
    vals = [v for v in soc.values() if np.isfinite(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 100.0)
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    fig, (ax_nyq, ax_mag, ax_ph) = plt.subplots(1, 3, figsize=(18, 5.5))
    for eid, g in _spectra_by_soc(df, soc):
        g = g.sort_values("frequency")
        color = _soc_color(soc.get(eid), cmap, norm)
        ax_nyq.plot(g["Z_real"], -g["Z_imag"], "o-", ms=3, lw=1, color=color)
        ax_mag.plot(g["frequency"], g["Z_abs"], "o-", ms=3, lw=1, color=color)
        ax_ph.plot(g["frequency"], g["phase"], "o-", ms=3, lw=1, color=color)

    ax_nyq.set_xlabel("Z_real (mΩ)")
    ax_nyq.set_ylabel("-Z_imag (mΩ)")
    ax_nyq.grid(alpha=0.3)
    ax_nyq.set_title("Nyquist (measured)")
    _frame_nyquist(ax_nyq, (df["Z_real"], -df["Z_imag"]))
    add_hf_inset(ax_nyq, df, table, soc, cmap, norm)

    # Bode: the frequency axis is log (pad it there, not in linear space), the
    # value axes are linear and get framed on the measured range — the phase
    # one especially, which was left entirely to autoscale.
    for ax in (ax_mag, ax_ph):
        ax.set_xscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
    ax_mag.set_ylabel("|Z| (mΩ)")
    ax_mag.set_title("Bode — magnitude")
    _autoscale(ax_mag, y=df["Z_abs"])
    ax_ph.set_ylabel("phase")
    ax_ph.set_title("Bode — phase")
    _autoscale(ax_ph, y=df["phase"])

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax_nyq, ax_mag, ax_ph],
                 label="SOC (%)", shrink=0.85, pad=0.02)
    fig.suptitle(f"Raw EIS spectra (measured) — {title}", fontsize=11)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("raw EIS spectra plot -> %s", out_png)


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
    # The 2RC / 2RC+Warburg grids went with their fits — see the retired-plotter
    # banner above.
    plot_zarc_vs_soc(table, f"{stem}_eis_zarc_warburg_vs_SOC.png", title=title)
    plot_fit_overlay(df, table, f"{stem}_eis_fit_overlay.png", title=title)
    plot_nyquist_by_soc(df, table, f"{stem}_eis_nyquist_by_SOC.png", title=title)


if __name__ == "__main__":
    main()
