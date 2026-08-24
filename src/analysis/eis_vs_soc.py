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

#: Diffusion element for the 2×ZARC fit. Read by both the fit and the overlay
#: plot so the drawn curve always matches the fitted one.
#:
#: * ``"generalized"`` — R·coth((jωτ)^φ)/((jωτ)^φ) with φ **fitted**. Only form
#:   whose low-frequency slope adapts, so the only one that describes both the
#:   steep tail at full charge and the flat ones mid-sweep.
#: * ``"coth"`` — as above with φ pinned to 0.5 (ISEA FittingGUI's element).
#: * ``"tanh"`` — transmissive finite-length Warburg; plateaus at low frequency.
ZARC_DIFFUSION_ELEMENT = "generalized"


def fit_zarc_warburg_eis(spec: pd.DataFrame, seed: dict = None, r_tot0=None,
                         element: str = None) -> dict:
    """2×ZARC + series-L + finite-length Warburg fit.

        Z(ω) = R0 + jωL + R1/(1+(jω τ1)^α1) + R2/(1+(jω τ2)^α2) + Z_diff(ω)

    Replaces the two ideal RC branches of :func:`fit_warburg_eis` with **ZARC**
    (depressed-arc) branches — each gains a CPE exponent ``α ∈ (0,1]`` so a real,
    flattened Nyquist semicircle is captured without inflating R/τ. Seeded from
    the 2RC+L+Warburg fit (``seed``). Params namespaced ``*_z``.

    ``element`` picks the diffusion branch — see :data:`ZARC_DIFFUSION_ELEMENT`
    for the three forms; ``None`` takes that module default. Only
    ``"generalized"`` fits the diffusion exponent φ; the fixed-exponent forms
    can each reproduce one low-frequency slope and misfit every other one.

    **Identifiability.** The ZARC τ are boxed into the measured band and the
    diffusion τ into :data:`DIFFUSION_TAU_BOX`, disjoint from it, so the slow
    ZARC and the diffusion branch cannot trade roles — without this the two
    swap between spectra and the vs-SOC curves jump decades at the swap points.
    All three τ are ordered and the fit is multistarted. τ_d normally lands on
    a box edge and carries no information; use ``R_d_z``.

    Returns ``{R0_z, L_z, R1_z, tau1_z, alpha1_z, R2_z, tau2_z, alpha2_z, R_d_z,
    tau_d_z, phi_d_z, zarc_rmse, zarc_degenerate}`` (NaN/True on failure).
    ``phi_d_z`` is the fitted φ, or the fixed exponent of the chosen element.
    """
    from scipy.optimize import least_squares

    keys = ("R0_z", "L_z", "R1_z", "tau1_z", "alpha1_z", "R2_z", "tau2_z",
            "alpha2_z", "R_d_z", "tau_d_z", "phi_d_z", "zarc_rmse")
    fail = {k: np.nan for k in keys}
    fail["zarc_degenerate"] = True

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
    tz_lo = 1.0 / (2 * np.pi * f.max()) / 3.0
    tz_hi = min(1.0 / (2 * np.pi * f.min()) * 3.0, td_lo)

    r0 = seed.get("R0_w", seed.get("R0", s["Z_real"].to_numpy(float)[int(np.argmax(w))]))
    l0 = seed.get("L_w", seed.get("L", 1e-6))
    rtot = r_tot0 if r_tot0 is not None else s["Z_real"].to_numpy(float)[int(np.argmin(w))]
    rspan = max(rtot - r0, 0.5)

    # Free-parameter layout. Always [R0, L, R1, R2, lt1, lt2, a1, a2, Rd], then
    # ltd only when DIFFUSION_TAU_BOX has width (equal bounds pin it), then phi
    # only when it is fitted.
    ph_lo, ph_hi = DIFFUSION_PHI_BOX
    fit_tau_d = td_hi > td_lo
    lb = [0.0, 0.0, 0.0, 0.0, np.log(tz_lo), np.log(tz_lo),
          ZARC_ALPHA_MIN, ZARC_ALPHA_MIN, 0.0]
    ub = [np.inf, np.inf, np.inf, np.inf, np.log(tz_hi), np.log(tz_hi),
          1.0, 1.0, np.inf]
    if fit_tau_d:
        lb, ub = lb + [np.log(td_lo)], ub + [np.log(td_hi)]
    if free_phi:
        lb, ub = lb + [ph_lo], ub + [ph_hi]
    i_ltd = 9 if fit_tau_d else None
    i_phi = (10 if fit_tau_d else 9) if free_phi else None

    def _unpack(x):
        ltd = x[i_ltd] if fit_tau_d else np.log(td_lo)
        phi = x[i_phi] if free_phi else 0.5
        return tuple(x[:9]) + (ltd, phi)

    def model(x, wv):
        R0_, L_, R1_, R2_, lt1, lt2, a1, a2, Rd_, ltd, phi = _unpack(x)
        return (R0_ + 1j * wv * L_
                + _z_zarc(R1_, np.exp(lt1), a1, wv)
                + _z_zarc(R2_, np.exp(lt2), a2, wv)
                + z_diff(Rd_, np.exp(ltd), phi, wv))

    def resid(x):
        r = (model(x, w) - zd) / absz
        return np.concatenate([r.real, r.imag])

    # Multistart: one seeded from the 2RC+W fit plus two fixed spreads. A single
    # start lands in a different local minimum from spectrum to spectrum, which
    # is the other half of why the vs-SOC curves used to jump.
    starts = [
        (seed.get("tau1_w", 1e-3), seed.get("tau2_w", 1e-2), seed.get("tau_d", 30.0), 0.5),
        (3e-4, 8e-3, 3.0, 0.4),
        (2e-3, 3e-2, 100.0, 0.3),
    ]
    best = None
    for t1s, t2s, tds, phs in starts:
        x0 = [max(r0, 1e-3), max(l0, 1e-9), 0.3 * rspan, 0.5 * rspan,
              np.log(t1s), np.log(t2s), 0.85, 0.85, 0.3 * rspan]
        if fit_tau_d:
            x0.append(np.log(tds))
        if free_phi:
            x0.append(phs)
        x0 = np.clip(x0, lb, ub)
        try:
            res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
        except Exception as exc:  # noqa: BLE001
            logging.warning("fit_zarc_warburg_eis start failed: %s", exc)
            continue
        err = float(np.sqrt(np.mean(np.abs(model(res.x, w) - zd) ** 2)))
        if best is None or err < best[0]:
            best = (err, res.x)
    if best is None:
        return fail

    rmse, x = best
    R0_, L_, R1_, R2_, lt1, lt2, a1, a2, Rd_, ltd, phi = _unpack(x)
    phi = float(phi)
    ta, tb, aa, ab = np.exp(lt1), np.exp(lt2), a1, a2
    if ta > tb:
        R1_, R2_, ta, tb, aa, ab = R2_, R1_, tb, ta, ab, aa
    td = float(np.exp(ltd))

    # Degenerate when the diffusion branch collapses, a ZARC τ sits on the edge
    # of the resolvable band, or a CPE / diffusion exponent pins to a bound —
    # each means the reported parameters are not constrained by the data.
    def _on_edge(v, lo, hi):
        return bool(v <= lo * 1.05 or v >= hi * 0.95)

    degenerate = bool(
        Rd_ < 1e-2
        or _on_edge(ta, tz_lo, tz_hi) or _on_edge(tb, tz_lo, tz_hi)
        or aa <= ZARC_ALPHA_MIN * 1.02 or ab <= ZARC_ALPHA_MIN * 1.02
        or aa >= 0.999 or ab >= 0.999
        or (free_phi and _on_edge(phi, ph_lo, ph_hi))
    )
    return {
        "R0_z": float(R0_), "L_z": float(L_),
        "R1_z": float(R1_), "tau1_z": float(ta), "alpha1_z": float(aa),
        "R2_z": float(R2_), "tau2_z": float(tb), "alpha2_z": float(ab),
        "R_d_z": float(Rd_), "tau_d_z": td, "phi_d_z": phi,
        "zarc_rmse": rmse, "zarc_degenerate": degenerate,
    }


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
        # No order-based SOC ladder: `100 - step * i` assumes every step moved
        # the same charge, which the measured voltages contradict (the first
        # NFPP step drops 155 mV, the next ones ~15 mV, all labelled "5 %").
        # SOC is measured from the run's own qOCV curve
        # (`analysis/soc_from_qocv.py`) using each measurement's own U.
        feat.update(
            {
                "eis_number": eid,
                "SOC_pct": np.nan,
                "U": float(spec["U"].mean()),
                "Time": spec["Time"].min(),
            }
        )
        rows.append(feat)
    out = pd.DataFrame(rows)
    logging.info(
        "eis_vs_soc: %d measurements (%s sweep) — SOC_pct left NaN, filled "
        "from the qOCV curve",
        len(out), direction,
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
        ("R_d_z", "R_d — diffusion (mΩ)"), ("tau_d_z", "τ_d (s) — shape only"),
        ("phi_d_z", "φ_d — diffusion exponent"),
        ("zarc_rmse", "2ZARC+W rmse (mΩ)"),
    ]
    metrics = [m for m in metrics if m[0] in t.columns]
    fig, axes = plt.subplots(2, 6, figsize=(26, 8))
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
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
                  + _z_diffusion_from_row(row, w))
            ax.plot(zz.real, -zz.imag, "-", color="#16a085", label=f"2ZARC+W ({row['zarc_rmse']:.3f})")
        title_bits = [f"SOC {row['SOC_pct']:.0f}%"]
        if "zarc_rmse" in table.columns and np.isfinite(row.get("zarc_rmse", np.nan)):
            title_bits.append(f"zarc rmse={row['zarc_rmse']:.3f}")
        if bool(row.get("zarc_degenerate", False)):
            title_bits.append("DEGENERATE")
        ax.set_title(" | ".join(title_bits), fontsize=9)
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


#: Inset geometry on the parent Nyquist axes: [left, bottom, width, height] in
#: axes fractions. Two side by side along the bottom-right, clear of the arcs
#: which rise to the left: the narrow R0 zoom then the wider MF-arc zoom.
R0_INSET_RECT = (0.34, 0.08, 0.29, 0.29)
MF_INSET_RECT = (0.68, 0.08, 0.29, 0.29)

#: Zoom widths, as multiples of a fitted arc diameter. ``R0`` frames just the
#: real-axis intercept region (first ZARC only); ``MF`` frames the whole
#: mid-frequency arc, which is *both* ZARCs — on the NFPP bundle ``R1_z`` alone
#: is 0.41 mOhm while the arc plainly runs to ~3 mOhm because ``R2_z``
#: (1.62 mOhm) is the bulk of it.
R0_INSET_SPAN_ARCS = 2.2
MF_INSET_SPAN_ARCS = 1.6


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
    for eid, g in df.groupby("eis_number"):
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
    for eid, g in df.groupby("eis_number"):
        g = g.sort_values("frequency")
        ax.plot(g["Z_real"], -g["Z_imag"], "-", lw=1,
                color=_soc_color(soc.get(eid), cmap, norm))
    ax.set_xlabel("Z_real (mΩ)")
    ax.set_ylabel("-Z_imag (mΩ)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    add_hf_inset(ax, df, table, soc, cmap, norm, marker="-")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="SOC (%)")
    fig.suptitle(f"EIS Nyquist by SOC — {title}", fontsize=11)
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
    for eid, g in df.groupby("eis_number"):
        g = g.sort_values("frequency")
        color = _soc_color(soc.get(eid), cmap, norm)
        ax_nyq.plot(g["Z_real"], -g["Z_imag"], "o-", ms=3, lw=1, color=color)
        ax_mag.plot(g["frequency"], g["Z_abs"], "o-", ms=3, lw=1, color=color)
        ax_ph.plot(g["frequency"], g["phase"], "o-", ms=3, lw=1, color=color)

    ax_nyq.set_xlabel("Z_real (mΩ)")
    ax_nyq.set_ylabel("-Z_imag (mΩ)")
    ax_nyq.set_aspect("equal", adjustable="datalim")
    ax_nyq.grid(alpha=0.3)
    ax_nyq.set_title("Nyquist (measured)")
    add_hf_inset(ax_nyq, df, table, soc, cmap, norm)

    ax_mag.set_xscale("log")
    ax_mag.set_xlabel("frequency (Hz)")
    ax_mag.set_ylabel("|Z| (mΩ)")
    ax_mag.grid(alpha=0.3, which="both")
    ax_mag.set_title("Bode — magnitude")

    ax_ph.set_xscale("log")
    ax_ph.set_xlabel("frequency (Hz)")
    ax_ph.set_ylabel("phase")
    ax_ph.grid(alpha=0.3, which="both")
    ax_ph.set_title("Bode — phase")

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
    plot_2rc_vs_soc(table, f"{stem}_eis_2RC_vs_SOC.png", title=title)
    plot_warburg_vs_soc(table, f"{stem}_eis_2RC_warburg_vs_SOC.png", title=title)
    plot_zarc_vs_soc(table, f"{stem}_eis_2ZARC_warburg_vs_SOC.png", title=title)
    plot_fit_overlay(df, table, f"{stem}_eis_fit_overlay.png", title=title)
    plot_nyquist_by_soc(df, table, f"{stem}_eis_nyquist_by_SOC.png", title=title)


if __name__ == "__main__":
    main()
