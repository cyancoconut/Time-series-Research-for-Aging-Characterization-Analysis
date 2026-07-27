"""Sequential (Bruch 2021) decomposition of a pulse into a **fractional-order** ECM.

This applies the parameterization *strategy* of

    M. Bruch, L. Millet, J. Kowal, M. Vetter, "Novel method for the
    parameterization of a reliable equivalent circuit model for the precise
    simulation of a battery cell's electric behavior", J. Power Sources 490
    (2021) 229513, https://doi.org/10.1016/j.jpowsour.2021.229513

to a model built from **fractional** circuit elements instead of the paper's
integer-order RC ladder.

Why the strategy, not the model
-------------------------------
Bruch's contribution is not the RC ladder — it is the *order of identification*.
A joint fit of an n-element ECM is a nasty multi-modal problem whose answer is
dominated by the initial guess (their §1.3). Their fix rests on one observation:
at the **end** of a relaxation only the slowest process is still detectable.
So they fit the slowest element there, subtract it, extend the fit window one
logarithmic step *back toward* the current interruption, fit the next-faster
element, and repeat. Every sub-problem has 2-3 parameters and an initial guess
computed analytically from the measured voltage itself (their eqs. 9-15), so no
user-supplied guesses or bounds are needed anywhere. Crucially the model **order
is decided by the data**: a new element is only introduced when the not-yet-
explained voltage at the start of the next window still exceeds a threshold
``dU_fit2`` (their decision D1).

That argument is entirely independent of what the elements *are*. It transfers
unchanged to a fractional model, which is what this module does.

The model
---------
Terminal voltage of one CC pulse (``0 < t <= t_p``, current ``I``) followed by
its relaxation (``t > t_p``, zero current)::

    U(t) = U_OCV(t) + I(t)*R_s + sum_n eta_ZARC,n(t)      [+ eta_W(t), optional]

* ``U_OCV(t)`` — Bruch eq. 1: ramps from the settled pre-pulse OCV ``U_OCVs`` to
  the end-of-relaxation OCV ``U_OCVe`` in proportion to the charge *actually
  delivered* (running integral of the measured current), then holds. ``U_OCVs``
  and ``U_OCVe`` are both **measured, not fitted**: the test runs a C/2 restore
  pulse after each test pulse, so every pulse starts from the same relaxed
  voltage (``U_OCVs`` = the prior pause) and the ~30 min pause after the test
  pulse relaxes to ``U_OCVe`` (the tail median). Pinning ``U_OCVe`` removes the
  OCV-vs-slow-element degeneracy that otherwise lets a small SOC-driven OCV drift
  be absorbed by the slow element (which then rails).
* ``R_s`` — series (pure ohmic) resistance.
* ``eta_ZARC,n`` — **ZARC** elements ``Z = R_n / (1 + (s tau_n)^alpha_n)``, i.e. a
  resistor in parallel with a constant-phase element. ``alpha = 1`` degenerates
  to an ordinary RC. The slowest is fitted first in the rearmost relaxation
  window (paper steps P3-P5), the rest subsequently, fastest last (P6-P8).
* ``eta_W`` — **optional** finite-length transmissive Warburg
  ``Z = R_d tanh(sqrt(s tau_d)) / sqrt(s tau_d)`` in place of the slowest ZARC
  (``--slowest warburg``, Bruch's original choice). It is **off by default**: its
  differenced two-step response is tiny in the relaxation tail, so on a modest
  slow tail ``R_d`` rails (observed on VTC6 mid-SOC charge pulses). A ZARC's
  step response decays cleanly and fits the same tail without railing.

Fractional elements are the reason to bother: a real electrode is a *distributed*
system (porous, with a spread of particle sizes and path lengths), so its
relaxation is a depressed arc, not a single exponential. An RC ladder fakes that
spread by stacking exponentials, which is why Bruch's own results need 3-6 RC
elements whose count and time constants scatter pulse-to-pulse (their Fig. 4b,
§3.2) — the ladder is fitting the *width* of one physical process, not several
distinct processes. A ZARC carries that width in one extra parameter
(``alpha``), so the element count stays small and stable, and ``alpha`` itself
becomes an aging observable.

Time-domain evaluation of a ZARC
--------------------------------
The unit-step response of a ZARC is the Mittag-Leffler function::

    s_n(t) = 1 - E_{alpha}(-(t/tau)^alpha)

Our current profile is exactly two steps (on at ``t=0``, off at ``t=t_p``), so
the full pulse+relaxation response is an exact superposition of two of these — no
time-stepping and no uniform grid are required, which matters because our export
is sampled at ~0.2 s under load and ~0.8 s during the rest.

``E_alpha(-x)`` is evaluated (``_ml_neg``) through the Cole-Cole relaxation-time
distribution, which is exact rather than an approximation::

    E_alpha(-(t/tau)^alpha) = integral G_alpha(u) exp(-t/(tau e^u)) du
    G_alpha(u) = sin(alpha pi) / (2 pi (cosh(alpha u) + cos(alpha pi)))

i.e. a ZARC *is* a continuum of RC branches whose weight is the depressed-arc
DRT. The integral is taken with panel Gauss-Legendre on a fixed, alpha-independent
node set geometrically graded toward ``u=0`` (``G`` collapses to a delta as
alpha -> 1). Validated in ``--self-test``: the quadrature normalizes to 1 within
1e-11 and matches the Mittag-Leffler Taylor series to <2e-5 relative over
alpha in [0.4, 1). Fixed nodes mean every alpha-dependence sits in smooth
``cosh``/``cos`` terms, so the whole model is differentiable in ``alpha`` — which
is what step P10 needs.

Steps, mapped to the paper's flowchart (their Fig. 2)
-----------------------------------------------------
=====  =========================================  ==========================
Step   Action                                     Solver
=====  =========================================  ==========================
P1-P2  window prep; back-scan the relaxation to   numpy
       the first sample exceeding ``dU_fit1``
       (index ``K1``)
P3-P4  split ``[t_N, t_K1]`` into ``N_MAX``       numpy
       log-spaced sections
P5     pin ``U_OCVe`` = tail; fit slowest ZARC    ``scipy.curve_fit`` (3 par)
       (or Warburg) on the rearmost section
P6-P8  subtract; extend one section; if the       ``scipy.curve_fit`` (3 par
       residual there still exceeds ``dU_fit2``,  each)
       fit one more ZARC; repeat
P9     extend into the pulse; fit ``R_s`` (init   ``scipy.curve_fit``
       from Ohm's law on the current edge) and
       refit the fastest ZARC
P10    joint refit of all parameters over the     JAX autodiff + L-BFGS-B
       relaxation only
=====  =========================================  ==========================

The pulse part is deliberately excluded from P10 (paper eq. 16 and their Fig. 5):
between two pulses the true OCV curve is unknown and eq. 1 interpolates it
linearly, so including the pulse would let the optimizer absorb that known
interpolation error into the element parameters.

P10 kernels
-----------
``--p10-kernel ml`` (default) differentiates through the exact Mittag-Leffler
quadrature above. ``--p10-kernel gl`` instead integrates each ZARC's defining
fractional ODE ``u + tau^alpha D^alpha u = R I`` with a Grunwald-Letnikov
recursion (short-memory truncated) on a uniformly resampled grid, as a
cross-check. Both give exact gradients w.r.t. every ``alpha`` via ``jax.grad``;
``ml`` is the default because it needs no resampling, adds no discretization
error, and is roughly two orders of magnitude cheaper. ``--self-test`` asserts
the two kernels agree.

Usage (run from ``src/``)::

    python -m analysis.fit_frac_pulse --self-test
    python -m analysis.fit_frac_pulse <pulse.parquet> [--plot]
    python -m analysis.fit_frac_pulse <pulse_folder>/ [--plot]

Standalone analysis utility; it does not touch the pipeline.
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from analysis.fit_2rc_pulse import (
    REST_CURRENT_A,
    _proc_id,
    _warburg_step,
    label_time_diff,
    select_pulse_segments,
    EXCLUDE_ZUSTAND_CURRENT,
    REMOVE_PULSE_BEFORE_MIN,
    _parse_soh,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --- method parameters (Bruch's dU_fit1 / dU_fit2 / N_RCmax) -----------------
# Both voltage limits are taken as a fraction of the total relaxation drop
# |U[N] - U[M]| (their eqs. 17/18) with an absolute floor, so they scale with the
# pulse amplitude instead of being hard-coded per cell.
D_FIT1 = 20.0          # dU_fit1 = drop / D_FIT1 -- opens the first fit window
D_FIT2 = 200.0         # dU_fit2 = drop / D_FIT2 -- threshold to add an element
# Absolute floors. Kept small because on a real high-SOC plateau the settled tail
# is genuinely clean (~0.1 mV) and a per-element slow relaxation there can be
# sub-mV; a large floor would reject those real elements. The noise-multiple
# floors in ``voltage_limits`` do the actual guarding when the tail is noisy.
DU_FIT1_MIN_V = 2e-4   # floor: must exceed the measurement noise (paper §2.2)
DU_FIT2_MIN_V = 1e-4
N_MAX = 8              # number of logarithmic sections = max element count
                       # (paper: overestimating this is harmless and recommended)

# A fitted element whose resistance falls below this is dropped as noise
# (the paper neglects RC circuits under 0.2 mOhm, their §3.2).
R_MIN_OHM = 2e-4

# Slow-element (Warburg) presence gate: the drop across the rearmost window
# [K1, M] must exceed this multiple of the measured noise for P5 to fit a slow
# branch at all. On the OCV plateau (high SOC) there is little diffusion-driven
# slow relaxation, so the tail is flat and this window drop is ~noise; below the
# gate the Warburg is dropped (R_d = 0, ``slow_degenerate`` flagged) rather than
# railed to a meaningless value. See project note on the 90%-SOC plateau.
SLOW_SIGNAL_NOISE_MULT = 4.0

# alpha is bounded away from 0 (the ZARC would lose all shape) and from 1 (where
# the Cole-Cole DRT becomes a delta and tau/alpha trade off freely).
ALPHA_MIN, ALPHA_MAX = 0.30, 0.999

# --- Mittag-Leffler quadrature nodes ----------------------------------------
# Fixed (alpha-independent) panel Gauss-Legendre grid in u = ln(tau'/tau). The
# kernel decays like exp(-alpha|u|), so the half-width is set from ALPHA_MIN;
# panels are graded geometrically toward u=0 because G collapses to a delta as
# alpha -> 1. 240 nodes reach ~1e-5 relative accuracy for every alpha in range,
# far below the ~0.1 mV voltage noise. See --self-test.
_GL_ORDER = 6
_N_PANEL = 20


def _make_ml_nodes(n_gl=_GL_ORDER, n_panel=_N_PANEL, half_width=None):
    """Fixed quadrature nodes/weights for the Cole-Cole DRT integral."""
    L = half_width if half_width is not None else 40.0 / ALPHA_MIN
    b = np.concatenate(([0.0], np.geomspace(1e-8, L, n_panel)))
    edges = np.concatenate((-b[::-1][:-1], b))
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    lo, hi = edges[:-1], edges[1:]
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    u = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    w = (half[:, None] * np.broadcast_to(wg, (len(lo), n_gl))).ravel()
    return u, w


_ML_U, _ML_W = _make_ml_nodes()
_ML_EXP_MINUS_U = np.exp(-_ML_U)


def _ml_neg(alpha, x):
    """``E_alpha(-x)`` for ``x >= 0``, ``0 < alpha <= 1``, via the Cole-Cole DRT.

    Uses ``E_alpha(-(t/tau)^alpha) = int G_alpha(u) exp(-t/(tau e^u)) du``, so the
    caller's ``x`` is converted back to ``t/tau = x^(1/alpha)``. Exact (not an
    approximation) up to the quadrature error; see the module docstring.
    """
    g = (np.sin(alpha * np.pi) / (2.0 * np.pi)) / (
        np.cosh(alpha * _ML_U) + np.cos(alpha * np.pi)
    )
    weights = _ML_W * g
    s = np.asarray(x, dtype=float) ** (1.0 / alpha)
    return weights @ np.exp(-np.outer(_ML_EXP_MINUS_U, s))


def zarc_step(t, tau, alpha):
    """Unit-current step response of a ZARC: ``1 - E_alpha(-(t/tau)^alpha)``.

    Zero for ``t <= 0`` (causal), rises to 1 (the DC resistance) as t -> inf.
    """
    t = np.asarray(t, dtype=float)
    pos = t > 0
    out = np.zeros_like(t)
    if not np.any(pos):
        return out
    x = (t[pos] / tau) ** alpha
    out[pos] = 1.0 - _ml_neg(alpha, x)
    return out


def _superpose(step_fn, t, t_p):
    """Two-step (on at 0, off at ``t_p``) response from a unit step response."""
    t = np.asarray(t, dtype=float)
    return step_fn(t) - np.where(t > t_p, step_fn(t - t_p), 0.0)


def zarc_pulse(t, t_p, r, tau, alpha):
    """ZARC overvoltage for a unit current pulse of length ``t_p``."""
    return r * _superpose(lambda tt: zarc_step(tt, tau, alpha), t, t_p)


def warburg_pulse(t, t_p, r_d, tau_d):
    """Finite-length transmissive Warburg overvoltage for a unit current pulse."""
    return r_d * _superpose(lambda tt: _warburg_step(tt, tau_d), t, t_p)


# ---------------------------------------------------------------------------
# OCV ramp (Bruch eq. 1)
# ---------------------------------------------------------------------------
def ocv_ramp(t, i_arr, ocv_s, ocv_e):
    """OCV interpolated over the charge actually delivered (Bruch eq. 1).

    ``U_OCV[i] = U_OCVs + (U_OCVe - U_OCVs) * (sum_0^i I dt) / (sum_0^M I dt)``.
    The running integral uses the *measured* current, so it saturates exactly
    when the pulse ends and holds flat through the relaxation.
    """
    t = np.asarray(t, dtype=float)
    q = np.concatenate(([0.0], np.cumsum(np.diff(t) * i_arr[:-1])))
    total = q[-1]
    frac = q / total if abs(total) > 1e-12 else np.zeros_like(q)
    return ocv_s + (ocv_e - ocv_s) * frac


# ---------------------------------------------------------------------------
# P1-P4: window prep, dU limits, logarithmic sections
# ---------------------------------------------------------------------------
def estimate_noise(v, i_relax_start, tail=200):
    """Measurement-noise level from the settled relaxation tail.

    Uses the std of the tail after a rolling-median detrend, not the raw MAD:
    the cycler *quantizes and holds* voltage once the cell settles, so on a flat
    high-SOC plateau many tail samples are byte-identical and a MAD collapses to
    ~0 — which would then let the ``dU_fit`` floors vanish and the decomposition
    fit quantization steps as spurious elements. The detrended std keeps the
    residual scatter (including the quantization step size) in the estimate, so it
    tracks the *effective* noise the fit actually has to clear (Bruch §2.2: both
    ``dU`` limits must exceed the measurement noise). Floored to 50 uV.
    """
    seg = np.asarray(v[max(i_relax_start, len(v) - tail):], dtype=float)
    if len(seg) < 8:
        return 5e-5
    resid = seg - _rolling_median(seg, 11)
    return float(max(np.std(resid), 5e-5))


def voltage_limits(v_relax_start, v_end, d_fit1=D_FIT1, d_fit2=D_FIT2, noise=0.0):
    """``dU_fit1`` / ``dU_fit2`` from the relaxation drop (Bruch eqs. 17/18).

    Each limit is a fraction of the total relaxation drop, floored to the larger
    of an absolute minimum and a multiple of the measured ``noise`` so neither the
    window search (``dU_fit1``) nor the add-an-element decision (``dU_fit2``) can
    key on noise (Bruch §2.2). ``dU_fit1`` gets the higher multiple because a
    window opened on noise rails the slow-element fit outright.
    """
    drop = abs(v_relax_start - v_end)
    return (
        max(DU_FIT1_MIN_V, 5.0 * noise, drop / d_fit1),
        max(DU_FIT2_MIN_V, 3.0 * noise, drop / d_fit2),
    )


def find_k1(t, v, i_relax_start, du_fit1):
    """P2 — back-scan the relaxation for the first sample ``du_fit1`` off the end.

    ``K1`` is the point where the relaxation, coming *back from the settled end*,
    has just risen ``du_fit1`` above ``U[M]``; everything from ``K1`` to the end is
    the rearmost window, in which (per the paper's core assumption) only the
    slowest element is still relaxing.

    The relaxation deviation ``|U - U[M]|`` decreases monotonically (in trend)
    from the interruption toward the end, so ``K1`` is the boundary where it drops
    below ``du_fit1``: from there to the end only the slowest element is still
    relaxing above the limit. Taking the *first trend crossing* scanned forward
    from ``t[N]`` — rather than the last sample that happens to exceed the limit —
    makes this robust to noise. Both matter with ``du_fit1`` near the noise floor:
    a last-crossing scan latches onto a lone late spike and collapses the window
    onto a flat 20 s tail (railing the slow-element fit), while a raw first-
    crossing latches onto a lone early *dip* and puts ``K1`` far too early (the
    window then still contains the faster elements, which the slow fit wrongly
    absorbs). Smoothing ``dev`` with a rolling median before the crossing removes
    both single-sample artefacts. ``U[M]`` is likewise a short median. Falls back
    to the relaxation start when the whole rest is flatter than ``du_fit1``.
    """
    v_end = float(np.median(v[-15:]))
    dev = np.abs(v[i_relax_start:] - v_end)
    dev_s = _rolling_median(dev, 15)
    below = np.where(dev_s < du_fit1)[0]
    if not len(below):
        return int(i_relax_start)
    return int(i_relax_start + max(below[0] - 1, 0))


def _rolling_median(x, win):
    """Centered rolling median (odd ``win``); edges shrink the window."""
    x = np.asarray(x, dtype=float)
    h = win // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def log_sections(t, i_relax_start, k1, n_max=N_MAX):
    """P4 — ``n_max`` logarithmically spaced window-start times (Bruch eq. for T_Lj).

    Section ``j=1`` starts at ``t[K1]`` (rearmost, smallest window); ``j=n_max``
    starts at the first relaxation sample, i.e. the whole relaxation. Successive
    starts are geometrically spaced in time-since-current-interruption, which is
    what makes each decade of the relaxation carry equal weight even though the
    late decades hold far more samples.

    Note: the printed exponent in the paper's ``T_Lj`` definition runs the
    opposite way to its own prose ("j=1 is the rearmost window ... T_L,NRCmax is
    equivalent to T_R"). The prose is implemented here.
    """
    t_ref = float(t[max(i_relax_start - 1, 0)])       # t[N-1]
    hi = float(t[k1]) - t_ref
    lo = float(t[min(i_relax_start + 1, len(t) - 1)]) - t_ref
    lo = max(lo, 1e-6)
    if hi <= lo:
        return [k1]
    ratio = lo / hi
    starts = []
    for j in range(1, n_max + 1):
        thresh = hi * ratio ** ((j - 1) / (n_max - 1))
        idx = int(np.searchsorted(t - t_ref, thresh))
        idx = min(max(idx, i_relax_start), len(t) - 2)
        if not starts or idx != starts[-1]:
            starts.append(idx)
    return starts


# ---------------------------------------------------------------------------
# Analytic initial guesses (Bruch eqs. 10-15, generalized to any element)
# ---------------------------------------------------------------------------
def _initial_guess(t, resid, k_start, i_relax_start, t_p, i_pulse, ocv_e,
                   step_fn):
    """Initial ``(tau, R)`` for the element about to be fitted.

    Follows Bruch eqs. 10-15 but written for a general element step response
    rather than only ``1 - exp(-t/tau)``:

    * ``tau_ini = t[Kn] - t[N]`` (eq. 10) — the time constant is simply where the
      fit window opens, since that is the timescale still visible there.
    * the element's overvoltage at the *start of the relaxation* is recovered
      from the residual at ``t[Kn]`` divided by how far the element has already
      decayed by then (eqs. 11/12; for an RC with ``tau_ini`` that factor is
      exactly ``1/e``, which is where the paper's ``e-1`` comes from).
    * that is scaled up by the fraction of full charge the element reached during
      a pulse of length ``t_p`` (eq. 13) and turned into a resistance by Ohm's law
      with the mean pulse current (eqs. 14/15).

    Because the guesses are read off the measurement itself they land close to
    the optimum, which is what removes the user-supplied-guess dependence.
    """
    t_n = float(t[i_relax_start])
    tau_ini = max(float(t[k_start]) - t_n, 1e-3)

    decay = float(step_fn(np.array([float(t[k_start]) - t_n]), tau_ini)[0])
    decayed = 1.0 - decay                     # fraction still remaining at t[Kn]
    if decayed < 1e-6:
        decayed = 1e-6
    u_max = abs(float(resid[k_start]) - ocv_e) / decayed

    buildup = float(step_fn(np.array([t_p]), tau_ini)[0])
    if buildup < 1e-6:
        buildup = 1e-6
    r_ini = u_max / buildup / abs(i_pulse)
    return tau_ini, float(np.clip(r_ini, R_MIN_OHM, 1.0))


# ---------------------------------------------------------------------------
# P5-P9: the sequential decomposition
# ---------------------------------------------------------------------------
def decompose_pulse(t, v, i_arr, t_p, i_pulse, ocv_s, *, n_max=N_MAX,
                    d_fit1=D_FIT1, d_fit2=D_FIT2, slowest="zarc", verbose=False):
    """Run steps P1-P9 on one pulse+relaxation window.

    ``slowest`` selects the slowest element's form: ``"zarc"`` (default) makes the
    whole model a ZARC ladder; ``"warburg"`` fits a finite-length Warburg as the
    slowest element (Bruch's original choice). The Warburg's differenced two-step
    response is tiny in the tail, so on a modest slow tail it rails ``R_d`` (seen
    on VTC6 mid-SOC charge pulses); a ZARC's step response decays cleanly and fits
    the same tail without railing, which is why ZARC is the default.

    Returns ``(params, info)`` where ``params`` is
    ``dict(ocv_s, ocv_e, r_s, r_d, tau_d, zarcs=[(R, tau, alpha), ...])`` ordered
    slow -> fast (``r_d``/``tau_d`` are 0/NaN in ZARC mode), and ``info`` carries
    the diagnostics of each stage.
    """
    n = int(np.searchsorted(t, t_p, side="right"))    # first relaxation sample
    n = min(max(n, 1), len(t) - 2)
    noise = estimate_noise(v, n)
    du1, du2 = voltage_limits(float(v[n]), float(v[-1]), d_fit1, d_fit2, noise)
    k1 = find_k1(t, v, n, du1)
    starts = log_sections(t, n, k1, n_max)
    if verbose:
        logging.info(
            "  P2/P4: noise=%.3f mV dU_fit1=%.2f mV dU_fit2=%.2f mV  K1 at "
            "t=%.1f s  %d log sections", noise * 1e3, du1 * 1e3, du2 * 1e3,
            t[k1] - t[n], len(starts),
        )

    # ---- P5: slowest element + U_OCVe, on the rearmost window ------------------
    zarcs = []
    if slowest == "warburg":
        ocv_e, r_d, tau_d, slow_degenerate = _fit_slow_branch(
            t, v, i_arr, t_p, i_pulse, ocv_s, starts[0], n, noise, verbose=verbose,
        )
        slow_contrib = (0.0 if slow_degenerate
                        else i_pulse * warburg_pulse(t, t_p, r_d, tau_d))
    else:
        r_d, tau_d = 0.0, np.nan
        ocv_e, slow_zarc, slow_degenerate = _fit_slow_zarc(
            t, v, i_arr, t_p, i_pulse, ocv_s, starts[0], n, noise, verbose=verbose,
        )
        if slow_zarc is not None:
            zarcs.append(slow_zarc)
            slow_contrib = i_pulse * zarc_pulse(t, t_p, *slow_zarc)
        else:
            slow_contrib = 0.0

    # ---- P6: subtract the identified slowest element (eq. 6). When it was
    # dropped (plateau), there is nothing to subtract and the faster ZARCs are
    # fitted against the full overvoltage.
    resid = v - slow_contrib

    # ---- P7/P8: extend one log section at a time, add a ZARC where warranted
    skipped = 0
    for k_start in starts[1:]:
        # D1 — is there still unexplained overpotential at the window start?
        excess = abs(float(resid[k_start]) - ocv_e)
        if excess < du2:
            skipped += 1
            if verbose:
                logging.info("  P7 skip section t=%.1f s (excess %.3f mV < %.3f mV)",
                             t[k_start] - t[n], excess * 1e3, du2 * 1e3)
            continue
        sec = slice(k_start, len(t))
        tau_ini, r_ini = _initial_guess(
            t, resid, k_start, n, t_p, i_pulse, ocv_e,
            lambda tt, tau: zarc_step(tt, tau, 0.8),
        )

        def _zarc_model(tt, r, tau, alpha, _sec=sec):
            full = ocv_ramp(t, i_arr, ocv_s, ocv_e) + i_pulse * zarc_pulse(
                t, t_p, r, tau, alpha
            )
            return full[_sec]

        try:
            popt, _ = curve_fit(
                _zarc_model, t[sec], resid[sec],
                p0=[r_ini, tau_ini, 0.8],
                bounds=([R_MIN_OHM, 1e-3, ALPHA_MIN],
                        [1.0, 1e5, ALPHA_MAX]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            if verbose:
                logging.info("  P8 fit failed at t=%.1f s", t[k_start] - t[n])
            continue
        r_n, tau_n, alpha_n = (float(x) for x in popt)
        # Reject an element that neither carries a resolvable voltage nor a
        # resolvable time constant: (a) its peak overvoltage must clear dU_fit2
        # (the same significance bar D1 uses -- otherwise it is fitting noise, the
        # source of the railed 0.2-0.5 mOhm micro-ZARCs on flat pulses), and (b)
        # tau must exceed a few pulse-sampling intervals or the exponential is
        # unresolved and R/tau trade off freely.
        contribution = float(np.max(np.abs(i_pulse * zarc_pulse(
            t, t_p, r_n, tau_n, alpha_n))))
        dt_pulse = float(np.median(np.diff(t[:max(n, 2)]))) if n >= 2 else 0.2
        if (r_n <= R_MIN_OHM * 1.01 or contribution < du2
                or tau_n < 3.0 * dt_pulse):
            if verbose:
                logging.info("  P8 reject at t=%.1f s (R=%.3f mOhm tau=%.2f s "
                             "contrib=%.3f mV)", t[k_start] - t[n], r_n * 1e3,
                             tau_n, contribution * 1e3)
            continue
        zarcs.append((r_n, tau_n, alpha_n))
        resid = resid - i_pulse * zarc_pulse(t, t_p, r_n, tau_n, alpha_n)
        if verbose:
            logging.info("  P8: ZARC %d  R=%.2f mOhm  tau=%.2f s  alpha=%.3f",
                         len(zarcs), r_n * 1e3, tau_n, alpha_n)

    # ---- P9: extend into the pulse; fit R_s and refit the fastest ZARC
    r_s_ini = _ohmic_init(t, v, i_arr, n)
    r_s, zarcs = _fit_series_resistance(
        t, v, i_arr, t_p, i_pulse, ocv_s, ocv_e, r_d, tau_d, zarcs, r_s_ini
    )
    if verbose:
        logging.info("  P9: R_s=%.2f mOhm (init %.2f mOhm), %d ZARC(s)",
                     r_s * 1e3, r_s_ini * 1e3, len(zarcs))

    params = {
        "ocv_s": ocv_s, "ocv_e": ocv_e, "r_s": r_s,
        "r_d": r_d, "tau_d": tau_d, "zarcs": zarcs,
        # has_slow gates the *Warburg* term only; in ZARC mode the slow element is
        # a normal ZARC so has_slow stays False (r_d=0/tau_d=NaN).
        "has_slow": slowest == "warburg" and not slow_degenerate,
    }
    info = {
        "noise_mV": noise * 1e3,
        "du_fit1_mV": du1 * 1e3, "du_fit2_mV": du2 * 1e3,
        "k1_t_s": float(t[k1] - t[n]), "n_sections": len(starts),
        "n_sections_skipped": skipped, "i_relax_start": n,
        "r_s_init_ohm": r_s_ini, "slow_degenerate": slow_degenerate,
        "slowest": slowest,
    }
    return params, info


def _fit_slow_branch(t, v, i_arr, t_p, i_pulse, ocv_s, k1, n, noise, *,
                     verbose=False):
    """P5 — fit the slowest element (Warburg) + ``U_OCVe`` on ``[K1, M]``.

    Returns ``(ocv_e, r_d, tau_d, slow_degenerate)``. The slow branch is *dropped*
    (``r_d=0``, ``tau_d=NaN``, ``slow_degenerate=True``) in two cases:

    * **No slow signal.** The relaxation drop across the rearmost window is below
      ``SLOW_SIGNAL_NOISE_MULT * noise``. On the high-SOC OCV plateau the diffusion
      tail is essentially flat, so there is no slow process to identify; fitting a
      Warburg to that flat tail rails it to a meaningless huge ``R_d`` (observed:
      309 mOhm on a 90 %-SOC charge pulse). Dropping it is the honest outcome.
    * **Railed fit.** The Warburg converged to the resistance ceiling, or to a
      ``tau_d`` longer than the observation window can constrain (``> 3x`` the rest
      length) -- unidentifiable, so not trusted.

    ``U_OCVe`` is **pinned to the measured settled tail median**, not fitted. The
    test cell runs a C/2 restore pulse after each test pulse, so every pulse starts
    from the same relaxed voltage (``U_OCVs``) and the ~30 min pause after the test
    pulse relaxes to the post-pulse OCV, which the tail directly measures. Fitting
    ``U_OCVe`` jointly with the slow element (Bruch's approach) is degenerate when
    the SOC-driven OCV drift is small (a low-current charge pulse, ~1.6 mV): the
    two trade off and the slow branch absorbs the OCV move, railing ``R_d``
    (observed: 309 mOhm on a mid-SOC +1.5 A pulse whose tail said the OCV rose
    1.6 mV but the fit reported 0.08 mV). Pinning it to the measured asymptote
    removes that freedom, so the slow branch fits only the *approach shape*.
    """
    # Robust settled-OCV estimate: median over the last ~3 % of the relaxation
    # (min 30 samples). A short 15-sample median is noise-sensitive, and since
    # U_OCVe is now *pinned* to this value a noisy estimate distorts the whole
    # fit; the rest is flat over its last few percent so a wider median is safe.
    tail_n = max(30, (len(v) - n) // 30)
    v_end = float(np.median(v[-tail_n:]))
    window_drop = abs(float(v[k1]) - v_end)
    if window_drop < SLOW_SIGNAL_NOISE_MULT * noise:
        if verbose:
            logging.info("  P5: slow window drop %.3f mV < %.1fx noise -- "
                         "no slow branch (plateau); U_OCVe=%.4f V",
                         window_drop * 1e3, SLOW_SIGNAL_NOISE_MULT, v_end)
        return v_end, 0.0, np.nan, True

    sec = slice(k1, len(t))
    tau_d_ini, r_d_ini = _initial_guess(
        t, v, k1, n, t_p, i_pulse, v_end, lambda tt, tau: _warburg_step(tt, tau),
    )
    rest_len = float(t[-1] - t[n])
    relax_drop = abs(float(v[n]) - v_end)
    tau_d_hi = 1e5
    # OCV ramp is fully determined (U_OCVs from the prior pause, U_OCVe = tail),
    # so only (R_d, tau_d) are fitted here.
    ocv_ramp_fixed = ocv_ramp(t, i_arr, ocv_s, v_end)

    def _warburg_model(tt, r_d, tau_d):
        full = ocv_ramp_fixed + i_pulse * warburg_pulse(t, t_p, r_d, tau_d)
        return full[sec]

    try:
        popt, _ = curve_fit(
            _warburg_model, t[sec], v[sec],
            p0=[r_d_ini, tau_d_ini],
            bounds=([R_MIN_OHM, 1.0], [1.0, tau_d_hi]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        if verbose:
            logging.info("  P5: Warburg fit failed -- no slow branch")
        return v_end, 0.0, np.nan, True

    ocv_e, r_d, tau_d = v_end, float(popt[0]), float(popt[1])
    # A Warburg is unidentifiable / railed when: it hits the resistance ceiling;
    # tau_d exceeds ~1.5x the observation window (a time constant longer than we
    # watch cannot be pinned -- R_d and tau_d then trade off through the early-sqrt
    # regime); or R_d exceeds 3x the *entire* measured polarization (relax_drop /
    # |I|), which is the symptom of exactly that tau_d >> window degeneracy
    # inflating R_d (observed: 309 mOhm on a plateau charge pulse). Any of these
    # drops the slow branch rather than reporting a meaningless value.
    r_d_ceiling = 3.0 * relax_drop / abs(i_pulse) if abs(i_pulse) > 1e-9 else 0.5
    railed = (r_d >= 0.5 or tau_d >= 0.999 * tau_d_hi
              or tau_d > 1.5 * rest_len or r_d > r_d_ceiling)
    if railed:
        if verbose:
            logging.info("  P5: Warburg railed (R_d=%.1f mOhm tau_d=%.0f s, "
                         "window %.0f s) -- dropping slow branch",
                         r_d * 1e3, tau_d, rest_len)
        return v_end, 0.0, np.nan, True
    if verbose:
        logging.info("  P5: U_OCVe=%.4f V  R_d=%.2f mOhm  tau_d=%.1f s",
                     ocv_e, r_d * 1e3, tau_d)
    return ocv_e, r_d, tau_d, False


def _fit_slow_zarc(t, v, i_arr, t_p, i_pulse, ocv_s, k1, n, noise, *,
                   verbose=False):
    """P5 (ZARC mode) — fit the slowest **ZARC** + pin ``U_OCVe`` on ``[K1, M]``.

    Returns ``(ocv_e, (R, tau, alpha) | None, slow_degenerate)``. Same structure
    and plateau/rail guards as ``_fit_slow_branch``, but the slowest element is a
    ZARC whose Mittag-Leffler step response decays cleanly in the tail (unlike the
    Warburg's tiny differenced two-step response), so a modest slow tail no longer
    rails it. The slowest ZARC is dropped (``slow_degenerate=True``, returns
    ``None``) when there is no slow signal above the noise (the genuine high-SOC
    plateau) or when its ``tau`` runs past the observation window.

    ``U_OCVe`` is pinned to the measured settled-tail median exactly as in the
    Warburg path (see ``_fit_slow_branch`` for why).
    """
    tail_n = max(30, (len(v) - n) // 30)
    v_end = float(np.median(v[-tail_n:]))
    window_drop = abs(float(v[k1]) - v_end)
    if window_drop < SLOW_SIGNAL_NOISE_MULT * noise:
        if verbose:
            logging.info("  P5: slow window drop %.3f mV < %.1fx noise -- "
                         "no slow ZARC (plateau); U_OCVe=%.4f V",
                         window_drop * 1e3, SLOW_SIGNAL_NOISE_MULT, v_end)
        return v_end, None, True

    sec = slice(k1, len(t))
    rest_len = float(t[-1] - t[n])
    tau_ini, r_ini = _initial_guess(
        t, v, k1, n, t_p, i_pulse, v_end, lambda tt, tau: zarc_step(tt, tau, 0.9),
    )
    ocv_ramp_fixed = ocv_ramp(t, i_arr, ocv_s, v_end)

    def _zarc_model(tt, r, tau, alpha):
        return (ocv_ramp_fixed + i_pulse * zarc_pulse(t, t_p, r, tau, alpha))[sec]

    try:
        popt, _ = curve_fit(
            _zarc_model, t[sec], v[sec], p0=[r_ini, tau_ini, 0.9],
            bounds=([R_MIN_OHM, 1.0, ALPHA_MIN], [1.0, 1e5, ALPHA_MAX]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        if verbose:
            logging.info("  P5: slow ZARC fit failed -- no slow branch")
        return v_end, None, True

    r_n, tau_n, alpha_n = (float(x) for x in popt)
    # A ZARC with tau past ~1.5x the window is unidentifiable (its decay is not
    # observed), same idea as the Warburg tau guard.
    if r_n <= R_MIN_OHM * 1.01 or tau_n > 1.5 * rest_len:
        if verbose:
            logging.info("  P5: slow ZARC unidentifiable (R=%.2f mOhm tau=%.0f s, "
                         "window %.0f s) -- dropping", r_n * 1e3, tau_n, rest_len)
        return v_end, None, True
    if verbose:
        logging.info("  P5: U_OCVe=%.4f V  slow ZARC R=%.2f mOhm tau=%.1f s "
                     "alpha=%.3f", v_end, r_n * 1e3, tau_n, alpha_n)
    return v_end, (r_n, tau_n, alpha_n), False


def _ohmic_init(t, v, i_arr, n):
    """``R_s,ini`` from the current interruption by Ohm's law (Bruch eq. 9)."""
    d_i = float(i_arr[n - 1]) - float(i_arr[n])
    if abs(d_i) < 1e-6:
        return 1e-3
    return abs((float(v[n - 1]) - float(v[n])) / d_i)


def _fit_series_resistance(t, v, i_arr, t_p, i_pulse, ocv_s, ocv_e, r_d, tau_d,
                           zarcs, r_s_ini):
    """P9 — fit ``R_s`` together with the fastest ZARC over pulse + relaxation.

    The paper refits the last (lowest-tau) element here because the instantaneous
    ohmic step and the fastest element's rise are entangled: neither is separable
    from the relaxation alone, and only the pulse part carries the ohmic step.

    ``R_s`` is *bounded* to ``[0.5, 1.5] x R_s,ini`` rather than ``[0, 1]``. The
    Ohm's-law jump across the current interruption is a direct, reliable ohmic
    measurement, whereas the fastest element (tau of a few seconds) nearly
    saturates within a 20 s pulse and mimics a step — so with a free lower bound
    the two swap and R_s collapses to 0 (observed). The band keeps R_s anchored to
    the measured jump while still letting the entangled part refine.
    """
    r_s_lo, r_s_hi = 0.5 * r_s_ini, 1.5 * r_s_ini
    has_slow = r_d > 0 and np.isfinite(tau_d)
    warb = i_pulse * warburg_pulse(t, t_p, r_d, tau_d) if has_slow else 0.0

    if not zarcs:
        def _model_rs(tt, r_s):
            return ocv_ramp(t, i_arr, ocv_s, ocv_e) + i_arr * r_s + warb
        try:
            popt, _ = curve_fit(_model_rs, t, v, p0=[r_s_ini],
                                bounds=([r_s_lo], [r_s_hi]), maxfev=20000)
            return float(popt[0]), zarcs
        except (RuntimeError, ValueError):
            return r_s_ini, zarcs

    fixed = zarcs[:-1]
    r_f, tau_f, alpha_f = zarcs[-1]
    base = (
        ocv_ramp(t, i_arr, ocv_s, ocv_e) + warb
        + sum(i_pulse * zarc_pulse(t, t_p, r, tau, a) for r, tau, a in fixed)
    )

    def _model(tt, r_s, r, tau, alpha):
        return base + i_arr * r_s + i_pulse * zarc_pulse(t, t_p, r, tau, alpha)

    try:
        popt, _ = curve_fit(
            _model, t, v, p0=[r_s_ini, r_f, tau_f, alpha_f],
            bounds=([r_s_lo, R_MIN_OHM, 1e-3, ALPHA_MIN],
                    [r_s_hi, 1.0, 1e5, ALPHA_MAX]),
            maxfev=30000,
        )
    except (RuntimeError, ValueError):
        return r_s_ini, zarcs
    r_s = float(popt[0])
    return r_s, fixed + [(float(popt[1]), float(popt[2]), float(popt[3]))]


# ---------------------------------------------------------------------------
# Forward model (numpy) + parameter packing
# ---------------------------------------------------------------------------
def _has_slow(params):
    """Whether a slow (Warburg) branch is present and identified."""
    return bool(params.get("has_slow", params["r_d"] > 0
                           and np.isfinite(params["tau_d"])))


def simulate(t, i_arr, t_p, i_pulse, params):
    """Terminal voltage of the fractional ECM over the whole window."""
    v = ocv_ramp(t, i_arr, params["ocv_s"], params["ocv_e"])
    v = v + i_arr * params["r_s"]
    if _has_slow(params):
        v = v + i_pulse * warburg_pulse(t, t_p, params["r_d"], params["tau_d"])
    for r, tau, alpha in params["zarcs"]:
        v = v + i_pulse * zarc_pulse(t, t_p, r, tau, alpha)
    return v


def _pack(params):
    """Flatten to an unconstrained vector: log for R/tau, logit for alpha.

    ``U_OCVe`` is **not** in the vector — it is pinned to the measured relaxation
    tail (see ``_fit_slow_branch``) and passed to the model as a constant, so P10
    cannot reopen the OCV-vs-slow-branch degeneracy. ``r_d``/``tau_d`` always
    occupy a slot for a fixed-length vector, but a dropped slow branch
    (``has_slow`` False) is carried by the non-parameter ``slow_on`` mask in the
    models — finite sentinels are packed and P10 leaves them untouched.
    """
    has_slow = _has_slow(params)
    r_d = params["r_d"] if has_slow else 1e-6
    tau_d = params["tau_d"] if has_slow else 1.0
    x = [np.log(max(params["r_s"], 1e-6)),
         np.log(max(r_d, 1e-6)),
         np.log(max(tau_d, 1e-6))]
    for r, tau, alpha in params["zarcs"]:
        a = np.clip((alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN), 1e-4, 1 - 1e-4)
        x += [np.log(max(r, 1e-6)), np.log(max(tau, 1e-6)), np.log(a / (1 - a))]
    return np.asarray(x, dtype=float)


def _unpack(x, n_zarc, ocv_s, ocv_e, has_slow=True, xp=np):
    """Inverse of ``_pack``; ``xp`` is numpy or jax.numpy. ``ocv_e`` is the pinned
    (non-parameter) end-of-relaxation OCV.

    ``has_slow=False`` reports the slow branch as absent (``r_d=0``, ``tau_d=NaN``)
    regardless of the packed sentinels.
    """
    r_s = xp.exp(x[0])
    r_d = xp.exp(x[1]) if has_slow else 0.0
    tau_d = xp.exp(x[2]) if has_slow else np.nan
    zarcs = []
    for k in range(n_zarc):
        r = xp.exp(x[3 + 3 * k])
        tau = xp.exp(x[4 + 3 * k])
        a = 1.0 / (1.0 + xp.exp(-x[5 + 3 * k]))
        zarcs.append((r, tau, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * a))
    return {"ocv_s": ocv_s, "ocv_e": ocv_e, "r_s": r_s,
            "r_d": r_d, "tau_d": tau_d, "zarcs": zarcs, "has_slow": has_slow}


# ---------------------------------------------------------------------------
# P10: joint refit with exact gradients
# ---------------------------------------------------------------------------
def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "P10 needs JAX for exact gradients. Install with "
            "`pip install 'jax[cpu]'`, or pass --no-p10 to stop after P9."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _jax_ml_neg(jnp, alpha, x):
    """``E_alpha(-x)`` on the same fixed quadrature, differentiable in alpha."""
    u = jnp.asarray(_ML_U)
    w = jnp.asarray(_ML_W)
    g = (jnp.sin(alpha * jnp.pi) / (2.0 * jnp.pi)) / (
        jnp.cosh(alpha * u) + jnp.cos(alpha * jnp.pi)
    )
    weights = w * g
    s = jnp.clip(x, 1e-300, None) ** (1.0 / alpha)
    return weights @ jnp.exp(-jnp.outer(jnp.exp(-u), s))


def _jax_zarc_step(jnp, t, tau, alpha):
    tt = jnp.clip(t, 1e-12, None)
    val = 1.0 - _jax_ml_neg(jnp, alpha, (tt / tau) ** alpha)
    return jnp.where(t > 0, val, 0.0)


def _jax_warburg_step(jnp, t, tau_d, n_terms=20):
    tt = jnp.clip(t, 0.0, None)
    m = jnp.arange(1, 2 * n_terms, 2, dtype=float)
    acc = (jnp.exp(-(m[:, None] ** 2) * jnp.pi ** 2 * tt[None, :]
                   / (4.0 * tau_d)) / m[:, None] ** 2).sum(axis=0)
    return jnp.where(t > 0, 1.0 - (8.0 / jnp.pi ** 2) * acc, 0.0)


def _jax_model_ml(jnp, x, n_zarc, t, i_arr, q_frac, t_p, i_pulse, ocv_s, ocv_e,
                  slow_on=1.0):
    # unpack with has_slow=True so tau_d stays finite (the packed sentinel); the
    # slow_on mask, not a NaN, is what removes a dropped slow branch.
    p = _unpack(x, n_zarc, ocv_s, ocv_e, has_slow=True, xp=jnp)
    v = ocv_s + (ocv_e - ocv_s) * q_frac + i_arr * p["r_s"]
    wb = _jax_warburg_step(jnp, t, p["tau_d"]) - jnp.where(
        t > t_p, _jax_warburg_step(jnp, t - t_p, p["tau_d"]), 0.0
    )
    v = v + slow_on * i_pulse * p["r_d"] * wb
    for r, tau, alpha in p["zarcs"]:
        s = _jax_zarc_step(jnp, t, tau, alpha) - jnp.where(
            t > t_p, _jax_zarc_step(jnp, t - t_p, tau, alpha), 0.0
        )
        v = v + i_pulse * r * s
    return v


# --- Grunwald-Letnikov kernel (alternative P10 route) ------------------------
# Short-memory length for the GL history sum. The binomial weights decay like
# k^{-1-alpha}, so truncating is the standard "fixed memory principle"; 512 terms
# at the resampled step keep the truncation error well under the voltage noise.
GL_MEMORY = 512


def _gl_weights(jnp, alpha, m=GL_MEMORY):
    """Grunwald-Letnikov binomial weights ``w_k`` of fractional order ``alpha``.

    ``w_0 = 1``, ``w_k = w_{k-1} (1 - (alpha+1)/k)``. Built with a cumulative
    product so the recursion is differentiable in ``alpha``.
    """
    k = jnp.arange(1, m, dtype=float)
    return jnp.concatenate([jnp.ones(1), jnp.cumprod(1.0 - (alpha + 1.0) / k)])


def _gl_zarc(jnp, lax, i_seq, h, r, tau, alpha, m=GL_MEMORY):
    """Integrate ``u + tau^alpha D^alpha u = R I`` by Grunwald-Letnikov recursion.

    Discretizing ``D^alpha u|_n ~ h^-alpha sum_k w_k u_{n-k}`` and solving for the
    new sample gives::

        u_n = (R I_n - c * sum_{k>=1} w_k u_{n-k}) / (1 + c),   c = (tau/h)^alpha

    Runs on a uniform grid (hence the resampling in ``refit_joint``) and carries
    a rolling ``m``-sample history. This is the route the paper's integer-order
    RC solves in closed form; for a fractional element the history term is what
    makes the element "remember", so it cannot be dropped.
    """
    w = _gl_weights(jnp, alpha, m)
    c = (tau / h) ** alpha
    w_hist = w[1:]

    def step(buf, i_n):
        u_n = (r * i_n - c * (w_hist @ buf)) / (1.0 + c)
        return jnp.concatenate([u_n[None], buf[:-1]]), u_n

    _, out = lax.scan(step, jnp.zeros(m - 1), i_seq)
    return out


def _jax_model_gl(jnp, lax, x, n_zarc, t_u, i_u, q_frac_u, h, t_p, i_pulse,
                  ocv_s, ocv_e, slow_on=1.0):
    """Same model as ``_jax_model_ml`` but with GL-integrated ZARCs."""
    p = _unpack(x, n_zarc, ocv_s, ocv_e, has_slow=True, xp=jnp)
    v = ocv_s + (ocv_e - ocv_s) * q_frac_u + i_u * p["r_s"]
    wb = _jax_warburg_step(jnp, t_u, p["tau_d"]) - jnp.where(
        t_u > t_p, _jax_warburg_step(jnp, t_u - t_p, p["tau_d"]), 0.0
    )
    v = v + slow_on * i_pulse * p["r_d"] * wb
    for r, tau, alpha in p["zarcs"]:
        v = v + _gl_zarc(jnp, lax, i_u, h, r, tau, alpha)
    return v


def refit_joint(t, v, i_arr, t_p, i_pulse, params, *, kernel="ml", verbose=False):
    """P10 — joint refit of every parameter with exact gradients + L-BFGS-B.

    The cost is evaluated on the **relaxation only** (Bruch eq. 16): during the
    pulse the true OCV path is unknown and eq. 1 replaces it with a straight
    line, so including those samples would push that known interpolation error
    into the element parameters instead of leaving it in the residual.

    ``kernel='ml'`` differentiates the exact Mittag-Leffler quadrature on the
    measured (non-uniform) time base. ``kernel='gl'`` resamples onto a uniform
    grid and differentiates the Grunwald-Letnikov recursion instead.
    """
    jax, jnp = _require_jax()
    from jax.scipy.optimize import minimize as jminimize   # noqa: F401  (probe)

    n_zarc = len(params["zarcs"])
    ocv_s = params["ocv_s"]
    ocv_e = params["ocv_e"]                           # pinned, not optimized
    has_slow = _has_slow(params)
    slow_on = 1.0 if has_slow else 0.0
    x0 = _pack(params)

    q = np.concatenate(([0.0], np.cumsum(np.diff(t) * i_arr[:-1])))
    q_frac = q / q[-1] if abs(q[-1]) > 1e-12 else np.zeros_like(q)
    mask = t > t_p                                   # relaxation-only cost

    if kernel == "ml":
        tj, ij, qj, vj = (jnp.asarray(a) for a in (t, i_arr, q_frac, v))
        mj = jnp.asarray(mask)

        def loss(x):
            pred = _jax_model_ml(jnp, x, n_zarc, tj, ij, qj, t_p, i_pulse, ocv_s,
                                 ocv_e, slow_on)
            r = jnp.where(mj, pred - vj, 0.0)
            return jnp.sum(r ** 2) / jnp.sum(mj)
    elif kernel == "gl":
        from jax import lax
        h = float(np.median(np.diff(t[t <= t_p]))) if np.any(t <= t_p) else 0.2
        h = max(h, 1e-3)
        t_u = np.arange(t[0], t[-1] + h, h)
        i_u = np.interp(t_u, t, i_arr)
        v_u = np.interp(t_u, t, v)
        q_u = np.concatenate(([0.0], np.cumsum(np.diff(t_u) * i_u[:-1])))
        qf_u = q_u / q_u[-1] if abs(q_u[-1]) > 1e-12 else np.zeros_like(q_u)
        mask_u = t_u > t_p
        tj, ij, qj, vj = (jnp.asarray(a) for a in (t_u, i_u, qf_u, v_u))
        mj = jnp.asarray(mask_u)
        if verbose:
            logging.info("  P10(gl): resampled %d -> %d samples at h=%.3f s",
                         len(t), len(t_u), h)

        def loss(x):
            pred = _jax_model_gl(jnp, lax, x, n_zarc, tj, ij, qj, h, t_p,
                                 i_pulse, ocv_s, ocv_e, slow_on)
            r = jnp.where(mj, pred - vj, 0.0)
            return jnp.sum(r ** 2) / jnp.sum(mj)
    else:
        raise ValueError(f"unknown P10 kernel {kernel!r} (expected 'ml' or 'gl')")

    val_grad = jax.jit(jax.value_and_grad(loss))

    def fun(x):
        f, g = val_grad(jnp.asarray(x))
        return float(f), np.asarray(g, dtype=float)

    from scipy.optimize import minimize
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12})
    refit = _unpack(res.x, n_zarc, ocv_s, ocv_e, has_slow=has_slow, xp=np)
    refit["zarcs"] = [(float(r), float(tau), float(a))
                      for r, tau, a in refit["zarcs"]]
    refit = {k: (val if k in ("zarcs", "has_slow")
                 else float(val)) for k, val in refit.items()}
    # order slow -> fast, matching the identification order
    refit["zarcs"].sort(key=lambda z: -z[1])
    if verbose:
        logging.info("  P10(%s): %d L-BFGS iters, cost %.3e -> %.3e",
                     kernel, res.nit, float(val_grad(jnp.asarray(x0))[0]),
                     float(res.fun))
    return refit, res


# ---------------------------------------------------------------------------
# Per-pulse driver
# ---------------------------------------------------------------------------
def _rmse_mv(t, v, i_arr, t_p, i_pulse, params, relax_only=True):
    pred = simulate(t, i_arr, t_p, i_pulse, params)
    m = (t > t_p) if relax_only else np.ones_like(t, dtype=bool)
    return float(np.sqrt(np.mean((pred[m] - v[m]) ** 2)) * 1e3)


def fit_one_pulse_frac(window, t_p, i_pulse, pre_rest_v, *, n_max=N_MAX,
                       d_fit1=D_FIT1, d_fit2=D_FIT2, slowest="zarc", p10=True,
                       kernel="ml", verbose=False):
    """Full P1-P10 identification for one pulse window. Returns a result dict."""
    t = (window["Time"] - window["Time"].iloc[0]).dt.total_seconds().to_numpy()
    v = window["Voltage"].to_numpy(dtype=float)
    i_arr = window["Current"].to_numpy(dtype=float)
    if len(t) < 20 or t_p <= 0:
        return None

    # U_OCVs: the settled voltage of the pause *before* the pulse. Without one,
    # the ramp collapses to a constant OCV (which is what a 2RC fit assumes).
    ocv_s = float(pre_rest_v) if pre_rest_v is not None else float(v[-1])

    params, info = decompose_pulse(
        t, v, i_arr, t_p, i_pulse, ocv_s,
        n_max=n_max, d_fit1=d_fit1, d_fit2=d_fit2, slowest=slowest, verbose=verbose,
    )
    if params is None:
        return None
    rmse_p9 = _rmse_mv(t, v, i_arr, t_p, i_pulse, params)

    params_final, rmse_p10 = params, np.nan
    if p10:
        try:
            params_final, _ = refit_joint(
                t, v, i_arr, t_p, i_pulse, params, kernel=kernel, verbose=verbose,
            )
            rmse_p10 = _rmse_mv(t, v, i_arr, t_p, i_pulse, params_final)
            if not np.isfinite(rmse_p10) or rmse_p10 > rmse_p9:
                # a refit that made things worse is discarded, not trusted
                logging.info("  P10 did not improve (%.3f -> %.3f mV); keeping P9",
                             rmse_p9, rmse_p10)
                params_final, rmse_p10 = params, np.nan
        except Exception as exc:                       # noqa: BLE001
            logging.warning("  P10 refit failed: %s", exc)

    zarcs = params_final["zarcs"]
    out = {
        "U_OCVs_V": round(params_final["ocv_s"], 4),
        "U_OCVe_V": round(params_final["ocv_e"], 4),
        "dOCV_pulse_mV": round(
            (params_final["ocv_e"] - params_final["ocv_s"]) * 1e3, 2
        ),
        "R_s_ohm": round(params_final["r_s"], 5),
        "R_d_ohm": round(params_final["r_d"], 5),
        "tau_d_s": round(params_final["tau_d"], 2),
        "n_zarc": len(zarcs),
        "rmse_P9_mV": round(rmse_p9, 4),
        "rmse_P10_mV": round(rmse_p10, 4) if np.isfinite(rmse_p10) else np.nan,
        "rmse_mV": round(min(rmse_p9, rmse_p10) if np.isfinite(rmse_p10)
                         else rmse_p9, 4),
        "R_total_ohm": round(
            params_final["r_s"] + params_final["r_d"]
            + sum(z[0] for z in zarcs), 5,
        ),
        **{f"R{k+1}_ohm": round(z[0], 5) for k, z in enumerate(zarcs)},
        **{f"tau{k+1}_s": round(z[1], 4) for k, z in enumerate(zarcs)},
        **{f"alpha{k+1}": round(z[2], 4) for k, z in enumerate(zarcs)},
        **{f"{k}": round(val, 4) if isinstance(val, float) else val
           for k, val in info.items()},
        "n_points": int(len(t)),
        "_t": t, "_v": v, "_vfit": simulate(t, i_arr, t_p, i_pulse, params_final),
        "_params": params_final,
    }
    return out


def fit_frac(labeled, seg_ids, nom_capacity, **kw):
    """Fit every selected pulse segment. Mirrors ``fit_2rc_pulse.fit_2rc``."""
    df = labeled.sort_values("Time").reset_index(drop=True)
    seg_bounds = df.groupby("pulse_segment_id")["Time"].agg(["min", "max"])
    rows, curves = [], []
    for seg_id in seg_ids:
        pulse_rows = df[df["pulse_segment_id"] == seg_id].sort_values("Time")
        if pulse_rows.empty:
            continue
        cur_id = pulse_rows["ID"].iloc[0]
        relax_rows = df[df["ID"] == _proc_id(cur_id, +1)].sort_values("Time")
        if relax_rows.empty:
            logging.info("skip pulse %s: no following pause", cur_id)
            continue
        i_pulse = float(pulse_rows["Current"].mean())
        if abs(i_pulse) < REST_CURRENT_A:
            continue
        t_p = (seg_bounds.loc[seg_id, "max"] - seg_bounds.loc[seg_id, "min"]).total_seconds()
        window = pd.concat([pulse_rows, relax_rows]).sort_values("Time")
        pre_rows = df[(df["ID"] == _proc_id(cur_id, -1))
                      & (df["Current"].abs() < REST_CURRENT_A)]
        pre_rest_v = float(pre_rows["Voltage"].iloc[-1]) if not pre_rows.empty else None

        logging.info("pulse %s  I=%.2f A  t_p=%.1f s  (%d samples)",
                     cur_id, i_pulse, t_p, len(window))
        res = fit_one_pulse_frac(window, t_p, i_pulse, pre_rest_v, **kw)
        if res is None:
            continue
        curves.append((seg_id, res.pop("_t"), res.pop("_v"), res.pop("_vfit")))
        res.pop("_params")
        meta = pulse_rows.iloc[0]
        rows.append({
            "File": meta.get("File", ""),
            "SOH": meta.get("SOH", ""),
            "SOC": meta.get("SOC", ""),
            "pulse_segment_id": seg_id,
            "ID": cur_id,
            "direction": "CHA" if i_pulse > 0 else "DCH",
            "I_A": round(i_pulse, 3),
            "C_rate": round(i_pulse / nom_capacity, 3),
            "pulse_dur_s": round(t_p, 2),
            **res,
        })
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fits(curves, results, out_png):
    """Overlay measured vs fitted voltage (log-time, the axis the method works in)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    if not n:
        return
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1] * nrow})
    for k, (seg_id, t, v, vfit) in enumerate(curves):
        r, c = divmod(k, ncol)
        ax, axr = axes[2 * r][c], axes[2 * r + 1][c]
        ax.plot(t, v * 1e3, ".", ms=2, label="measured", color="0.35")
        ax.plot(t, vfit * 1e3, "-", lw=1.3, label="fractional ECM", color="C3")
        ax.set_xscale("symlog", linthresh=1.0)
        row = results[results["pulse_segment_id"] == seg_id]
        ttl = seg_id if row.empty else (
            f"{seg_id} — {int(row['n_zarc'].iloc[0])} ZARC, "
            f"RMSE {row['rmse_mV'].iloc[0]:.3f} mV"
        )
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("U [mV]")
        ax.legend(fontsize=7)
        axr.plot(t, (vfit - v) * 1e3, "-", lw=0.8, color="C0")
        axr.axhline(0, color="0.6", lw=0.6)
        axr.set_xscale("symlog", linthresh=1.0)
        axr.set_ylabel("res [mV]", fontsize=8)
        axr.set_xlabel("t since pulse start [s]", fontsize=8)
    for k in range(n, nrow * ncol):
        r, c = divmod(k, ncol)
        axes[2 * r][c].axis("off")
        axes[2 * r + 1][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    logging.info("plot -> %s", out_png)


# tau-decade bins (seconds): each ZARC is grouped by which decade its time
# constant falls in, giving a consistent physical identity ("the ~1 s process",
# "the ~100 s process") across pulses regardless of how many elements each one
# resolved -- unlike an element *index*, which is not comparable once the slow
# branch is dropped on some pulses and kept on others.
TAU_DECADE_EDGES = [0.3, 3.0, 30.0, 300.0, 3000.0, 30000.0]
TAU_DECADE_LABELS = ["~1 s", "~10 s", "~100 s", "~1000 s", "~10⁴ s"]


def _iter_zarcs(row):
    """Yield ``(R, tau, alpha)`` for every resolved ZARC in a results row."""
    n = int(row.get("n_zarc", 0) or 0)
    for k in range(1, n + 1):
        r, tau, a = row.get(f"R{k}_ohm"), row.get(f"tau{k}_s"), row.get(f"alpha{k}")
        if pd.notna(r) and pd.notna(tau):
            yield float(r), float(tau), float(a)


def _decade_series(sub, value="R"):
    """Per-tau-decade {SOH: value} tables. ``value`` is 'R' or 'alpha'."""
    cols = {i: {} for i in range(len(TAU_DECADE_LABELS))}
    for _, row in sub.iterrows():
        soh = row["SOH_num"]
        for r, tau, a in _iter_zarcs(row):
            b = int(np.clip(np.searchsorted(TAU_DECADE_EDGES, tau) - 1,
                            0, len(TAU_DECADE_LABELS) - 1))
            # keep the largest-R element if two land in one decade for one pulse
            v = r * 1e3 if value == "R" else a
            if soh not in cols[b] or (value == "R" and v > cols[b][soh]):
                cols[b][soh] = v
    return cols


def plot_vs_soh(results, out_png, title=""):
    """Fractional-ECM parameters vs SOH: CHA/DCH split, tau-decade binned.

    Layout: two columns (charge | discharge), three rows:

    * **Headline resistances** -- ``R_s``, ``R_total`` and the slow-branch ``R_d``
      (only where the Warburg was identified; ``slow_degenerate`` rows are hidden
      so the plateau's non-identifiable slow branch does not inject spikes).
    * **ZARC R by tau-decade** -- one trace per decade, so a given physical
      process is one line even though pulses resolve different element counts.
    * **ZARC alpha by tau-decade** -- the depression of each process's arc.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "SOH_num" not in results.columns or results.empty:
        return
    slow_ok = ~results.get("slow_degenerate", pd.Series(False, index=results.index)) \
        .astype(str).str.lower().isin(["true", "1"])

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    cmap = plt.get_cmap("viridis")
    dec_colors = [cmap(i / (len(TAU_DECADE_LABELS) - 1))
                  for i in range(len(TAU_DECADE_LABELS))]

    for col, direction in enumerate(("CHA", "DCH")):
        sub = results[results["direction"] == direction].sort_values("SOH_num")
        a0, a1, a2 = axes[0][col], axes[1][col], axes[2][col]
        if sub.empty:
            for ax in (a0, a1, a2):
                ax.axis("off")
            continue

        # row 0: robust resistances
        a0.plot(sub["SOH_num"], sub["R_s_ohm"] * 1e3, "o-", color="C0",
                label="R_s", ms=4)
        a0.plot(sub["SOH_num"], sub["R_total_ohm"] * 1e3, "s--", color="0.4",
                label="R_total", ms=3)
        slow = sub[slow_ok.loc[sub.index] & sub["R_d_ohm"].gt(0)]
        if not slow.empty:
            a0.plot(slow["SOH_num"], slow["R_d_ohm"] * 1e3, "D:", color="C3",
                    label="R_d (Warburg, identified)", ms=4)
        a0.set_title(f"{direction}", fontsize=11)
        a0.set_ylabel("R [mΩ]")

        # rows 1-2: ZARC R and alpha, binned by tau-decade
        for value, ax in (("R", a1), ("alpha", a2)):
            series = _decade_series(sub, value)
            for b, lab in enumerate(TAU_DECADE_LABELS):
                if not series[b]:
                    continue
                xs = sorted(series[b])
                ys = [series[b][x] for x in xs]
                ax.plot(xs, ys, "o-", color=dec_colors[b], ms=4,
                        label=f"tau {lab}")
        a1.set_ylabel("ZARC R [mΩ]")
        a2.set_ylabel("ZARC alpha [-]")
        a2.set_xlabel("SOH [%]")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.3)
            if ax.has_data():
                ax.legend(fontsize=7, ncol=2)
    axes[0][0].invert_xaxis()      # shared x: SOH high -> low
    fig.suptitle(f"Fractional ECM (Bruch decomposition) vs SOH — {title}\n"
                 "R_s / R_total headline; ZARCs binned by tau-decade; R_d shown "
                 "only where a Warburg was identified (--slowest warburg)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    logging.info("plot -> %s", out_png)


# ---------------------------------------------------------------------------
# Folder mode
# ---------------------------------------------------------------------------
def fit_folder(folder, nom_capacity, remove_before_min, exclude_zc, **kw):
    files = sorted(glob.glob(os.path.join(folder, "*_pulse_*SOH.parquet")))
    logging.info("folder mode: %d pulse files", len(files))
    frames = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        labeled = label_time_diff(pd.read_parquet(path), os.path.basename(path))
        seg_ids = select_pulse_segments(labeled, remove_before_min, exclude_zc)
        res, _ = fit_frac(labeled, seg_ids, nom_capacity, **kw)
        if res.empty:
            continue
        res["SOH_num"] = _parse_soh(stem)
        res["BM_Programm"] = labeled["BM_Programm"].iloc[0]
        frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Self-test (mirrors the paper's own verification, their §3.1)
# ---------------------------------------------------------------------------
def self_test(kernel_check=True):
    """Validate the kernels, then round-trip a synthetic cell with known values."""
    from scipy.special import gamma

    print("\n=== 1. Mittag-Leffler quadrature vs Taylor series ===")
    ok = True
    for a in (0.4, 0.5, 0.7, 0.9, 0.95, 0.99):
        xs = np.array([1e-6, 1e-3, 0.05, 0.3, 1.0, 2.0])
        quad = _ml_neg(a, xs)
        ser = sum((-xs) ** k / gamma(a * k + 1) for k in range(60))
        norm = float(np.sum(_ML_W * (np.sin(a * np.pi) / (2 * np.pi))
                            / (np.cosh(a * _ML_U) + np.cos(a * np.pi))))
        rel = float(np.max(np.abs(quad - ser) / np.abs(ser)))
        ok &= rel < 2e-4 and abs(norm - 1) < 5e-6
        print(f"  alpha={a:<5} |norm-1|={abs(norm-1):.2e}  max rel err={rel:.2e}")
    print("  ->", "PASS" if ok else "FAIL")

    print("\n=== 2. alpha=1 ZARC must equal a plain RC ===")
    t = np.linspace(0, 200, 400)
    z = zarc_step(t, 30.0, ALPHA_MAX)
    rc = 1 - np.exp(-t / 30.0)
    err = float(np.max(np.abs(z - rc)))
    print(f"  max |ZARC(alpha->1) - RC| = {err:.2e}  ->",
          "PASS" if err < 5e-3 else "FAIL")

    if kernel_check:
        print("\n=== 3. Grunwald-Letnikov recursion vs exact Mittag-Leffler ===")
        try:
            jax, jnp = _require_jax()
            from jax import lax
            h, t_p = 0.05, 20.0
            t_u = np.arange(0.0, 400.0, h)
            i_u = np.where(t_u <= t_p, 1.0, 0.0)
            for a, tau, r in ((1.0 - 1e-6, 10.0, 0.01), (0.8, 10.0, 0.01),
                              (0.6, 30.0, 0.02)):
                gl = np.asarray(_gl_zarc(jnp, lax, jnp.asarray(i_u), h,
                                         r, tau, a))
                ml = zarc_pulse(t_u, t_p, r, tau, a)
                # GL is O(h^1); compare where both are resolved (skip t < 5h)
                m = t_u > 5 * h
                err = float(np.max(np.abs(gl[m] - ml[m]))) * 1e3
                rel = err / (abs(r) * 1e3)
                print(f"  alpha={a:<8.3f} tau={tau:<5} max|GL-ML|={err:.4f} mV "
                      f"({rel*100:.2f} % of R*I)  ->",
                      "PASS" if rel < 0.05 else "FAIL")
        except ImportError as exc:
            print("  skipped:", exc)

    print("\n=== 4. Synthetic cell round-trip (paper §3.1) ===")
    rng = np.random.default_rng(0)
    truth = {
        "ocv_s": 3.700, "ocv_e": 3.688, "r_s": 0.0180,
        "r_d": 0.0090, "tau_d": 900.0,
        "zarcs": [(0.0060, 60.0, 0.75), (0.0035, 3.0, 0.85)],
    }
    t_p = 20.0
    t = np.concatenate([np.arange(0, t_p, 0.2), np.arange(t_p, 1800, 0.8)])
    i_pulse = -1.5
    i_arr = np.where(t <= t_p, i_pulse, 0.0)
    v_true = simulate(t, i_arr, t_p, i_pulse, truth)
    v = v_true + rng.normal(0, 5e-4, size=len(t))     # paper's +-0.5 mV noise

    params, info = decompose_pulse(t, v, i_arr, t_p, i_pulse, truth["ocv_s"],
                                   verbose=True)
    if params is None:
        print("  -> FAIL (decomposition returned nothing)")
        return
    rmse9 = _rmse_mv(t, v, i_arr, t_p, i_pulse, params)
    print(f"  after P9 : {len(params['zarcs'])} ZARC(s), RMSE {rmse9:.4f} mV")
    try:
        refit, _ = refit_joint(t, v, i_arr, t_p, i_pulse, params, verbose=True)
        rmse10 = _rmse_mv(t, v, i_arr, t_p, i_pulse, refit)
        print(f"  after P10: RMSE {rmse10:.4f} mV  (noise floor 0.5 mV)")
        best = refit if rmse10 < rmse9 else params
    except ImportError as exc:
        print("  P10 skipped:", exc)
        best = params
    print(f"  truth  R_s={truth['r_s']*1e3:.2f} R_d={truth['r_d']*1e3:.2f} "
          f"tau_d={truth['tau_d']:.0f} "
          + " ".join(f"[R={r*1e3:.2f} tau={tt:.1f} a={a:.2f}]"
                     for r, tt, a in truth["zarcs"]))
    print(f"  fitted R_s={best['r_s']*1e3:.2f} R_d={best['r_d']*1e3:.2f} "
          f"tau_d={best['tau_d']:.0f} "
          + " ".join(f"[R={r*1e3:.2f} tau={tt:.1f} a={a:.2f}]"
                     for r, tt, a in best["zarcs"]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Bruch-2021 sequential decomposition with a fractional-order ECM."
    )
    ap.add_argument("parquet", nargs="?", help="pulse parquet or folder")
    ap.add_argument("--nom-capacity", type=float, default=3.0, help="Ah (VTC6=3.0)")
    ap.add_argument("--n-max", type=int, default=N_MAX,
                    help="number of logarithmic sections = max element count")
    ap.add_argument("--d-fit1", type=float, default=D_FIT1,
                    help="dU_fit1 = relaxation drop / this")
    ap.add_argument("--d-fit2", type=float, default=D_FIT2,
                    help="dU_fit2 = relaxation drop / this (element threshold)")
    ap.add_argument("--slowest", choices=("zarc", "warburg"), default="zarc",
                    help="slowest element: 'zarc' (default, pure ZARC ladder) or "
                         "'warburg' (finite-length Warburg, Bruch's original). "
                         "ZARC avoids the Warburg tail-degeneracy railing.")
    ap.add_argument("--p10-kernel", choices=("ml", "gl"), default="ml",
                    help="P10 forward kernel: exact Mittag-Leffler (default) or "
                         "Grunwald-Letnikov recursion on a resampled grid")
    ap.add_argument("--no-p10", action="store_true", help="stop after P9")
    ap.add_argument("--remove-pulse-before-min", type=float,
                    default=REMOVE_PULSE_BEFORE_MIN)
    ap.add_argument("--exclude-zc", nargs="*", default=EXCLUDE_ZUSTAND_CURRENT)
    ap.add_argument("-o", "--out", help="output CSV")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the kernels and round-trip a synthetic cell")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log every decomposition step")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.parquet:
        ap.error("a pulse parquet/folder is required (or use --self-test)")

    kw = dict(n_max=args.n_max, d_fit1=args.d_fit1, d_fit2=args.d_fit2,
              slowest=args.slowest, p10=not args.no_p10, kernel=args.p10_kernel,
              verbose=args.verbose)

    if os.path.isdir(args.parquet):
        results = fit_folder(args.parquet, args.nom_capacity,
                             args.remove_pulse_before_min, args.exclude_zc, **kw)
        if results.empty:
            logging.warning("no pulses fit in %s", args.parquet)
            return
        out_csv = args.out or os.path.join(args.parquet, "frac_vs_SOH.csv")
        results.to_csv(out_csv, index=False)
        logging.info("results -> %s", out_csv)
        plot_vs_soh(results, os.path.join(args.parquet, "frac_vs_SOH.png"),
                    title=os.path.basename(os.path.normpath(args.parquet)))
        pd.set_option("display.width", 220, "display.max_columns", 40)
        cols = [c for c in ("BM_Programm", "SOH_num", "direction", "R_s_ohm",
                            "R_d_ohm", "tau_d_s", "n_zarc", "rmse_mV")
                if c in results.columns]
        print(results[cols].to_string(index=False))
        return

    labeled = label_time_diff(pd.read_parquet(args.parquet),
                              os.path.basename(args.parquet))
    seg_ids = select_pulse_segments(labeled, args.remove_pulse_before_min,
                                    args.exclude_zc)
    results, curves = fit_frac(labeled, seg_ids, args.nom_capacity, **kw)
    if results.empty:
        logging.warning("no pulses fit")
        return

    pd.set_option("display.width", 220, "display.max_columns", 60)
    print("\n=== fractional ECM fit results ===")
    print(results.to_string(index=False))
    stem = os.path.splitext(args.parquet)[0]
    out_csv = args.out or f"{stem}_frac.csv"
    results.to_csv(out_csv, index=False)
    logging.info("results -> %s", out_csv)
    if args.plot:
        plot_fits(curves, results, f"{stem}_frac.png")


if __name__ == "__main__":
    main()
