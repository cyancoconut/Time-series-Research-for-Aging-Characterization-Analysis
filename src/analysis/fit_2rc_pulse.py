"""Fit a Thevenin 2RC equivalent-circuit model to HPPC pulse exports.

Pipeline
--------
1. **Preprocessing** (``label_time_diff`` / ``build_pulse_sequence``) — adapted
   from the user's ``build_time_diff_sequence``. Time-difference gaps split the
   record into *cycles*; each cycle maps to an SOC stadium via ``SOC_ORDER``.
   **If a cell does not have exactly len(SOC_ORDER) cycles (3 time sequences),
   every row gets the default SOC ``DEFAULT_SOC`` (50%).** Segments are cut on
   ``Zustand`` changes (``pulse_segment_id``); current-unstable pulses are
   dropped; pulses earlier than ``REMOVE_PULSE_BEFORE_MIN`` into their cycle are
   removed (this drops the leading un-rested pulse of a cycle).

2. **2RC fit** (``fit_2rc``) — per surviving pulse segment, the *pulse* and its
   following *relaxation* are fit jointly as one continuous current profile with
   a shared parameter set ``{OCV, R0, R1, tau1, R2, tau2}``:

       during pulse  (I = I_pulse):
           V = OCV + I * (R0 + R1*(1 - e^{-t/tau1}) + R2*(1 - e^{-t/tau2}))
       during rest   (I = 0):
           V = OCV + I*R1*(1 - e^{-tp/tau1}) * e^{-(t-tp)/tau1}
                   + I*R2*(1 - e^{-tp/tau2}) * e^{-(t-tp)/tau2}

   The 20 s pulse alone cannot resolve the slow tau and the relaxation alone
   under-weights R0; fitting both together identifies all six parameters.
   ``OCV`` is a fit parameter (the relaxation asymptote), so a pulse missing its
   *pre*-relaxation is fine. ``R0`` is **not** fit: it is pinned to the DC pulse
   resistance R_DC,Δt = ΔU/ΔI (Ludwig et al., J. Power Sources 490 (2021)
   229523) — measured over a fixed Δt (default 0.5 s) from the onset using the
   actual current step, which is reproducible across check-ups and ramp-robust.
   The termination R_DC and the instantaneous jumps are reported as cross-checks
   (``R0_consistent`` flags onset vs termination R_DC agreement). C_k = tau_k/R_k.
   ``R0_extrap_onset_ohm`` / ``R0_extrap_term_ohm`` are pure-ohmic R0 estimates:
   the ohmic step extrapolated back to t=0 from the pulse *rise* (onset, on the
   dense ~0.18 s cadence — the most reproducible) and from the relaxation *decay*
   (termination, coarse ~0.8 s side). Neither depends on where a single sample
   lands; the two agree when both are well-posed, and both read ~1.5-2 mΩ below
   R_DC,0.5 s (the difference is the RC that develops within 0.5 s).

3. **Coupled staged fit** (``_staged_branches``) — a physically-separated
   decomposition reported next to the joint fit (``R0_staged``/``R1_fast``/
   ``tau1_fast``/``R2_slow``/``tau2_slow``/``staged_rmse_mV``). R0 is the onset
   ohmic; the slow branch is read from the fast-free part of the relaxation; its
   contribution *during* the pulse (a non-negligible several mV — the slow branch
   is partly built up by t_p) is reconstructed and subtracted, together with the
   OCV ramp, before the fast branch is fit on the dense onset residual. Each
   branch is thus read where the cadence resolves it. It re-simulates the full
   curve to sub-mV, giving reproducible charge-transfer params for aging while the
   joint fit stays the (lowest-RMSE) simulation model.

4. **2RC + finite-length Warburg** (``_fit_2rc_warburg``) — reported *alongside*
   the joint 2RC (columns ``R_d_ohm``/``tau_d_s``/``R{1,2}_w_ohm``/
   ``warburg_rmse_mV``). The lumped slow RC branch is a crude stand-in for the
   distributed diffusion impedance (``2RC_parameters_research_notes.md`` §2.3);
   this fit adds a proper **transmissive finite-length Warburg** element
   (``Z_Ws = R_d tanh(√(jω τ_d))/√(jω τ_d)``, step response ``_warburg_step``)
   on top of the two RC branches, so the √t-early diffusion shape is captured
   rather than approximated. R0 is pinned to the same R_DC as the 2RC fit, so the
   two models are directly comparable (``rmse_mV`` vs ``warburg_rmse_mV``). The
   joint 2RC stays the primary simulation model; the Warburg fit is a diagnostic.

Standalone analysis utility — does not touch the pipeline. Run from ``src/``::

    # every cell: takes working_path (and nom_capacity) from the pipeline config
    python -m analysis.fit_2rc_pulse --battery-config ../battery_config_VTC_linux.json
    # one cell, by name fragment
    python -m analysis.fit_2rc_pulse --battery-config <cfg.json> --cell 003
    # an explicit folder or file (wins over the config)
    python -m analysis.fit_2rc_pulse <cell_folder|pulse.parquet> [--plot]
"""

import argparse
import glob
import json
import logging
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- preprocessing constants (user-supplied) --------------------------------
SOC_ORDER = ["90%", "50%", "10%"]
DEFAULT_SOC = "50%"  # used when a cell lacks len(SOC_ORDER) distinct cycles
REMOVE_PULSE_BEFORE_MIN = 0       # drop pulses earlier than this into a cycle
                                  # (0 = keep all; kept inert here)
# Pulses to exclude from the fit by their ``Zustand/Current`` identity. The
# lead DCH/-1.5 pulse has no pre-pulse pause, so its onset-R0 cross-check is
# unavailable and only the termination jump remains; the RC fit then degenerates
# (the step is undersampled at ~0.2 s, so the early double-layer region the RC
# needs is unmeasured). Dropped by label rather than by timing.
EXCLUDE_ZUSTAND_CURRENT = ["DCH/-1.5"]
CYCLE_ACTIVE_LIMIT_HOUR = 4.0     # time_diff > this starts a new cycle
# Amplitude-agnostic pulse-stability gate: a segment is a pulse when its current
# plateau is *relatively* stable, std(I) <= PULSE_STD_FRACTION * |I|. Scales with
# cell size (VTC6 ~1.5/3 A, the 28 Ah sweep ~9.87 A), so no per-chemistry level list.
PULSE_STD_FRACTION = 0.05
REST_CURRENT_A = 0.05            # |Current| below this counts as rest
# Resistance-calculation-period for the DC pulse resistance R_DC,Δt, after
# Ludwig et al. (J. Power Sources 490 (2021) 229523): R_DC,Δt = ΔU/ΔI measured
# Δt s after a current change, using the *actual* current step. They constrain
# t_rise < Δt < 1 s (charge transfer + diffusion creep in past 1 s; the rise
# time must be cleared first). At our ~0.2 s cadence 0.5 s is the smoothest
# choice inside that window; 10-100 ms (their temperature-optimal band) needs
# ms sampling we do not have.
R_DC_DELTA_T = 0.5

# --- SOC-sweep labeling (assign_pulse_soc) ----------------------------------
# A full-SOC-sweep HPPC pulses at a series of SOC plateaus. Pulses are grouped
# into plateaus by their Ah_throughput: a jump larger than the plateau gap between
# consecutive pulses (the between-plateau charge/discharge step) starts the next
# plateau; a small jump (only the pulse+restore throughput) keeps them together.
# The gap defaults to one full SOC step in Ah, nom_capacity * SOC_SWEEP_STEP_PCT/100
# (config `plateau_gap_ah` overrides it).
#
# SOC_SWEEP_DIRECTION is the *sweep* direction (not a single pulse's CHA/DCH):
#   "discharge" -> sweep starts full and empties: SOC = 100 - step*plateau
#   "charge"    -> sweep starts empty and fills:  SOC =   0 + step*plateau
# so plateau 0 is 100% for a discharge sweep and 0% for a charge sweep.
SOC_SWEEP_DIRECTION = "discharge"
SOC_SWEEP_STEP_PCT = 5.0
PLATEAU_GAP_AH = None  # None -> derive nom_capacity * SOC_SWEEP_STEP_PCT / 100

#: Nominal capacity fallback (VTC6) when neither config supplies one.
DEFAULT_NOM_CAPACITY = 3.0

#: Pulse-export folder under ``working_path`` — one sub-folder per cell stem,
#: written by ``output/export_pulse.py``.
PULSE_EXPORT_DIR = "20_export_pulse"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Config (dedicated JSON; see config_2rc_example.json)
# ---------------------------------------------------------------------------
# Maps config keys -> the module globals they override. Every key is optional;
# omitted keys keep the module default. Values that are not globals (nom_capacity,
# the SOC-sweep params) are read straight off the returned dict by ``main``.
_CONFIG_GLOBALS = {
    "pulse_std_fraction": "PULSE_STD_FRACTION",
    "rest_current_a": "REST_CURRENT_A",
    "cycle_active_limit_hour": "CYCLE_ACTIVE_LIMIT_HOUR",
    "r_dc_delta_t_s": "R_DC_DELTA_T",
    "exclude_zustand_current": "EXCLUDE_ZUSTAND_CURRENT",
    "remove_pulse_before_min": "REMOVE_PULSE_BEFORE_MIN",
    "soc_sweep_direction": "SOC_SWEEP_DIRECTION",
    "soc_step_pct": "SOC_SWEEP_STEP_PCT",
    "plateau_gap_ah": "PLATEAU_GAP_AH",
}


def load_2rc_config(path):
    """Load a dedicated 2RC JSON config and apply it to the module globals.

    Underscore-prefixed keys (``_description`` etc.) are ignored so the JSON can
    carry inline docs. Recognised keys in ``_CONFIG_GLOBALS`` overwrite the
    matching module-level constant (the analyzer's tunables live in one place);
    ``nom_capacity`` and the SOC-sweep params are returned for ``main`` to consume.
    Returns the parsed dict. Missing keys keep the module defaults.
    """
    with open(path) as fh:
        cfg = json.load(fh)
    for key, glob_name in _CONFIG_GLOBALS.items():
        if key in cfg and cfg[key] is not None:
            globals()[glob_name] = cfg[key]
    logging.info("loaded 2RC config %s", os.path.basename(path))
    return cfg


def load_battery_config(path):
    """Load the pipeline's battery config (the one ``main.py`` takes).

    Only two keys are used here — ``working_path`` (to locate the pulse exports)
    and ``nom_capacity`` — so the analyzer stays runnable on any machine without
    a hardcoded data path. A dedicated 2RC ``--config`` still wins on
    ``nom_capacity``: it is the cell-specific one.
    """
    with open(path) as fh:
        cfg = json.load(fh)
    logging.info("loaded battery config %s", os.path.basename(path))
    return cfg


def resolve_cell_folders(cfg, cell_filters=None):
    """Every cell's pulse-export folder under ``<working_path>/20_export_pulse``.

    One sub-folder per ``cell_stem``. ``cell_filters`` subsets by name fragment,
    matching ``main.py --cells``; ``None`` returns all of them. Folders holding
    no ``*_pulse_BM*.parquet`` are skipped (a cell can be exported but have no
    pulses). Raises ``FileNotFoundError`` with a pointed hint when the export
    folder itself is missing — the usual cause is that the pipeline ran with
    ``export_pulse`` at its default ``false``.
    """
    working_path = cfg.get("working_path")
    if not working_path:
        raise ValueError("battery config has no 'working_path'")
    root = os.path.join(working_path, PULSE_EXPORT_DIR)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"no pulse exports at {root} — the pipeline writes them only with "
            "'export_pulse': true (default false). With download_from='minio' "
            "they may exist only in MinIO and need syncing to working_path first."
        )

    folders = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if cell_filters and not any(c in name for c in cell_filters):
            continue
        if not glob.glob(os.path.join(path, "*_pulse_BM*.parquet")):
            logging.info("skip %s: no pulse exports in folder", name)
            continue
        folders.append(path)

    if not folders:
        raise FileNotFoundError(
            f"no cell folder with pulse exports under {root}"
            + (f" matching {cell_filters}" if cell_filters else "")
        )
    logging.info("%d cell folder(s) under %s", len(folders), root)
    return folders


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _parse_soh(file_stem):
    """Pull the SOH value out of a ``..._BM<n>_<SOH>SOH`` filename stem."""
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?|NA)SOH", file_stem)
    return m.group(1) if m else "NA"


def label_time_diff(df, file_name):
    """Label the *full* time series with cycle_id / SOC / pulse_segment_id.

    Mirrors the user's ``build_time_diff_sequence`` but keeps every row (no
    collapse to one representative point) so the 2RC fit sees the full curve.
    """
    out = df.copy()
    out["File"] = file_name
    out["SOH"] = _parse_soh(os.path.splitext(os.path.basename(file_name))[0])

    out["Time"] = pd.to_datetime(out["Time"], utc=True, errors="coerce")
    out["Current"] = pd.to_numeric(out["Current"], errors="coerce")
    out["Voltage"] = pd.to_numeric(out["Voltage"], errors="coerce")
    out = out.dropna(subset=["Time", "Current", "Voltage"]).copy()
    out = out.sort_values(["File", "Time"]).reset_index(drop=True)

    # cycle boundaries from intra-file time gaps
    out["time_diff_hour"] = (
        out.groupby("File")["Time"].diff() / pd.Timedelta(hours=1)
    )
    out["is_new_cycle"] = (
        out["time_diff_hour"].isna()
        | (out["time_diff_hour"] > CYCLE_ACTIVE_LIMIT_HOUR)
    )
    out["cycle_id"] = out.groupby("File")["is_new_cycle"].cumsum().astype(int)

    # SOC: only a cell with exactly len(SOC_ORDER) cycles gets the 90/50/10
    # schema; anything else (incl. this single-SOC HPPC file) -> default 50%.
    n_cycles = out.groupby("File")["cycle_id"].transform("nunique")
    soc_from_order = out["cycle_id"].map(
        lambda c: SOC_ORDER[(c - 1) % len(SOC_ORDER)]
    )
    out["SOC"] = np.where(n_cycles == len(SOC_ORDER), soc_from_order, DEFAULT_SOC)

    cycle_start = out.groupby(["File", "cycle_id"])["Time"].transform("min")
    out["time_from_cycle_start_min"] = (out["Time"] - cycle_start) / pd.Timedelta(
        minutes=1
    )

    # normalise Zustand to CHA / DCH (rest keeps its own label)
    out["Zustand"] = out["Zustand"].astype(str)
    out.loc[out["Zustand"].str.startswith("DCH", na=False), "Zustand"] = "DCH"
    out.loc[out["Zustand"].str.startswith("CHA", na=False), "Zustand"] = "CHA"

    # segment id: new segment on File change or Zustand change
    out["pulse_segment_id"] = (
        out["File"].ne(out["File"].shift())
        | out["Zustand"].ne(out["Zustand"].shift())
    ).cumsum()

    out["Zustand/Current"] = (
        out["Zustand"] + "/" + out["Current"].round(1).astype(str)
    )
    return out


def _is_bad_current_segment(group):
    """True if a pulse segment's current is too unstable or has no real step.

    Amplitude-agnostic: a real pulse has a genuine current step (|I| plateau above
    the rest threshold) that is *relatively* stable (``std <= PULSE_STD_FRACTION *
    |I|``). Keying on relative stability rather than a fixed level list lets the
    gate scale with cell size (VTC6 ~1.5/3 A, the 28 Ah sweep ~9.87 A, …) without
    per-chemistry tuning.
    """
    g = group.sort_values("Time")
    cur = g["Current"]
    if len(g) >= 2 and cur.iloc[0] == 0:  # ignore a leading zero sample
        cur = cur.iloc[1:]
    amp = float(cur.abs().median())
    if amp < REST_CURRENT_A:
        return True
    return bool(cur.std() > PULSE_STD_FRACTION * amp)


def _segment_zc(grp):
    """Representative ``Zustand/Current`` label of a pulse segment (e.g. DCH/-1.5)."""
    active = grp.loc[grp["Current"].abs() > REST_CURRENT_A, "Zustand/Current"]
    src = active if not active.empty else grp["Zustand/Current"]
    return src.mode().iloc[0]


def select_pulse_segments(
    labeled,
    remove_before_min=REMOVE_PULSE_BEFORE_MIN,
    exclude_zc=EXCLUDE_ZUSTAND_CURRENT,
):
    """Return the list of *good* pulse ``pulse_segment_id`` values, in time order.

    A good pulse: Zustand in {CHA, DCH}, stable current, not in ``exclude_zc``
    (matched on the ``Zustand/Current`` label), and starting at least
    ``remove_before_min`` minutes into its cycle.
    """
    exclude_zc = set(exclude_zc or [])
    pulses = labeled[labeled["Zustand"].isin(["CHA", "DCH"])]
    good = []
    for seg_id, grp in pulses.groupby("pulse_segment_id", sort=False):
        zc = _segment_zc(grp)
        if zc in exclude_zc:
            logging.info("skip pulse seg %s: excluded by Zustand/Current=%s", seg_id, zc)
            continue
        if grp["time_from_cycle_start_min"].min() < remove_before_min:
            logging.info(
                "skip pulse seg %s: starts %.1f min into cycle (< %s)",
                seg_id, grp["time_from_cycle_start_min"].min(), remove_before_min,
            )
            continue
        if _is_bad_current_segment(grp):
            logging.info("skip pulse seg %s: unstable/unknown current", seg_id)
            continue
        good.append((grp["Time"].min(), seg_id))
    good.sort()
    return [seg_id for _, seg_id in good]


def build_pulse_sequence(labeled, output_columns):
    """The user's collapsed one-row-per-pulse table (for inspection/export)."""
    pulses = labeled[labeled["Zustand"].isin(["CHA", "DCH"])]
    pulses = pulses.groupby(["File", "pulse_segment_id"], sort=False).filter(
        lambda g: not _is_bad_current_segment(g)
    )

    def pick(group):
        g = group.sort_values("Time")
        if g["Current"].iloc[0] == 0 and len(g) >= 2:
            return g.iloc[1]
        return g.iloc[0]

    seq = (
        pulses.groupby(["File", "pulse_segment_id"], group_keys=False)
        .apply(pick)
        .reset_index(drop=True)
    )
    keep = [c for c in output_columns if c in seq.columns]
    return seq[keep].copy()


def assign_pulse_soc(
    labeled,
    nom_capacity,
    sweep_direction=None,
    soc_step_pct=None,
    plateau_gap_ah=None,
):
    """Label each pulse segment with amplitude, direction and sweep SOC.

    A full-SOC-sweep HPPC steps the cell through equal SOC plateaus, doing one or
    two test pulses at each. Pulses are ordered in time and grouped into plateaus
    by their ``Ah_throughput``: a jump larger than ``plateau_gap_ah`` (the
    between-plateau charge/discharge step) starts a new plateau, while the small
    jump between a plateau's paired pulses (only the pulse+restore throughput)
    keeps them together. Each plateau then gets an SOC from the *sweep* direction:

        discharge sweep -> SOC = 100 - soc_step_pct * plateau   (100% at the top)
        charge sweep    -> SOC =   0 + soc_step_pct * plateau   (0% at the bottom)

    ``plateau_gap_ah`` defaults to one full SOC step in Ah,
    ``nom_capacity * soc_step_pct / 100``.

    Returns a per-segment DataFrame indexed by ``pulse_segment_id`` with columns
    ``t0, Ah_throughput, pulse_amplitude_A, pulse_C_rate, direction, soc_plateau,
    SOC_pct`` (in time order). Empty if no good pulse segments or no
    ``Ah_throughput`` column.
    """
    sweep_direction = sweep_direction if sweep_direction is not None else SOC_SWEEP_DIRECTION
    soc_step_pct = soc_step_pct if soc_step_pct is not None else SOC_SWEEP_STEP_PCT
    if plateau_gap_ah is None:
        plateau_gap_ah = PLATEAU_GAP_AH
    if plateau_gap_ah is None:
        plateau_gap_ah = nom_capacity * soc_step_pct / 100.0

    if "Ah_throughput" not in labeled.columns:
        logging.warning("assign_pulse_soc: no Ah_throughput column — cannot label SOC")
        return pd.DataFrame()

    pulses = labeled[labeled["Zustand"].isin(["CHA", "DCH"])]
    recs = []
    for seg_id, grp in pulses.groupby("pulse_segment_id", sort=False):
        if _is_bad_current_segment(grp):
            continue
        g = grp.sort_values("Time")
        cur = g["Current"]
        active = cur[cur.abs() > REST_CURRENT_A]
        i_signed = float(active.mean()) if not active.empty else float(cur.mean())
        amp = float(active.abs().median()) if not active.empty else float(cur.abs().median())
        recs.append(
            {
                "pulse_segment_id": seg_id,
                "t0": g["Time"].min(),
                "Ah_throughput": float(g["Ah_throughput"].mean()),
                "pulse_amplitude_A": round(amp, 3),
                "pulse_C_rate": round(amp / nom_capacity, 3) if nom_capacity else np.nan,
                "direction": "CHA" if i_signed > 0 else "DCH",
            }
        )
    seg = pd.DataFrame(recs)
    if seg.empty:
        return seg
    seg = seg.sort_values("t0").reset_index(drop=True)

    # plateau index from Ah_throughput jumps (monotone-increasing cumulative Ah)
    dah = seg["Ah_throughput"].diff()
    new_plateau = dah.isna() | (dah.abs() > plateau_gap_ah)
    seg["soc_plateau"] = new_plateau.cumsum().astype(int) - 1  # 0-based, top plateau = 0

    if str(sweep_direction).lower().startswith("cha"):
        seg["SOC_pct"] = 0.0 + soc_step_pct * seg["soc_plateau"]
    else:  # discharge sweep
        seg["SOC_pct"] = 100.0 - soc_step_pct * seg["soc_plateau"]

    n_plateaus = seg["soc_plateau"].nunique()
    logging.info(
        "assign_pulse_soc: %d pulses -> %d plateaus (gap %.3f Ah, %s sweep, SOC %.0f..%.0f%%)",
        len(seg), n_plateaus, plateau_gap_ah, sweep_direction,
        seg["SOC_pct"].iloc[0], seg["SOC_pct"].iloc[-1],
    )
    return seg


# ---------------------------------------------------------------------------
# 2RC model
# ---------------------------------------------------------------------------
def _v_2rc(t, ocv_post, r0, r1, tau1, r2, tau2, *, i_pulse, t_p, ocv_pre):
    """Terminal voltage of a 2RC cell for a single CC pulse + relaxation.

    ``t`` is seconds from pulse start; the pulse lasts ``t_p`` s at ``i_pulse``
    (signed), then current is zero.

    OCV is **not** held constant: during the pulse it ramps linearly from
    ``ocv_pre`` to ``ocv_post`` as the (≈ linear-in-time) charge moves SOC, then
    holds at ``ocv_post`` once current stops. On the steep low-SOC OCV curve this
    ramp is tens of mV; modelling it stops that SOC-driven drift from being
    loaded onto the slow RC branch (which otherwise rails). With
    ``ocv_pre == ocv_post`` this reduces to the constant-OCV model.
    """
    t = np.asarray(t, dtype=float)
    during = t <= t_p
    frac = np.clip(t / t_p, 0.0, 1.0)            # charge fraction delivered
    ocv_t = ocv_pre + (ocv_post - ocv_pre) * frac
    # polarisation built up during the pulse, evaluated at each t (capped at t_p)
    tc = np.minimum(t, t_p)
    eta1 = i_pulse * r1 * (1.0 - np.exp(-tc / tau1))
    eta2 = i_pulse * r2 * (1.0 - np.exp(-tc / tau2))
    # during rest the RC voltages decay from their value at t_p
    decay = np.where(during, 1.0, np.exp(-(t - t_p) / tau1))
    eta1 = np.where(during, eta1, i_pulse * r1 * (1.0 - np.exp(-t_p / tau1)) * decay)
    decay2 = np.where(during, 1.0, np.exp(-(t - t_p) / tau2))
    eta2 = np.where(during, eta2, i_pulse * r2 * (1.0 - np.exp(-t_p / tau2)) * decay2)
    ohmic = np.where(during, i_pulse * r0, 0.0)
    return ocv_t + ohmic + eta1 + eta2


def _onset_step_voltage(t, v, t_p, i_pulse, slope_k=0.005):
    """Voltage at the end of the onset ohmic step, robust to current ramp-up.

    The first logged pulse sample can land *mid-ramp* (current not yet at its
    plateau, voltage still settling toward the IR step), which under-measures the
    onset jump and produces spurious R0 dips. Reading ``v[0]`` blindly is the
    failure mode; a plain current-plateau gate misses it too (the current can
    already read ~99 % while the voltage is still moving).

    Instead, find the **knee**: the onset step shows a large ``dV/dt`` spike that
    collapses into the gentle RC decay once the IR step is complete. Return the
    voltage at the first early sample (within the first ``min(2 s, t_p)``) whose
    ``|dV/dt|`` has dropped below ``slope_k * |i_pulse|`` (V/s) — the threshold
    scales with current so it auto-adapts across C-rates. Falls back to ``v[0]``
    when there are too few early samples to form a slope.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    early = np.where((t > 0) & (t <= min(2.0, t_p)))[0]
    if len(early) < 1:
        return float(v[0])
    dvdt = np.abs((v[early] - v[early - 1]) / (t[early] - t[early - 1]))
    below = early[dvdt < slope_k * abs(i_pulse)]
    k = below[0] if len(below) else early[-1]
    return float(v[k])


def _r_dc_delta_t(t, v, i_arr, t_change, dt, u_ref, i_ref):
    """DC pulse resistance R_DC,Δt after Ludwig et al. (2021), eq. R_DC,Δt=ΔU/ΔI.

    Measures the resistance over a fixed *resistance-calculation-period* ``dt``
    following a current change at ``t_change`` (seconds, same clock as ``t``)::

        R_DC,Δt = |(u_ref - U(t_change+dt)) / (i_ref - I(t_change+dt))|

    ``u_ref`` / ``i_ref`` are the voltage / current just before the change.
    Crucially the denominator uses the **measured** current step, not the nominal
    pulse current, so a ramped / imperfect current edge does not bias R0 (the
    failure mode behind the aged-state R0 dips). A *fixed* ``dt`` (vs. the first
    available sample) also makes the resistance reproducible across check-ups.
    Returns ``np.nan`` when there is no sample at ``t_change+dt`` or the current
    step is too small to divide by.
    """
    k = int(np.searchsorted(t, t_change + dt))
    if k >= len(t):
        return np.nan
    d_i = i_ref - float(i_arr[k])
    if abs(d_i) < 1e-6:
        return np.nan
    return abs((u_ref - float(v[k])) / d_i)


# Early-relaxation window (s) fit for the extrapolated termination R0. The t=0
# intercept must be reconstructed from the *fast* decay near current-off; fitting
# the whole (~30 min) rest lets the two exponentials chase the slow SOC/diffusion
# drift and biases the intercept high (a ~3 mOhm over-read on VTC6). A short early
# window isolates the fast+medium branches (the slow tail is ~flat over it and
# folds into OCV), so the extrapolation converges toward the true ohmic R0.
EXTRAP_REST_WINDOW_S = 60.0


def _extrap_termination_r0(t, v, t_p, i_pulse, v_pulse_last,
                           window_s=EXTRAP_REST_WINDOW_S):
    """Fit-extrapolated termination R0 — the ohmic step at current-off, sampling-lag-free.

    The raw termination jump ``(v_relax_first - v_pulse_last)/I`` reads the *first*
    relaxation sample, which on a coarse rest cadence (here ~0.8 s) lands well
    after current-off. In that gap the RC overvoltages have already begun to
    decay, so the raw jump over-reads R0 by the RC that leaked in during the lag
    (a systematic bias, not averageable). Instead, fit the *early* relaxation
    (first ``window_s`` s) to the 2RC decay
    ``V(t') = OCV + V1(0)e^{-t'/tau1} + V2(0)e^{-t'/tau2}`` and reconstruct the
    voltage **at the current-off instant** ``t'=0``::

        V(0+) = OCV + V1(0) + V2(0)          # RC still full, ohmic already gone
        R0    = |(v_pulse_last - V(0+)) / i_pulse|

    Because ``V(0+)`` is extrapolated from the decay shape, the result does not
    depend on where the first rest sample happens to fall — it is reproducible
    across check-ups regardless of the relaxation cadence. The fit is restricted
    to ``window_s`` because the t=0 intercept is set by the fast branch; fitting
    the full rest lets the slow tail dominate and biases R0 high. Mirrors the t=0
    extrapolation ``fit_one_relaxation`` uses for cycling rest curves.
    **Caveat**: at a coarse rest cadence the sub-second double-layer decay is
    unmeasured, so the intercept still carries some window sensitivity — read this
    as a *true-ohmic* estimate, cross-checked against the (densely sampled) onset.
    Returns ``np.nan`` when the early rest tail is too short to fit.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    rest = (t > t_p) & (t - t_p <= window_s)
    t_rest = t[rest] - t_p
    v_rest = v[rest]
    if len(t_rest) < 8:
        return np.nan

    v_settled = float(v_rest[-1])                     # ~OCV guess (rest tail)
    amp0 = float(v_rest[0] - v_settled)               # total overvoltage at t'~0
    tau2_hi = 6000.0
    bounds = (
        [v_settled - 0.5, -0.5, -0.5, 0.2, 20.0],
        [v_settled + 0.5, 0.5, 0.5, 60.0, tau2_hi],
    )
    best_popt, best_rmse = None, np.inf
    for tau1_0, tau2_0 in [(5.0, 200.0), (2.0, 30.0), (8.0, 60.0), (3.0, 100.0)]:
        p0 = [v_settled, amp0 / 2, amp0 / 2, tau1_0, tau2_0]
        try:
            popt, _ = curve_fit(_v_relax, t_rest, v_rest, p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((_v_relax(t_rest, *popt) - v_rest) ** 2)))
        if rmse < best_rmse:
            best_popt, best_rmse = popt, rmse
    if best_popt is None:
        return np.nan
    ocv, v1_0, v2_0, _, _ = best_popt
    v0_plus = ocv + v1_0 + v2_0                        # extrapolated to current-off
    return abs((v_pulse_last - v0_plus) / i_pulse)


# Window (s) of the pulse rise used for the onset-extrapolated R0. The whole 20 s
# pulse is densely sampled (~0.18 s), so the fast rise is well resolved; a window
# a little longer than a few tau1 anchors the t=0 intercept without letting slow
# SOC drift over a long pulse bias it.
EXTRAP_ONSET_WINDOW_S = 20.0


def _extrap_onset_r0(t, v, t_p, i_pulse, ocv_pre, window_s=EXTRAP_ONSET_WINDOW_S,
                     r0_cap=None):
    """Onset-extrapolated R0 — pure ohmic step read from the *densely-sampled* rise.

    The rest->pulse transition is on the fine (~0.18 s) pulse cadence, so unlike
    the termination extrapolation (which fights the coarse ~0.8 s rest) the fast
    branch is directly resolved here. Fit the first ``window_s`` s of the pulse to
    the 2RC charging curve with OCV anchored at the settled pre-pulse voltage::

        V(t) = ocv_pre + I*R0 + I*R1*(1-e^{-t/tau1}) + I*R2*(1-e^{-t/tau2})

    and read R0 (the t=0 intercept above OCV) directly. Because the intercept is
    anchored by the dense early samples, this is the most reproducible pure-ohmic
    R0 for coarse-rest data — cross-checked against ``_extrap_termination_r0`` (the
    two land on the same value when both are well-posed). Returns ``np.nan`` when
    the early rise is too short to fit.

    ``r0_cap`` (when given, the fixed-window R_DC,0.5s) bounds the intercept from
    above: the pure-ohmic R0 is ohmic-only and R_DC,Δt is ohmic + the fast-RC
    overpotential grown within Δt, so R0 <= R_DC,Δt by construction. Capping the
    R0 bound at R_DC,0.5s and seeding from it keeps the intercept from swapping
    magnitude with the fast branch across check-ups (the vs-SOH rumble).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    rise = (t > 0) & (t <= min(t_p, window_s))
    tt, vv = t[rise], v[rise]
    if len(tt) < 8:
        return np.nan

    def model(x, r0, r1, tau1, r2, tau2):
        eta1 = i_pulse * r1 * (1.0 - np.exp(-x / tau1))
        eta2 = i_pulse * r2 * (1.0 - np.exp(-x / tau2))
        return ocv_pre + i_pulse * r0 + eta1 + eta2

    # R0 upper bound: R_DC,0.5s when supplied (physical ceiling), else the loose
    # default. A hair of slack (2 %) above the cap absorbs measurement noise.
    r0_hi = 0.2
    r0_seed = 0.02
    if r0_cap is not None and np.isfinite(r0_cap) and r0_cap > 0:
        r0_hi = min(0.2, r0_cap * 1.02)
        r0_seed = min(r0_hi, 0.8 * r0_cap)
    bounds = ([1e-4, 1e-5, 0.2, 1e-5, 20.0], [r0_hi, 1.0, 60.0, 1.0, 6000.0])
    best, brmse = None, np.inf
    for tau1_0, tau2_0 in [(5.0, 200.0), (2.0, 30.0), (8.0, 60.0), (3.0, 100.0)]:
        try:
            popt, _ = curve_fit(model, tt, vv, p0=[r0_seed, 0.01, tau1_0, 0.01, tau2_0],
                                bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((model(tt, *popt) - vv) ** 2)))
        if rmse < brmse:
            best, brmse = popt, rmse
    if best is None:
        return np.nan
    return abs(best[0])


# Rest time (s) after which the fast branch has effectively decayed, so the
# relaxation is slow-branch-only — used to isolate R2/tau2 before reconstructing
# the slow branch's contribution *during* the pulse.
STAGED_SLOW_CUT_S = 5.0


def _staged_branches(t, v, t_p, i_pulse, ocv_pre, r0):
    """Coupled staged 2RC decomposition — each branch read where it is resolved.

    A fraction of the slow branch already builds up *during* the pulse (up to tens
    of %  of R2, several mV by t_p), so the dense onset rise is **not** fast-only.
    This fit removes that overlap explicitly:

    1. **Slow branch from the relaxation.** Past ``STAGED_SLOW_CUT_S`` s the fast
       branch has decayed, so the (coarsely sampled, but tau2 >> 0.8 s) rest tail
       is pure slow decay: fit ``V = OCV + A2 e^{-t'/tau2}``. The slow amplitude at
       current-off is ``A2`` (tau2 >> the cut, so it is ~unchanged over it), giving
       ``R2 = A2 / (I (1 - e^{-t_p/tau2}))`` via the pulse development factor.
    2. **Reconstruct the slow ramp in the pulse** ``eta2(t) = I R2 (1-e^{-t/tau2})``
       and subtract it — with the ohmic ``I*r0`` (onset-extrapolated) — from the
       dense onset rise.
    3. **Fast branch from the residual onset** (0.18 s cadence):
       ``resid(t) = I R1 (1-e^{-t/tau1})`` -> clean R1, tau1.

    Returns ``{R1_fast, tau1_fast, R2_slow, tau2_slow, staged_rmse_mV}`` (the RMSE
    is the full pulse+relaxation curve re-simulated from the staged params, so it
    is comparable to the joint fit's rmse) or ``None`` if either stage fails.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    # --- stage 1: slow branch from the fast-free part of the relaxation ---
    rest = (t > t_p) & (t - t_p >= STAGED_SLOW_CUT_S)
    tr, vr = t[rest] - t_p, v[rest]
    if len(tr) < 8:
        return None
    v_settled = float(vr[-1])

    def slow(tt, ocv, a2, tau2):
        return ocv + a2 * np.exp(-tt / tau2)

    sbounds = ([v_settled - 0.5, -0.5, 20.0], [v_settled + 0.5, 0.5, 6000.0])
    a0 = float(vr[0] - v_settled)
    best_s, rmse_s = None, np.inf
    for tau2_0 in (60.0, 150.0, 400.0, 1000.0):
        try:
            popt, _ = curve_fit(slow, tr, vr, p0=[v_settled, a0, tau2_0],
                                bounds=sbounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((slow(tr, *popt) - vr) ** 2)))
        if rmse < rmse_s:
            best_s, rmse_s = popt, rmse
    if best_s is None:
        return None
    ocv_post, a2, tau2 = best_s          # ocv_post = relaxation asymptote (settled OCV)
    # A2 is the slow amplitude at t'=0 (current-off, extrapolated across the cut).
    # It was built over the pulse: A2 = I*R2*(1 - e^{-t_p/tau2}) -> solve for R2.
    dev2 = 1.0 - np.exp(-t_p / tau2)
    if abs(dev2) < 1e-6:
        return None
    r2 = a2 / (i_pulse * dev2)

    # --- stage 2+3: subtract ohmic + OCV ramp + reconstructed slow ramp, fit fast ---
    onset = (t > 0) & (t <= t_p)
    to, vo = t[onset], v[onset]
    if len(to) < 8:
        return None
    # OCV ramps ocv_pre -> ocv_post over the pulse as SOC moves (matters at low SOC
    # where the OCV curve is steep); remove it so it does not leak into the fast RC.
    ocv_ramp = ocv_pre + (ocv_post - ocv_pre) * np.clip(to / t_p, 0.0, 1.0)
    eta2_pulse = i_pulse * r2 * (1.0 - np.exp(-to / tau2))
    resid = vo - ocv_ramp - i_pulse * r0 - eta2_pulse

    def fast(tt, r1, tau1):
        return i_pulse * r1 * (1.0 - np.exp(-tt / tau1))

    fbounds = ([1e-5, 0.2], [1.0, 60.0])
    best_f, rmse_f = None, np.inf
    for tau1_0 in (1.0, 2.0, 5.0, 10.0):
        try:
            popt, _ = curve_fit(fast, to, resid, p0=[0.01, tau1_0],
                                bounds=fbounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((fast(to, *popt) - resid) ** 2)))
        if rmse < rmse_f:
            best_f, rmse_f = popt, rmse
    if best_f is None:
        return None
    r1, tau1 = best_f

    # order fast < slow (guard against a swapped/degenerate slow branch)
    if tau1 > tau2:
        return None
    # full-curve validation: reconstruct the whole window from the staged params
    # with the same OCV-ramp model the joint fit uses, so staged_rmse is comparable.
    vsim = _v_2rc(t, ocv_post, r0, r1, tau1, r2, tau2,
                  i_pulse=i_pulse, t_p=t_p, ocv_pre=ocv_pre)
    staged_rmse = float(np.sqrt(np.mean((vsim - v) ** 2)) * 1000)
    return {
        "R1_fast_ohm": round(r1, 5),
        "tau1_fast_s": round(tau1, 3),
        "R2_slow_ohm": round(r2, 5),
        "tau2_slow_s": round(tau2, 2),
        "staged_rmse_mV": round(staged_rmse, 3),
    }


# ---------------------------------------------------------------------------
# 2RC + finite-length Warburg (transmissive Ws) extension
# ---------------------------------------------------------------------------
# Number of terms kept in the transmissive-FLW step-response series. The n-th
# odd term decays as exp(-n^2 ...), so the sum converges quadratically; 20 odd
# terms is exact to well below the fit noise for any t.
WARBURG_N_TERMS = 20


def _warburg_step(t, tau_d, n_terms=WARBURG_N_TERMS):
    """Dimensionless step response g(t) of a finite-length transmissive Warburg.

    The finite-length *transmissive* (short/absorbing terminus) Warburg element
    has impedance ``Z_Ws = R_d * tanh(sqrt(jω τ_d)) / sqrt(jω τ_d)`` — the proper
    distributed diffusion branch that the lumped slow RC only approximates (see
    ``2RC_parameters_research_notes.md`` §2.3/§3.2). Its unit current-step voltage
    response is ``I * R_d * g(t)`` with::

        g(t) = 1 - (8/π²) Σ_{k=0..} 1/(2k+1)² · exp(-(2k+1)² π² t / (4 τ_d))

    ``g(0)=0`` (Σ 1/(2k+1)² = π²/8), ``g(∞)=1`` (settles to the DC resistance
    ``R_d``), and the early rise is ∝ √t — the diffusion signature a single RC
    cannot reproduce. ``τ_d = L²/D`` is the diffusion time constant. Bounded and
    settling, so it fits both the pulse and the 30-min relaxation asymptote.
    """
    t = np.asarray(t, dtype=float)
    acc = np.zeros_like(t)
    for k in range(n_terms):
        m = 2 * k + 1
        acc += np.exp(-(m ** 2) * np.pi ** 2 * np.maximum(t, 0.0) / (4.0 * tau_d)) / m ** 2
    g = 1.0 - (8.0 / np.pi ** 2) * acc
    return np.where(t > 0, g, 0.0)


def _v_2rc_warburg(t, ocv_post, r0, r1, tau1, r2, tau2, r_d, tau_d,
                   *, i_pulse, t_p, ocv_pre):
    """Terminal voltage of a 2RC **+ finite-length Warburg** cell for one pulse+rest.

    Extends ``_v_2rc`` (OCV ramp + ohmic + two RC branches) with a transmissive
    Warburg diffusion branch added by superposition, exactly as the RC branches
    build up during the pulse and decay after it: the response to a current step
    ``+I`` at t=0 minus the step ``+I`` removed at ``t_p``::

        during pulse (t ≤ t_p):  η_W = I R_d g(t)
        during rest  (t > t_p):  η_W = I R_d (g(t) - g(t - t_p))

    ``g`` is the transmissive-FLW step response (``_warburg_step``).
    """
    base = _v_2rc(t, ocv_post, r0, r1, tau1, r2, tau2,
                  i_pulse=i_pulse, t_p=t_p, ocv_pre=ocv_pre)
    t = np.asarray(t, dtype=float)
    during = t <= t_p
    g_now = _warburg_step(t, tau_d)
    g_off = np.where(during, 0.0, _warburg_step(t - t_p, tau_d))
    eta_w = i_pulse * r_d * (g_now - g_off)
    return base + eta_w


def _fit_2rc_warburg(t, v, t_p, i_pulse, ocv_pre, r0, v_rest, r0_guess):
    """Joint 2RC + finite-length-Warburg fit (R0 pinned). Returns a dict or None.

    Fits ``{OCV, R1, τ1, R2, τ2, R_d, τ_d}`` over the same pulse+relaxation window
    the 2RC fit uses, with R0 held at the pinned R_DC value so the Warburg result
    is directly comparable to the 2RC one. Multi-start over (τ1, τ2, τ_d) seeds —
    the extra diffusion branch adds a slow-RC↔Warburg trade-off, so a single start
    is unreliable; the lowest-RMSE fit is kept. ``_vfit_w`` carries the fitted curve
    for the overlay plot.
    """
    tau2_hi = 6000.0
    tau_d_hi = 20000.0

    def model(tt, ocv_post, r1, tau1, r2, tau2, r_d, tau_d):
        return _v_2rc_warburg(
            tt, ocv_post, r0, r1, tau1, r2, tau2, r_d, tau_d,
            i_pulse=i_pulse, t_p=t_p, ocv_pre=ocv_pre,
        )

    bounds = (
        [v_rest - 0.5, 1e-5, 0.2, 1e-5, 20.0, 1e-5, 5.0],
        [v_rest + 0.5, 1.0, 60.0, 1.0, tau2_hi, 1.0, tau_d_hi],
    )
    best_popt, best_rmse = None, np.inf
    seeds = [(5.0, 200.0), (2.0, 30.0), (8.0, 60.0), (3.0, 100.0)]
    for tau1_0, tau2_0 in seeds:
        for tau_d_0 in (100.0, 500.0, 2000.0):
            p0 = [v_rest, r0_guess, tau1_0, r0_guess, tau2_0, r0_guess, tau_d_0]
            try:
                popt, _ = curve_fit(model, t, v, p0=p0, bounds=bounds, maxfev=30000)
            except (RuntimeError, ValueError):
                continue
            rmse = float(np.sqrt(np.mean((model(t, *popt) - v) ** 2)))
            if rmse < best_rmse:
                best_popt, best_rmse = popt, rmse
    if best_popt is None:
        return None

    ocv, r1, tau1, r2, tau2, r_d, tau_d = best_popt
    if tau1 > tau2:  # order the two RC branches fast -> slow
        r1, tau1, r2, tau2 = r2, tau2, r1, tau1
    # a collapsed Warburg (railed τ_d / vanishing R_d) means the slow RC absorbed
    # the diffusion branch — flag it, same idea as the 2RC `degenerate` flag.
    degenerate = bool(tau_d >= 0.999 * tau_d_hi or r_d <= 1e-4 or tau2 >= 0.999 * tau2_hi)
    return {
        "OCV_w_V": round(ocv, 4),
        "R1_w_ohm": round(r1, 5),
        "tau1_w_s": round(tau1, 3),
        "R2_w_ohm": round(r2, 5),
        "tau2_w_s": round(tau2, 2),
        "R_d_ohm": round(r_d, 5),
        "tau_d_s": round(tau_d, 2),
        "C_d_F": round(tau_d / r_d, 1) if r_d else np.nan,
        "warburg_rmse_mV": round(best_rmse * 1000, 3),
        "warburg_degenerate": degenerate,
        "_vfit_w": model(t, *best_popt),
    }


def fit_one_pulse(window, t_p, i_pulse, v_pulse_last, v_relax_first, pre_rest_v):
    """Fit the 2RC model to one pulse window. Returns a result dict or None."""
    t = (window["Time"] - window["Time"].iloc[0]).dt.total_seconds().to_numpy()
    v = window["Voltage"].to_numpy(dtype=float)
    if len(t) < 8 or t_p <= 0:
        return None

    v_rest = float(window["Voltage"].iloc[-1])      # ~settled OCV guess
    i_arr = window["Current"].to_numpy(dtype=float)
    # Model-free instantaneous ohmic jumps (first sample after the step), kept as
    # timescale diagnostics next to R_DC,Δt (the paper stresses Δt-dependence).
    r0_term = abs((v_relax_first - v_pulse_last) / i_pulse)
    r0_guess = max(r0_term, 1e-3)

    # OCV at pulse start: the measured pre-pulse settled voltage. Falls back to
    # the relaxation tail (-> zero ramp = constant OCV) when no pre-rest exists.
    ocv_pre = pre_rest_v if pre_rest_v is not None else v_rest

    # DC pulse resistance R_DC,Δt (Ludwig et al. 2021): ΔU/ΔI over a fixed Δt,
    # using the measured current step. Onset (rested reference) and termination
    # (current-off) give two estimates; both are smooth across aging. The last
    # under-load sample supplies the termination reference current.
    i_load_last = float(i_arr[t <= t_p][-1]) if np.any(t <= t_p) else i_pulse
    r_dc_onset = (
        _r_dc_delta_t(t, v, i_arr, 0.0, R_DC_DELTA_T, pre_rest_v, 0.0)
        if pre_rest_v is not None else np.nan
    )
    r_dc_term = _r_dc_delta_t(t, v, i_arr, t_p, R_DC_DELTA_T, v_pulse_last, i_load_last)
    # Fit-extrapolated termination R0: the ohmic step at current-off with the RC
    # decay extrapolated back to t=0, so it is free of the raw jump's coarse-rest
    # sampling-lag bias. Reported as a cadence-independent R0 candidate.
    r0_extrap_term = _extrap_termination_r0(t, v, t_p, i_pulse, v_pulse_last)
    # Onset-extrapolated R0: pure ohmic intercept read from the densely-sampled
    # pulse rise (the fine 0.18 s side), so it does not fight the coarse rest.
    # Most reproducible pure-ohmic estimate; cross-checks the termination one.
    r0_extrap_onset = (
        _extrap_onset_r0(t, v, t_p, i_pulse, pre_rest_v, r0_cap=r_dc_onset)
        if pre_rest_v is not None else np.nan
    )
    # Coupled staged decomposition: slow branch from the relaxation, its in-pulse
    # contribution removed from the dense onset, then the fast branch — with R0 at
    # the onset-extrapolated ohmic value. Physical/reproducible params for aging;
    # the joint fit above stays the simulation model.
    staged = (
        _staged_branches(t, v, t_p, i_pulse, ocv_pre, r0_extrap_onset)
        if np.isfinite(r0_extrap_onset) else None
    )

    tau2_hi = 6000.0
    # Pin R0 to R_DC,Δt (onset preferred -> rested reference). Pinning removes the
    # R0-vs-fast-RC fit degeneracy (which otherwise makes a fitted R0 hop between
    # check-ups); Δt=0.5 s sits well below the fitted RC time constants (15-200 s)
    # so R0 absorbs only the sub-second resistance and does not double-count the
    # RC branches. Fall back to the termination R_DC, then the jump, if needed.
    r0 = next(
        (x for x in (r_dc_onset, r_dc_term, r0_extrap_term, r0_term)
         if x is not None and np.isfinite(x)),
        r0_term,
    )

    def model(tt, ocv_post, r1, tau1, r2, tau2):
        return _v_2rc(
            tt, ocv_post, r0, r1, tau1, r2, tau2,
            i_pulse=i_pulse, t_p=t_p, ocv_pre=ocv_pre,
        )

    bounds = (
        [v_rest - 0.5, 1e-5, 0.2, 1e-5, 20.0],
        [v_rest + 0.5, 1.0, 60.0, 1.0, tau2_hi],
    )
    # Multi-start over (tau1, tau2) seeds: the two-exponential fit has a 1RC
    # local minimum (R2->0, tau2 rails) that traps a single fixed start when the
    # two time constants are close (aged low-SOC pulses). Keep the lowest-rmse.
    best_popt, best_rmse = None, np.inf
    for tau1_0, tau2_0 in [(5.0, 200.0), (2.0, 30.0), (8.0, 60.0), (3.0, 100.0)]:
        p0 = [v_rest, r0_guess, tau1_0, r0_guess, tau2_0]
        try:
            popt, _ = curve_fit(model, t, v, p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((model(t, *popt) - v) ** 2)))
        if rmse < best_rmse:
            best_popt, best_rmse = popt, rmse
    if best_popt is None:
        logging.warning("fit failed for all seeds")
        return None

    ocv, r1, tau1, r2, tau2 = best_popt
    # order so tau1 < tau2 (fast then slow)
    if tau1 > tau2:
        r1, tau1, r2, tau2 = r2, tau2, r1, tau1
    rmse_mv = best_rmse * 1000
    # flag a collapsed 2nd RC (railed slow branch / vanishing R2) as unreliable
    degenerate = bool(tau2 >= 0.999 * tau2_hi or r2 <= 1e-4)

    # instantaneous onset jump (dV/dt-knee voltage, ramp-robust) — diagnostic
    r0_onset = (
        abs((_onset_step_voltage(t, v, t_p, i_pulse) - pre_rest_v) / i_pulse)
        if pre_rest_v is not None
        else np.nan
    )
    # 2RC + finite-length Warburg (transmissive) fit, reported alongside the joint
    # 2RC — the proper distributed diffusion branch in place of the lumped slow RC
    # (see 2RC_parameters_research_notes.md §2.3/§3.2). R0 pinned to the same value.
    warburg = _fit_2rc_warburg(t, v, t_p, i_pulse, ocv_pre, r0, v_rest, r0_guess)
    vfit_w = warburg.pop("_vfit_w") if warburg else None

    # cross-check the onset vs termination R_DC,Δt (the two pinning candidates);
    # they estimate the same resistance and should agree within 10 %.
    r0_consistent = (
        bool(
            np.isfinite(r_dc_onset) and r_dc_onset > 0
            and abs(r_dc_onset - r_dc_term) <= 0.1 * max(r_dc_onset, r_dc_term)
        )
        if np.isfinite(r_dc_onset) and np.isfinite(r_dc_term)
        else np.nan
    )
    return {
        "OCV_V": round(ocv, 4),
        "OCV_pre_V": round(ocv_pre, 4),
        "dOCV_pulse_mV": round((ocv - ocv_pre) * 1000, 2),
        "R0_ohm": round(r0, 5),
        "R1_ohm": round(r1, 5),
        "tau1_s": round(tau1, 3),
        "C1_F": round(tau1 / r1, 1) if r1 else np.nan,
        "R2_ohm": round(r2, 5),
        "tau2_s": round(tau2, 2),
        "C2_F": round(tau2 / r2, 1) if r2 else np.nan,
        "R_DC_dt_s": R_DC_DELTA_T,
        "R_DC_onset_ohm": round(r_dc_onset, 5) if np.isfinite(r_dc_onset) else np.nan,
        "R_DC_term_ohm": round(r_dc_term, 5) if np.isfinite(r_dc_term) else np.nan,
        "R0_jump_term_ohm": round(r0_term, 5),
        "R0_jump_onset_ohm": round(r0_onset, 5) if not np.isnan(r0_onset) else np.nan,
        "R0_extrap_term_ohm": round(r0_extrap_term, 5) if np.isfinite(r0_extrap_term) else np.nan,
        "R0_extrap_onset_ohm": round(r0_extrap_onset, 5) if np.isfinite(r0_extrap_onset) else np.nan,
        # coupled staged decomposition (R0=onset ohmic, fast from onset, slow from
        # relaxation); NaN-filled when either stage failed to converge
        "R0_staged_ohm": round(r0_extrap_onset, 5) if staged is not None else np.nan,
        "R1_fast_ohm": staged["R1_fast_ohm"] if staged else np.nan,
        "tau1_fast_s": staged["tau1_fast_s"] if staged else np.nan,
        "R2_slow_ohm": staged["R2_slow_ohm"] if staged else np.nan,
        "tau2_slow_s": staged["tau2_slow_s"] if staged else np.nan,
        "staged_rmse_mV": staged["staged_rmse_mV"] if staged else np.nan,
        "R0_consistent": r0_consistent,
        "rmse_mV": round(rmse_mv, 3),
        "degenerate": degenerate,
        # 2RC + finite-length Warburg (transmissive Ws) fit; NaN when it failed to
        # converge. R_d/tau_d are the diffusion resistance / time constant.
        "OCV_w_V": warburg["OCV_w_V"] if warburg else np.nan,
        "R1_w_ohm": warburg["R1_w_ohm"] if warburg else np.nan,
        "tau1_w_s": warburg["tau1_w_s"] if warburg else np.nan,
        "R2_w_ohm": warburg["R2_w_ohm"] if warburg else np.nan,
        "tau2_w_s": warburg["tau2_w_s"] if warburg else np.nan,
        "R_d_ohm": warburg["R_d_ohm"] if warburg else np.nan,
        "tau_d_s": warburg["tau_d_s"] if warburg else np.nan,
        "C_d_F": warburg["C_d_F"] if warburg else np.nan,
        "warburg_rmse_mV": warburg["warburg_rmse_mV"] if warburg else np.nan,
        "warburg_degenerate": warburg["warburg_degenerate"] if warburg else np.nan,
        "n_points": int(len(t)),
        "_t": t, "_v": v, "_vfit": model(t, *popt),  # for plotting
        "_vfit_w": vfit_w,
    }


def _v_relax(t, ocv, v1_0, v2_0, tau1, tau2):
    """Terminal voltage of a 2RC cell relaxing at zero current.

    ``t`` is seconds from the current-off instant. ``v1_0`` / ``v2_0`` are the
    two RC overvoltages at that instant (signed: positive after a charge, the
    cell sits above OCV and decays down). No ohmic term — current is zero.
    """
    t = np.asarray(t, dtype=float)
    return ocv + v1_0 * np.exp(-t / tau1) + v2_0 * np.exp(-t / tau2)


def fit_one_relaxation(relax_rows, i_prev, v_cycle_last, *, t0=None, t_prev=None):
    """Fit the 2RC model to a *rest* curve (e.g. the pause between two cycles).

    Unlike ``fit_one_pulse``, the window carries no current, so:

    * OCV is a single constant (the relaxation asymptote) — there is no
      ``ocv_pre -> ocv_post`` ramp, because SOC does not move during the rest.
    * The fit yields ``{OCV, V1(0), V2(0), tau1, tau2}``. The RC *resistances*
      come from the current that built the polarisation up — ``i_prev``, the
      signed mean current of the cycle step that **ended** at this pause::

          R_k = V_k(0) / (i_prev * (1 - e^{-t_prev/tau_k}))

      With ``t_prev`` unset the step is assumed fully developed (factor -> 1),
      i.e. ``R_k = V_k(0) / i_prev`` (good when the preceding cycle half is long
      vs tau_k, which it usually is).
    * **R0 is reverse-calculated**, not fit. At current-off the ohmic term
      ``i_prev*R0`` vanishes instantly while the RC voltages do not, so the
      ohmic step is ``v_cycle_last - V(0)``:

          R0 (primary)  = (v_cycle_last - (OCV + V1(0) + V2(0))) / i_prev   # fit-extrapolated to t=0
          R0 (jump)     = (v_cycle_last - v_first_rest) / i_prev            # model-free, 2 samples

      The fit-extrapolated form removes the sampling-lag bias in the raw jump
      (the first rest sample is logged after the fast RC has already started
      decaying). ``R0_consistent`` flags when the two agree within 10 %.

    Parameters
    ----------
    relax_rows : DataFrame with ``Time`` and ``Voltage`` (the rest, current ~0),
        sorted by time.
    i_prev : signed mean current (A) of the preceding cycle step.
    v_cycle_last : last terminal voltage (V) under current at the end of that step.
    t0 : Timestamp of the current-off instant (defaults to the first rest sample).
    t_prev : duration (s) of the preceding step, for the development factor
        (defaults to None -> assume fully developed).

    Returns a result dict (same key style as ``fit_one_pulse``) or ``None``.
    """
    rr = relax_rows.sort_values("Time")
    if len(rr) < 8 or abs(i_prev) < REST_CURRENT_A:
        return None
    origin = t0 if t0 is not None else rr["Time"].iloc[0]
    t = (rr["Time"] - origin).dt.total_seconds().to_numpy(dtype=float)
    v = rr["Voltage"].to_numpy(dtype=float)

    v_first_rest = float(v[0])
    v_settled = float(v[-1])                          # ~OCV guess (rest tail)
    amp0 = (v_first_rest - v_settled)                 # total overvoltage at t~0

    tau2_hi = 6000.0
    bounds = (
        [v_settled - 0.5, -0.5, -0.5, 0.2, 20.0],
        [v_settled + 0.5, 0.5, 0.5, 60.0, tau2_hi],
    )
    # Multi-start over (tau1, tau2) seeds — same 1RC local-minimum trap as the
    # pulse fit (R2 -> 0, tau2 rails) when the two time constants are close.
    best_popt, best_rmse = None, np.inf
    for tau1_0, tau2_0 in [(5.0, 200.0), (2.0, 30.0), (8.0, 60.0), (3.0, 100.0)]:
        p0 = [v_settled, amp0 / 2, amp0 / 2, tau1_0, tau2_0]
        try:
            popt, _ = curve_fit(_v_relax, t, v, p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((_v_relax(t, *popt) - v) ** 2)))
        if rmse < best_rmse:
            best_popt, best_rmse = popt, rmse
    if best_popt is None:
        logging.warning("relaxation fit failed for all seeds")
        return None

    ocv, v1_0, v2_0, tau1, tau2 = best_popt
    # order so tau1 < tau2 (fast then slow), carrying the amplitudes along
    if tau1 > tau2:
        v1_0, tau1, v2_0, tau2 = v2_0, tau2, v1_0, tau1
    degenerate = bool(tau2 >= 0.999 * tau2_hi)

    # amplitudes -> resistances via the preceding-step current (+ development)
    dev1 = (1.0 - np.exp(-t_prev / tau1)) if t_prev else 1.0
    dev2 = (1.0 - np.exp(-t_prev / tau2)) if t_prev else 1.0
    r1 = v1_0 / (i_prev * dev1) if dev1 else np.nan
    r2 = v2_0 / (i_prev * dev2) if dev2 else np.nan

    # reverse-calculate R0: ohmic step at current-off
    v_relax0 = ocv + v1_0 + v2_0                       # fit extrapolated to t=0
    r0 = abs((v_cycle_last - v_relax0) / i_prev)
    r0_jump = abs((v_cycle_last - v_first_rest) / i_prev)
    r0_consistent = bool(
        r0 > 0 and abs(r0 - r0_jump) <= 0.1 * max(r0, r0_jump)
    )

    return {
        "OCV_V": round(ocv, 4),
        "R0_ohm": round(r0, 5),
        "R1_ohm": round(r1, 5) if np.isfinite(r1) else np.nan,
        "tau1_s": round(tau1, 3),
        "C1_F": round(tau1 / r1, 1) if np.isfinite(r1) and r1 else np.nan,
        "R2_ohm": round(r2, 5) if np.isfinite(r2) else np.nan,
        "tau2_s": round(tau2, 2),
        "C2_F": round(tau2 / r2, 1) if np.isfinite(r2) and r2 else np.nan,
        "V1_0_mV": round(v1_0 * 1000, 2),
        "V2_0_mV": round(v2_0 * 1000, 2),
        "R0_jump_ohm": round(r0_jump, 5),
        "R0_consistent": r0_consistent,
        "rmse_mV": round(best_rmse * 1000, 3),
        "degenerate": degenerate,
        "n_points": int(len(t)),
        "_t": t, "_v": v, "_vfit": _v_relax(t, *best_popt),  # for plotting
    }


def _proc_id(id_str, delta):
    """``5_21`` + delta=+1 -> ``5_22`` (the BM_Programm prefix is preserved)."""
    bm, _, proc = str(id_str).rpartition("_")
    try:
        return f"{bm}_{int(proc) + delta}"
    except ValueError:
        return None


def fit_2rc(labeled, seg_ids, nom_capacity):
    """Fit every selected pulse segment; return a results DataFrame (+ fit curves).

    Each pulse is paired with the single rest that follows it by ID: pulse
    ``<BM>_<n>`` -> relaxation ``<BM>_<n+1>`` (one ~30-min pause). The pause
    before it (``<BM>_<n-1>``) supplies the settled pre-pulse voltage for the
    onset-R0 cross-check.
    """
    df = labeled.sort_values("Time").reset_index(drop=True)
    seg_bounds = df.groupby("pulse_segment_id")["Time"].agg(["min", "max"])
    rows, curves, records = [], [], []
    for seg_id in seg_ids:
        pulse_rows = df[df["pulse_segment_id"] == seg_id].sort_values("Time")
        if pulse_rows.empty:
            continue
        cur_id = pulse_rows["ID"].iloc[0]
        relax_id = _proc_id(cur_id, +1)
        pre_id = _proc_id(cur_id, -1)
        relax_rows = df[df["ID"] == relax_id].sort_values("Time")
        if relax_rows.empty:
            logging.info("skip pulse %s: no following pause %s", cur_id, relax_id)
            continue

        pstart = seg_bounds.loc[seg_id, "min"]
        pend = seg_bounds.loc[seg_id, "max"]
        i_pulse = float(pulse_rows["Current"].mean())
        if abs(i_pulse) < REST_CURRENT_A:
            continue
        t_p = (pend - pstart).total_seconds()
        v_pulse_last = float(pulse_rows["Voltage"].iloc[-1])
        v_relax_first = float(relax_rows["Voltage"].iloc[0])

        window = pd.concat([pulse_rows, relax_rows]).sort_values("Time")
        # settled voltage from the preceding pause, for the onset-R0 cross-check
        pre_rows = df[(df["ID"] == pre_id) & (df["Current"].abs() < REST_CURRENT_A)]
        pre_rest_v = float(pre_rows["Voltage"].iloc[-1]) if not pre_rows.empty else None

        res = fit_one_pulse(window, t_p, i_pulse, v_pulse_last, v_relax_first, pre_rest_v)
        if res is None:
            continue
        curves.append((seg_id, res.pop("_t"), res.pop("_v"), res.pop("_vfit"),
                       res.pop("_vfit_w")))
        meta = pulse_rows.iloc[0]
        row = {
            "File": meta["File"],
            "SOH": meta["SOH"],
            "SOC": meta["SOC"],
            "pulse_segment_id": seg_id,
            "ID": meta.get("ID", ""),
            "direction": "CHA" if i_pulse > 0 else "DCH",
            "I_A": round(i_pulse, 3),
            "C_rate": round(i_pulse / nom_capacity, 3),
            "pulse_dur_s": round(t_p, 2),
            **res,
        }
        rows.append(row)
        # keep the raw window + params so the pulse can be re-simulated later
        records.append(
            {
                "seg_id": seg_id,
                "ID": row["ID"],
                "direction": row["direction"],
                "I_A": row["I_A"],
                "t": (window["Time"] - window["Time"].iloc[0]).dt.total_seconds().to_numpy(),
                "current": window["Current"].to_numpy(dtype=float),
                "voltage": window["Voltage"].to_numpy(dtype=float),
                "ocv0": pre_rest_v if pre_rest_v is not None else row["OCV_V"],
                "params": (
                    row["R0_ohm"], row["R1_ohm"], row["tau1_s"],
                    row["R2_ohm"], row["tau2_s"],
                ),
            }
        )
    return pd.DataFrame(rows), curves, records


# ---------------------------------------------------------------------------
# Relaxation (inter-cycle pause) fit
# ---------------------------------------------------------------------------
def build_relax_pairs(df, file_name):
    """Label a cycling series into rest / charge / discharge runs.

    Unlike ``label_time_diff`` this does **not** need a ``Zustand`` column — the
    state is derived from current sign/magnitude, so it works on generic cycling
    parquets. A new segment starts on a state change (rest<->cha<->dch) or a time
    gap > ``CYCLE_ACTIVE_LIMIT_HOUR`` (so a step never spans a cycle boundary).
    Returns the labeled frame with a ``state`` and ``seg_id`` column.
    """
    out = df.copy()
    out["File"] = file_name
    out["SOH"] = _parse_soh(os.path.splitext(os.path.basename(file_name))[0])
    out["Time"] = pd.to_datetime(out["Time"], utc=True, errors="coerce")
    out["Current"] = pd.to_numeric(out["Current"], errors="coerce")
    out["Voltage"] = pd.to_numeric(out["Voltage"], errors="coerce")
    out = out.dropna(subset=["Time", "Current", "Voltage"]).copy()
    out = out.sort_values("Time").reset_index(drop=True)

    out["state"] = np.where(
        out["Current"].abs() < REST_CURRENT_A, "rest",
        np.where(out["Current"] > 0, "cha", "dch"),
    )
    gap_h = out["Time"].diff() / pd.Timedelta(hours=1)
    new_seg = (
        out["state"].ne(out["state"].shift())
        | gap_h.isna()
        | (gap_h > CYCLE_ACTIVE_LIMIT_HOUR)
    )
    out["seg_id"] = new_seg.cumsum().astype(int)
    return out


def fit_relax(df, file_name, nom_capacity):
    """Fit every active-step -> following-rest pair in a cycling file.

    For each charge/discharge segment immediately followed by a rest segment,
    the rest curve is fit with ``fit_one_relaxation``; the preceding step
    supplies ``i_prev`` (median current), ``v_cycle_last``, the current-off time
    ``t0`` and the step duration ``t_prev``. Returns a results DataFrame and the
    per-pair fit curves (keyed on the active segment's ``seg_id``).
    """
    labeled = build_relax_pairs(df, file_name)
    segs = list(labeled.groupby("seg_id", sort=True))
    rows, curves = [], []
    for (a_id, a), (_, r) in zip(segs, segs[1:]):
        if a["state"].iloc[0] == "rest" or r["state"].iloc[0] != "rest":
            continue
        if len(r) < 8:
            continue
        i_prev = float(a["Current"].median())          # CC level (robust to a CV tail)
        if abs(i_prev) < REST_CURRENT_A:
            continue
        v_cycle_last = float(a["Voltage"].iloc[-1])
        t0 = a["Time"].iloc[-1]
        t_prev = (a["Time"].iloc[-1] - a["Time"].iloc[0]).total_seconds()
        res = fit_one_relaxation(r, i_prev, v_cycle_last, t0=t0, t_prev=t_prev)
        if res is None:
            continue
        curves.append((a_id, res.pop("_t"), res.pop("_v"), res.pop("_vfit")))
        meta = a.iloc[0]
        rows.append({
            "File": meta["File"],
            "SOH": meta["SOH"],
            "seg_id": a_id,
            "direction": "CHA" if i_prev > 0 else "DCH",
            "I_A": round(i_prev, 3),
            "C_rate": round(i_prev / nom_capacity, 3),
            "step_dur_s": round(t_prev, 1),
            **res,
        })
    return pd.DataFrame(rows), curves


def fit_relax_folder(folder, nom_capacity):
    """Fit relaxations in every ``*.parquet`` under ``folder``; one combined table.

    ``seg_id`` is namespaced per file (``<stem>#<seg_id>``) so ids stay unique
    across the folder. Adds ``SOH_num`` for plotting vs aging where filenames
    carry an SOH tag.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    all_results = []
    for f in files:
        name = os.path.basename(f)
        res, _ = fit_relax(pd.read_parquet(f), name, nom_capacity)
        if res.empty:
            logging.warning("%s: no relaxations fit", name)
            continue
        stem = os.path.splitext(name)[0]
        res["seg_id"] = stem + "#" + res["seg_id"].astype(str)
        all_results.append(res)
        logging.info("%s: fit %d relaxation(s)", name, len(res))
    if not all_results:
        return pd.DataFrame()
    out = pd.concat(all_results, ignore_index=True)
    out["SOH_num"] = pd.to_numeric(out["SOH"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
def plot_fits(curves, results, out_png, id_col="pulse_segment_id"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    if n == 0:
        return
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), squeeze=False)
    for ax, (seg_id, t, v, vfit, *rest) in zip(axes.ravel(), curves):
        row = results[results[id_col] == seg_id].iloc[0]
        ax.plot(t, v, ".", ms=2, label="measured", color="0.5")
        ax.plot(t, vfit, "-", lw=1.5, label="2RC fit", color="C3")
        # overlay the 2RC+Warburg fit when present (pulse curves carry it)
        vfit_w = rest[0] if rest else None
        if vfit_w is not None:
            ax.plot(t, vfit_w, "-", lw=1.2, label="2RC+Warburg", color="C0", alpha=0.8)
        ax.set_title(
            f"{row['direction']} {row['I_A']} A | R0={row['R0_ohm']*1000:.1f} mΩ "
            f"R1={row['R1_ohm']*1000:.1f} R2={row['R2_ohm']*1000:.1f} | rmse={row['rmse_mV']:.1f} mV",
            fontsize=8,
        )
        ax.set_xlabel("t (s)")
        ax.set_ylabel("V")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    logging.info("plot -> %s", out_png)


def simulate_2rc(t, current, ocv, r0, r1, tau1, r2, tau2):
    """Time-domain 2RC voltage for an *arbitrary* current profile (signed A).

    Discrete zero-order-hold update of the two RC voltages — works for any
    I(t), unlike ``_v_2rc`` which assumes a single constant-current pulse.
    ``ocv`` is held constant (SOC change over a pulse is negligible).
    """
    t = np.asarray(t, dtype=float)
    current = np.asarray(current, dtype=float)
    v = np.empty_like(t)
    v1 = v2 = 0.0
    for n in range(len(t)):
        dt = t[n] - t[n - 1] if n > 0 else 0.0
        a1, a2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
        v1 = v1 * a1 + current[n] * r1 * (1.0 - a1)
        v2 = v2 * a2 + current[n] * r2 * (1.0 - a2)
        v[n] = ocv + current[n] * r0 + v1 + v2
    return v


def validate_loo(records):
    """Leave-one-out validation: predict each pulse from the *other* pulses' params.

    Borrowed params = element-wise median of every other pulse's
    ``(R0, R1, tau1, R2, tau2)``. The held-out pulse is then re-simulated from
    its own measured current profile + measured pre-pulse OCV, and compared to
    its measured voltage. Returns a results DataFrame and per-pulse curves.
    """
    rows, curves = [], []
    if len(records) < 2:
        logging.warning("validation needs >= 2 pulses (have %d)", len(records))
        return pd.DataFrame(rows), curves
    for i, rec in enumerate(records):
        others = [r["params"] for j, r in enumerate(records) if j != i]
        r0, r1, tau1, r2, tau2 = np.median(np.array(others), axis=0)
        vsim = simulate_2rc(rec["t"], rec["current"], rec["ocv0"], r0, r1, tau1, r2, tau2)
        vmeas = rec["voltage"]
        rmse_mv = float(np.sqrt(np.mean((vsim - vmeas) ** 2)) * 1000)
        max_mv = float(np.max(np.abs(vsim - vmeas)) * 1000)
        rows.append(
            {
                "ID": rec["ID"],
                "direction": rec["direction"],
                "I_A": rec["I_A"],
                "borrowed_R0_ohm": round(r0, 5),
                "borrowed_R1_ohm": round(r1, 5),
                "borrowed_tau1_s": round(tau1, 2),
                "borrowed_R2_ohm": round(r2, 5),
                "borrowed_tau2_s": round(tau2, 2),
                "val_rmse_mV": round(rmse_mv, 3),
                "val_max_err_mV": round(max_mv, 3),
            }
        )
        curves.append((rec["ID"], rec["t"], vmeas, vsim))
    return pd.DataFrame(rows), curves


def plot_validation(curves, val_df, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    if n == 0:
        return
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), squeeze=False)
    for ax, (pid, t, vmeas, vsim) in zip(axes.ravel(), curves):
        row = val_df[val_df["ID"] == pid].iloc[0]
        ax.plot(t, vmeas, ".", ms=2, label="measured", color="0.5")
        ax.plot(t, vsim, "-", lw=1.5, label="predicted (LOO)", color="C0")
        ax.set_title(
            f"{pid} {row['direction']} {row['I_A']} A | "
            f"val rmse={row['val_rmse_mV']:.1f} mV  max={row['val_max_err_mV']:.1f} mV",
            fontsize=8,
        )
        ax.set_xlabel("t (s)")
        ax.set_ylabel("V")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    logging.info("validation plot -> %s", out_png)


def fit_folder(folder, nom_capacity, remove_before_min, exclude_zc):
    """Fit every pulse file in ``folder`` and return one combined results table.

    Each BM_Programm can have several files (stale stubs + the rehydrated
    export); the **largest** file per BM is the rehydrated one, so that is the
    one kept. Files with SOH=NA are skipped. Adds ``BM_Programm``, ``SOH_num``
    and a ``pulse_type`` label (direction + |current| + SOC) for plotting vs SOH.
    """
    files = glob.glob(os.path.join(folder, "*_pulse_BM*.parquet"))
    best = {}  # BM_Programm -> (size, path), keep the largest (rehydrated) file
    for f in files:
        m = re.search(r"_BM(\d+)_", os.path.basename(f))
        if not m:
            continue
        bm = int(m.group(1))
        size = os.path.getsize(f)
        if bm not in best or size > best[bm][0]:
            best[bm] = (size, f)

    all_results = []
    for bm in sorted(best):
        f = best[bm][1]
        soh = _parse_soh(os.path.splitext(os.path.basename(f))[0])
        if soh == "NA":
            logging.info("skip BM%s: SOH=NA", bm)
            continue
        labeled = label_time_diff(pd.read_parquet(f), os.path.basename(f))
        seg_ids = select_pulse_segments(labeled, remove_before_min, exclude_zc)
        res, _, _ = fit_2rc(labeled, seg_ids, nom_capacity)
        if res.empty:
            logging.warning("BM%s (SOH=%s): no pulses fit", bm, soh)
            continue
        res["BM_Programm"] = bm
        all_results.append(res)
        logging.info("BM%s SOH=%s: fit %d pulses", bm, soh, len(res))

    if not all_results:
        return pd.DataFrame()
    out = pd.concat(all_results, ignore_index=True)
    out["SOH_num"] = pd.to_numeric(out["SOH"], errors="coerce")
    # include SOC so multi-SOC cells (90/50/10) don't collapse onto one line
    out["pulse_type"] = (
        out["direction"] + " " + out["I_A"].abs().round(1).astype(str)
        + "A @" + out["SOC"].astype(str)
    )
    return out.sort_values(["pulse_type", "SOH_num"]).reset_index(drop=True)


def plot_vs_soh(results, out_png, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("R0_ohm", "R0 (mΩ)", 1000),
        ("R1_ohm", "R1 (mΩ)", 1000),
        ("R2_ohm", "R2 (mΩ)", 1000),
        ("tau1_s", "τ1 (s)", 1),
        ("tau2_s", "τ2 (s)", 1),
        ("rmse_mV", "fit rmse (mV)", 1),
    ]
    # drop railed/collapsed fits so they don't distort the trends
    if "degenerate" in results.columns:
        n_bad = int(results["degenerate"].sum())
        if n_bad:
            logging.info("vs-SOH plot: hiding %d degenerate fit(s)", n_bad)
        results = results[~results["degenerate"]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        for ptype, g in results.groupby("pulse_type"):
            g = g.sort_values("SOH_num")
            ax.plot(g["SOH_num"], g[col] * scale, "o-", ms=4, label=ptype)
        ax.set_xlabel("SOH (%)")
        ax.set_ylabel(label)
        ax.invert_xaxis()  # aging reads left (fresh) -> right (aged)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, title="pulse")
    fig.suptitle(f"2RC parameters vs SOH — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logging.info("vs-SOH plot -> %s", out_png)


def plot_vs_soc(results, out_png, title=""):
    """Plot the 2RC parameters vs SOC across a single-checkup SOC sweep.

    Companion to ``plot_vs_soh`` (which walks aging at fixed SOC). Needs the
    numeric ``SOC_pct`` column from ``assign_pulse_soc``; charge and discharge
    pulses are drawn as separate series (one line per direction × amplitude), so
    the CHA/DCH resistance asymmetry is visible. Degenerate fits are hidden.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "SOC_pct" not in results.columns:
        logging.info("vs-SOC plot: no SOC_pct column, skipping")
        return
    metrics = [
        ("R0_ohm", "R0 (mΩ)", 1000),
        ("R1_ohm", "R1 (mΩ)", 1000),
        ("R2_ohm", "R2 (mΩ)", 1000),
        ("tau1_s", "τ1 (s)", 1),
        ("tau2_s", "τ2 (s)", 1),
        ("rmse_mV", "fit rmse (mV)", 1),
    ]
    good = results.dropna(subset=["SOC_pct"])
    if "degenerate" in good.columns:
        n_bad = int(good["degenerate"].sum())
        if n_bad:
            logging.info("vs-SOC plot: hiding %d degenerate fit(s)", n_bad)
        good = good[~good["degenerate"]]
    if good.empty:
        logging.info("vs-SOC plot: nothing to plot")
        return

    # one series per (direction, amplitude); DCH cool, CHA warm
    good = good.copy()
    good["series"] = good["direction"] + " " + good["pulse_amplitude_A"].round(1).astype(str) + " A"
    colors = {"DCH": "#2f6fdb", "CHA": "#e08a1e"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        for series, g in good.groupby("series"):
            g = g.sort_values("SOC_pct")
            direction = g["direction"].iloc[0]
            ax.plot(g["SOC_pct"], g[col] * scale, "o-", ms=4, label=series,
                    color=colors.get(direction))
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, title="pulse")
    fig.suptitle(f"2RC parameters vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logging.info("vs-SOC plot -> %s", out_png)


def plot_warburg_vs_soc(results, out_png, title=""):
    """Plot all 7 2RC+Warburg parameters vs SOC across a single-checkup sweep.

    SOC-sweep companion to ``plot_warburg_vs_soh``: the finite-length-Warburg
    parameter set — R0 (pinned ohmic), the two RC branches (R1/τ1, R2/τ2) and the
    diffusion branch (R_d/τ_d) — plus ``warburg_rmse_mV`` on a 2×4 grid, with
    charge and discharge pulses as separate series. Needs the numeric ``SOC_pct``
    column from ``assign_pulse_soc``; rows where the Warburg fit did not converge
    (NaN) or collapsed (``warburg_degenerate``) are hidden.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "SOC_pct" not in results.columns:
        logging.info("Warburg vs-SOC plot: no SOC_pct column, skipping")
        return
    if "R_d_ohm" not in results.columns:
        logging.info("Warburg vs-SOC plot: no Warburg columns, skipping")
        return
    metrics = [
        ("R0_ohm", "R0 ohmic (mΩ)", 1000),
        ("R1_w_ohm", "R1 (mΩ)", 1000),
        ("tau1_w_s", "τ1 (s)", 1),
        ("R2_w_ohm", "R2 (mΩ)", 1000),
        ("tau2_w_s", "τ2 (s)", 1),
        ("R_d_ohm", "R_d diffusion (mΩ)", 1000),
        ("tau_d_s", "τ_d diffusion (s)", 1),
        ("warburg_rmse_mV", "2RC+W rmse (mV)", 1),
    ]
    good = results.dropna(subset=["SOC_pct", "R_d_ohm", "tau_d_s"])
    if "warburg_degenerate" in good.columns:
        good = good[good["warburg_degenerate"] != True]  # noqa: E712 — keep False/NaN
    n_drop = len(results) - len(good)
    if n_drop:
        logging.info("Warburg vs-SOC plot: hiding %d non-converged/degenerate fit(s)", n_drop)
    if good.empty:
        logging.info("Warburg vs-SOC plot: nothing to plot")
        return

    good = good.copy()
    good["series"] = good["direction"] + " " + good["pulse_amplitude_A"].round(1).astype(str) + " A"
    colors = {"DCH": "#2f6fdb", "CHA": "#e08a1e"}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        for series, g in good.groupby("series"):
            g = g.sort_values("SOC_pct")
            direction = g["direction"].iloc[0]
            ax.plot(g["SOC_pct"], g[col] * scale, "o-", ms=4, label=series,
                    color=colors.get(direction))
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, title="pulse")
    fig.suptitle(f"2RC + finite-length Warburg vs SOC — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logging.info("Warburg vs-SOC plot -> %s", out_png)


def plot_staged_vs_soh(results, out_png, title=""):
    """Plot the coupled staged decomposition (pure-ohmic R0 + fast/slow branches) vs SOH.

    Companion to ``plot_vs_soh`` (which shows the joint-fit params). Uses the
    ``R0_staged``/``R1_fast``/``R2_slow`` columns — the physically-separated set —
    and hides rows where the staged fit did not converge (NaN) or whose full-curve
    ``staged_rmse_mV`` shows it did not reconstruct the pulse.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R0_staged_ohm" not in results.columns:
        logging.info("staged vs-SOH plot: no staged columns, skipping")
        return
    metrics = [
        ("R0_staged_ohm", "R0 ohmic (mΩ)", 1000),
        ("R1_fast_ohm", "R1 fast (mΩ)", 1000),
        ("tau1_fast_s", "τ1 fast (s)", 1),
        ("R2_slow_ohm", "R2 slow (mΩ)", 1000),
        ("tau2_slow_s", "τ2 slow (s)", 1),
        ("staged_rmse_mV", "staged rmse (mV)", 1),
    ]
    # keep only rows where the staged fit converged and reconstructed the curve
    good = results.dropna(subset=["R0_staged_ohm", "R1_fast_ohm", "R2_slow_ohm"])
    # also drop ill-posed R0 estimates: a collapsed joint 2RC branch (degenerate)
    # and pulses whose onset/termination R_DC disagree (R0_consistent False) both
    # make the extrapolated ohmic intercept unreliable and cause the vs-SOH rumble.
    if "degenerate" in good.columns:
        good = good[~good["degenerate"].fillna(False)]
    if "R0_consistent" in good.columns:
        good = good[good["R0_consistent"] != False]  # noqa: E712 — keep True and NaN
    n_drop = len(results) - len(good)
    if n_drop:
        logging.info("staged vs-SOH plot: hiding %d unreliable/non-converged staged fit(s)", n_drop)
    if good.empty:
        logging.info("staged vs-SOH plot: nothing to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        for ptype, g in good.groupby("pulse_type"):
            g = g.sort_values("SOH_num")
            ax.plot(g["SOH_num"], g[col] * scale, "o-", ms=4, label=ptype)
        ax.set_xlabel("SOH (%)")
        ax.set_ylabel(label)
        ax.invert_xaxis()  # aging reads left (fresh) -> right (aged)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, title="pulse")
    fig.suptitle(f"Staged 2RC (pure-ohmic R0 + fast/slow) vs SOH — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logging.info("staged vs-SOH plot -> %s", out_png)


def plot_warburg_vs_soh(results, out_png, title=""):
    """Plot all 7 2RC+Warburg parameters vs SOH (+ the fit RMSE).

    Companion to ``plot_vs_soh``. Shows the full parameter set of the
    finite-length-Warburg fit — R0 (pinned ohmic), the two RC branches
    (R1/τ1, R2/τ2) and the diffusion branch (R_d/τ_d) — plus ``warburg_rmse_mV``,
    on a 2×4 grid. Rows where the Warburg fit did not converge (NaN) or collapsed
    (``warburg_degenerate``) are hidden.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "R_d_ohm" not in results.columns:
        logging.info("Warburg vs-SOH plot: no Warburg columns, skipping")
        return
    # all 7 model parameters (R0 is pinned -> the shared R0_ohm column) + fit rmse
    metrics = [
        ("R0_ohm", "R0 ohmic (mΩ)", 1000),
        ("R1_w_ohm", "R1 (mΩ)", 1000),
        ("tau1_w_s", "τ1 (s)", 1),
        ("R2_w_ohm", "R2 (mΩ)", 1000),
        ("tau2_w_s", "τ2 (s)", 1),
        ("R_d_ohm", "R_d diffusion (mΩ)", 1000),
        ("tau_d_s", "τ_d diffusion (s)", 1),
        ("warburg_rmse_mV", "2RC+W rmse (mV)", 1),
    ]
    good = results.dropna(subset=["R_d_ohm", "tau_d_s"])
    if "warburg_degenerate" in good.columns:
        good = good[good["warburg_degenerate"] != True]  # noqa: E712 — keep False/NaN
    n_drop = len(results) - len(good)
    if n_drop:
        logging.info("Warburg vs-SOH plot: hiding %d non-converged/degenerate fit(s)", n_drop)
    if good.empty:
        logging.info("Warburg vs-SOH plot: nothing to plot")
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, label, scale) in zip(axes.ravel(), metrics):
        for ptype, g in good.groupby("pulse_type"):
            g = g.sort_values("SOH_num")
            ax.plot(g["SOH_num"], g[col] * scale, "o-", ms=4, label=ptype)
        ax.set_xlabel("SOH (%)")
        ax.set_ylabel(label)
        ax.invert_xaxis()  # aging reads left (fresh) -> right (aged)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, title="pulse")
    fig.suptitle(f"2RC + finite-length Warburg vs SOH — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logging.info("Warburg vs-SOH plot -> %s", out_png)


def run_cell_folder(folder, args, out_csv=None):
    """Fit every pulse export in one cell folder and plot the params vs SOH.

    Outputs land beside the data, in the cell folder itself: ``2RC_vs_SOH.csv``
    plus the joint / staged / Warburg vs-SOH plots. Returns the results frame
    (empty when nothing fit), so a multi-cell run can concatenate them.
    """
    results = fit_folder(
        folder, args.nom_capacity, args.remove_pulse_before_min, args.exclude_zc
    )
    if results.empty:
        logging.warning("no pulses fit in %s", folder)
        return results

    folder_title = os.path.basename(os.path.normpath(folder))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols = ["BM_Programm", "SOH_num", "pulse_type", "R0_ohm", "R1_ohm",
            "tau1_s", "R2_ohm", "tau2_s", "rmse_mV"]
    print(f"\n=== 2RC parameters across folder — {folder_title} ===")
    print(results[cols].to_string(index=False))

    out_csv = out_csv or os.path.join(folder, "2RC_vs_SOH.csv")
    results.to_csv(out_csv, index=False)
    logging.info("combined results -> %s", out_csv)

    plot_vs_soh(results, os.path.join(folder, "2RC_vs_SOH.png"), title=folder_title)
    plot_staged_vs_soh(
        results, os.path.join(folder, "2RC_staged_vs_SOH.png"), title=folder_title,
    )
    plot_warburg_vs_soh(
        results, os.path.join(folder, "2RC_warburg_vs_SOH.png"), title=folder_title,
    )
    return results


def run_all_cells(battery_cfg, args):
    """Fit every cell folder under ``<working_path>/20_export_pulse``.

    One ``run_cell_folder`` pass per cell stem, each writing its own CSV/plots,
    plus a combined table with a ``cell`` column at the export root. A cell that
    raises is logged and skipped so one bad export cannot abort the fleet run.

    Without ``--cell`` the selection falls back to the config's ``type_cell``
    fragment (``main.py``'s own convention). This matters: one ``working_path``
    can hold several chemistries side by side, and ``nom_capacity`` comes from
    the config — fitting an A123 folder with a VTC6 capacity silently
    misnormalizes every C-rate.
    """
    cell_filters = args.cell
    if not cell_filters and battery_cfg.get("type_cell"):
        cell_filters = [battery_cfg["type_cell"]]
        logging.info("no --cell: restricting to type_cell=%r", cell_filters[0])
    folders = resolve_cell_folders(battery_cfg, cell_filters)
    frames, failed = [], []
    for folder in folders:
        cell = os.path.basename(os.path.normpath(folder))
        logging.info("=== %s ===", cell)
        try:
            results = run_cell_folder(folder, args)
        except Exception as exc:  # noqa: BLE001 — one cell must not kill the run
            logging.warning("%s: fit failed (%s)", cell, exc)
            failed.append(cell)
            continue
        if results.empty:
            continue
        frames.append(results.assign(cell=cell))

    if not frames:
        logging.warning("no pulses fit in any of the %d cell folder(s)", len(folders))
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["cell"] + [c for c in combined.columns if c != "cell"]]
    root = os.path.join(battery_cfg["working_path"], PULSE_EXPORT_DIR)
    out_csv = args.out or os.path.join(root, "2RC_vs_SOH_all_cells.csv")
    combined.to_csv(out_csv, index=False)
    logging.info(
        "%d cell(s), %d pulse fit(s) -> %s",
        combined["cell"].nunique(), len(combined), out_csv,
    )
    if failed:
        logging.warning("%d cell(s) failed: %s", len(failed), ", ".join(failed))


def main():
    ap = argparse.ArgumentParser(description="Fit a 2RC ECM to HPPC pulse exports.")
    ap.add_argument(
        "parquet", nargs="?",
        help="pulse parquet, or a cell folder of them. Omit and pass "
        "--battery-config to fit every cell under <working_path>/20_export_pulse.",
    )
    ap.add_argument(
        "--battery-config",
        help="the pipeline battery config (as given to main.py): supplies "
        "working_path (-> 20_export_pulse/<cell_stem>/) and nom_capacity",
    )
    ap.add_argument(
        "--cell", nargs="*",
        help="with --battery-config: only cells whose folder name contains one "
        "of these fragments (same semantics as main.py --cells). Default: all.",
    )
    ap.add_argument(
        "--nom-capacity", type=float,
        help=f"Ah (VTC6={DEFAULT_NOM_CAPACITY}). Default: from --battery-config, "
        f"else {DEFAULT_NOM_CAPACITY}. A --config value wins over both.",
    )
    ap.add_argument(
        "--remove-pulse-before-min", type=float, default=REMOVE_PULSE_BEFORE_MIN,
        help="drop pulses earlier than this many minutes into a cycle (0=keep all)",
    )
    ap.add_argument(
        "--exclude-zc", nargs="*", default=EXCLUDE_ZUSTAND_CURRENT,
        help="Zustand/Current labels to exclude from the fit (e.g. DCH/-1.5)",
    )
    ap.add_argument(
        "--config",
        help="dedicated 2RC JSON config (see config_2rc_example.json): pulse gate, "
        "nom_capacity and SOC-sweep params. Config values win over the flag defaults.",
    )
    ap.add_argument(
        "-o", "--out",
        help="output CSV (default: <stem>_2RC.csv; per-folder 2RC_vs_SOH.csv; "
        "all-cells 2RC_vs_SOH_all_cells.csv at the export root)",
    )
    ap.add_argument("--plot", action="store_true", help="also save a fit overlay PNG")
    ap.add_argument(
        "--validate", action="store_true",
        help="leave-one-out validation: predict each pulse from the others' params",
    )
    ap.add_argument(
        "--relax", action="store_true",
        help="fit the inter-cycle rest curves instead of pulses: pair each "
        "charge/discharge step with the pause that follows it, fit "
        "{OCV,R1,tau1,R2,tau2} from the relaxation and reverse-calc R0.",
    )
    args = ap.parse_args()

    # nom_capacity precedence: --config (cell-specific) > --nom-capacity >
    # battery config > module default.
    battery_cfg = {}
    if args.battery_config:
        battery_cfg = load_battery_config(args.battery_config)
        if args.nom_capacity is None and battery_cfg.get("nom_capacity") is not None:
            args.nom_capacity = float(battery_cfg["nom_capacity"])

    # A dedicated config overrides the module tunables (applied to globals) and the
    # flag defaults for nom_capacity / exclude-zc / remove-before.
    cfg = {}
    if args.config:
        cfg = load_2rc_config(args.config)
        if cfg.get("nom_capacity") is not None:
            args.nom_capacity = cfg["nom_capacity"]
        args.exclude_zc = EXCLUDE_ZUSTAND_CURRENT
        args.remove_pulse_before_min = REMOVE_PULSE_BEFORE_MIN

    if args.nom_capacity is None:
        args.nom_capacity = DEFAULT_NOM_CAPACITY
    logging.info("nom_capacity = %.3f Ah", args.nom_capacity)

    # No explicit path: fit every cell folder the battery config points at.
    if args.parquet is None:
        if not battery_cfg:
            ap.error(
                "give a pulse parquet/folder, or --battery-config <cfg.json> to "
                "fit every cell under <working_path>/20_export_pulse"
            )
        if args.relax:
            ap.error("--relax needs an explicit path (it reads cycling data, "
                     "not the pulse exports)")
        try:
            run_all_cells(battery_cfg, args)
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))  # a config/layout problem, not a stack trace
        return
    if args.cell:
        logging.info("--cell ignored: an explicit path was given")

    # Relaxation mode: fit rest curves (cycling data), not HPPC pulses.
    if args.relax:
        if os.path.isdir(args.parquet):
            results = fit_relax_folder(args.parquet, args.nom_capacity)
            if results.empty:
                logging.warning("no relaxations fit in %s", args.parquet)
                return
            out_csv = args.out or os.path.join(args.parquet, "relax_2RC.csv")
        else:
            results, curves = fit_relax(
                pd.read_parquet(args.parquet),
                os.path.basename(args.parquet),
                args.nom_capacity,
            )
            if results.empty:
                logging.warning("no relaxations fit")
                return
            stem = os.path.splitext(args.parquet)[0]
            out_csv = args.out or f"{stem}_relax_2RC.csv"
            if args.plot:
                plot_fits(curves, results, f"{stem}_relax_2RC.png", id_col="seg_id")
        pd.set_option("display.width", 200, "display.max_columns", 30)
        cols = ["File", "seg_id", "direction", "I_A", "OCV_V", "R0_ohm",
                "R0_jump_ohm", "R0_consistent", "R1_ohm", "tau1_s", "R2_ohm",
                "tau2_s", "rmse_mV", "degenerate"]
        cols = [c for c in cols if c in results.columns]
        print("\n=== 2RC relaxation fit results ===")
        print(results[cols].to_string(index=False))
        results.to_csv(out_csv, index=False)
        logging.info("results -> %s", out_csv)
        return

    # Folder mode: fit every pulse file and plot the parameters vs SOH.
    if os.path.isdir(args.parquet):
        run_cell_folder(args.parquet, args, out_csv=args.out)
        return

    df = pd.read_parquet(args.parquet)
    file_name = os.path.basename(args.parquet)
    labeled = label_time_diff(df, file_name)

    n_cycles = labeled["cycle_id"].nunique()
    logging.info(
        "%s: %d rows, %d cycle(s) -> SOC=%s",
        file_name, len(labeled), n_cycles, sorted(labeled["SOC"].unique()),
    )

    seq = build_pulse_sequence(labeled, OUTPUT_COLUMNS)
    logging.info("pulse_sequence (%d representative pulses):\n%s", len(seq), seq.to_string())

    seg_ids = select_pulse_segments(
        labeled, args.remove_pulse_before_min, args.exclude_zc
    )
    results, curves, records = fit_2rc(labeled, seg_ids, args.nom_capacity)
    if results.empty:
        logging.warning("no pulses fit")
        return

    # SOC-sweep labeling: attach amplitude / direction / numeric SOC per pulse and,
    # when it resolves, plot the 2RC parameters vs SOC (a single-checkup sweep).
    soc_seg = assign_pulse_soc(labeled, args.nom_capacity)
    if not soc_seg.empty:
        results = results.merge(
            soc_seg[["pulse_segment_id", "pulse_amplitude_A", "pulse_C_rate",
                     "soc_plateau", "SOC_pct"]],
            on="pulse_segment_id", how="left",
        )

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== 2RC fit results ===")
    print(results.to_string(index=False))

    stem = os.path.splitext(args.parquet)[0]
    out_csv = args.out or f"{stem}_2RC.csv"
    results.to_csv(out_csv, index=False)
    logging.info("results -> %s", out_csv)

    if args.plot:
        plot_fits(curves, results, f"{stem}_2RC.png")
        if "SOC_pct" in results.columns and results["SOC_pct"].notna().any():
            plot_vs_soc(results, f"{stem}_2RC_vs_SOC.png",
                        title=os.path.basename(stem))
            plot_warburg_vs_soc(results, f"{stem}_2RC_warburg_vs_SOC.png",
                                title=os.path.basename(stem))

    if args.validate:
        val_df, val_curves = validate_loo(records)
        if not val_df.empty:
            print("\n=== leave-one-out validation (predict each pulse from the others) ===")
            print(val_df.to_string(index=False))
            val_df.to_csv(f"{stem}_2RC_validation.csv", index=False)
            logging.info("validation -> %s", f"{stem}_2RC_validation.csv")
            plot_validation(val_curves, val_df, f"{stem}_2RC_validation.png")


# user-supplied output columns for the collapsed pulse sequence
OUTPUT_COLUMNS = [
    "SOH", "SOC", "File", "Time", "Current", "Zustand", "ID", "Zustand/Current",
]

if __name__ == "__main__":
    main()
