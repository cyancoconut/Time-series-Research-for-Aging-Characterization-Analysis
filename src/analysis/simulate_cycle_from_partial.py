"""Predict a full discharge from a *partial* cycling discharge (100 -> 40% SOC).

Demonstrates the practically useful case: field/cycling data rarely contains a
full discharge, only partial swings. Given one partial discharge from an aging
cycling test (here 003's 70%SOC / 60%DOD / 0.5C block) plus a 2RC ECM and a
qOCV(SOC) reference from a nearby check-up, we:

  1. drive the 2RC+OCV model over the measured partial discharge (100->40%) and
     check it reproduces the measured voltage (validation on data we have);
  2. extrapolate the model past 40% down to the cutoff -> predicted full
     discharge curve + capacity (no SOH injected);
  3. validate the *extrapolated* lower half (40->0%) against the measured
     ``jri_Discharge_C2`` segment that lives in the same file and reaches 2.5 V.

Non-circular: the OCV(Ah) reference and RC come from an independent check-up;
capacity falls out of where the simulated voltage hits the cutoff; and the
predicted region is checked against a real discharge never used to build it.

Usage (from src/):  python -m analysis.simulate_cycle_from_partial
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

DATA = "/home/ann/Documents/Data_Metabatt"
CELL = "METABatt_Sony_Murata_18650VTC6_003"
CYCLING_FILE = (
    f"{DATA}/J8005_BMWK_METABatt={CELL}=2024-10-03_142512="
    "jri_Aging_VTC6_Cyc_25grad_70SOC_60DOD_05C=TS014653 _ Format01="
    "Kreis M3-034=filesize-109888838=finished.parquet"
)
V_MIN = 2.5
NOM = 3.0


def _cum_ah(t_s, current):
    """Cumulative |Ah| removed along a segment (trapezoid on |I| vs time)."""
    dt = np.diff(t_s, prepend=t_s[0])
    return np.cumsum(np.abs(current) * dt) / 3600.0


def extract_segments(path):
    """Return (partial 100->40 discharge, jri_Discharge_C2 lower discharge)."""
    df = pd.read_parquet(path, columns=["Zeit", "Spannung", "Strom", "Zustand", "Prozedur", "AhAkku"])
    df["Zeit"] = pd.to_datetime(df["Zeit"])
    df = df.reset_index(drop=True)
    seg = (df["Prozedur"].ne(df["Prozedur"].shift()) | df["Zustand"].ne(df["Zustand"].shift())).cumsum()

    partial = lower = None
    for _, g in df.groupby(seg):
        if g["Zustand"].iloc[0] != "DCH" or len(g) < 2000:
            continue
        proc = str(g["Prozedur"].iloc[0])
        if partial is None and "Aging" in proc:
            partial = g
        if lower is None and "Discharge_C2" in proc and g["Spannung"].min() <= 2.55:
            lower = g
        if partial is not None and lower is not None:
            break
    return partial, lower


def load_reference(cell, bm):
    """OCV(Ah_removed) from the check-up qOCV (with IR correction) + RC params."""
    r = pd.read_csv(f"{DATA}/20_export_pulse/{cell}/2RC_vs_SOH.csv")
    row = r[(r["BM_Programm"] == bm) & (r["pulse_type"] == "DCH 3.0A @50%")].iloc[0]
    rc = {k: float(row[k]) for k in ("R0_ohm", "R1_ohm", "tau1_s", "R2_ohm", "tau2_s")}
    soh = float(row["SOH_num"])

    qf = glob.glob(f"{DATA}/30_export_qocv/{cell}/*qocv_dch_BM{bm}_*.parquet")[0]
    q = pd.read_parquet(qf).sort_values("Time")
    ah = q["Ah_throughput"].to_numpy()
    ah_removed = ah - ah[0]                          # 0 (full) -> Q (empty)
    ocv = q["Voltage"].to_numpy() + abs(float(q["Current"].mean())) * rc["R0_ohm"]
    order = np.argsort(ah_removed)
    return rc, soh, ah_removed[order], ocv[order]


def ocv_of(ah, grid, ocv):
    return np.interp(ah, grid, ocv)


def ah_of_ocv(ocv_val, grid, ocv):
    """Inverse: Ah_removed at a given OCV (ocv decreases as ah increases)."""
    return float(np.interp(ocv_val, ocv[::-1], grid[::-1]))


def simulate_over(rc, grid, ocv, ah_start, t_s, current):
    """Drive the 2RC+OCV model over a measured segment (RC from zero at start).

    Anchored at ``ah_start`` on the OCV(Ah) curve; Ah advances by the measured
    current. Returns the modelled voltage aligned to the segment samples.
    """
    a1, a2 = np.exp(-np.diff(t_s, prepend=t_s[0]) / rc["tau1_s"]), \
             np.exp(-np.diff(t_s, prepend=t_s[0]) / rc["tau2_s"])
    v = np.empty(len(t_s))
    ah, v1, v2 = ah_start, 0.0, 0.0
    dt = np.diff(t_s, prepend=t_s[0])
    for k in range(len(t_s)):
        i = -abs(current[k])
        v1 = v1 * a1[k] + i * rc["R1_ohm"] * (1 - a1[k])
        v2 = v2 * a2[k] + i * rc["R2_ohm"] * (1 - a2[k])
        v[k] = ocv_of(ah, grid, ocv) + i * rc["R0_ohm"] + v1 + v2
        ah += abs(i) * dt[k] / 3600.0
    return v


def simulate_full(rc, grid, ocv, current, dt=1.0):
    """CC discharge from full (Ah=0) to the cutoff. Returns (Ah_removed, V)."""
    i = -abs(current)
    a1, a2 = np.exp(-dt / rc["tau1_s"]), np.exp(-dt / rc["tau2_s"])
    ah, v1, v2 = 0.0, 0.0, 0.0
    out_ah, out_v = [], []
    while True:
        vt = ocv_of(ah, grid, ocv) + i * rc["R0_ohm"] + v1 + v2
        out_ah.append(ah)
        out_v.append(vt)
        if vt <= V_MIN:
            break
        v1 = v1 * a1 + i * rc["R1_ohm"] * (1 - a1)
        v2 = v2 * a2 + i * rc["R2_ohm"] * (1 - a2)
        ah += abs(i) * dt / 3600.0
        if ah > 1.5 * grid[-1]:           # safety stop
            break
    return np.array(out_ah), np.array(out_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", type=int, default=7, help="reference check-up BM_Programm")
    args = ap.parse_args()

    partial, lower = extract_segments(CYCLING_FILE)
    rc, soh, grid, ocv = load_reference(CELL, args.bm)

    # --- simulate the full discharge at the cycling rate (predicted curve) ---
    i_cyc = abs(float(partial["Strom"].mean()))
    sim_ah, sim_v = simulate_full(rc, grid, ocv, i_cyc)
    q_pred = sim_ah[-1]

    # --- measured partial discharge (100 -> 40%) ---
    # Anchor on the OCV(Ah) curve by its starting OCV (V0 under load + the IR
    # drop, since RC ~ 0 at the discharge onset) -- it does not start at full.
    pt = (partial["Zeit"] - partial["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    p_cum = _cum_ah(pt, partial["Strom"].to_numpy())
    p_v = partial["Spannung"].to_numpy()
    p_ah0 = ah_of_ocv(p_v[0] + i_cyc * rc["R0_ohm"], grid, ocv)
    p_ah = p_ah0 + p_cum
    p_sim = simulate_over(rc, grid, ocv, p_ah0, pt, partial["Strom"].to_numpy())
    rmse_partial = np.sqrt(np.mean((p_sim - p_v) ** 2)) * 1000

    # --- measured lower discharge (jri_Discharge_C2, 40 -> 0%) ---
    # Anchored by its END: it genuinely reaches empty (2.5 V), so it must line
    # up with the predicted curve's cutoff at Ah = q_pred.
    lt = (lower["Zeit"] - lower["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    l_cum = _cum_ah(lt, lower["Strom"].to_numpy())
    l_v = lower["Spannung"].to_numpy()
    l_ah0 = q_pred - l_cum[-1]
    l_ah = l_ah0 + l_cum
    l_sim = simulate_over(rc, grid, ocv, l_ah0, lt, lower["Strom"].to_numpy())
    rmse_lower = np.sqrt(np.mean((l_sim - l_v) ** 2)) * 1000

    print(f"reference check-up BM{args.bm}  SOH {soh}%")
    print(f"RC: R0={rc['R0_ohm']*1000:.1f} R1={rc['R1_ohm']*1000:.1f} R2={rc['R2_ohm']*1000:.1f} mOhm  "
          f"tau1={rc['tau1_s']:.1f} tau2={rc['tau2_s']:.1f} s")
    print(f"partial cycling DCH: {p_v[0]:.3f}->{p_v[-1]:.3f} V, {p_cum[-1]:.3f} Ah at {i_cyc:.2f} A")
    print(f"  anchored at Ah_removed {p_ah0:.3f} (OCV {p_v[0]+i_cyc*rc['R0_ohm']:.3f} V) -> ends {p_ah[-1]:.3f} Ah")
    print(f"  -> implied capacity {p_cum[-1]/0.6:.3f} Ah (60% DOD), SOH ~{p_cum[-1]/0.6/NOM*100:.1f}%")
    print(f"PREDICTED full discharge: {q_pred:.3f} Ah to {V_MIN} V  -> SOH {q_pred/NOM*100:.1f}%")
    print(f"validation RMSE: partial(100->40%)={rmse_partial:.1f} mV   "
          f"extrapolated(40->0%) vs jri_Discharge_C2={rmse_lower:.1f} mV")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sim_ah, sim_v, "-", color="C3", lw=1.8,
            label=f"2RC+qOCV predicted full @ {i_cyc:.1f} A")
    ax.plot(p_ah, p_v, "-", color="0.35", lw=2.5,
            label=f"measured partial cycle (100->40%, RMSE {rmse_partial:.1f} mV)")
    ax.plot(l_ah, l_v, "-", color="C0", lw=2.0,
            label=f"measured jri_Discharge_C2 (40->0%, RMSE {rmse_lower:.1f} mV)")
    ax.axhline(V_MIN, color="0.6", ls=":", lw=1)
    ax.axvline(p_ah[-1], color="green", ls="--", lw=1, label="40% SOC (end of partial)")
    ax.set_xlabel("Ah removed from full")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{CELL}: full discharge predicted from a partial cycle  (ref BM{args.bm}, {soh}% SOH)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = f"{DATA}/20_export_pulse/{CELL}/sim_cycle_from_partial_BM{args.bm}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
