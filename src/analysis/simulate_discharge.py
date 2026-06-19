"""Simulate a full constant-current discharge from a 2RC ECM + qOCV curve.

Combines three per-cell artifacts (all keyed by check-up / SOH):
  * 2RC parameters  -> ``20_export_pulse/<cell>/2RC_vs_SOH.csv``  (R0,R1,tau1,R2,tau2)
  * OCV(SOC) curve  -> ``30_export_qocv/<cell>/*qocv_dch_BM<bm>_*``  (V vs SOC)
  * capacity Q      -> ``40_capacity_monitore/<cell>_capacity.csv``  (Capacity_py)

Model (constant-current discharge, signed current I < 0):
    SOC(t) = SOC0 - |I|*t / Q                     (coulomb counting)
    OCV(t) = interp(qOCV curve, SOC(t))
    V1,V2  : discrete 2RC update under I
    V(t)   = OCV(t) + I*R0 + V1 + V2              (stop at V_min)

**Caveat for single-SOC cells (e.g. 003):** the 2RC params exist at only one
SOC, so they are held constant over the whole discharge. Good in the flat
mid-SOC region; under-predicts the polarization sag near empty, where R2/tau2
genuinely grow (visible on multi-SOC cells).

Validation: the capacity check-up itself is a measured full discharge (~C/2),
read back from GOLD and overlaid.

Usage (from src/):
    python -m analysis.simulate_discharge [cell_stem] [--bm 5] [--rate-A 1.5]
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

DATA = "/home/ann/Documents/Data_Metabatt"
DEFAULT_CELL = "METABatt_Sony_Murata_18650VTC6_003"
V_MIN = 2.5
NOM_CAPACITY = 3.0


def _bm_from_name(path):
    m = re.search(r"_BM(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_ocv_soc(cell, bm, r0):
    """Build OCV(SOC) from the qOCV discharge for this check-up.

    SOC is the *fractional* state of charge over the qOCV sweep (1 at the top,
    0 at the cutoff). The small IR offset of the C/20 sweep is added back so the
    curve is true OCV: during discharge the terminal V sits I*R0 below OCV.
    Returns (soc_grid ascending, ocv) for interpolation.
    """
    files = glob.glob(f"{DATA}/30_export_qocv/{cell}/*qocv_dch_BM{bm}_*.parquet")
    if not files:
        raise FileNotFoundError(f"no qocv_dch for {cell} BM{bm}")
    q = pd.read_parquet(files[0]).sort_values("Time")
    ah = q["Ah_throughput"].to_numpy()
    dod = (ah - ah[0]) / (ah[-1] - ah[0])          # 0 (full) -> 1 (empty)
    soc = 1.0 - dod
    i_qocv = abs(float(q["Current"].mean()))
    ocv = q["Voltage"].to_numpy() + i_qocv * r0     # undo the C/20 IR drop
    order = np.argsort(soc)
    return soc[order], ocv[order]


def get_rc_params(cell, bm, pulse_type="DCH 3.0A @50%"):
    csv = f"{DATA}/20_export_pulse/{cell}/2RC_vs_SOH.csv"
    r = pd.read_csv(csv)
    row = r[(r["BM_Programm"] == bm) & (r["pulse_type"] == pulse_type)]
    if row.empty:
        raise ValueError(f"no 2RC row for BM{bm} / {pulse_type}")
    row = row.iloc[0]
    return {k: float(row[k]) for k in ("R0_ohm", "R1_ohm", "tau1_s", "R2_ohm", "tau2_s")}, float(row["SOH_num"])


def get_capacity(cell, bm):
    c = pd.read_csv(f"{DATA}/40_capacity_monitore/{cell}_capacity.csv")
    row = c[c["BM_Programm"] == bm]
    return float(row["Capacity_py"].iloc[0]) if not row.empty else NOM_CAPACITY


def simulate(rc, soc_grid, ocv, q_ah, rate_a, v_min=V_MIN, dt=1.0):
    """March a constant-current discharge; return (Ah_delivered, V, SOC)."""
    i = -abs(rate_a)                                # discharge, signed
    r0, r1, t1, r2, t2 = (rc["R0_ohm"], rc["R1_ohm"], rc["tau1_s"], rc["R2_ohm"], rc["tau2_s"])
    soc, v1, v2, t = 1.0, 0.0, 0.0, 0.0
    a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)
    ah_list, v_list, soc_list = [], [], []
    while soc > 0:
        ocv_t = float(np.interp(soc, soc_grid, ocv))
        v1 = v1 * a1 + i * r1 * (1.0 - a1)
        v2 = v2 * a2 + i * r2 * (1.0 - a2)
        v = ocv_t + i * r0 + v1 + v2
        ah_list.append(abs(i) * t / 3600.0)
        v_list.append(v)
        soc_list.append(soc)
        if v <= v_min:
            break
        t += dt
        soc -= abs(i) * dt / 3600.0 / q_ah
    return np.array(ah_list), np.array(v_list), np.array(soc_list)


def measured_cap_discharge(cell, bm):
    """The measured CAP full discharge from GOLD: (Ah_delivered, V, I_mean)."""
    g = pd.read_parquet(
        f"{DATA}/GOLD/{cell}.parquet",
        columns=["Time", "Current", "Voltage", "target", "BM_Programm", "Ah_throughput"],
    )
    cap = g[(g["BM_Programm"] == bm) & (g["target"] == "CAP")].sort_values("Time")
    if cap.empty:
        return None
    ah = cap["Ah_throughput"].to_numpy()
    return ah - ah[0], cap["Voltage"].to_numpy(), float(cap["Current"].mean())


def main():
    ap = argparse.ArgumentParser(description="Simulate a full discharge from 2RC + qOCV.")
    ap.add_argument("cell", nargs="?", default=DEFAULT_CELL)
    ap.add_argument("--bm", type=int, default=5, help="check-up BM_Programm")
    ap.add_argument("--rate-A", type=float, default=None,
                    help="discharge current (A); default = measured CAP test rate")
    ap.add_argument("--pulse-type", default="DCH 3.0A @50%", help="2RC row to use")
    args = ap.parse_args()

    rc, soh = get_rc_params(args.cell, args.bm, args.pulse_type)
    q_ah = get_capacity(args.cell, args.bm)
    soc_grid, ocv = load_ocv_soc(args.cell, args.bm, rc["R0_ohm"])
    meas = measured_cap_discharge(args.cell, args.bm)

    rate = args.rate_A if args.rate_A is not None else (abs(meas[2]) if meas else 1.5)
    ah, v, soc = simulate(rc, soc_grid, ocv, q_ah, rate)

    print(f"cell {args.cell}  BM{args.bm}  SOH {soh}%  Q={q_ah:.3f} Ah")
    print(f"RC: R0={rc['R0_ohm']*1000:.1f} R1={rc['R1_ohm']*1000:.1f} R2={rc['R2_ohm']*1000:.1f} mOhm  "
          f"tau1={rc['tau1_s']:.1f} tau2={rc['tau2_s']:.1f} s")
    print(f"sim discharge @ {rate:.2f} A (C/{q_ah/rate:.1f}) -> {ah[-1]:.3f} Ah to {V_MIN} V")
    if meas is not None:
        print(f"measured CAP @ {abs(meas[2]):.2f} A -> {meas[0][-1]:.3f} Ah")
        # interpolate sim onto measured Ah grid for a fair residual
        vsim_on_meas = np.interp(meas[0], ah, v)
        valid = meas[0] <= ah[-1]
        rmse = np.sqrt(np.mean((vsim_on_meas[valid] - meas[1][valid]) ** 2)) * 1000
        print(f"sim-vs-measured discharge RMSE = {rmse:.1f} mV  (over {valid.sum()} pts)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if meas is not None:
        ax.plot(meas[0], meas[1], "-", color="0.5", lw=2, label=f"measured CAP @ {abs(meas[2]):.1f} A")
    ax.plot(ah, v, "-", color="C3", lw=1.6, label=f"2RC+qOCV sim @ {rate:.1f} A")
    ax.axhline(V_MIN, color="C0", ls=":", lw=1, label=f"cutoff {V_MIN} V")
    ax.set_xlabel("Ah delivered")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{args.cell}  BM{args.bm}  SOH {soh}%  —  simulated vs measured full discharge")
    ax.legend()
    ax.grid(alpha=0.3)
    out = f"{DATA}/20_export_pulse/{args.cell}/sim_discharge_BM{args.bm}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
