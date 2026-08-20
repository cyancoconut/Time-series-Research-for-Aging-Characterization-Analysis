"""Track R0 across an aging block from the inter-cycle relaxation pauses.

Companion to ``predict_capacity_over_cycles.py``: instead of predicting capacity
from each partial cycle, we fit the 2RC model to the *relaxation pause* that
follows every Nth aging discharge (``fit_one_relaxation``) and pull out R0. R0 is
reverse-calculated from the ohmic step at current-off two ways -- fit-extrapolated
to t=0 and the model-free jump -- so the soft R0/fast-RC split is visible.

The relaxation-R0 trend is then plotted against the pulse-test R0 from the
*adjacent* HPPC check-ups that bracket the block (BM7 before, BM9 after for the
003 70%SOC/60%DOD/0.5C block), read from ``2RC_vs_SOH.csv``.

Usage (from src/):  python -m analysis.relax_r0_over_cycles [--every 5]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from analysis.fit_2rc_pulse import fit_one_relaxation
from analysis.simulate_cycle_from_partial import CELL as DEFAULT_CELL
from analysis.simulate_cycle_from_partial import CYCLING_FILE as DEFAULT_CYCLING_FILE
from analysis.simulate_cycle_from_partial import DATA, NOM
from analysis.simulate_cycle_soc_interp_rc import block_throughput, discover_checkups

# Default cycling rate is ~C/2 (1.4 A): CHA pulses exist at C/2 (rate-matched),
# DCH pulses only at 1C (3.0 A). Override with --pulse-rate.
DEFAULT_PULSE_RATE = {"DCH": 3.0, "CHA": 1.5}
# Where the pause sits in the 70%SOC/60%DOD swing, for the caption.
PAUSE_SOC = {"DCH": "bottom of swing (~40% SOC)", "CHA": "top of swing (~100% SOC)"}


def extract_cycle_pauses(path, direction="DCH", min_step_rows=2000, min_pau_rows=300):
    """Every aging step (``direction``) paired with the PAU pause that follows it.

    Same segmentation as ``predict_capacity_over_cycles.extract_cycles`` (split on
    Prozedur/Zustand change) but keeps the *full pause rows* so the relaxation can
    be fit, not just its settled voltage.
    """
    df = pd.read_parquet(path, columns=["Zeit", "Spannung", "Strom", "Zustand", "Prozedur"])
    df["Zeit"] = pd.to_datetime(df["Zeit"])
    df = df.reset_index(drop=True)
    seg = (df["Prozedur"].ne(df["Prozedur"].shift()) | df["Zustand"].ne(df["Zustand"].shift())).cumsum()
    groups = [g for _, g in df.groupby(seg)]
    out = []
    for idx, g in enumerate(groups):
        if not (g["Zustand"].iloc[0] == direction and len(g) >= min_step_rows
                and "Aging" in str(g["Prozedur"].iloc[0])):
            continue
        after = groups[idx + 1] if idx + 1 < len(groups) else None
        if after is None or after["Zustand"].iloc[0] not in ("PAU", "PAUO") or len(after) < min_pau_rows:
            continue
        out.append((g, after))
    return out


def fit_relax_r0(dch, pau):
    """Reverse-calc R0 (+ RC) from one DCH step's following relaxation pause."""
    rest = pau.rename(columns={"Zeit": "Time", "Spannung": "Voltage", "Strom": "Current"})
    i_prev = float(dch["Strom"].median())
    v_cycle_last = float(dch["Spannung"].iloc[-1])
    t0 = pd.to_datetime(dch["Zeit"].iloc[-1])
    t_prev = (dch["Zeit"].iloc[-1] - dch["Zeit"].iloc[0]).total_seconds()
    rest = rest.assign(Time=pd.to_datetime(rest["Time"]))
    return fit_one_relaxation(rest, i_prev, v_cycle_last, t0=t0, t_prev=t_prev)


def pulse_r0(cell, bms, pulse_type):
    """Pulse-test R0 (fit + model-free jump) and check-up time for each BM."""
    r = pd.read_csv(f"{DATA}/20_export_pulse/{cell}/2RC_vs_SOH.csv")
    rows = []
    for bm in bms:
        sub = r[(r["BM_Programm"] == bm) & (r["pulse_type"] == pulse_type)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        f = glob.glob(f"{DATA}/20_export_pulse/{cell}/*_pulse_BM{bm}_*.parquet")[0]
        t = pd.to_datetime(pd.read_parquet(f, columns=["Time"])["Time"])
        rows.append({
            "BM": bm, "SOH": float(row["SOH_num"]), "time": t.min(),
            "R0_ohm": float(row["R0_ohm"]),
            "R0_jump_ohm": float(row["R0_jump_term_ohm"]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5, help="fit every Nth aging cycle")
    ap.add_argument("--direction", choices=["DCH", "CHA"], default="DCH",
                    help="cycling half to fit (DCH pause at swing bottom vs CHA pause at top)")
    ap.add_argument("--cell", default=DEFAULT_CELL, help="cell folder name under 20_export_pulse/")
    ap.add_argument("--cycling-file", default=DEFAULT_CYCLING_FILE, help="aging cycling parquet")
    ap.add_argument("--adjacent-bms", type=int, nargs="+", default=[7, 9],
                    help="check-up BM_Programms bracketing the block (before after)")
    ap.add_argument("--pulse-soc", default="50%", help="pulse SOC to overlay, e.g. 50%% or 90%%")
    ap.add_argument("--pulse-rate", type=float, default=None,
                    help="pulse C-rate current (A); default 1.5 (CHA) / 3.0 (DCH)")
    args = ap.parse_args()
    direction = args.direction
    cell = args.cell
    rate = args.pulse_rate if args.pulse_rate is not None else DEFAULT_PULSE_RATE[direction]
    rate_str = f"{rate:.1f}"                       # match CSV convention ("3.0A", "1.5A")
    pulse_type = f"{direction} {rate_str}A @{args.pulse_soc}"

    pairs = extract_cycle_pauses(args.cycling_file, direction)
    picks = pairs[:: args.every]
    print(f"{len(pairs)} aging {direction} cycles w/ pause; fitting every {args.every} -> {len(picks)}")

    rows = []
    for dch, pau in picks:
        res = fit_relax_r0(dch, pau)
        if res is None or res["degenerate"]:
            continue
        rows.append({
            "time": pd.to_datetime(dch["Zeit"].iloc[0]),
            "I_A": round(float(dch["Strom"].median()), 3),
            "R0_ohm": res["R0_ohm"], "R0_jump_ohm": res["R0_jump_ohm"],
            "R0_consistent": res["R0_consistent"],
            "R1_ohm": res["R1_ohm"], "tau1_s": res["tau1_s"],
            "R2_ohm": res["R2_ohm"], "tau2_s": res["tau2_s"],
            "rmse_mV": res["rmse_mV"],
        })
    relax = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    pulse = pulse_r0(cell, args.adjacent_bms, pulse_type)

    # check-up-interpolated R0 at each cycle: map the cycle time onto the
    # cumulative-throughput axis, then linearly blend the pulse-test R0 between
    # the bracketing check-ups (same interpolation as the ECM in
    # simulate_cycle_soc_interp_rc -- here applied to R0 only).
    checkups = discover_checkups(cell, pulse_type)
    thr_cycles = np.array([block_throughput(t, checkups) for t in relax["time"]])
    relax["throughput_Ah"] = thr_cycles
    relax["R0_checkup_interp_ohm"] = np.interp(
        thr_cycles, checkups["throughput"], checkups["R0_ohm"])
    relax["R0_jump_checkup_interp_ohm"] = np.interp(
        thr_cycles, checkups["throughput"], checkups["R0_jump_ohm"])

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== %s relaxation R0 over the block (every %d cycles) ===" % (direction, args.every))
    print(relax.to_string(index=False))
    print("\n=== adjacent pulse-test R0 (%s) ===" % pulse_type)
    print(pulse.to_string(index=False))

    out_csv = f"{DATA}/20_export_pulse/{cell}/relax_r0_over_cycles_{direction}.csv"
    relax.to_csv(out_csv, index=False)
    print(f"\nresults -> {out_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(relax["time"], relax["R0_ohm"] * 1000, "o-", color="C0", ms=4,
            label="relaxation R0 (fit-extrapolated)")
    ax.plot(relax["time"], relax["R0_jump_ohm"] * 1000, "x--", color="C0", ms=5,
            alpha=0.5, label="relaxation R0 (model-free jump)")
    ax.plot(relax["time"], relax["R0_checkup_interp_ohm"] * 1000, "-", color="C2", lw=2,
            label="check-up-interpolated R0 (throughput axis)")
    for _, p in pulse.iterrows():
        ax.plot(p["time"], p["R0_ohm"] * 1000, "*", color="C3", ms=18,
                label=f"pulse BM{int(p['BM'])} R0 ({p['SOH']:.1f}% SOH)")
        ax.plot(p["time"], p["R0_jump_ohm"] * 1000, "P", color="C3", ms=9, alpha=0.5)
    ax.set_xlabel("date")
    ax.set_ylabel("R0 (mΩ)")
    ax.set_title(f"{cell}: {direction} relaxation R0 across the aging block vs adjacent pulse-test R0\n"
                 f"(70%SOC/60%DOD/0.5C cycling, {direction} relaxation @ {PAUSE_SOC[direction]}; "
                 f"pulses {pulse_type})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out = f"{DATA}/20_export_pulse/{cell}/relax_r0_over_cycles_{direction}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
