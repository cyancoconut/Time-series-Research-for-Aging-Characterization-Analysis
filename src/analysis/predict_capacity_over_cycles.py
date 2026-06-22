"""Predict full capacity from each partial cycle and track it across the block.

The 003 cycling file (70%SOC / 60%DOD / 0.5C) holds ~115 partial discharges over
17 days, bracketed by two check-ups (BM7 before, BM9 after). For every 5th
partial cycle we predict the cell's *full* capacity from that single partial
swing and compare the trajectory to the bracketing check-ups' **qOCV C/20**
capacities (not the C/2 CAP SOH).

Method (non-circular, aging-tracking): a partial discharge removes a measured
``dAh`` between two terminal voltages. IR/polarization-correct those voltages to
OCV, read the SOC endpoints off the qOCV OCV-SOC *shape* (aging-stable), and
extrapolate the partial to a full 0..100% swing:

    capacity = dAh / (SOC_start - SOC_end)

No SOH is injected; capacity falls out of the measured Ah-per-SOC at that point
in aging. The qOCV reference (OCV shape + @50% RC) comes from the BM7 check-up.

Usage (from src/):  python -m analysis.predict_capacity_over_cycles [--every 5] [--ref-bm 7]
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd

from analysis.simulate_cycle_from_partial import (
    CELL,
    CYCLING_FILE,
    DATA,
    NOM,
    _cum_ah,
    load_reference,
)


def check_up_capacities(cell):
    """Per-check-up *C/2* capacity from the export filename SOH (the C/2 test).

    The export filename carries the C/2-test SOH (e.g. ``..._BM7_93.9SOH``);
    C/2 capacity = SOH/100 * nominal. Date comes from the parquet's Time.
    Returns a DataFrame [bm, date, cap_Ah] sorted by date.
    """
    rows = []
    for f in glob.glob(f"{DATA}/30_export_qocv/{cell}/*qocv_dch*.parquet"):
        bm = int(re.search(r"BM(\d+)", f).group(1))
        soh = float(re.search(r"_([\d.]+)SOH", f).group(1))
        q = pd.read_parquet(f, columns=["Time"])
        rows.append((bm, pd.to_datetime(q["Time"]).min(), round(soh / 100.0 * NOM, 4)))
    return pd.DataFrame(rows, columns=["bm", "date", "cap_Ah"]).sort_values("date").reset_index(drop=True)


def extract_cycles(path, min_rows=2000):
    """Partial aging cycles with their bracketing relaxation pauses.

    Each cycle is the DCH segment plus the relaxed (equilibrium) voltage of the
    PAU pause immediately before and after it -- those settled voltages are
    OCV(SOC) at the swing endpoints, far more reliable than guessing the
    under-load polarization with a fixed RC.
    """
    df = pd.read_parquet(path, columns=["Zeit", "Spannung", "Strom", "Zustand", "Prozedur"])
    df["Zeit"] = pd.to_datetime(df["Zeit"])
    df = df.reset_index(drop=True)
    seg = (df["Prozedur"].ne(df["Prozedur"].shift()) | df["Zustand"].ne(df["Zustand"].shift())).cumsum()
    groups = [g for _, g in df.groupby(seg)]
    out = []
    for idx, g in enumerate(groups):
        if not (g["Zustand"].iloc[0] == "DCH" and len(g) >= min_rows
                and "Aging" in str(g["Prozedur"].iloc[0])):
            continue
        before = groups[idx - 1] if idx > 0 else None
        after = groups[idx + 1] if idx + 1 < len(groups) else None
        v_before = before["Spannung"].iloc[-1] if before is not None and before["Zustand"].iloc[0] == "PAU" else None
        v_after = after["Spannung"].iloc[-1] if after is not None and after["Zustand"].iloc[0] == "PAU" else None
        out.append(dict(dch=g, v_relax_before=v_before, v_relax_after=v_after))
    return out


def soc_of_ocv(v, soc_grid, ocv_grid):
    """SOC fraction at a given OCV (OCV rises with SOC); clamped to the curve."""
    order = np.argsort(ocv_grid)
    return float(np.interp(v, ocv_grid[order], soc_grid[order]))


def predict_from_partial(cyc, soc_grid, ocv_grid, rc):
    """Capacity = dAh / dSOC for one partial discharge, two endpoint variants.

    ``cap``  -- SOC endpoints from the *relaxed* bracketing pauses (equilibrium OCV).
    ``cap_rc`` -- SOC endpoints from the under-load DCH voltages corrected by the
                  @50% RC: start by I*R0 (onset), end by I*(R0+R1+R2) (settled).
    ``dAh`` is the coulomb-counted charge removed during the DCH segment.
    """
    g = cyc["dch"]
    t = (g["Zeit"] - g["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    cur = g["Strom"].to_numpy()
    d_ah = _cum_ah(t, cur)[-1]
    i = abs(float(np.mean(cur)))

    # relaxed-pause endpoints
    v0 = cyc["v_relax_before"] if cyc["v_relax_before"] is not None else g["Spannung"].iloc[0]
    v1 = cyc["v_relax_after"] if cyc["v_relax_after"] is not None else g["Spannung"].iloc[-1]
    soc0 = soc_of_ocv(v0, soc_grid, ocv_grid)
    soc1 = soc_of_ocv(v1, soc_grid, ocv_grid)
    d_soc = soc0 - soc1
    cap = d_ah / d_soc if d_soc > 0 else np.nan

    # RC-corrected under-load endpoints
    ocv0 = g["Spannung"].iloc[0] + i * rc["R0_ohm"]
    ocv1 = g["Spannung"].iloc[-1] + i * (rc["R0_ohm"] + rc["R1_ohm"] + rc["R2_ohm"])
    s0_rc = soc_of_ocv(ocv0, soc_grid, ocv_grid)
    s1_rc = soc_of_ocv(ocv1, soc_grid, ocv_grid)
    d_soc_rc = s0_rc - s1_rc
    cap_rc = d_ah / d_soc_rc if d_soc_rc > 0 else np.nan

    return dict(time=g["Zeit"].iloc[0], v0=v0, v1=v1, d_ah=d_ah, i=i,
                soc0=soc0, soc1=soc1, d_soc=d_soc, cap=cap, cap_rc=cap_rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5, help="predict on every Nth partial cycle")
    ap.add_argument("--ref-bm", type=int, default=7, help="reference check-up BM (OCV shape + RC)")
    args = ap.parse_args()

    rc, _, grid, ocv = load_reference(CELL, args.ref_bm)
    cap_ref = grid[-1]
    soc_grid = 1.0 - grid / cap_ref                     # 1 (full) -> 0 (empty)

    cycles = extract_cycles(CYCLING_FILE)
    block_start = cycles[0]["dch"]["Zeit"].iloc[0]
    block_end = cycles[-1]["dch"]["Zeit"].iloc[-1]

    # bracketing check-ups: latest before the block, earliest after it.
    qcaps = check_up_capacities(CELL)
    before = qcaps[qcaps["date"] <= block_start].iloc[-1]
    after = qcaps[qcaps["date"] >= block_end].iloc[0]

    picks = cycles[:: args.every]
    rows = [predict_from_partial(c, soc_grid, ocv, rc) for c in picks]
    res = pd.DataFrame(rows)
    # one-time calibration: anchor the first predicted cycle to the "before"
    # check-up's C/2 capacity (a constant offset; the predicted *fade* is
    # unchanged). Removes the systematic OCV-hysteresis/plateau bias.
    offset = before["cap_Ah"] - res["cap"].iloc[0]
    res["cap_cal"] = res["cap"] + offset
    offset_rc = before["cap_Ah"] - res["cap_rc"].iloc[0]
    res["cap_rc_cal"] = res["cap_rc"] + offset_rc

    print(f"reference BM{args.ref_bm}: qOCV cap {cap_ref:.3f} Ah, "
          f"RC R0/R1/R2 = {rc['R0_ohm']*1e3:.1f}/{rc['R1_ohm']*1e3:.1f}/{rc['R2_ohm']*1e3:.1f} mOhm")
    print(f"cycling block {block_start:%Y-%m-%d} -> {block_end:%Y-%m-%d}")
    print(f"bracket (C/2 capacity from filename SOH): "
          f"BM{before['bm']} {before['date']:%Y-%m-%d} {before['cap_Ah']:.3f} Ah  ->  "
          f"BM{after['bm']} {after['date']:%Y-%m-%d} {after['cap_Ah']:.3f} Ah")
    print(f"{len(cycles)} partial cycles, predicting every {args.every} -> {len(picks)} cycles")
    print(f"calibration offset to BM{before['bm']}: {offset:+.3f} Ah\n")
    show = res.copy()
    show["time"] = show["time"].dt.strftime("%m-%d %H:%M")
    print(show[["time", "v0", "v1", "d_ah", "soc0", "soc1", "cap", "cap_rc", "cap_cal"]].round(3).to_string(index=False))
    print(f"\nrelaxed-pause cap : first {res['cap'].iloc[0]:.3f}  last {res['cap'].iloc[-1]:.3f}  "
          f"(fade {res['cap'].iloc[0]-res['cap'].iloc[-1]:.3f} Ah)")
    print(f"RC-corrected  cap : first {res['cap_rc'].iloc[0]:.3f}  last {res['cap_rc'].iloc[-1]:.3f}  "
          f"(fade {res['cap_rc'].iloc[0]-res['cap_rc'].iloc[-1]:.3f} Ah)")
    print(f"difference (relax - rc): mean {(res['cap']-res['cap_rc']).mean():+.3f} Ah")
    print(f"calibrated (relax): first {res['cap_cal'].iloc[0]:.3f}  last {res['cap_cal'].iloc[-1]:.3f}  (offset {offset:+.3f})")
    print(f"calibrated (rc)   : first {res['cap_rc_cal'].iloc[0]:.3f}  last {res['cap_rc_cal'].iloc[-1]:.3f}  (offset {offset_rc:+.3f})")
    print(f"vs bracket        : before {before['cap_Ah']:.3f}  after {after['cap_Ah']:.3f} Ah  "
          f"(actual fade {before['cap_Ah']-after['cap_Ah']:.3f} Ah)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res["time"], res["cap_cal"], "o-", color="C3", ms=4,
            label=f"calibrated relaxed ({offset:+.3f} Ah)")
    ax.plot(res["time"], res["cap_rc_cal"], "s-", color="C1", ms=4,
            label=f"calibrated @50% RC ({offset_rc:+.3f} Ah)")
    ax.plot(res["time"], res["cap"], "o--", color="0.6", ms=3, label="raw relaxed pause")
    ax.plot(res["time"], res["cap_rc"], "s--", color="C4", ms=3, label="raw @50% RC")
    for b, c in ((before, "C0"), (after, "C2")):
        ax.scatter(b["date"], b["cap_Ah"], color=c, s=90, zorder=5,
                   label=f"BM{b['bm']} C/2 capacity = {b['cap_Ah']:.3f} Ah")
    ax.set_xlabel("date")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title(f"{CELL}: full capacity predicted from partial cycles vs C/2 check-ups")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out = f"{DATA}/20_export_pulse/{CELL}/predict_capacity_over_cycles.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nplot -> {out}")

    _plot_bm_detail(picks[0], soc_grid, ocv, rc, cap_ref, before, args.ref_bm)


def _plot_bm_detail(cyc, soc_grid, ocv_grid, rc, cap_ref, before, ref_bm):
    """Method-2 detail for the first (BM-reference) partial cycle: the predicted
    full discharge (OCV shape scaled to the predicted capacity) with the measured
    partial swing and its relaxed SOC endpoints overlaid."""
    r = predict_from_partial(cyc, soc_grid, ocv_grid, rc)
    cap_pred = r["cap"]
    g = cyc["dch"]
    # measured partial, placed on the predicted-capacity Ah axis via its start SOC
    t = (g["Zeit"] - g["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    p_cum = _cum_ah(t, g["Strom"].to_numpy())
    ah0 = (1.0 - r["soc0"]) * cap_pred
    p_ah = ah0 + p_cum

    # predicted full discharge OCV curve, scaled so SOC=0 at cap_pred
    soc_line = np.linspace(1.0, 0.0, 300)
    ocv_line = np.interp(soc_line[::-1], np.sort(soc_grid),
                         ocv_grid[np.argsort(soc_grid)])[::-1]
    ah_line = (1.0 - soc_line) * cap_pred

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ah_line, ocv_line, "-", color="C3", lw=1.8,
            label=f"predicted full discharge (qOCV shape, cap {cap_pred:.3f} Ah)")
    ax.plot(p_ah, g["Spannung"].to_numpy(), "-", color="0.35", lw=2.2,
            label="measured partial swing (under load)")
    ax.scatter([ah0, ah0 + p_cum[-1]], [r["v0"], r["v1"]], color="C0", zorder=5,
               label=f"relaxed SOC endpoints {r['soc0']:.2f} -> {r['soc1']:.2f}")
    ax.axvline(cap_pred, color="green", ls="--", lw=1, label=f"cap_pred {cap_pred:.3f} Ah")
    ax.axvline(before["cap_Ah"], color="C0", ls=":", lw=1, label=f"BM{before['bm']} C/2 {before['cap_Ah']:.3f} Ah")
    ax.set_xlabel("Ah removed from full")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{CELL}: method-2 capacity from partial cycle (ref BM{ref_bm})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = f"{DATA}/20_export_pulse/{CELL}/sim_cycle_from_partial_method2_BM{ref_bm}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
