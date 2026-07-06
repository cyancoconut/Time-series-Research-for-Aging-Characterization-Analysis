"""Partial->full discharge prediction on 003's cycling data with SOC-interpolated RC.

Cell 003 only carries a single-SOC (@50%) pulse schema, so its 2RC ECM cannot
capture the rate-dependent polarization that grows near depletion -- which is
why the fixed-@50% prediction (``simulate_cycle_from_partial.py``) overshoots the
measured C/2 capacity: at 5% SOC it still applies @50% RC.

007 is the *same cell type* (VTC6) but carries the full 90/50/10 SOC pulse
schema. Here we keep 003's own qOCV(Ah) reference and capacity anchoring (the
non-circular SOC/capacity backbone) and **borrow 007's DCH RC**, interpolated
over SOC, so the slow diffusion branch (R2/tau2) ramps up toward empty the way it
physically should.

We compare three RC treatments on the identical 003 partial->full task:
  (a) 003-own @50% fixed   (the existing baseline)
  (b) 007 @50% fixed       (isolates cell-to-cell RC difference)
  (c) 007 SOC-interpolated (isolates the SOC-interpolation effect)

Validation is unchanged: reproduce the measured partial (100->40%), extrapolate
to cutoff, and check the lower half against the in-file ``jri_Discharge_C2``.

Usage (from src/):  python -m analysis.simulate_cycle_soc_interp_rc
"""

import argparse

import numpy as np
import pandas as pd

from analysis.simulate_cycle_from_partial import (
    CELL,
    CYCLING_FILE,
    DATA,
    NOM,
    V_MIN,
    _cum_ah,
    ah_of_ocv,
    extract_segments,
    load_reference,
    ocv_of,
)

# 007 = same VTC6 cell type, full 90/50/10 SOC pulse schema.
RC_CELL = "METABatt_Sony_Murata_18650VTC6_007"


def load_soc_rc(cell, soh_target, direction="DCH"):
    """SOC-resolved DCH RC table from `cell` at the SOH nearest `soh_target`.

    Returns (soh_used, DataFrame[soc, R0_ohm, R1_ohm, tau1_s, R2_ohm, tau2_s])
    sorted by SOC, dropping `degenerate` fits (railed slow branch).
    """
    r = pd.read_csv(f"{DATA}/20_export_pulse/{cell}/2RC_vs_SOH.csv")
    d = r[r["pulse_type"].str.startswith(direction)].copy()
    soh = d["SOH_num"].iloc[(d["SOH_num"] - soh_target).abs().to_numpy().argmin()]
    d = d[(d["SOH_num"] == soh) & (~d["degenerate"])].copy()
    d["soc"] = d["SOC"].astype(str).str.rstrip("%").astype(float)
    cols = ["soc", "R0_ohm", "R1_ohm", "tau1_s", "R2_ohm", "tau2_s"]
    return float(soh), d[cols].sort_values("soc").reset_index(drop=True)


def rc_at(soc, tab):
    """Linear interpolation of each RC parameter at `soc` (clamped to anchors)."""
    s = tab["soc"].to_numpy()
    return {
        "R0_ohm": float(np.interp(soc, s, tab["R0_ohm"])),
        "R1_ohm": float(np.interp(soc, s, tab["R1_ohm"])),
        "tau1_s": float(np.interp(soc, s, tab["tau1_s"])),
        "R2_ohm": float(np.interp(soc, s, tab["R2_ohm"])),
        "tau2_s": float(np.interp(soc, s, tab["tau2_s"])),
    }


def _const_tab(rc):
    """A one-row SOC table that yields `rc` at every SOC (fixed-RC baseline)."""
    return pd.DataFrame([{"soc": 50.0, **{k: rc[k] for k in
                          ("R0_ohm", "R1_ohm", "tau1_s", "R2_ohm", "tau2_s")}}])


def simulate_over_soc(tab, grid, ocv, ah_start, t_s, current, q_full):
    """Drive the 2RC+OCV model over a measured segment with SOC-dependent RC."""
    dt = np.diff(t_s, prepend=t_s[0])
    v = np.empty(len(t_s))
    ah, v1, v2 = ah_start, 0.0, 0.0
    for k in range(len(t_s)):
        i = -abs(current[k])
        rc = rc_at(100.0 * (1.0 - ah / q_full), tab)
        a1 = np.exp(-dt[k] / rc["tau1_s"])
        a2 = np.exp(-dt[k] / rc["tau2_s"])
        v1 = v1 * a1 + i * rc["R1_ohm"] * (1 - a1)
        v2 = v2 * a2 + i * rc["R2_ohm"] * (1 - a2)
        v[k] = ocv_of(ah, grid, ocv) + i * rc["R0_ohm"] + v1 + v2
        ah += abs(i) * dt[k] / 3600.0
    return v


def simulate_full_soc(tab, grid, ocv, current, q_full, dt=1.0):
    """CC discharge from full (Ah=0) to cutoff with SOC-dependent RC."""
    i = -abs(current)
    ah, v1, v2 = 0.0, 0.0, 0.0
    out_ah, out_v = [], []
    while True:
        rc = rc_at(100.0 * (1.0 - ah / q_full), tab)
        vt = ocv_of(ah, grid, ocv) + i * rc["R0_ohm"] + v1 + v2
        out_ah.append(ah)
        out_v.append(vt)
        if vt <= V_MIN:
            break
        a1 = np.exp(-dt / rc["tau1_s"])
        a2 = np.exp(-dt / rc["tau2_s"])
        v1 = v1 * a1 + i * rc["R1_ohm"] * (1 - a1)
        v2 = v2 * a2 + i * rc["R2_ohm"] * (1 - a2)
        ah += abs(i) * dt / 3600.0
        if ah > 1.5 * grid[-1]:
            break
    return np.array(out_ah), np.array(out_v)


def run_case(name, tab, grid, ocv, q_full, i_cyc, partial, lower):
    """Run the partial->full prediction for one RC table; return a result dict."""
    # predicted full discharge at the cycling rate
    sim_ah, sim_v = simulate_full_soc(tab, grid, ocv, i_cyc, q_full)
    q_pred = sim_ah[-1]

    # measured partial (100->40%), anchored on OCV by its starting voltage
    pt = (partial["Zeit"] - partial["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    p_cum = _cum_ah(pt, partial["Strom"].to_numpy())
    p_v = partial["Spannung"].to_numpy()
    r0_hi = rc_at(95.0, tab)["R0_ohm"]                 # ~start SOC for IR anchor
    p_ah0 = ah_of_ocv(p_v[0] + i_cyc * r0_hi, grid, ocv)
    p_ah = p_ah0 + p_cum
    p_sim = simulate_over_soc(tab, grid, ocv, p_ah0, pt, partial["Strom"].to_numpy(), q_full)
    rmse_p = np.sqrt(np.mean((p_sim - p_v) ** 2)) * 1000

    # measured lower (jri_Discharge_C2, 40->0%), end-anchored at q_pred
    lt = (lower["Zeit"] - lower["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    l_cum = _cum_ah(lt, lower["Strom"].to_numpy())
    l_v = lower["Spannung"].to_numpy()
    l_ah0 = q_pred - l_cum[-1]
    l_ah = l_ah0 + l_cum
    l_sim = simulate_over_soc(tab, grid, ocv, l_ah0, lt, lower["Strom"].to_numpy(), q_full)
    rmse_l = np.sqrt(np.mean((l_sim - l_v) ** 2)) * 1000

    return dict(name=name, sim_ah=sim_ah, sim_v=sim_v, q_pred=q_pred,
                p_ah=p_ah, p_v=p_v, l_ah=l_ah, l_v=l_v,
                rmse_p=rmse_p, rmse_l=rmse_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", type=int, default=7, help="003 reference check-up BM_Programm")
    args = ap.parse_args()

    partial, lower = extract_segments(CYCLING_FILE)
    # 003's own qOCV(Ah) reference + capacity backbone; ignore its @50% rc here.
    rc003, soh003, grid, ocv = load_reference(CELL, args.bm)
    q_full = grid[-1]                                  # full capacity (qOCV extent)
    i_cyc = abs(float(partial["Strom"].mean()))

    # 007 SOC-resolved DCH RC at the SOH nearest 003's reference SOH.
    soh007, tab007 = load_soc_rc(RC_CELL, soh003)

    cases = [
        run_case("003 @50% fixed", _const_tab(rc003), grid, ocv, q_full, i_cyc, partial, lower),
        run_case("007 @50% fixed",
                 _const_tab(rc_at(50.0, tab007)), grid, ocv, q_full, i_cyc, partial, lower),
        run_case("007 SOC-interp", tab007, grid, ocv, q_full, i_cyc, partial, lower),
    ]

    print(f"003 reference check-up BM{args.bm}  SOH {soh003}%   full capacity (qOCV) {q_full:.3f} Ah")
    print(f"007 RC borrowed at SOH {soh007}%  ({RC_CELL.split('_')[-1]}, same VTC6 type)")
    print(f"  SOC anchors: " + ", ".join(
        f"{int(s)}%->R0={rc_at(s, tab007)['R0_ohm']*1e3:.1f} R2={rc_at(s, tab007)['R2_ohm']*1e3:.1f}mOhm "
        f"tau2={rc_at(s, tab007)['tau2_s']:.0f}s" for s in (90, 50, 10)))
    p0 = cases[0]
    pt = (partial["Zeit"] - partial["Zeit"].iloc[0]).dt.total_seconds().to_numpy()
    p_removed = _cum_ah(pt, partial["Strom"].to_numpy())[-1]
    print(f"partial cycling DCH: {p0['p_v'][0]:.3f}->{p0['p_v'][-1]:.3f} V at {i_cyc:.2f} A, "
          f"{p_removed:.3f} Ah over 60% DOD; implied SOH ~{p_removed/0.6/NOM*100:.1f}%")
    print(f"\n{'RC treatment':<18}{'pred Ah':>9}{'pred SOH':>10}{'partial RMSE':>14}{'lower RMSE':>12}")
    for c in cases:
        print(f"{c['name']:<18}{c['q_pred']:>9.3f}{c['q_pred']/NOM*100:>9.1f}%"
              f"{c['rmse_p']:>13.1f}{c['rmse_l']:>12.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"003 @50% fixed": "C1", "007 @50% fixed": "C2", "007 SOC-interp": "C3"}
    for c in cases:
        ax.plot(c["sim_ah"], c["sim_v"], "-", color=colors[c["name"]], lw=1.6,
                label=f"pred {c['name']}: {c['q_pred']:.3f} Ah ({c['q_pred']/NOM*100:.1f}%)")
    c = cases[-1]
    ax.plot(c["p_ah"], c["p_v"], "-", color="0.35", lw=2.5,
            label=f"measured partial (100->40%, RMSE {c['rmse_p']:.1f} mV)")
    ax.plot(c["l_ah"], c["l_v"], "-", color="C0", lw=2.0,
            label=f"measured jri_Discharge_C2 (40->0%, RMSE {c['rmse_l']:.1f} mV)")
    ax.axhline(V_MIN, color="0.6", ls=":", lw=1)
    ax.axvline(c["p_ah"][-1], color="green", ls="--", lw=1, label="40% SOC (end of partial)")
    ax.set_xlabel("Ah removed from full")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{CELL}: full discharge from partial cycle, 007 SOC-interpolated RC "
                 f"(ref BM{args.bm} {soh003}%, RC@{soh007}%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = f"{DATA}/20_export_pulse/{CELL}/sim_cycle_soc_interp_rc_BM{args.bm}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
