"""Measure SOC by coulomb-counting the SOC-adjust step before each EIS block.

The charge-counting twin of :mod:`util.soc_from_qocv`. Both answer "what SOC was
this pulse / EIS spectrum measured at"; they differ in what they read.

**The ladder is built on the EIS measurements, and pulses inherit from it.** A
full-SOC-sweep parametrization run moves the cell one step, then measures it —
and the measurement is an *EIS block*, one or more consecutive ``EIS`` segments,
sitting directly behind the step that set the SOC. On the NFPP cell (BM22)::

    22_13  partial_cha_c14 DCH   0.695 Ah    <- the SOC-adjust step
    22_14  EIS                   0
    22_15  EIS                   0           <- the block: same SOC as 22_14
    22_16  PAU                   0
    22_17  PUL DCH               0.148       <- a *test* pulse at that SOC
    ...
    22_21  partial_cha_c14 DCH   0.693 Ah    <- the next step

So a block's step is simply the segment immediately before the block, skipping
only the block's own ``EIS`` segments, and every pulse after a block is at that
block's SOC until the next block starts. Reading the step off the EIS block is
what makes this robust: the segment literally one before a *pulse* is a ``PAU``,
and the nearest charge-moving segment before one can be the plateau's own
earlier test pulse (BM22 splits ``22_33`` and ``22_39`` across six procedure
numbers), which would book a pulse's charge as a SOC step.

Why count charge at all rather than assume the protocol's nominal step: the
run's "5 %" steps are not equal. On the NFPP cell the first EIS step drops
155 mV and the following ones ~15 mV, all labelled 5 %. An order-based ladder
(``100 - 5*i``) prices them the same and is wrong. Each block keeps its own
measured step for the same reason — freezing one scalar for the whole sweep
would put the equal-step assumption straight back in.

**Relation to the qOCV mapping.** This module is tried first and
``soc_from_qocv`` is the fallback, because a coulomb count is a direct
measurement of what moved while the qOCV route is an interpolation through a
voltage curve that itself needs an IR correction. But it needs GOLD (for the
segment IDs and ``Ah_throughput``) and at least one EIS block; where that is
missing the qOCV route still works from the bundle parquets alone. Whichever ran
is recorded in the ``soc_source`` diagnostic, so a fallback is never mistaken for
a step measurement.

**What it cannot do.** Counting charge gives *differences* in SOC, never an
absolute value, so the ladder is anchored at a rail: a discharge sweep starts at
100 % and a charge sweep at 0 %. A sweep that starts part-way in is offset by
however far in it started. The accumulated span is checked against the rails and
warns when it overflows, which is the signature of a wrong capacity reference or
a missed step.

**The step size is only as good as ``Ah_throughput``.** It is read straight from
BRONZE, so a BRONZE built before the per-file throughput fix reports half the
charge on a 1 s-cadence sweep and the steps come out at 2.5 % where the protocol
says 5 %. That is the diluted counter, not the protocol — rebuild BRONZE rather
than rescaling here.
"""

import logging

import numpy as np
import pandas as pd

#: Below this |mean current| a segment is a rest, not a charge-moving step.
REST_CURRENT_A = 1e-3

#: Target marking a segment as part of an EIS measurement block.
EIS_TARGET = "EIS"

#: How far outside 0–100 % the accumulated ladder may land before warning.
SOC_RAIL_TOLERANCE_PCT = 5.0


def parse_proc_id(seg_id):
    """``"13_16"`` -> ``(13, 16)``; ``(None, None)`` when unparseable.

    Mirrors ``characterize.pulse_fit._parse_proc_id`` — the ID convention is
    ``<BM_Programm>_<procedure_number>``, set in ``dismember_raw_cell``.
    """
    try:
        bm, proc = str(seg_id).rsplit("_", 1)
        return int(bm), int(proc)
    except (ValueError, AttributeError):
        return None, None


def segment_steps(gold: pd.DataFrame) -> pd.DataFrame:
    """Per GOLD segment: ``ID, BM_Programm, proc_num, target, signed_ah``.

    ``Ah_throughput`` counts ``|I|dt``, so its span across a segment is the
    charge magnitude and the direction has to come from the segment's own mean
    current. A rest segment (mean |I| below :data:`REST_CURRENT_A`) moves no
    charge regardless of its span. ``target`` is carried because the ladder is
    anchored on the ``EIS`` segments.

    Returns an empty frame when GOLD lacks ``ID`` or ``Ah_throughput``.
    """
    if gold is None or gold.empty:
        return pd.DataFrame()
    missing = {"ID", "Ah_throughput"} - set(gold.columns)
    if missing:
        logging.warning(
            "segment_steps: GOLD has no %s column — cannot count step charge",
            "/".join(sorted(missing)),
        )
        return pd.DataFrame()

    df = gold.copy()
    df["Ah_throughput"] = pd.to_numeric(df["Ah_throughput"], errors="coerce")
    rows = []
    for seg_id, grp in df.groupby("ID", sort=False):
        ah = grp["Ah_throughput"].dropna()
        bm, proc = parse_proc_id(seg_id)
        if ah.empty or proc is None:
            continue
        current = pd.to_numeric(grp.get("Current"), errors="coerce")
        i_mean = float(current.mean()) if current is not None and current.notna().any() else np.nan
        delta = float(ah.max() - ah.min())
        if not np.isfinite(i_mean) or abs(i_mean) <= REST_CURRENT_A:
            signed = 0.0
        else:
            signed = delta * (1.0 if i_mean > 0 else -1.0)
        rows.append({
            "ID": seg_id,
            "BM_Programm": bm,
            "proc_num": proc,
            "target": str(grp["target"].iloc[0]) if "target" in grp else "",
            "signed_ah": signed,
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values(["BM_Programm", "proc_num"])
              .reset_index(drop=True))


def eis_blocks(steps: pd.DataFrame, bm: int) -> pd.DataFrame:
    """The EIS measurement blocks of one program, in sweep order.

    A block is a run of consecutive ``EIS`` segments (consecutive in the
    program's ordered segment list, not necessarily in procedure number — the
    numbering has holes where dismember dropped short segments). Each block gets
    ``step_ah``: the signed charge of the segment immediately before it, which
    is the step that moved the cell to the SOC the block measured.

    Columns: ``block, first_proc, last_proc, ids, step_id, step_ah``.
    """
    prog = steps[steps["BM_Programm"] == bm].reset_index(drop=True)
    if prog.empty:
        return pd.DataFrame()

    is_eis = prog["target"] == EIS_TARGET
    blocks, i, n = [], 0, len(prog)
    while i < n:
        if not is_eis.iloc[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_eis.iloc[j + 1]:
            j += 1
        prev = prog.iloc[i - 1] if i > 0 else None
        blocks.append({
            "block": len(blocks),
            "first_proc": int(prog["proc_num"].iloc[i]),
            "last_proc": int(prog["proc_num"].iloc[j]),
            "ids": prog["ID"].iloc[i:j + 1].tolist(),
            "step_id": None if prev is None else prev["ID"],
            "step_ah": np.nan if prev is None else float(prev["signed_ah"]),
        })
        i = j + 1
    return pd.DataFrame(blocks)


def block_soc(blocks: pd.DataFrame, direction: str, capacity: float,
              label: str = "") -> tuple:
    """SOC (%) per EIS block, from the charge each block's step moved.

    The first block anchors at the rail its sweep starts from (100 % for a
    discharge sweep, 0 % for a charge sweep) — a coulomb count measures
    *differences*, so the absolute level has to come from somewhere. Its own
    ``step_ah`` is deliberately ignored: the segment before the first block is
    the full charge (or discharge) that set the sweep up, not a step within it.

        SOC_k = SOC_0 + 100 * cumsum(step_ah) / capacity

    Returns ``(blocks_with_SOC, info)``; ``info["reason"]`` is set and the SOC
    column is all-NaN when the ladder could not be built.
    """
    out = blocks.copy() if blocks is not None else pd.DataFrame()
    if out.empty:
        return out, {"reason": "no EIS block in this program"}
    out["SOC_pct"] = np.nan
    if not capacity or not np.isfinite(capacity) or capacity <= 0:
        return out, {"reason": f"no usable capacity reference ({capacity})"}
    if len(out) < 2:
        # One block measures no step: its own predecessor is the setup charge,
        # which the anchor ignores. Assigning the anchor alone would pin every
        # measurement in the program to 100 % (or 0 %) — a flat, confident,
        # wrong axis. The programs this hits are the capacity/qOCV ones that
        # carry a single EIS marker, not the sweep, and the qOCV mapping serves
        # them properly.
        return out, {"reason": f"only {len(out)} EIS block — no step measured"}

    steps_ah = out["step_ah"].to_numpy(dtype=float).copy()
    if len(steps_ah) > 1 and not np.isfinite(steps_ah[1:]).all():
        n_bad = int((~np.isfinite(steps_ah[1:])).sum())
        return out, {
            "reason": f"{n_bad} of {len(steps_ah) - 1} EIS block(s) have no "
                      f"preceding SOC-adjust step"
        }
    steps_ah[0] = 0.0                      # anchor: setup charge is not a step

    start = 100.0 if direction == "discharge" else 0.0
    soc = start + 100.0 * np.cumsum(steps_ah) / float(capacity)
    out["SOC_pct"] = np.round(soc, 2)

    step_pct = np.abs(steps_ah[1:]) * 100.0 / float(capacity)
    info = {
        "soc_anchor_pct": start,
        "capacity_ref_ah": round(float(capacity), 4),
        "n_eis_blocks": int(len(out)),
        "step_pct_mean": round(float(step_pct.mean()), 3) if len(step_pct) else None,
        "step_pct_min": round(float(step_pct.min()), 3) if len(step_pct) else None,
        "step_pct_max": round(float(step_pct.max()), 3) if len(step_pct) else None,
        "soc_span_pct": round(float(soc.max() - soc.min()), 2),
    }

    lo, hi = -SOC_RAIL_TOLERANCE_PCT, 100.0 + SOC_RAIL_TOLERANCE_PCT
    n_out = int(((soc < lo) | (soc > hi)).sum())
    if n_out:
        # Overflow means the counted charge does not fit the capacity it was
        # divided by — a wrong reference, a missed step, or a sweep that did not
        # start at the rail. Worth saying, but the numbers are the measured
        # ones, so they are kept rather than blanked.
        logging.warning(
            "%s: %d of %d step-counted SOC values fall outside %.0f–%.0f%% "
            "(span %.1f%%, capacity %.3f Ah) — check the capacity reference "
            "and that no SOC-adjust step was dropped",
            label or "soc_from_steps", n_out, len(soc), lo, hi,
            info["soc_span_pct"], capacity,
        )
        info["n_outside_rails"] = n_out
    return out, info


def _held_soc(blocks: pd.DataFrame, proc: float) -> float:
    """SOC of the most recent block at or before ``proc``.

    This is the "hold it until the next EIS" rule: everything measured after a
    block and before the next one sits at that block's SOC. A pulse *before* the
    first block has no measured SOC (NaN) rather than being given the anchor —
    it is on the setup ramp, not on the sweep.
    """
    if blocks.empty or not np.isfinite(proc):
        return np.nan
    prior = blocks[blocks["first_proc"] <= proc]
    if prior.empty:
        return np.nan
    return float(prior["SOC_pct"].iloc[-1])


def assign_eis_soc(table: pd.DataFrame, blocks: pd.DataFrame,
                   anchor_id_col: str = "segment_ID") -> int:
    """Set ``SOC_pct`` on an EIS measurement table from its block. -> n filled.

    Each row carries the GOLD segment it was measured in; that segment belongs
    to exactly one block, and the block carries the SOC.
    """
    if blocks.empty or anchor_id_col not in table.columns:
        return 0
    soc_by_id = {
        seg_id: row["SOC_pct"]
        for _, row in blocks.iterrows() for seg_id in row["ids"]
    }
    mapped = table[anchor_id_col].map(soc_by_id)
    table["SOC_pct"] = mapped
    return int(mapped.notna().sum())


def assign_pulse_soc(table: pd.DataFrame, blocks: pd.DataFrame,
                     id_col: str = "ID") -> int:
    """Set ``SOC_pct`` on a pulse table by holding the last block's SOC.

    -> number of rows filled.
    """
    if blocks.empty or id_col not in table.columns:
        return 0
    procs = table[id_col].map(lambda i: parse_proc_id(i)[1])
    mapped = procs.map(lambda p: _held_soc(blocks, p if p is not None else np.nan))
    table["SOC_pct"] = mapped
    return int(mapped.notna().sum())
