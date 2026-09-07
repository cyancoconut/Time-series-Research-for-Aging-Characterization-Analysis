# Result — segmentation metadata-leak audit (paper validation item 2)

**Date:** 2026-09-07
**Branch:** `feat/segmentation-metadata-audit`
**Answers:** the open item in `2026-07-23-paper-scope-checkup-segmentation.md`,
correction #4 — *"confirm how much of the boundary detection depends on
`Prozedur`/`Zustand` vs. pure signal + physical config."*

**Verdict: the claim survives, and the honest version of it is stronger than
the wording we had been protecting.**

---

## What was measured

`src/evaluation/segmentation_audit.py` — read-only, changes no pipeline
behaviour. An ablation ladder over the four boundary rules in
`DismemblerFunctions.dismembling`:

| | boundary rules | protocol columns read |
|---|---|---|
| **L0** | long-PAU start, long-PAU exit, `Prozedur` change, `Zustand` change inside a `qocv_procedure_filter` match | `Prozedur`, `Zustand` |
| **L1** | L0 minus the `Prozedur`-change rule | `Zustand` |
| **L2** | L1 with `Zustand` rebuilt from the current (`\|I\| < i_tol` → PAU, else sign), qOCV rule ungated | **none** |

L2 is not "L1 with more removed" — it also promotes the qOCV rule to a general
signal rule. That is the point: the question is not how much a stripped
segmenter loses, but whether the *signal* carries the structure the protocol
columns were supplying.

Fleet: 7 NMC (Sony/Murata VTC6), 2 LFP (A123 APR18650M1B), 1 SIB (Hina 10 Ah)
— 131.4 M rows, 13 025 L0 segments.

**Fidelity gate.** `_dismember_at_level` is a level-parameterised copy of the
production segmenter. Its L0 output is compared row for row against
`dismember_raw_cell` run live; one differing row aborts the cell. All 10 cells
passed. The gate is deliberately not built on the stored label CSVs — those
date from 2026-06-30 and predate the 2026-08-24 AGING-stub change, and the
first version of this audit failed against them for exactly that reason.

---

## 1. `Zustand` carries no information beyond `sign(I)`

**1 100 of 131 442 261 rows disagree (0.0008 %).** PAU recall is exactly 1.000
on every cell of every chemistry; PAU precision ≥ 0.99986 — and it is flat
across `i_tol` from C/10 000 to C/33, i.e. two and a half orders of magnitude.
There is no threshold to tune.

The residue is not a chemistry effect: it is a few hundred rows of cycler
bookkeeping states (`STO`, `INT`, `SYNCLINE`, `SAVE**`, `;`) that sit at zero
current and are read as PAU, plus ~600 rows where a CHA/DCH step is caught at
its zero crossing.

**For the paper:** the cycler's state column is decorative. Anything the
pipeline does with `Zustand` it can do with the measured current.

## 2. `Prozedur` fires constantly but decides almost nothing

| chemistry | `Prozedur` firings | fire where no other rule does | of those, land in the discard bucket |
|---|---|---|---|
| NMC | 2 177 | 1 242 (57.1 %) | 1 037 (83.5 %) |
| LFP | 585 | 358 (61.2 %) | 312 (87.2 %) |
| SIB | 78 | 16 (20.5 %) | 13 (81.2 %) |

A majority of `Prozedur`-change boundaries are unique at the mask level — which
is why this looked dangerous — but 81–87 % of those cut *inside* a long rest or
a sub-`min_rows` fragment that is discarded regardless. The mask-level number
overstates the dependence by roughly 6×.

## 3. Check-up segment recovery — the number the claim rests on

Fate of every L0 segment carrying a final `CAP` / `PUL` / `PUL*RES` /
`qOCV_DCH` / `qOCV_CHA` label. *Intact* = exact or split; a split segment still
exists, cut into pieces, whereas a merged or lost one has been swallowed.

| chemistry | n | L1 exact | L1 intact | L2 exact | **L2 intact** |
|---|---|---|---|---|---|
| NMC | 3 054 | 89.3 % | 89.3 % | 99.9 % | **100 %** |
| LFP | 332 | 85.5 % | 85.5 % | 99.7 % | **100 %** |
| SIB | 90 | 100 % | 100 % | 95.6 % | **100 %** |

**Zero merged and zero lost check-up segments at L2, on all three chemistries.**

The shape of the L1 loss is diagnostic: every single one is a `PUL` (349) or
`PUL*RES` (27) merge. **`CAP` and `qOCV_*` never break at any level.** Pulse
trains are where consecutive steps are separated only by a procedure change and
a *sub-threshold* pause, so `Prozedur` was the only thing cutting them — and a
state change, which is precisely what a pulse boundary is in the signal,
recovers them at L2.

### Why L2 beats L1 — the mechanism, checked by hand

Of the 142 `Prozedur`-unique firings on NMC cell `…VTC6_003`, 108 land in the
discard bucket. Inspecting the 34 that survive shows every one is the same
thing:

```
 _row Zustand         Prozedur  Current  Voltage
   68     PAU jri_CU_VTC6_Hyst  0.00000 3.834484
   69     PAU jri_CU_VTC6_Hyst  0.00000 3.834484
   70     CHA    jri_Charge_C2  0.00000 3.862045   <- boundary
   71     CHA    jri_Charge_C2  1.49972 3.863712
```

A rest shorter than `pau_duration` followed by a new step. Neither pause rule
can fire (the pause is sub-threshold), so at L1 nothing cuts here and the step
is swallowed by its predecessor. But the **state changes**, `PAU → CHA`, and
that is visible in the current alone — so L2's ungated state rule cuts exactly
where `Prozedur` did.

This is the whole result in one frame: `Prozedur` was never contributing
information, only *timing*. The same boundary is present in the signal; the
production segmenter simply reads it off the cheaper column.

---

## What this means for the manuscript

The defensible claim is now stronger than the hedge we had prepared:

> Boundary placement requires no protocol column. Using only the measured
> current and two scalar thresholds (`pau_duration`, `min_rows`), segmentation
> recovers 100 % of the capacity, pulse and quasi-OCV segments intact across
> NMC, LFP and sodium-ion cells.

`Prozedur` and `Zustand` are used in the production path as a convenience, not
a necessity — and that is now a measured statement, not an assertion.

**Say honestly, do not skip:**

1. **L0 is still the shipped path.** L2 was measured, not adopted. Building it
   as a config-switchable segmenter was deliberately deferred (it is the L3
   step in the original plan).
2. **This measures segmentation, not the full pipeline.** Splits are counted
   as intact because the segment survives, but a split `PUL` reaches the
   classifier as two fragments. No end-to-end SOH comparison under L2 was run.
3. **L2 damages the PAU structure** (NMC: 1 161 merged, 302 lost non-check-up
   segments). PAU stubs feed `prev_end_voltage_norm` and the pulse relaxation
   windows, so a real L2 segmenter needs to handle rests explicitly. It does
   not touch the taxonomy claim.
4. **EIS windows are not signal-separable.** The `Prozedur.str.contains("_EIS_")`
   relabel has no L2 equivalent — at zero current an EIS dwell reads as a rest.
   That is labelling rather than boundary placement, but `export_eis` anchors on
   it, so an EIS-carrying source needs the instrument channel or the procedure
   name.
5. **Two metadata uses remain outside boundary detection** and should be
   disclosed rather than defended: `prefiltering` drops rows with
   `Zustand ∈ {SAVE, REST}`, and `BM_Programm` comes from `Ahjo_Test_ID` (file
   identity, not protocol semantics — there is already a single-group fallback).

---

## Reproducing

```bash
cd src
for c in VTC APR Hina; do
  python -m evaluation.segmentation_audit \
    /home/ann/Documents/Data_Metabatt/battery_config_${c}_linux.json \
    -o /home/ann/Documents/Data_Metabatt/50_evaluation
done
python -m evaluation.segmentation_audit --pool \
  -o /home/ann/Documents/Data_Metabatt/50_evaluation
```

~25 min for the fleet. Per-config outputs are namespaced by `type_cell`;
`--pool` writes `segmentation_audit_fleet_{boundary,checkup,rule3,
zustand_confusion}.csv` — the four tables above.

## Next (unchanged from the original plan)

Validation items 3–5: Pozzato–Onori ingestion shim, hand-labelled segment F1 +
check-up detection recall, and the LFP-plateau ablation (note `norm_duration`
in `create_features.py` already implements the coulombic feature that item
proposed).
