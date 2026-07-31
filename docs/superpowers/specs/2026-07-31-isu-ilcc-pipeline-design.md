# ISU-ILCC through the METAbatt pipeline — design

**Date:** 2026-07-31
**Branch:** `feat/isu-ilcc-pipeline`
**Goal (paper):** Demonstrate the METAbatt signal-based check-up pipeline generalizes
to an independent public NMC dataset. **Success criterion:** the pipeline
*auto-detects* the capacity segment per reference-performance-test (RPT) and its
extracted `Capacity_py` matches ISU-ILCC's own author-extracted
`capacity_discharge_C_2` per RPT, within a small tolerance (target MAPE < ~5 %).

## Dataset facts (verified)

ISU-ILCC = Iowa State "Iowa Long-term Lithium-ion Cell Cycling" dataset.
- Location: `/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/`,
  `Release 1.0/` (238 cells) + `Release 2.0/` (13 cells). Files `G<group>C<cell>.json`.
- Each `.json` is **double-encoded** (a JSON string containing JSON — decode twice;
  mirrors the shipped `Example_Load_Matlab.m` which does `jsondecode(fileread(...))`).
- Per cell, top-level keys: `capacity_{charge,discharge}_{C_5,C_2}` (scalar per RPT),
  `QV_{charge,discharge}_{C_5,C_2}` (per-RPT time series `{Q, V, t, E, I}`),
  `start_stop_time`.
- Each cell has **N RPTs** (varies 2–31). Every RPT contains four sweeps:
  C/2 charge, C/2 discharge, C/5 charge, C/5 discharge.
- **Chemistry:** 4.2 V max / 3.0 V min → NMC/LCO. **Small format:** ~0.27 Ah,
  ~1 Wh (verified: `Q` range = capacity scalar; `E` ≈ Q·V̄; `|I|·t` ≈ Q).
- **Sign convention:** charge `I > 0`, discharge `I < 0` — matches the pipeline
  (`Strom > 0` = charge). No negation needed.
- **Measured rates (normalized by ~0.28 Ah):** C/2 discharge |I| ≈ 0.125 A ≈ **0.45C**;
  C/5 discharge |I| ≈ 0.05 A ≈ **0.18C**. The "C/2"/"C/5" names are nominal.
- **No temperature field** in the JSON → must be synthesized.
- **No aging-condition metadata** (DOD/SOC/C_Rate) in the RPT-only JSON — it lives in
  the group number / external docs, not in procedure names.

## Approach

ISU is not the German per-test-file cycler format, so `build_bronze_cu_with_ah.py`
is **not** touched. Instead a standalone adapter synthesizes a BRONZE_CU parquet per
cell that satisfies the dismember contract, and the **unchanged** `main.py` classifier
path runs it. Rejected alternatives: generalizing the existing builder (too coupled to
German columns + MinIO); a bespoke mini-pipeline (throws away the exact
GOLD/`export_capacity` machinery we want to demo).

## Components

### 1. `src/download/build_bronze_isu.py` (new)

CLI: `python download/build_bronze_isu.py <isu_config.json> [--cells G14C1 …] [--overwrite]`

Per cell JSON (decode twice), for each RPT index `r`:
- Build four sub-frames from `QV_{charge,discharge}_{C_2,C_5}[r]`:
  `Zeit ← t` (real acquisition timestamps), `Strom ← I` (sign already correct),
  `Spannung ← V`. Concatenate the four in acquisition-time order.
- Synthesize the columns the dismember contract (`read_and_fix_format`) requires:
  - `T1 = 25.0` (constant; no temperature in source).
  - `AhAkku = NaN` (a column must exist — `read_and_fix_format` references it
    unconditionally in the EIS mask; absence → KeyError).
  - `Prozedur = "ISU_RPT"` (constant; the CU / `procedure_filter` marker).
  - `Zustand` **unique per sweep**: `CHA_C2 / DCH_C2 / CHA_C5 / DCH_C5`. With
    `qocv_procedure_filter` matching, every Zustand change cuts a boundary → four
    clean segments per RPT.
  - `Ahjo_Test_ID = "{cell}_RPT{r:03d}"` (**zero-padded**) → dismember's
    `groupby(Ahjo_Test_ID).ngroup()` gives **one `BM_Programm` per RPT** in RPT order.
    Padding matters: `ngroup()` sorts the key as a string, so an unpadded `RPT10`
    would sort before `RPT2`. `export_capacity` then emits one CAP/SOH row per RPT.

Concatenate all RPTs, sort by `Zeit`, compute `Ah_throughput` via
`util.add_ah_throughput`, write `<working_path>/BRONZE_CU/<cell>.parquet`.

Output column set (per BRONZE_CU): `Zeit, Strom, Spannung, T1, Zustand, Prozedur,
AhAkku, Ahjo_Test_ID, Ah_throughput`.

### 2. `battery_config_ISU_linux.json` (new, repo root)

Mirrors `battery_config_example.json` schema (lowercase keys), with:
- `type_cell = "ISU"`, `v_max = 4.2`, `v_min = 3.0`, `v_nom = 3.7`,
  `nom_capacity = 0.28`.
- `cap_rate = 0.45` (measured normalized C/2 discharge; ±5 % → 0.43–0.47 locks onto
  the C/2 sweep). `cap_type = "CC"`, `cap_temp = 25`.
- `qocv_crate = 0.05` (kept low so the 0.18C C/5 sweep is **not** force-mapped to
  qOCV; its label is left to whatever the classifier predicts, then reported).
- `procedure_filter = "ISU_RPT"`; `qocv_procedure_filter = "ISU_RPT"` (reuses the
  Zustand-change→boundary cut for clean per-sweep segmentation).
- `classifier_model_path` / `classifier_meta_path` → the provided chemistry-robust
  model, copied into `<working_path>/60_classifier/models/`
  (`vtc_classifier_20260731T135816.joblib` + `_meta.json`; `label_source = "llm"`,
  classes include `cap`, `full_discharge_*`, `qocv`, `pulse`, `partial_*`).
- `download_from = "local"`, `upload_to = "local"`.
- `working_path = "/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline"`.
- `min_rows = 20`, `pau_duration = 9.9`, standard `tolerances`,
  `feature_columns = ["Voltage", "Current", "Temperature"]`, standard HDBSCAN
  params (present but unused on the classifier path).

### 3. `src/evaluation/validate_isu.py` (new — paper deliverable)

Joins pipeline `40_capacity_monitore/<cell>_capacity.csv` against ISU
`capacity_discharge_C_2[r]` per (cell, RPT). The join key is the **CAP-segment
timestamp**: each pipeline CAP row's `CAP_start_time` is matched to the RPT `r` whose
C/2-discharge time window (from the ISU `t[r]` array) contains it. This is robust to a
mid-run RPT yielding no CAP (that RPT is simply reported as "n missing" without shifting
the others). Emits into
`<working_path>/50_evaluation/`:
- `isu_capacity_validation.csv` — per (cell, RPT): pipeline `Capacity_py`, ISU ground
  truth, absolute + percent error, and the classifier label the CAP segment got.
- error summary (MAPE, bias, n matched / n missing).
- a scatter/overlay plot (pipeline vs ground truth; SOH-vs-RPT per cell).

## Data flow

```
ISU G<g>C<c>.json
  └─ build_bronze_isu.py ──> BRONZE_CU/<cell>.parquet
       └─ main.py (classifier path) ──> dismember ──> features
            ──> predict_classifier ──> GOLD ──> export_capacity
                 └─ 40_capacity_monitore/<cell>_capacity.csv
                      └─ validate_isu.py  vs  capacity_discharge_C_2  ──> 50_evaluation/
```

## Segmentation & labeling logic (the subtle part)

- One RPT → one `BM_Programm`; unique `Zustand` per sweep + `qocv_procedure_filter`
  match ⇒ dismember cuts on every Zustand change ⇒ 4 segments/RPT.
- Classifier (`label_source = "llm"`) maps at inference:
  `cap → CAP*` directly; a residual `full_discharge` resolved by **measured**
  `abs_Current_mean` against `cap_rate` (0.45 ±5 %) → `CAP*`. So the **C/2 discharge**
  becomes CAP either way.
- The **C/5 sweep is out-of-distribution** (no `full_discharge_c5` class): it lands in
  the nearest trained class (likely `qocv` or a `partial_*`). We **report** what it
  becomes; we do not force it. `qocv_crate = 0.05` prevents the residual-rate path from
  turning the 0.18C sweep into qOCV.
- `export_capacity` picks the CAP segment per `BM_Programm` → `Capacity_py` = the
  trapezoidal Ah of the C/2 discharge ⇒ the SOH timeline.

## Error handling

- Cells with few RPTs (min 2) still work; a missing QV rate for an RPT → skip that sweep.
- If the classifier finds no CAP in an RPT, the pipeline's `ClusterNotFound` path skips
  GOLD for that cell (logged, not a failure); `validate_isu` flags the gap as
  "n missing".
- Malformed / non-decodable JSON → skip the cell with a warning.

## Out of scope

- **Aging matrix** (`evaluation/aging_matrix.py`): ISU's RPT-only JSON has no
  DOD/SOC/C_Rate in procedure names, so the metadata parse would be degenerate.
- **MinIO** routing: local-only for this demo.
- HDBSCAN comparison run: the chosen path is the classifier; HDBSCAN is not exercised
  here.

## Testing & rollout

1. **Unit-check `build_bronze_isu` on `G14C1`**: column contract present; exactly one
   `BM_Programm` per RPT; discharge `Strom < 0`; `Ah_throughput` monotonic non-decreasing;
   row count sane.
2. **3-cell pilot** (e.g. `G4C1` = 31 RPTs, plus two others with differing RPT counts):
   run `main.py` on the classifier path, then `validate_isu`. Confirm `Capacity_py` vs
   `capacity_discharge_C_2` MAPE < ~5 %, and inspect what label the C/5 sweep received.
3. **Full run** (all 238 R1.0 cells, optionally +13 R2.0) once the pilot passes.

## Key parameters (summary)

| Config key | Value | Why |
|---|---|---|
| `nom_capacity` | 0.28 | Verified fresh C/2 discharge capacity (Ah) |
| `cap_rate` | 0.45 | Measured normalized C/2 discharge (0.125/0.28); ±5 % locks CAP |
| `qocv_crate` | 0.05 | Low, so 0.18C C/5 is not force-mapped to qOCV |
| `cap_temp` | 25 | Synthesized constant `T1` |
| `v_max / v_min / v_nom` | 4.2 / 3.0 / 3.7 | NMC window |
| `procedure_filter` | `"ISU_RPT"` | CU marker (constant `Prozedur`) |
| `qocv_procedure_filter` | `"ISU_RPT"` | Enables per-Zustand segment boundaries |
