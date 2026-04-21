# METAbatt Pipeline — Technical Report

## 1. Overview

The METAbatt pipeline automatically processes raw battery cycling data from a cycler (Ahjo/BaSyTec), identifies and classifies test procedures, and computes electrochemical metrics (capacity, pulse resistance, quasi-OCV). The pipeline eliminates manual labelling of test protocols by using unsupervised machine learning (HDBSCAN clustering) combined with physics-based supervision rules.

The entry point is `src/main.py`. Configuration is provided via a JSON file (e.g. `battery_config_VTC_linux.json`).

---

## 2. Data Layers

Data flows through five progressively refined stages:

| Layer | Location | Content |
|-------|----------|---------|
| **BRONZE_CU** | `BRONZE_CU/<cell>.parquet` | Raw cycler export (German column names) |
| **preSILVER** | `preSILVER/<cell>.parquet` | Segmented procedures with IDs assigned |
| **with_features_pre_labeled** | `with_features_pre_labeled/<cell>.csv` | Per-segment statistical features |
| **with_features_post_labeled** | `with_features_post_labeled/<cell>.csv` | Features after clustering and labelling |
| **GOLD** | `GOLD/<type>/<cell>.parquet` | Final results with capacity, pulse, qOCV values |

---

## 3. Pipeline Stages

### 3.1 Dismembering (`dismember/dismember_raw_cell.py`)

**Purpose:** Convert a continuous time-series recording into discrete procedure segments.

**Steps:**
1. Filter cells that do not contain the required procedure string (`jri_CU`).
2. Rename German columns: `Spannung → Voltage`, `Strom → Current`, `Zeit → Time`, `T1 → Temperature`.
3. Compute `Power = Current × Voltage` and add output columns (`Capacity_py`, `Pulse_py`, `target`).
4. Remove pause states (`PAU`, `PAUO`) and rest/save segments.
5. Split each `BM_Programm` (grouped by `Ahjo_Test_ID`) by:
   - Changes in the `Prozedur` string
   - PAU pauses longer than `pau_duration` minutes (default 9.9 min)
6. Drop segments with fewer than `min_rows` rows (default 20).
7. Assign string IDs in the form `<BM_Programm>_<procedure_number>` (e.g. `13_16`).

**Key classes/functions:**
- `DismembererFunctions` (`dismember/cluster_preparation.py`) — `prefiltering()`, `dismembling()`
- `allocate_IDs()` — assigns the `ID` column

---

### 3.2 Feature Extraction (`feature_extraction/create_features.py`, `feature_extraction/classification.py`)

**Purpose:** Summarise each procedure segment into a fixed-length feature vector for clustering.

**Features computed per segment:**

| Feature | Description |
|---------|-------------|
| `Voltage_mean/std/min/max/range` | Normalised by `(V_max − V_min)` |
| `Current_mean/std/min/max` | Normalised by `Nom_Capacity` |
| `Temperature_mean/std/min/max` | Raw °C |
| `Duration_minutes` | Segment duration |
| `Duration_quartile` | `log1p(Duration_minutes)` — used for clustering |
| `abs_Current_mean` | `|Current_mean|` — separates qOCV from rest |
| `BM_Programm` | Program group index |

**Normalisation rationale:**
- Voltage is normalised to [0, 1] within the cell's operating window.
- Current is normalised by nominal capacity so that values directly represent the C-rate. A value of −0.5 corresponds to a C/2 discharge.

The feature CSV is saved to `with_features_pre_labeled/` and also returned directly for in-memory use.

---

### 3.3 Clustering (`cluster/model_and_supervise.py`, `cluster/feature_extract_HDBSCAN.py`)

The pipeline uses a two-layer HDBSCAN approach to identify test types without pre-labelled training data.

#### Layer 1 — Coarse clustering

**Input features:** `["Duration_quartile", "abs_Current_mean", "ID"]`

`abs_Current_mean` is critical for separating:
- **qOCV procedures** (|I| ≈ 0.05 C-rate, very long duration ~1200 min): qOCV runs come in discharge + charge pairs; their *signed* mean ≈ 0 so they would otherwise merge with rest segments. Using the absolute value breaks this degeneracy.
- **Rest / pause segments** (|I| ≈ 0)
- **Capacity tests** (|I| ≈ 0.5 C-rate, long duration)
- **Pulse tests** (|I| >> 0.5, very short duration)

HDBSCAN parameters for Layer 1:
```
min_cluster_size = max(2, n_programs − 1)
min_samples      = 1
cluster_selection_epsilon = 0.5
```

#### Supervised filter (`cluster/post_cluster_filter.py`)

After Layer 1, rule-based masks are applied to identify which cluster corresponds to which test type:

- **`find_capacity`** — looks for a cluster where:
  - `|Current_mean|` is within ±10 % of `CAP_Rate` (default C/2)
  - `Current_mean < 0` (discharge direction)
  - `Duration_minutes` is within ±20 % of `60 / CAP_Rate`
  - `Voltage_min < 0.005` (fully discharged)

- **`find_pulses`** — selects the cluster with the shortest `Duration_minutes`.

- **`find_qocv`** — selects the cluster whose `Duration_minutes` is closest to `60 / qOCV_CRate` (default C/20 → 1200 min), with a ceiling of `(60 / qOCV_CRate) + 100` min.

If Layer 1 cannot unambiguously identify the capacity cluster, `counter` is set to 1 and Layer 2 is triggered.

#### Layer 2 — Fine clustering on capacity candidates

**Input features:** `["Current_mean", "ID"]`

Applies stricter current masks (±2 % of `CAP_Rate`) and selects the smallest valid cluster to isolate true capacity measurements from cycling data.

After labelling, cluster targets are renamed:
- `CAP*` — capacity measurements
- `PUL*` — pulse resistance measurements
- `QOCV*` — quasi-OCV measurements
- `-1` — noise / unclassified

Additional post-clustering filters:
- `check_temperature` — keeps only capacity segments within ±3 °C of `CAP_Temp`
- `check_CRate` — ensures current is within ±5 % of `CAP_Rate`
- `check_previous_voltage` — verifies the preceding segment had `Voltage_max > 0.99` (full charge before capacity test)

---

### 3.4 Results Calculation (`calculate/results_fetching.py`)

The `calculation` class computes physical metrics on the labelled SILVER DataFrame.

#### Capacity (`update_capacity` → `fetch_capacity`)
- Integrates `|Current|` over time using the trapezoidal rule to obtain Ah throughput.
- Rejects segments where computed capacity < `Nom_Capacity / 3` (outlier check).
- Stores result in `Capacity_py` column.

#### Pulse Resistance (`update_pulse` → `fetch_pulse`)
Two resistance metrics are computed per pulse:

| Metric | Formula |
|--------|---------|
| **R_ct** (charge-transfer) | `|ΔV| / |I_mean|` over the full pulse |
| **R_0** (ohmic) | `(V_before_pulse − V_first_sample) / I_first_nonzero` |

**Restore pulse filtering** (`_filter_restore_pulses`):
After each test pulse, a restore pulse returns the cell to its original SoC. These must be excluded:
- Restore pulses always run at C/2 current.
- Within each `BM_Programm`, a pulse is identified as a restore if it has the **same current magnitude** (within 5 %) as the immediately preceding pulse **and opposite sign**.
- 1 C restore pulses (~40 s) are already rejected by the duration check.

#### Quasi-OCV (`update_qOCV` → `fetch_qOCV`)
- Confirms `|I_mean| < qOCV_CRate × Nom_Capacity + 0.01 A` and `I_std < 1 mA`.
- Labels as `qOCV_DCH` (discharge) or `qOCV_CHA` (charge) based on current sign.
- Rejects segments where computed capacity < `Nom_Capacity / 3`.

---

## 4. Configuration Parameters

All parameters are stored in a JSON file passed on the command line:

```json
{
    "working_path": "/path/to/data",
    "type_cell": "VTC",
    "nom_capacity": 3.0,
    "v_min": 2.5,  "v_max": 4.2,  "v_nom": 3.6,
    "qocv_crate": 0.05,
    "cap_type": "CC",
    "cap_rate": 0.5,
    "cap_temp": 25,
    "target_pulse_duration": 20,
    "pulse_type": 1,
    "pulse_target_unit": "Resistance",
    "min_rows": 20,
    "pau_duration": 9.9,
    "feature_columns": ["Voltage", "Current", "Temperature"],
    "hdbscan_para_layer_1": { "min_cluster_size": 8, "min_samples": 8,
                               "cluster_selection_epsilon": 0.3, "allow_single_cluster": false },
    "hdbscan_para_layer_2": { "min_cluster_size": 3, "min_samples": 3,
                               "cluster_selection_epsilon": 0.001, "allow_single_cluster": false }
}
```

---

## 5. Running the Pipeline

```bash
cd /home/ann/Documents/Project_METAbatt/src
source ../venv/bin/activate
python main.py /path/to/battery_config_VTC_linux.json

# Process a subset of cells:
python main.py /path/to/battery_config_VTC_linux.json --cells VTC_cell01 VTC_cell02
```

---

## 6. Module Summary

| Module | File | Role |
|--------|------|------|
| Entry point | `main.py` | CLI, orchestrates all stages per cell |
| Dismembering | `dismember/dismember_raw_cell.py` | Raw parquet → segmented DataFrame |
| Segment prep | `dismember/cluster_preparation.py` | Filtering, splitting, ID assignment |
| Feature extraction | `feature_extraction/create_features.py` | Segments → feature vectors |
| Feature maths | `feature_extraction/classification.py` | Normalised statistics per segment |
| HDBSCAN model | `cluster/feature_extract_HDBSCAN.py` | Autoencoder + HDBSCAN clustering |
| Cluster orchestration | `cluster/model_and_supervise.py` | Layer 1 + 2, merge targets |
| Cluster supervision | `cluster/post_cluster_filter.py` | Rule-based cluster identification |
| Results | `calculate/results_fetching.py` | Capacity, pulse resistance, qOCV |
| Visualise/label | `visualize/add_test_schedule.py` | Adds aging labels to GOLD output |
