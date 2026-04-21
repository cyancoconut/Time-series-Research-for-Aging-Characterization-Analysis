# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

All scripts and notebooks must be run from the `src/` directory (imports are relative to `src/`).

**Main pipeline (CLI)**: `src/main.py`

```bash
cd src
python main.py /path/to/battery_config_VTC_linux.json
python main.py /path/to/battery_config_VTC_linux.json --cells VTC_cell01 VTC_cell02
```

The legacy notebook `src/Process_Detection_via_Cluster_py_METABATT.ipynb` also works but `main.py` is the current entry point.

**Venv (Linux)**: `source venv/bin/activate` from project root.

## Architecture

The pipeline uses a medallion architecture. The main layers are all parquet (time-series):

```
BRONZE_CU → preSILVER → SILVER → GOLD
```

- **BRONZE_CU**: raw check-up cycler export (CU = check-up), German columns, unsegmented
- **preSILVER**: segmented into discrete procedures; PAU pauses removed, short segments dropped, columns renamed to English. Written to disk for debugging.
- **SILVER**: preSILVER with cluster labels merged back. Written to disk for debugging. In `main.py`, held in memory as `df_silver` — the `"silver"` path in `_build_paths` is unused dead code.
- **GOLD**: SILVER enriched with calculated metrics (capacity, pulse resistance, qOCV)

Two CSV helper layers support the clustering step (not time-series — one row per segment):

```
with_features_pre_labeled → [HDBSCAN clustering] → with_features_post_labeled
```

These are a per-segment projection of preSILVER. Labels from `with_features_post_labeled` are merged back into the time-series to produce SILVER.

**Pipeline stages and their modules:**

1. **`dismember/dismember_raw_cell.py`** — reads BRONZE_CU parquet, renames German columns (`Spannung→Voltage`, `Strom→Current`, `Zeit→Time`, `T1→Temperature`), segments into discrete procedures. Groups by `Ahjo_Test_ID` → `BM_Programm`, splits by `Prozedur` changes and PAU pauses > `pau_duration` minutes. Drops segments with < `min_rows` rows. Assigns string ID: `<BM_Programm>_<procedure_number>` (e.g. `13_16`). Core logic: `dismember/cluster_preparation.py` (`DismembererFunctions`, `allocate_IDs`).

2. **`feature_extraction/create_features.py`** + **`feature_extraction/classification.py`** — per-segment statistical features (mean, std, min, max of Voltage/Current/Temperature). Normalization: Voltage by `(V_max - V_min)`, Current/Power by `Nom_Capacity`. Adds `Duration_quartile = log1p(Duration_minutes)` and `abs_Current_mean = |Current_mean|`. Saves to `with_features_pre_labeled/<cell>.csv`.

3. **`cluster/model_and_supervise.py`** — two-layer HDBSCAN clustering:
   - Layer 1: clusters on `["Duration_quartile", "abs_Current_mean", "ID"]` via `TabularAutoencoderHDBSCAN.fit_cluster_only()`. `min_cluster_size = max(2, n_programs − 1)`.
   - Layer 2 (only if Layer 1 fails to identify a capacity cluster): re-clusters capacity candidates on `["Current_mean", "ID"]` with stricter masks.
   - `cluster/post_cluster_filter.py` (`cluster_filter`) — rule-based masks to label CAP* / PUL* / QOCV* / −1.

4. **`calculate/results_fetching.py`** — `calculation` class:
   - `update_capacity()` — trapezoidal Ah integration
   - `update_pulse()` — R_ct and R_0 pulse resistance. Calls `_filter_restore_pulses()` first to exclude restore pulses (same |current|, opposite sign, consecutive ID within BM_Programm).
   - `update_qOCV()` — labels qOCV_DCH / qOCV_CHA

5. **`output/`** — uploads to InfluxDB. **`util/connect_minio.py`** — uploads parquet to MinIO.

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `V_max`, `V_min`, `V_nom` | Voltage limits and nominal voltage |
| `Nom_Capacity` | Nominal capacity in Ah |
| `CAP_Rate` | Capacity C-rate vs normalized `Current_mean` (÷ `Nom_Capacity`). `0.5` → C/2. |
| `qOCV_CRate` | C-rate threshold for quasi-OCV (`0.05` → C/20, ~1200 min full discharge) |
| `pau_duration` | Pause threshold in minutes for procedure boundary detection (default 9.9) |
| `min_rows` | Minimum rows to keep a procedure segment (default 20) |
| `target_pulse_duration` | Expected pulse duration in seconds (default 20 s) |

`hdbscan_para_layer_1["min_cluster_size"]` is overridden at runtime to `max(2, n_programs − 1)`.
`hdbscan_para_layer_1["cluster_selection_epsilon"]` must be **0.3** (not 3.0) for correct qOCV separation.

## qOCV detection note

qOCV procedures come in discharge+charge pairs per aging cycle. Their signed `Current_mean` ≈ 0 (cancels), making them indistinguishable from rest segments if only signed current is used. `abs_Current_mean` is added to Layer 1 features to break this degeneracy. Layer 1 `cluster_selection_epsilon = 0.3` (tight) is required — 3.0 merges qOCV with rests.

## Restore pulse structure

After each test pulse a restore pulse returns the cell to its original SoC. All restore pulses run at C/2. The C/2 restores (20 s) are filtered in `update_pulse` by `_filter_restore_pulses`: within each BM_Programm, a PUL* segment is a restore if it has the same |current| (±5 %) as the immediately preceding pulse but opposite sign. 1C restores (~40 s at C/2) are already rejected by the duration check in `fetch_pulse`.

## Configuration

- **Battery parameters**: JSON config file passed to `main.py` (e.g. `battery_config_VTC_linux.json` in the data directory).
- **MinIO/Ahjo credentials**: `config.json` at project root (gitignored). Copy structure from `config_SE_example.json`. Also available via env vars: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `INFLUX_TOKEN`.
- **Data path**: set `working_path` in the config JSON. BRONZE_CU parquet files must be under `<working_path>/BRONZE_CU/`.

## Documentation

- `METAbatt_Pipeline_Report.md` — full technical report of the pipeline
- `METAbatt_Pipeline_Flowchart.svg` — visual flowchart of all pipeline stages
