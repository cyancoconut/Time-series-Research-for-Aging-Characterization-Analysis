# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always tell me what the plan is before changing code.
Working data path is at /home/ann/Documents/Data_Metabatt.
Update every changes made to the pipeline also here.

## Git workflow

Before making any code changes, always create a new branch:

```bash
git checkout -b feat/my-feature     # new feature
git checkout -b fix/my-bugfix       # bug fix
git checkout -b refactor/my-change  # refactor
git checkout -b config/my-config    # config / parameters
```

Push the branch and open a PR into `J8005_BMWK_METABatt` on GitHub. CodeRabbit will auto-review the PR. Never commit directly to `J8005_BMWK_METABatt`.

## Running the pipeline

All scripts and notebooks must be run from the `src/` directory (imports are relative to `src/`).

**Main pipeline (CLI)**: `src/main.py`

```bash
cd src
python main.py /path/to/battery_config_VTC_linux.json
python main.py /path/to/battery_config_VTC_linux.json --cells VTC_cell01 VTC_cell02
```

The legacy notebook `src/Process_Detection_via_Cluster_py_METABATT.ipynb` also works but `main.py` is the current entry point.

**Venv (Linux)**: `source .venv/bin/activate` from project root.

**Unified UI**: `src/pipeline_ui.py` — customtkinter desktop app that wraps all five stages (Download / Build BRONZE_CU / Run Pipeline / Monitor / Evaluation) in one window with a shared battery-config picker, per-tab Run buttons, a "Run all 1→2→3→4→5" chain button, a Stop button, and a live console for subprocess output. Persists last-used paths to `~/.config/metabatt_ui.json`.

```bash
cd src
python pipeline_ui.py
```

Each Run button spawns a `subprocess.Popen` of the relevant CLI:
- Tab 1 → `download/run_download.py <download_cfg.json>` (headless wrapper around the same logic as `download/download_GUI.py`)
- Tab 2 → `download/build_bronze_cu_with_ah.py <battery_cfg> [--cells …] [--overwrite]`. Honors `download_from` and `upload_to` from the battery config (same semantics as `main.py`). `download_from="local"` reads per-test parquets from `<working_path>/<cell>/*.parquet`; `download_from="minio"` (default) pulls from `<minio_prefix>/<cell>/`. Legacy `save_local` / `upload_s3` keys are still accepted when `upload_to` is absent. Skip-checks: if `upload_to` writes locally and the local BRONZE_CU exists, skip; if `upload_to` writes to MinIO and the MinIO object exists, skip. CU-file detection (`_is_cu`) uses the config's `procedure_filter`: a per-test file is treated as a check-up when its 4th `=`-delimited filename field (the programme name) contains `procedure_filter`. `procedure_filter` is **required** — `process_cell` raises `ValueError` if it is unset.
- Tab 3 → `main.py <battery_cfg> [--cells …] [--overwrite]`
- Tab 4 → `python -m monitor.aging_status <battery_cfg> [-o …]`
- Tab 5 → `python -m evaluation.export_cap_pulse <battery_cfg> [-o …]`

Prereq on Linux: `sudo apt install python3-tk` (Tk bindings are not provided by pip). The Download tab's "Save JSON" writes the same shape as `download/get_user_input.py`; for full-pipeline runs that config is auto-written to `.metabatt_ui_download.json` at the project root (gitignored).

## Architecture

The pipeline uses a medallion architecture. The main layers are all parquet (time-series):

```
BRONZE_CU → preSILVER → SILVER → GOLD
```

- **BRONZE_CU**: raw check-up cycler export (CU = check-up), German columns, unsegmented
- **preSILVER**: segmented into discrete procedures; long PAU pauses (> `pau_duration`) are reduced to first+last row stubs (target="PAU"), short pauses discarded, short segments dropped, columns renamed to English. Written to disk for debugging.
- **SILVER**: preSILVER with cluster labels merged back. Written to disk for debugging. In `main.py`, held in memory as `df_silver` — the `"silver"` path in `_build_paths` is unused dead code.
- **GOLD**: SILVER enriched with calculated metrics (capacity, pulse resistance, qOCV)

Two CSV helper layers support the clustering step (not time-series — one row per segment):

```
with_features_pre_labeled → [HDBSCAN clustering] → with_features_post_labeled
```

These are a per-segment projection of preSILVER. Labels from `with_features_post_labeled` are merged back into the time-series to produce SILVER.

**Procedure-filter gate**: `_process_cell` in `main.py` peeks at the BRONZE `Prozedur` column before pulling the full payload. If `procedure_filter` is set in the config and no `Prozedur` matches, the cell is skipped immediately with `INFO {cell}: no Prozedur matches filter '<filter>', skipping` — dismember is never entered and, on MinIO, `fetch_bronze` is never called. The check uses `processing_procedure_filter` (pyarrow, reads only `Prozedur` row-group by row-group, short-circuits on first match). On MinIO it runs against `io_router.open_bronze_range`, a seekable HTTP-range-read file-like, so a filtered-out cell only fetches the parquet footer + one column.

**Cells without a proper checkup**: if clustering cannot identify a CAP cluster (raised as `ClusterNotFoundException` from `post_cluster_filter.find_capacity` — either no layer-1 candidate in the CAP duration window, or no layer-2 capacity cluster), `_process_cell_inner` in `main.py` catches it and logs a single warning (`no proper checkup detected (no CAP cluster: …) — skipping GOLD`). The cell is skipped cleanly — no GOLD, no exports, no traceback — and the run continues. It is **not** counted as a failure in the exceptions dict.

**Pipeline stages and their modules:**

1. **`dismember/dismember_raw_cell.py`** — reads BRONZE_CU parquet, renames German columns (`Spannung→Voltage`, `Strom→Current`, `Zeit→Time`, `T1→Temperature`), segments into discrete procedures. Groups by `Ahjo_Test_ID` → `BM_Programm`, splits by `Prozedur` changes and PAU pauses > `pau_duration` minutes. Drops segments with < `min_rows` rows. Assigns string ID: `<BM_Programm>_<procedure_number>` (e.g. `13_16`). Core logic: `dismember/cluster_preparation.py` (`DismembererFunctions`, `allocate_IDs`).
   - **qOCV Zustand split**: optional. When `qocv_procedure_filter` is set in the config, every `Zustand` change inside a procedure whose `Prozedur` contains that substring also fires a segment boundary. This is needed when a qOCV procedure's discharge and charge halves share one `Prozedur` and are separated only by a sub-`pau_duration` pause — without the split they collapse into one segment whose signed `Current_mean` (and therefore `abs_Current_mean`) cancels to ≈0, making the qOCV indistinguishable from a rest. The split isolates the `DCH` and `CHA` halves into separate segments. When the key is absent the boundary condition is inert and dismember behaves exactly as before; it is gated per-row on the `Prozedur` match, so all non-qOCV procedures (and cells with no qOCV) are unaffected.
   - **PAU stubs**: long PAU/PAUO segments (> `pau_duration`) are kept as 2-row stubs (first + last row) with their own ID and `target="PAU"`. `Duration_minutes` is the actual pause length (last − first timestamp). Short pauses (≤ `pau_duration`) and middle rows of long pauses are assigned `BM_Programm_procedure=0` (discard bucket). PAU stubs are exempt from the `min_rows` check. They are excluded from feature extraction and clustering (filtered out by the `target == -1` guard in `create_features.py`) but flow through to SILVER and GOLD, making the relaxed-cell voltage and pause duration available for pulse resistance calculations.

2. **`feature_extraction/create_features.py`** + **`feature_extraction/classification.py`** — per-segment statistical features (mean, std, min, max of Voltage/Current/Temperature). Normalization: Voltage by `(V_max - V_min)`, Current/Power by `Nom_Capacity`. Adds `Duration_quartile = log1p(Duration_minutes)` and `abs_Current_mean = |Current_mean|`. Saves to `with_features_pre_labeled/<cell>.csv`.

3. **`cluster/model_and_supervise.py`** — two-layer HDBSCAN clustering:
   - Layer 1: clusters on `["Duration_quartile", "abs_Current_mean", "ID"]` via `TabularAutoencoderHDBSCAN.fit_cluster_only()`. `min_cluster_size = max(2, n_programs − 1)`.
   - Layer 2 (only if Layer 1 fails to identify a capacity cluster): re-clusters capacity candidates on `["Current_mean", "ID"]` with stricter masks.
   - `cluster/post_cluster_filter.py` (`cluster_filter`) — rule-based masks to label CAP* / PUL* / QOCV* / −1.

4. **`calculate/results_fetching.py`** — `calculation` class (labeling + capacity only; pulse/qOCV numeric results live downstream, recomputed from the per-BM_Programm exports):
   - `update_capacity()` — trapezoidal Ah integration → `Capacity_py` column, refines `CAP*` → `CAP`.
   - `update_pulse()` — labels only. Calls `_filter_pulse_group()` to tag restore pulses as `PUL*RES`, then `fetch_pulse` labels remaining `PUL*` as `PUL` (passes duration check) or `-1` (outlier). No `R_ct` / `R_0` / `Pulse_py` columns are written.
   - `update_qOCV()` — labels `QOCV*` → `qOCV_DCH` / `qOCV_CHA` based on sign; outlier-size sanity check uses Ah but doesn't store it.

5. **`output/`** — uploads to InfluxDB. **`util/connect_minio.py`** — uploads parquet to MinIO.

6. **`output/export_pulse.py`, `output/export_qocv.py`** — optional per-`BM_Programm` exports of PUL / qOCV segments from GOLD. Gated by `export_pulse` and `export_qocv` flags in the battery config (both default off). For each BM_Programm, capacity is looked up from the same program's CAP segment (`Capacity_py`) and SOH is computed as `round(Capacity_py / nom_capacity * 100, 1)`; if no valid CAP capacity exists for a program, `SOH=NA` is used and a warning is logged. The pulse export also includes the adjacent PAU stubs (proc_num ±1 within the same BM_Programm) so the relaxation voltage before and after each pulse is preserved. Files:
   - `20_export_pulse/<cell_stem>/<cell_stem>_pulse_BM<BM_Programm>_<SOH>SOH.parquet`
   - `30_export_qocv/<cell_stem>/<cell_stem>_qocv_dch_BM<BM_Programm>_<SOH>SOH.parquet`, `..._qocv_cha_BM<BM_Programm>_<SOH>SOH.parquet`

   Routing follows `download_from` / `upload_to` like GOLD. MinIO keys for exports do **not** include the `10_TRACY` tag — they sit directly under `<minio_prefix>/`.

7. **`output/export_capacity.py`** — always runs at the end of `_process_cell` (no flag). Writes a compact per-cell capacity summary CSV (one row per BM_Programm) consumed by the aging-status monitor and the aging matrix. Columns: `BM_Programm, Capacity_py, Ah_throughput, SOH, CAP_start_time`. `Ah_throughput` is the cumulative throughput at the **start** of the CAP segment (the value at the check-up). Files sit flat under the folder (no per-cell subdirectory):
   - Local: `<working_path>/40_capacity_monitore/<cell_stem>_capacity.csv`
   - MinIO: `<minio_prefix>/40_capacity_monitore/<cell_stem>_capacity.csv` (untagged)

## Evaluation: fleet-wide capacity aggregation

`evaluation/export_cap_pulse.py` aggregates the per-cell `40_capacity_monitore/*_capacity.csv` files into one fleet-wide table for cross-cell analysis. Capacity-only port of the legacy `Export_cap_pulse.ipynb` notebook; pulse aggregation will be a separate script.

```bash
cd src
python -m evaluation.export_cap_pulse /path/to/battery_config.json
# optional: -o /custom/path.csv
```

- **Source** (driven by `download_from`): reads `40_capacity_monitore/*_capacity.csv`, then reads only the `Prozedur` column from each cell's GOLD parquet (via `pyarrow.ParquetFile` + `io_router.open_gold_range` for MinIO range-reads) to build the unique procedure list per cell.
- **Aging metadata**: `output.add_information_METABATT.add_additional_information` parses `DOD / SOC / C_Rate / Temperature` out of the `jri_Aging_DOD..SOC..C..grad..` procedure names.
- **Output** (driven by `upload_to`):
  - `<working_path>/50_evaluation/capacity_results.csv` — all CAP rows across the fleet.
  - MinIO: `<minio_prefix>/50_evaluation/capacity_results.csv` (untagged) when `upload_to` includes `minio`.

  Latest-per-cell SOH is already covered by the aging-status monitor, so this script only emits the full history.

## Evaluation: aging matrix

`evaluation/aging_matrix.py` builds the fleet-wide **Alterungsmatrix** — per-cell capacity loss normalized by Ah throughput, aggregated over the cell design space. Port of the exploratory `evaluation/alterungsmatrix.ipynb` notebook.

```bash
cd src
python -m evaluation.aging_matrix /path/to/battery_config.json
# optional: -o /custom/output_dir
```

- **Source** (driven by `download_from`): reuses `export_cap_pulse.build_capacity_table` for the fleet capacity table. Requires the `Ah_throughput` column in `40_capacity_monitore/*_capacity.csv` — runs predating that column need a pipeline re-run first; the script aborts with a clear error otherwise.
- **Per cell**: `capacity_lost` = max − min of `Capacity_py`; `Delta_Ah_throughput` = max − min of `Ah_throughput`, both across the cell's check-ups.
- **Matrix**: groups cells by `(C_Rate, Temperature, DOD, SOC)` → mean/std of both, `candidate_count`, the cell list, and `capacity_lost_norm = capacity_lost_mean / Delta_Ah_throughput_mean`.
- **Output** (driven by `upload_to`):
  - `<working_path>/50_evaluation/aging_matrix.csv` — the aggregated matrix.
  - `<working_path>/50_evaluation/aging_matrix.html` — interactive plotly report: per-`(C_Rate, Temperature)` 2D SOC×DOD variance scatter and 3D aging surface, plus a multi-temperature 3D surface per C-rate.
  - MinIO: `<minio_prefix>/50_evaluation/...` (untagged) when `upload_to` includes `minio`.

The notebook's cross-cell-type comparison (VTC vs A123) is intentionally dropped — the pipeline runs one battery config (one cell type) at a time.

## Aging-status monitor

`monitor/aging_status.py` builds a sortable HTML report of per-cell SOH for spotting cells approaching EOL while tests are still running. Run on demand:

```bash
cd src
python -m monitor.aging_status /path/to/battery_config.json
# optional: -o /custom/path.html
```

- **Source** (driven by `download_from`): reads `<...>/40_capacity_monitore/*_capacity.csv` for SOH/CU history, then loads only the **last row group** of each cell's GOLD parquet (via `pyarrow.ParquetFile.read_row_group`) to get the most recent `Time` and `Prozedur`. For MinIO, the parquet is opened through `io_router.open_gold_range`, a seekable file-like wrapper around `Minio.get_object(..., offset, length)` so pyarrow can HTTP-range-read just the footer + last row group instead of downloading the full file (per-cell network I/O drops from tens of MB to tens of KB, critical at fleet scale).
- **Output**: `<working_path>/40_capacity_monitore/aging_status.html` locally; uploaded to `<minio_prefix>/40_capacity_monitore/aging_status.html` when `upload_to` includes `minio` (untagged).
- **Columns**: `cell · latest_SOH_% · dSOH_per_CU · n_CU · last_row_time · last_Prozedur · status`.
- **Status** (three-valued, evaluated in this order):
  1. `unfinished` if any raw per-test parquet under `<prefix>/<cell_stem>/` has `=unfinished` in its filename. The downloader (`download/download_from_specimen.py` → `download_single_tests`) writes per-test files as `…=filesize-XXX=<status>.parquet` where `<status>` is `finished`/`unfinished` from Ahjo's `test.finished`. BRONZE_CU (the concatenated per-cell parquet) cannot itself carry the suffix, so the monitor inspects the per-test folder directly. The unfinished tag is authoritative and wins over the time heuristic.
  2. else `running` if last GOLD row's Time is within `running_window_days` (default 2).
  3. else `finished`.

  Unfinished / running / finished cells are rendered as three separate DataTables.
- **Coloring**: SOH `< 70%` row → yellow; SOH `< 60%` → red. Sort within each table is SOH ascending so the most-aged cell sits on top.
- SOH thresholds are constants at the top of `aging_status.py` (`YELLOW_THRESHOLD`, `RED_THRESHOLD`). The running window is read from the battery config key `running_window_days` (default `DEFAULT_RUNNING_WINDOW_DAYS = 2`).

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `V_max`, `V_min`, `V_nom` | Voltage limits and nominal voltage |
| `Nom_Capacity` | Nominal capacity in Ah |
| `CAP_Rate` | Capacity C-rate vs normalized `Current_mean` (÷ `Nom_Capacity`). `0.5` → C/2. |
| `cap_temp` | Target temperature(s) in °C for capacity segments. Scalar (`25`) or list (`[25, 35, 45]`); each value matches `Temperature_mean` within ±3 °C and the per-value masks are OR-combined. |
| `qOCV_CRate` | C-rate threshold for quasi-OCV (`0.05` → C/20, ~1200 min full discharge) |
| `tolerances.qocv_duration_tolerance` | Optional. Multiplier on the nominal qOCV duration (`60 / qOCV_CRate` min) for the upper bound in `find_qocv` (default `1.2`). A C/20 sweep runs longer than the nominal 20 h when a cell over-delivers vs `nom_capacity`, so the window needs headroom. |
| `pau_duration` | Pause threshold in minutes for procedure boundary detection (default 9.9) |
| `min_rows` | Minimum rows to keep a procedure segment (default 20) |
| `qocv_procedure_filter` | Optional. Substring matched against `Prozedur`; inside matching procedures every `Zustand` change also cuts a segment boundary, splitting a single-`Prozedur` qOCV into its `DCH` / `CHA` halves. Omit (default `None`) to disable — dismember then behaves unchanged. |
| `target_pulse_duration` | Expected pulse duration in seconds (default 20 s) |
| `export_pulse` | If true, write per-BM_Programm PUL parquet files to `20_export_pulse/` (default false) |
| `export_qocv` | If true, write per-BM_Programm qOCV_DCH / qOCV_CHA parquet files to `30_export_qocv/` (default false) |
| (always on) | `export_capacity` writes `<cell_stem>_capacity.csv` to `40_capacity_monitore/`, consumed by `monitor/aging_status.py` |
| `running_window_days` | Aging-status monitor: a cell is `running` if its last GOLD row's `Time` is within this many days of now, else `finished` (default 2) |

`hdbscan_para_layer_1["min_cluster_size"]` defaults to `max(2, n_programs − 1)` at runtime. If `min_cluster_size` is explicitly set in the config JSON, that value takes precedence (config key is merged last via `{defaults, **cfg["hdbscan_para_layer_1"]}`).
`hdbscan_para_layer_1["cluster_selection_epsilon"]` must be **0.3** (not 3.0) for correct qOCV separation.

## qOCV detection note

qOCV procedures come in discharge+charge pairs per aging cycle. Their signed `Current_mean` ≈ 0 (cancels), making them indistinguishable from rest segments if only signed current is used. `abs_Current_mean` is added to Layer 1 features to break this degeneracy. Layer 1 `cluster_selection_epsilon = 0.3` (tight) is required — 3.0 merges qOCV with rests.

**Type coercion fix** (`cluster/model_and_supervise.py` `merge_target`): previously used `fillna` to merge Layer 1 integer cluster labels with Layer 2 string labels (`"cap_layer_N"`). NaNs in the string column forced int64→float64 coercion, so labels became `1.0`, `2.0` etc., causing `isin([np.int32(N)])` in `concat_clusters` to match 0 rows and produce no `QOCV*` labels. Fixed by using `.where(notna, target_x.astype(object))` to preserve integer types.

**Pre-labeled target preservation fix** (`main.py` `_run_clustering`): both branches of the layer-1/layer-2 split previously rebuilt `df_clustered` via `df.drop(columns=["target"]).merge(X_clustered[...], how="left")`. This erased pre-labeled targets ("PAU", "EIS") for rows not present in `X_clustered` (which only contains clustered IDs), leaving them as NaN in GOLD. Fixed by replacing both merges with `merge_target(df, X_clustered)`, which falls back to the original target when no cluster result exists for an ID.

## Restore pulse structure

After each test pulse a restore pulse returns the cell to its original SoC. All restore pulses run at C/2. The C/2 restores (20 s) are filtered in `update_pulse` by `_filter_restore_pulses`: within each BM_Programm, a PUL* segment is a restore if **all three** conditions hold:
1. Its proc_num is exactly 1 more than the preceding PUL* segment (adjacent in the procedure sequence).
2. Same |current| (±5 %) as the preceding PUL* segment.
3. Opposite sign.

The proc_num gap check (condition 1) is critical: if the true 1C restore (C/2 current, ~40 s) is not labeled PUL* by HDBSCAN, consecutive test pulses of opposite sign would otherwise be wrongly flagged as restores. A gap > 1 between consecutive PUL* segments means a non-PUL* segment sits between them, so the pair are two tests, not a test+restore. 1C restores (~40 s at C/2) that do reach `fetch_pulse` are rejected by the duration check there.

Restore pulses are **not dropped** — they are labelled `PUL*RES` in both the GOLD parquet and the `with_features_post_labeled` CSV. Test pulses proceed to `fetch_pulse` and are labelled `PUL` after passing the duration check (no numeric resistance is computed in GOLD).

## Configuration

- **Battery parameters**: JSON config file passed to `main.py` (e.g. `battery_config_VTC_linux.json` in the data directory).
- **MinIO/Ahjo credentials**: `config.json` at project root (gitignored). Copy structure from `config_SE_example.json`. Also available via env vars: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `INFLUX_TOKEN`.
- **Data path**: set `working_path` in the config JSON. BRONZE_CU parquet files must be under `<working_path>/BRONZE_CU/`.

## CLI flags

| Flag | Description |
|------|-------------|
| `--cells` | Process a subset of cells by name fragment |
| `--overwrite` | Reprocess cells even if GOLD already exists |

## Post-labeling target sync

After all three `df_gold.update(...)` calls in `_process_cell`, final targets are propagated back to `with_features_post_labeled/<cell>.csv`:

```python
target_map = df_gold.groupby("ID")["target"].first()
X_silver["target"] = X_silver["ID"].map(target_map).fillna(X_silver["target"])
X_silver.to_csv(paths["X_silver"], index=False)
```

This overwrites the intermediate clustering labels (CAP\*, PUL\*, QOCV\*) with the final calculated targets (CAP, PUL, PUL\*RES, qOCV\_DCH, qOCV\_CHA, −1). Numeric HDBSCAN labels not matched to any test type are left as-is (to be set to −1 in a future cleanup step).

## Field-data track (`src/field/`)

Parallel pipeline for **EV field data** (driving + charging + parking telemetry from real vehicles), separate from the cycler-based CU pipeline. Lives in `src/field/` next to `dismember/`, `cluster/`, etc.

Reference dataset: **RWTH Aachen "Electric Vehicle and Battery Data"** (DOI 10.18154/RWTH-2024-01907, CC BY 4.0), 9 vehicles (1× iMiEV, 2× iOn, 6× Smart), 2014–2016 geriatric-care fleet. Unzipped at `<working_data>/field_data/rwth_aachen/`. Four tracks: `field_test/<vehicle>.parquet` (raw real-world time series), `capacity_test/<vehicle>_capacity_tests.parquet` (periodic dyno SOH refs — ground truth), `charging_curves/`, `trip_data/`.

The TUM FTM UDS dataset (https://github.com/TUMFTM/electric-vehicle-uds-dataset) was the first pick but its GitHub LFS budget is exhausted — only the 286 session JSONs came through, the per-vehicle parquet files are unreachable. Kept at `<working_data>/field_data/tum_uds/` for later.

**Stage F1 — `field/io_rwth.py`** (adapter, smoke-testable via `python -m field.io_rwth [base_dir]`):
- `load_field_test(path)` → DataFrame with canonical schema `Time / Voltage / Current / Temperature / SOC / Speed / Odometer / Power` (extras preserved). Renames `Temp_Ambient→Temperature`, `SoC_Real→SOC`. Converts `time_num` (MATLAB datenum, days since 0000-01-01) to UTC datetime via `pd.to_datetime(time_num - 719529, unit='D', utc=True)`. Drops leading rows where Voltage/Current/Temperature are all NaN (BMS signals lag SOC at log start).
- `load_capacity_test(path)` → same canonical columns plus `test_number`, `test_direction` (1=charge, 2=discharge), `Power_AC`.
- `list_field_test_vehicles(base_dir)`, `field_test_path()`, `capacity_test_path()` — directory helpers.

Data quirks surfaced by the smoke test: Smart-5 has zero `Power_AC` (no AC charging logged); iMiEV-1's field_test has ~50% non-null Temperature/SOC (sparser logger). Both vehicles still produce usable Voltage/Current/SOC streams.

**Stage F2 — `field/segment.py`** (DRIVE/CHARGE/REST segmenter, smoke-testable via `python -m field.segment [base_dir] [--vehicle V] [--trips ...]`):

Key insight from the RWTH data — the BMS logger sleeps when the car is parked with key off, so REST shows up as the *absence* of rows, not as low-current samples. The active sample coverage of wall-clock time is only ~2–3 %.

Segmentation model:
- **REST** = any inter-row Δt > `gap_threshold_s` (default 300 s) becomes a synthetic REST segment spanning `(Time[i], Time[i+1])` with `n_rows = 0`.
- **CHARGE** = a maximal run of consecutive rows with `Current > i_charge_threshold` (default 0.5 A) whose wall-clock duration is at least `min_charge_duration_s` (default 60 s). The duration requirement intentionally excludes brief regen-braking bursts during driving (median positive-current run is ~2 s on the Smart fleet).
- **DRIVE** = everything else inside an active session.

Sign convention (RWTH): positive Current = charging (SOC rises). Verified at runtime by `check_sign_convention` against dSOC/dt — using Speed is unreliable because regen produces positive current with nonzero Speed.

`to_segments(df_with_state, *, gap_threshold_s=300)` collapses the per-row state into one row per segment (columns: `segment_id, state, start_time, end_time, duration_s, n_rows, Current_mean, Voltage_mean, SOC_start, SOC_end, Speed_max, distance_km`).

`validate_against_trips(segments, trips, vehicle)` matches DRIVE segments to ground-truth `trip_data/*_datafile.parquet` rows by time overlap. Fleet-wide validation (vehicles present in GeriatricCare trip file):

| Vehicle | trips | DRIVE segs | recall | precision | median \|ΔSOC error\| |
|---------|-------|-----------|--------|-----------|----------------------|
| Smart-1 | 5 708 | 5 101 | 99.5 % | 99.0 % | 0.1 % |
| Smart-3 | 2 192 | 2 119 | 99.3 % | 98.8 % | 0.1 % |
| iMiEV-1 | 1 508 | 3 022 | 99.9 % | 45.5 % | 0.5 % |

iMiEV-1's low precision is a known over-segmentation case — its logger has intra-trip gaps > 300 s, so the gap rule splits single trips into multiple DRIVE segments. Each trip still gets recovered (recall is 99.9 %).

Charging is rarely visible in `field_test/*` because the logger is usually off while the car charges. Of 5 095 long gaps on Smart-1, only 6 show SOC rise > 5 % with valid SOC on both sides; most real charging activity lives in `capacity_test/*` and in cross-gap SOC jumps that F3 will detect later.

**Stage F1 (shiyunliu) — `field/io_shiyunliu.py`** (adapter for the on-road EV charging dataset, smoke-testable via `python -m field.io_shiyunliu [base_dir]`):

Scope pivot: after concluding that the RWTH dataset is pre-segmented by activity type (driving in `field_test/`, charging shapes as templates in `charging_curves/`, capacity in dyno `capacity_test/`), the field-data track refocused on the **shiyunliu on-road EV charging dataset** (20 production EVs, ~29 months each, MIT licence, accompanies Deng et al. Applied Energy 339:120954). Repo: `shiyunliu-battery/battery-charging-data-of-on-road-electric-vehicles`. Unzipped at `<working_data>/field_data/shiyunliu_20ev/` as `#1.csv`..`#20.csv` (~1.4 GB total, ~60 MB per vehicle, ~800 k rows per vehicle).

The data is **charging-only** — sessions are detected by 10-s time gaps and the timeline between sessions (driving, parking) is not recorded. This is a deliberate match for the refined goal: identify "capacity tests" = opportunistic full CC-CV charge events, using a modified main.py + HDBSCAN architecture.

`load_vehicle(path)` returns the canonical schema `Time / Voltage / Current / Temperature / SOC / Cell_V_max / Cell_V_min / Cell_T_min / Available_Energy_kWh / Available_Capacity_Ah` (extras preserved):
- Decodes `record_time` (integer `YYYYMMDDhhmmss`) to tz-aware UTC datetime.
- Strips unit suffixes from CSV column names (e.g. `pack_voltage (V)` → `pack_voltage`).
- **Negates `charge_current`** so positive Current means charging — the raw shiyunliu convention is `charge_current < 0` during charging, opposite of the field-track convention used elsewhere (e.g. RWTH).
- `check_sign_convention` re-verifies via dSOC/dt against the post-negation series; passes silently on all 20 vehicles.

Smoke-tested: all 20 vehicles load cleanly, 100 % non-null on every canonical column, spans align around 842–847 days each. Two outliers flagged: **vehicles #6 and #17** show post-negation max Current = 400 A vs. ~90–95 A on the rest — likely positive-current rows in the raw CSV (sensor glitches or discharge logging slipping in). Treat with care before HDBSCAN clustering.

Charging-session segmentation (per the dataset's own `capacity_extract.py`): `dt > 10 s` between consecutive rows. Vehicle #1 has 4 223 sessions over 843 days, with **197 sessions of ΔSOC > 70 %** (strong CAP candidates).

**Later stages (planned, not yet built):**
- F2 (shiyunliu) — session segmentation + per-session feature extraction (`duration_s, dSOC, I_mean, I_std, has_CV_tail, T_mean, SOC_start, SOC_end`).
- F3 — HDBSCAN cluster on the per-session feature matrix → pick the "full CC-CV" cluster as the CAP equivalent.
- F4 — coulomb-count CAP-cluster sessions → SOH timeline per vehicle; emit `40_capacity_monitore`-shaped CSV so the existing aging-status monitor and aging matrix run unchanged.
- F5 — benchmark our extracted capacities against the dataset author's published Fig1.png values.

**Legacy code from earlier scopes** (kept for reference): `io_rwth.py` (RWTH Aachen adapter) and `segment.py` (rule-based DRIVE/CHARGE/REST segmenter that was the F2 of an earlier plan). Both remain useful as sibling adapters / reference implementations; neither is on the path of the current shiyunliu work.

## Documentation

- `METAbatt_Pipeline_Report.md` — full technical report of the pipeline
- `METAbatt_Pipeline_Flowchart.svg` — visual flowchart of all pipeline stages
