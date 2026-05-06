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

**Venv (Linux)**: `source venv/bin/activate` from project root.

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

**Pipeline stages and their modules:**

1. **`dismember/dismember_raw_cell.py`** — reads BRONZE_CU parquet, renames German columns (`Spannung→Voltage`, `Strom→Current`, `Zeit→Time`, `T1→Temperature`), segments into discrete procedures. Groups by `Ahjo_Test_ID` → `BM_Programm`, splits by `Prozedur` changes and PAU pauses > `pau_duration` minutes. Drops segments with < `min_rows` rows. Assigns string ID: `<BM_Programm>_<procedure_number>` (e.g. `13_16`). Core logic: `dismember/cluster_preparation.py` (`DismembererFunctions`, `allocate_IDs`).
   - **PAU stubs**: long PAU/PAUO segments (> `pau_duration`) are kept as 2-row stubs (first + last row) with their own ID and `target="PAU"`. `Duration_minutes` is the actual pause length (last − first timestamp). Short pauses (≤ `pau_duration`) and middle rows of long pauses are assigned `BM_Programm_procedure=0` (discard bucket). PAU stubs are exempt from the `min_rows` check. They are excluded from feature extraction and clustering (filtered out by the `target == -1` guard in `create_features.py`) but flow through to SILVER and GOLD, making the relaxed-cell voltage and pause duration available for pulse resistance calculations.

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

Restore pulses are **not dropped** — they are labelled `PUL*RES` in both the GOLD parquet and the `with_features_post_labeled` CSV. Test pulses proceed to `fetch_pulse` and are labelled `PUL` after calculation.

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

## Documentation

- `METAbatt_Pipeline_Report.md` — full technical report of the pipeline
- `METAbatt_Pipeline_Flowchart.svg` — visual flowchart of all pipeline stages
