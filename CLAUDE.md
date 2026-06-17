# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always tell me what the plan is before changing code.
Working data path is at /home/ann/Documents/Data_Metabatt (data only — do **not** mirror code/doc edits there).

## Git workflow

Before any code change, branch off: `feat/…` / `fix/…` / `refactor/…` / `config/…`. Push and open a PR into `J8005_BMWK_METABatt` (CodeRabbit auto-reviews). **Never commit directly to `J8005_BMWK_METABatt`.**

## Running the pipeline

All scripts/notebooks run from `src/` (imports are relative to `src/`). Venv: `source .venv/bin/activate` from project root.

**Main CLI**: `src/main.py`

```bash
cd src
python main.py /path/to/battery_config_VTC_linux.json
python main.py /path/to/battery_config_VTC_linux.json --cells VTC_cell01 VTC_cell02
```

The legacy notebook `src/Process_Detection_via_Cluster_py_METABATT.ipynb` still works; `main.py` is the current entry point.

**Unified UI**: `src/pipeline_ui.py` — customtkinter app wrapping six stages (Download / Build BRONZE_CU / Run Pipeline / Monitor / Evaluation / Train Classifier) with a shared config picker, per-tab Run buttons, a "Run all 1→2→3→4→5" chain, Stop, and a live console. Persists paths to `~/.config/metabatt_ui.json`. Tab 6 (Train Classifier) is intentionally **outside** the chain — an offline build step. Prereq on Linux: `sudo apt install python3-tk`. Each Run button spawns the matching CLI:
- Tab 1 → `download/run_download.py <download_cfg.json>` (headless wrapper of `download/download_GUI.py`)
- Tab 2 → `download/build_bronze_cu_with_ah.py <battery_cfg> [--cells …] [--overwrite]`. Honors `download_from`/`upload_to` (same as `main.py`); `local` reads per-test parquets from `<working_path>/<cell>/*.parquet`, `minio` (default) from `<minio_prefix>/<cell>/`. Legacy `save_local`/`upload_s3` accepted when `upload_to` absent. Skips if the target BRONZE_CU already exists. CU detection (`_is_cu`): per-test file is a check-up when its 4th `=`-delimited filename field contains `procedure_filter`, which is **required** (`process_cell` raises `ValueError` if unset).
- Tab 3 → `main.py <battery_cfg> [--cells …] [--overwrite] [--clustering …]`. A **Clustering** toggle (Auto / HDBSCAN / Classifier) maps to `--clustering`, so HDBSCAN can be forced even when the config sets `classifier_model_path`. A separate **Interpret clusters (LLM)** button runs `cluster.interpret_clusters` (label-only) on the existing CSVs.
- Tab 4 → `python -m monitor.aging_status <battery_cfg> [-o …]`
- Tab 5 → a checklist of evaluation outputs run in sequence by one "Run evaluation" button: **Fleet-wide capacity aggregation** → `python -m evaluation.export_cap_pulse <battery_cfg>`; **Capacity evaluation (Alterungsmatrix)** → `python -m evaluation.aging_matrix <battery_cfg>`. Pulse / qOCV evaluations are placeholder checkboxes (disabled) for future stages. The "Run all 1→2→3→4→5" chain runs stages 1–4 then every ticked Tab 5 evaluation.
- Tab 6 → `python -m cluster.train_classifier <battery_cfg> [--model-out …] [--meta-out …] [--labels …]`. A **Labels** toggle (Config / target / llm) maps to `--labels`; `Config` omits it so the config's `classifier_label_source` (default `target`) decides.

The Download tab's "Save JSON" matches `download/get_user_input.py`; full-pipeline runs auto-write it to `.metabatt_ui_download.json` (gitignored).

## Architecture

Medallion architecture, all parquet (time-series):

```
BRONZE_CU → preSILVER → SILVER → GOLD
```

- **BRONZE_CU**: raw check-up cycler export (CU = check-up), German columns, unsegmented.
- **preSILVER**: segmented into procedures; long PAU pauses (> `pau_duration`) reduced to first+last row stubs (target="PAU"), short pauses/segments dropped, columns renamed to English. On disk for debugging.
- **SILVER**: preSILVER with cluster labels merged back. On disk for debugging; held in memory as `df_silver` in `main.py`.
- **GOLD**: SILVER + calculated metrics (capacity, pulse resistance, qOCV).

A per-segment CSV helper layer (one row per segment) supports clustering: an **in-memory** feature table → [HDBSCAN / classifier] → `with_features_post_labeled` (persists every segment + `cluster_id`; no separate pre-labeled CSV). Labels merge back into the time-series to make SILVER.

**Procedure-filter gate**: `_process_cell` peeks at BRONZE `Prozedur` before pulling the payload. If `procedure_filter` is set and nothing matches, the cell is skipped (`INFO {cell}: no Prozedur matches filter …`) — dismember never runs and MinIO never calls `fetch_bronze`. Uses `processing_procedure_filter` (pyarrow, reads only `Prozedur` row-group by row-group, short-circuits); on MinIO via `io_router.open_bronze_range` (HTTP-range read of footer + one column).

**Cells without a proper checkup**: if clustering finds no CAP cluster (`ClusterNotFoundException` from `post_cluster_filter.find_capacity`), `_process_cell_inner` catches it, logs one warning (`no proper checkup detected … — skipping GOLD`), and continues. Not counted as a failure.

**Pipeline stages:**

1. **`dismember/dismember_raw_cell.py`** — renames German columns (`Spannung→Voltage`, `Strom→Current`, `Zeit→Time`, `T1→Temperature`), segments into procedures. Groups by `Ahjo_Test_ID` → `BM_Programm`, splits on `Prozedur` changes and PAU pauses > `pau_duration`. Drops segments < `min_rows`. ID = `<BM_Programm>_<procedure_number>` (e.g. `13_16`). Core: `dismember/cluster_preparation.py`.
   - **qOCV Zustand split** (optional, when `qocv_procedure_filter` set): inside matching procedures, every `Zustand` change also cuts a boundary — splits a single-`Prozedur` qOCV's DCH/CHA halves (otherwise their signed `Current_mean` cancels to ≈0 and looks like a rest). Inert/gated per-row when the key is absent.
   - **PAU stubs**: long PAU/PAUO (> `pau_duration`) kept as 2-row stubs (first+last) with `target="PAU"`; `Duration_minutes` = actual pause length. Short/middle pause rows → discard bucket (`BM_Programm_procedure=0`). Stubs are exempt from `min_rows`, excluded from features/clustering (the `target == -1` guard in `create_features.py`), but flow to SILVER/GOLD so relaxed-cell voltage and pause duration are available for pulse resistance.

2. **`feature_extraction/create_features.py`** + **`classification.py`** — per-segment stats (mean/std/min/max of Voltage/Current/Temperature). Normalize Voltage by `(V_max - V_min)`, Current/Power by `Nom_Capacity`. Adds `Duration_quartile = log1p(Duration_minutes)`, `abs_Current_mean = |Current_mean|`, and `prev_end_voltage_norm` (see classifier section). Returns the table in memory.

3. **`cluster/model_and_supervise.py`** — two-layer HDBSCAN:
   - Layer 1: clusters on `["Duration_quartile", "abs_Current_mean", "ID"]`; `min_cluster_size = max(2, n_programs − 1)`.
   - Layer 2 (only if Layer 1 finds no capacity cluster): re-clusters candidates on `["Current_mean", "ID"]` with stricter masks.
   - `cluster/post_cluster_filter.py` (`cluster_filter`) — rule-based masks → CAP* / PUL* / QOCV* / −1.

4. **`calculate/results_fetching.py`** — `calculation` class (labeling + capacity only; numeric pulse/qOCV results are recomputed downstream from the exports):
   - `update_capacity()` — trapezoidal Ah → `Capacity_py`, refines `CAP*` → `CAP`.
   - `update_pulse()` — labels only: `_filter_pulse_group()` tags restores `PUL*RES`, then `fetch_pulse` labels remaining `PUL*` → `PUL` (passes duration check) or `-1`.
   - `update_qOCV()` — labels `QOCV*` → `qOCV_DCH`/`qOCV_CHA` by sign.

5. **`output/`** → InfluxDB. **`util/connect_minio.py`** → parquet to MinIO.

6. **`output/export_pulse.py`, `export_qocv.py`** — optional per-`BM_Programm` PUL / qOCV exports from GOLD (flags `export_pulse`, `export_qocv`, default off). Per program, capacity comes from that program's CAP segment; `SOH = round(Capacity_py / nom_capacity * 100, 1)` (or `NA` + warning if no CAP). Pulse export also bundles adjacent PAU stubs (proc_num ±1) for before/after relaxation voltage. Files:
   - `20_export_pulse/<cell_stem>/<cell_stem>_pulse_BM<BM_Programm>_<SOH>SOH.parquet`
   - `30_export_qocv/<cell_stem>/<cell_stem>_qocv_{dch,cha}_BM<BM_Programm>_<SOH>SOH.parquet`

   Routing follows `download_from`/`upload_to`. Export MinIO keys are **untagged** (no `10_TRACY`).

7. **`output/export_capacity.py`** — always runs at end of `_process_cell` (no flag). Per-cell capacity summary CSV (one row per BM_Programm) for the aging monitor/matrix. Columns: `BM_Programm, Capacity_py, Ah_throughput, SOH, CAP_start_time` (`Ah_throughput` = cumulative throughput at the CAP-segment start). Flat under the folder:
   - Local: `<working_path>/40_capacity_monitore/<cell_stem>_capacity.csv`
   - MinIO: `<minio_prefix>/40_capacity_monitore/<cell_stem>_capacity.csv` (untagged)

## Learned segment classifier (optional, replaces HDBSCAN)

`cluster/train_classifier.py` + `predict_classifier.py` — a RandomForest drop-in for HDBSCAN + `post_cluster_filter`. **Opt-in**: runs only when `classifier_model_path` is set; otherwise the unchanged HDBSCAN path. Motivation: the rule thresholds (`CAP_Rate ± 5 %`, `cap_temp ± 3 °C`, qOCV window) are per-chemistry and don't transfer to field data.

- **Feature basis** (`FEATURE_COLS`, 12 cols) — scale-/chemistry-portable: current profile (`Current_{mean,std,max,min,range}`, `abs_Current_mean`, normalized ÷ `Nom_Capacity`), voltage **edges** (`Voltage_{max,min,range}`, ÷ `(V_max − V_min)`; `Voltage_range` is the SoC-swing proxy), `Duration_{minutes,quartile}`, and `prev_end_voltage_norm`. Voltage *curve shape* (`Voltage_mean/std`) and `Temperature_mean` are **excluded on purpose** (sensor-/chemistry-bound, no accuracy cost). Contract lives in `_meta.json["feature_columns"]`; `predict_classifier` reads it from there. `create_features` still computes the dropped cols (inert).
- **Context feature** `prev_end_voltage_norm` (`create_features.py`): end-of-segment voltage of the most recent **non-PAU** predecessor (walk back ≤4 steps, skip PAU), normalized as `(V - V_min) / (V_max - V_min)` (same scale as `Voltage_*` features: 0 = bottom rail, 1 = top rail); `-1.0` if no predecessor exists. Mirrors `post_cluster_filter.previous_voltage` — the signal the CAP rule keys on (a true CAP discharge follows a full charge). Inert for HDBSCAN (3-col subset). **CSVs predating this column must be regenerated before training** — `train_classifier` aborts if it's absent.
- **Training** (`python -m cluster.train_classifier <battery_cfg>`): reads every `with_features_post_labeled/*.csv` (post-#37 these carry all segments + `cluster_id`, supplying negatives), weak-labels leftovers via `bootstrap_leftover_labels`, runs leave-one-cell-out CV, then refits on all cells. **Output**: pipeline artifact → `<working_path>/60_classifier/models/<type_cell>_classifier_<UTC-timestamp>.joblib` + `_meta.json` (falls back to `../models/` only if `working_path` unset). Timestamped, so nothing is overwritten — **prune manually**, and `classifier_model_path` must point at a specific file. `--model-out`/`--meta-out` override the stem but a `_<timestamp>` suffix is always appended. Uploaded **untagged** to `<minio_prefix>/60_classifier/models/` when `upload_to` includes minio. **Source routing** (`_load_cell_csvs`): `local` globs `<working_path>/with_features_post_labeled/*.csv`; `minio` lists/fetches the tagged `<prefix>/10_TRACY/with_features_post_labeled/`. One config = one source.
- **Bootstrap labels** (`bootstrap_leftover_labels`): names the stringified raw leftover clusters by scale-free signature, **without touching** final labels (`CAP`/`PUL`/`PUL*RES`/`qOCV_*`/`-1`). Runs in two places: (a) `main.py` HDBSCAN path — after target sync, applied to `X_silver` and mapped into `df_gold["target"]` so the CSV **and** GOLD carry `PREP_CHA`/`SOC_ADJUST`/`-1` (raw integer cluster kept in `cluster_id`); changes no numeric result. (b) `train_classifier._load_cell_csvs` — idempotent re-derivation for legacy CSVs. Rules:
  - `PREP_CHA` — charge (`Current_mean > 0`) at ≥ `min_prep_current` (default 0.1, ~C/10) ending near full (`Voltage_max > 0.95`).
  - `SOC_ADJUST` — partial charge/discharge at ≥ `min_prep_current` ending at intermediate SoC (`Voltage_max ≤ 0.95` charge, or `Voltage_min > 0.05` discharge). `PREP_CHA` wins ties.
  - Everything else leftover stays `-1` (OTHER). **No `CYCLE`** (BRONZE_CU has no cycling) and **no `PREP_DCH`** — CAP-vs-discharge is carried by `prev_end_voltage_norm`.
  - **`PUL*RES → PUL` for training**: the classifier needn't tell restores apart (the split is recomputed downstream). Merging lifted LOCO 0.952 → 0.985.
- **Inference** (`predict_classifier.predict_targets`, gated in `_process_cell_inner`): maps labels to the tagged form `calculate/` expects (`CAP→CAP*`, `PUL→PUL*`, `qOCV_{DCH,CHA}→QOCV*`); `update_pulse` still runs and re-splits `PUL*` into `PUL`/`PUL*RES`. `PREP_CHA`/`SOC_ADJUST` pass through into GOLD `target` (informational; no numeric effect). NaN-feature rows → `-1`.
  - **CSV routed to `60_classifier/`** (gated on `classifier_model_path`): written to `<working_path>/60_classifier/with_features_post_labeled/<stem>.csv` (local) / `<minio_prefix>/60_classifier/with_features_post_labeled/` (MinIO, untagged) — side by side with the HDBSCAN CSVs, kept out of the training set. **The classifier route writes only this CSV**: after the second `_write_x_silver`, `_process_cell_inner` returns early — `_write_gold`, `export_capacity`, and exports are all skipped. So `--overwrite` is safe on a classifier run. (Local skip-check still keys on the HDBSCAN GOLD path, so use `--overwrite` to produce the CSV when GOLD already exists.)
  - **`cluster_id` on the classifier path**: holds the **raw predicted label** (`CAP`/`PUL`/`qOCV_*`/`PREP_CHA`/`SOC_ADJUST`/`-1`), captured before the tagged-form map. **Caveat**: it's the classifier's own string label (provenance/audit), **not** a raw HDBSCAN cluster — building new training data still requires an HDBSCAN re-run (`classifier_model_path` unset). Type differs by path: integer (HDBSCAN) vs string (classifier); treat as opaque.

## LLM cluster interpretation (advisory, augment-only)

`cluster/interpret_clusters.py` (`python -m cluster.interpret_clusters <cfg> [--source local|minio] [--cells …] [--overwrite] [--dry-run] [-o …]`) — names HDBSCAN clusters with an LLM so leftover (`-1`) segments get descriptive names and the rule labels get an independent audit. Groups `with_features_post_labeled` rows by `(cell, cluster_id)`, builds a per-cluster signature (mean/min/max of the scale-free classifier features + member count + majority `target` + rule-bootstrap label), dedupes identical rounded signatures across cells (one API call each). The LLM invents its **own free-form snake_case label** (e.g. `full_discharge_c2`, `full_discharge_1c`, `full_charge_c20`, `mixed_*` for sign-mixed clusters) — deliberately *not* constrained to the pipeline taxonomy, so `llm_label` is a second opinion next to `target`, not a forced mapping. The prompt has **no `qocv` label** (nor a capacity-test label): a full charge/discharge at any C-rate — including a very low-rate quasi-OCV sweep — is named `full_charge`/`full_discharge` with its crate, and quasi-OCV-vs-capacity is decided downstream from the measured rate (see below), mirroring how CAP is handled. **Augment-only**: results land in `llm_label` / `llm_confidence` / `llm_rationale` columns written back to the same CSV (local in place / same MinIO key); `target`, `cluster_id`, and all numerics untouched. Audit summary → `<working_path>/50_evaluation/cluster_interpretation.csv` (one row per cluster: n, majority `target`, llm label/confidence/rationale; MinIO untagged); an all-skip run leaves the audit untouched. Cells already carrying `llm_label` are skipped unless `--overwrite`; CSVs predating the `cluster_id` column (pre-#37) are skipped with a regenerate hint — they need an HDBSCAN re-run (`classifier_model_path` unset).

- **Integrated label-only route** (`main.py --interpret`, or config `llm_interpret: true`): folds the same interpretation into the pipeline. After HDBSCAN clusters a cell, it adds `llm_*` to the in-memory `X_silver`, writes the `with_features_post_labeled` CSV, and **returns early — no GOLD, no calc, no capacity/pulse/qOCV exports** (the CSV is the sole output, like the classifier route). `--interpret` forces the HDBSCAN path (`classifier_model_path` ignored), since interpretation names HDBSCAN clusters. Dedup is per-cell here (one cache per cell), vs cross-cell in the standalone CLI. Use the standalone `cluster.interpret_clusters` to label CSVs that already exist; use `--interpret` to cluster-and-label in one pass.
- **Provider seam** (`util/llm_client.py`): `LLMClient` protocol (`interpret_cluster(signature) -> ClusterLabel{label, confidence, rationale}` — Pydantic-validated). Backend via config key `llm_provider` (default `openai`). `OpenAILLMClient`: official `openai` SDK, `responses.parse()` structured output, `reasoning effort: medium`; model via `llm_model` (default `gpt-5.4`). `AnthropicLLMClient`: official `anthropic` SDK, `messages.parse()` structured output, adaptive thinking + `effort: medium`, prompt caching (`cache_control: ephemeral`) on the stable system prompt; `llm_model` default `claude-opus-4-8`. Credentials: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var, fallback `openai_api_key` / `anthropic_api_key` in root `config.json` (gitignored).
- **LLM labels as training targets** (`train_classifier --labels llm`, or config `classifier_label_source: "llm"`; default `"target"`): trains on the `llm_label` column (from `interpret_clusters`), but **canonicalizes it first** (`_canonicalize_llm_labels`) into the resolved space the pipeline acts on — the LLM's raw `label` field is unreliable for training (it sometimes contradicts its own `true_voltage_range ≥ 0.9` rationale on the full-vs-partial cut, and fragments C-rates/pulses into unlearnable singletons: `pulse` vs `pulse_c1`, `full_charge_c20` vs `full_discharge_c20` vs `mixed_*_c20`). Each segment is **rebuilt deterministically from its own features** (sign, `true_voltage_range`, measured `abs_Current_mean`): any `*pulse*`→`pulse`; a full sweep (`true_voltage_range ≥ 0.9`) at/below the qOCV band (either sign)→`qocv`; a full **discharge** in the cap band→`cap`; any other full sweep→`full_charge_<crate>`/`full_discharge_<crate>` (informational, e.g. the C/2 prep charge); a partial→`partial_cha_<crate>`/`partial_dch_<crate>` (kept, with direction+crate); `unknown`/`rest`/`artifact` and sign-ambiguous rows pass through. `mixed_*` clusters **dissolve** because each member is resolved from its own sign. The cap/qocv bands use the **same** `cap_rate`/`qocv_crate` + tolerances (`cap_tol 0.05`, `qocv_tol 0.2`) as inference, so a training label and the inference tag agree; this folds the worst LOCO confusions (within-pulse, within-qocv) into single classes. Segments with no `llm_label` are dropped; no PUL*RES-merge / bootstrap. `_meta.json["label_source"]` records the space. At inference `predict_classifier._map_llm_label_to_tagged` maps the prediction back: `*pulse*→PUL*`, `*qocv*→QOCV*`, bare `cap→CAP*`, and a residual `full_charge`/`full_discharge` is resolved by its **measured** `abs_Current_mean`: `→QOCV*` when ≤ `qocv_crate × (1 + qocv_tol)` (charge or discharge), else `→CAP*` when a `full_discharge` within `cap_rate ± 5 %`. `qocv_crate ≪ cap_rate` so the bands never overlap; qocv is checked first. Every other label passes through into GOLD `target` informationally. `predict_targets` takes `cap_rate` and `qocv_rate` (passed from `main.py` as `cfg["cap_rate"]` / `cfg["qocv_crate"]`); with a rate unset, the corresponding tag is never produced.

## Evaluation

**Fleet-wide capacity** — `evaluation/export_cap_pulse.py` (`python -m evaluation.export_cap_pulse <cfg> [-o …]`): aggregates `40_capacity_monitore/*_capacity.csv` + the `Prozedur` column from each GOLD parquet into one table. Aging metadata (`DOD/SOC/C_Rate/Temperature`) parsed from procedure names via `output.add_information_METABATT`. Output (driven by `upload_to`): `<working_path>/50_evaluation/capacity_results.csv` (full CAP history), MinIO `<prefix>/50_evaluation/capacity_results.csv` (untagged).

**Aging matrix** — `evaluation/aging_matrix.py` (`python -m evaluation.aging_matrix <cfg> [-o …]`): per-cell capacity loss normalized by Ah throughput, aggregated over the design space. Reuses `export_cap_pulse.build_capacity_table`; needs `Ah_throughput` (aborts otherwise). Per cell: `capacity_lost` = max−min `Capacity_py`, `Delta_Ah_throughput` = max−min `Ah_throughput`. Matrix: group by `(C_Rate, Temperature, DOD, SOC)` → mean/std, `candidate_count`, cell list, `capacity_lost_norm`. Output: `<working_path>/50_evaluation/aging_matrix.{csv,html}` (interactive plotly), MinIO untagged.

**HDBSCAN vs classifier diff** — `evaluation/compare_labels.py` (`python -m evaluation.compare_labels <cfg> [--source local|minio] [--hdbscan-dir] [--classifier-dir] [-o]`): diffs the two per-segment label sets per cell (joined on `ID`) to see whether the classifier recovers CAP check-ups HDBSCAN missed (invisible to the LOCO report, which scores against HDBSCAN's own labels). Prereq: run the pipeline twice on the same cells (HDBSCAN, then classifier with `--overwrite`). `local` reads `<working_path>/with_features_post_labeled/` (HDBSCAN) + `…/60_classifier/with_features_post_labeled/` (classifier); `minio` reads tagged `10_TRACY/…` + untagged `60_classifier/…`. Reports cell coverage (both / classifier-only / HDBSCAN-only), per-`(cell, BM_Programm)` CAP-count delta, label transitions, and per-segment disagreements with explanatory features. Output (default `<working_path>/50_evaluation`): `label_diff_segments.csv`, `cap_count_diff.csv`, `cap_recovered_cells.csv` (always written locally).

## Aging-status monitor

`monitor/aging_status.py` (`python -m monitor.aging_status <cfg> [-o …]`) — sortable HTML report of per-cell SOH for spotting near-EOL cells mid-test.

- **Source** (`download_from`): `40_capacity_monitore/*_capacity.csv` for SOH/CU history, plus only the **last row group** of each **BRONZE_CU** parquet (`ParquetFile.read_row_group`) for the latest `Zeit`→`Time` and `Prozedur`. Reading BRONZE_CU (not GOLD) means a BRONZE_CU rebuild alone refreshes time/status; SOH refreshes only on a full pipeline run. MinIO uses `io_router.open_bronze_range` (range-read footer + last row group).
- **Output**: `<working_path>/40_capacity_monitore/aging_status.html`; MinIO `<prefix>/40_capacity_monitore/aging_status.html` (untagged).
- **Columns**: `cell · latest_SOH_% · dSOH_per_CU · n_CU · last_row_time · last_Prozedur · status`.
- **Status** (in order): `unfinished` if any per-test parquet under `<prefix>/<cell_stem>/` has `=unfinished` in its filename (authoritative — BRONZE_CU can't carry the suffix, so the per-test folder is inspected); else `running` if the last BRONZE_CU `Time` is within `running_window_days` (default 2); else `finished`. Rendered as three DataTables.
- **Coloring**: SOH `< 70%` → yellow, `< 60%` → red (`YELLOW_THRESHOLD`/`RED_THRESHOLD` constants). Sort SOH ascending.

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `V_max`, `V_min`, `V_nom` | Voltage limits and nominal voltage |
| `Nom_Capacity` | Nominal capacity in Ah |
| `CAP_Rate` | Capacity C-rate vs normalized `Current_mean` (÷ `Nom_Capacity`). `0.5` → C/2. |
| `cap_temp` | Target temperature(s) in °C for CAP segments. Scalar or list; each matches `Temperature_mean` within ±3 °C, OR-combined. |
| `qOCV_CRate` | C-rate threshold for quasi-OCV (`0.05` → C/20, ~1200 min) |
| `tolerances.qocv_duration_tolerance` | Multiplier on nominal qOCV duration (`60/qOCV_CRate` min) for `find_qocv`'s upper bound (default `1.2`) |
| `tolerances.qocv_std_tolerance` | Max current std for a qOCV in `fetch_qOCV`, as a C-rate (× `Nom_Capacity` → Amps; default `0.002`). Replaces the old absolute `0.001 A` cap that didn't scale with cell size. |
| `pau_duration` | Pause threshold (min) for procedure boundaries (default 9.9) |
| `min_rows` | Min rows to keep a segment (default 20) |
| `qocv_procedure_filter` | Optional substring; inside matching procedures every `Zustand` change cuts a boundary (splits qOCV DCH/CHA). Omit to disable. |
| `target_pulse_duration` | Expected pulse duration (s, default 20) |
| `export_pulse` / `export_qocv` | Write per-BM_Programm PUL / qOCV parquets (default false) |
| (always on) | `export_capacity` writes `<cell_stem>_capacity.csv` to `40_capacity_monitore/` |
| `running_window_days` | Monitor: `running` if last BRONZE_CU `Time` within N days (default 2) |
| `classifier_model_path` | Optional `.joblib` (relative to `src/` or absolute). When set, `_process_cell_inner` uses the classifier instead of HDBSCAN. Resolution (`_resolve_classifier_paths`): load locally if it exists; if absent **and** `download_from="minio"`, fetch `.joblib`+`_meta.json` by **basename** from `<minio_prefix>/60_classifier/models/` into a local cache. `local` + missing file → `FileNotFoundError`. Omit to keep HDBSCAN. |
| `classifier_meta_path` | Optional. Defaults to `<model stem>_meta.json`. |
| `classifier_label_source` | `target` (default) or `llm` — training-target space for `train_classifier` (CLI `--labels` overrides). `llm` trains on the free-form `llm_label` column; inference maps it back via measured C-rate. |
| `llm_interpret` | When true (or `main.py --interpret`), the HDBSCAN run is a label-only LLM pass: writes `llm_*` into the per-segment CSV and skips GOLD/exports. Default false. |

`hdbscan_para_layer_1["min_cluster_size"]` defaults to `max(2, n_programs − 1)`; an explicit config value wins (merged last). `cluster_selection_epsilon` must be **0.3** (not 3.0) for correct qOCV separation.

## Restore pulse structure

After each test pulse a C/2 restore returns the cell to its original SoC. `update_pulse._filter_restore_pulses` flags a `PUL*` as a restore when **all three** hold within a BM_Programm: (1) proc_num exactly 1 more than the preceding `PUL*` (adjacency), (2) same |current| (±5 %), (3) opposite sign. Condition 1 is critical — a gap > 1 means a non-`PUL*` sits between, so the pair are two tests, not test+restore. Restores are **not dropped** — labelled `PUL*RES` in GOLD and the CSV. Test pulses → `PUL` after the duration check (no numeric resistance in GOLD).

## Configuration

- **Battery params**: JSON config passed to `main.py` (e.g. `battery_config_VTC_linux.json`). See `battery_config_example.json` for the full schema (cell/voltage/CAP/pulse/qOCV, `tolerances`, HDBSCAN layers, `download_from`/`upload_to` + MinIO keys, optional `classifier_model_path`/`classifier_meta_path`).
- **MinIO/Ahjo credentials**: `config.json` at project root (gitignored; copy `config_example.json`). Also via env: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `INFLUX_TOKEN`.
- **Data path**: `working_path` in the config; BRONZE_CU parquets under `<working_path>/BRONZE_CU/`.

## CLI flags

| Flag | Description |
|------|-------------|
| `--cells` | Process a subset of cells by name fragment |
| `--overwrite` | Reprocess cells even if GOLD already exists |
| `--clustering {auto,hdbscan,classifier}` | Override the clustering path. `auto` (default) lets `classifier_model_path` decide; `hdbscan` forces HDBSCAN even if a classifier model is configured; `classifier` requires one. |
| `--interpret` | Label-only LLM route: force HDBSCAN, LLM-name each cluster into the per-segment CSV (`llm_*` columns), then stop — **no GOLD, no capacity/pulse/qOCV exports**. Sets config `llm_interpret`. Keys its skip-check on the GOLD path, so pair with `--overwrite` when GOLD already exists. |

## Post-labeling target sync

After `df_gold.update(...)` in `_process_cell`, final targets are mapped per-`ID` onto `X_silver` and re-saved to `with_features_post_labeled/<cell>.csv`, overwriting intermediate cluster labels (CAP*/PUL*/QOCV*) with finals (CAP, PUL, PUL*RES, qOCV_DCH, qOCV_CHA, −1). Unmatched numeric labels are left as-is.

## Field-data track (`src/field/`)

Parallel pipeline for **EV field data**, separate from the cycler CU pipeline. Lives in `src/field/`.

Current focus: the **shiyunliu on-road EV charging dataset** (20 production EVs, ~29 months each, MIT licence; Deng et al. Applied Energy 339:120954; repo `shiyunliu-battery/battery-charging-data-of-on-road-electric-vehicles`). At `<working_data>/field_data/shiyunliu_20ev/` as `#1.csv`..`#20.csv` (~1.4 GB, ~800 k rows/vehicle). **Charging-only** — sessions detected by 10-s gaps; the goal is to identify "capacity tests" = opportunistic full CC-CV charges via a modified main.py + HDBSCAN.

*(Earlier datasets, kept for reference: RWTH Aachen "Electric Vehicle and Battery Data" at `…/field_data/rwth_aachen/` — pre-segmented by activity, hence the pivot away. TUM FTM UDS at `…/field_data/tum_uds/` — only the 286 session JSONs, parquets blocked by LFS budget.)*

**Stage F1 — `field/io_shiyunliu.py`** (`python -m field.io_shiyunliu [base_dir]`): `load_vehicle(path)` returns canonical `Time / Voltage / Current / Temperature / SOC / Cell_V_max / Cell_V_min / Cell_T_min / Available_Energy_kWh / Available_Capacity_Ah` (extras preserved). Decodes `record_time` (int `YYYYMMDDhhmmss`) → UTC, strips unit suffixes, **negates `charge_current`** so positive = charging (raw convention is the opposite), and `check_sign_convention` re-verifies via dSOC/dt. Smoke-tested: all 20 load clean, 100 % non-null, ~842–847-day spans. **Outliers: vehicles #6 and #17** show post-negation max Current 400 A vs ~90–95 A elsewhere — treat with care. Session segmentation: `dt > 10 s`; vehicle #1 has 4 223 sessions, 197 with ΔSOC > 70 % (strong CAP candidates).

**Planned (not built):** F2 — session segmentation + per-session features (`duration_s, dSOC, I_mean, I_std, has_CV_tail, T_mean, SOC_start, SOC_end`). F3 — HDBSCAN → pick the full CC-CV cluster as CAP. F4 — coulomb-count CAP sessions → SOH timeline, emit `40_capacity_monitore`-shaped CSV so the existing monitor/matrix run unchanged. F5 — benchmark against the author's published Fig1.png.

Legacy (off the current path): `io_rwth.py`, `segment.py`.

## Documentation

- `METAbatt_Pipeline_Report.md` — full technical report
- `METAbatt_Pipeline_Flowchart.svg` — visual flowchart
