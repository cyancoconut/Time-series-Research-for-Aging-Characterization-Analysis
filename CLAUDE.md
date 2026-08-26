# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always tell me what the plan is before changing code.
Working data path is at /home/ann/Documents/Data_Metabatt (data only — do **not** mirror code/doc edits there).

## Git workflow

Before any code change, branch off: `feat/…` / `fix/…` / `refactor/…` / `config/…`. Push and open a PR into `main` (CodeRabbit auto-reviews). **Never commit directly to `main`.**

## Running the pipeline

All scripts/notebooks run from `src/` (imports are relative to `src/`). Venv: `source .venv/bin/activate` from project root.

**Main CLI**: `src/main.py`

```bash
cd src
python main.py /path/to/battery_config_VTC_linux.json
python main.py /path/to/battery_config_VTC_linux.json --cells VTC_cell01 VTC_cell02
```

The legacy notebook `src/Process_Detection_via_Cluster_py_METABATT.ipynb` still works; `main.py` is the current entry point.

**Unified UI**: `src/pipeline_ui.py` — customtkinter app wrapping seven stages (Download / Build BRONZE_CU / Run Pipeline / Monitor / Evaluation / Train Classifier / Initial Characterization) with a shared config picker, per-tab Run buttons, a "Run all 1→2→3→4→5" chain, Stop, and a live console. Persists paths to `~/.config/metabatt_ui.json`. Tab 6 (Train Classifier) and Tab 7 (Initial Characterization) are intentionally **outside** the chain — offline/one-off build steps. Prereq on Linux: `sudo apt install python3-tk`. Each Run button spawns the matching CLI:
- Tab 1 → `download/run_download.py <download_cfg.json>` (headless wrapper of `download/download_GUI.py`). Two independent substring filters select which Ahjo tests are fetched, ANDed, each a string or list matching **any** entry (`util.procedure_filter.matches_any`): **`test_type_filter`** ("Test type", default `TS`) on `test.name` — the measurement token, `TS…` for cycler tests / `EIS…` for EIS files — and **`test_name_filter`** ("Test name", optional, default all) on `test.parent`, the **programme** name (`jri_CU_VTC6`, `jri_Aging_…`). The programme is the same text `procedure_filter` matches later in `_is_cu`, so setting it here narrows what is downloaded at all, not just what BRONZE_CU treats as a check-up. EIS measurements live under their own programme (e.g. `zho_Namey_EIS`), so `test_name_filter: "jri_CU"` excludes them — list both to keep them. Legacy configs' `name_filter` is read as `test_type_filter`.
- Tab 2 → `download/build_bronze_cu_with_ah.py <battery_cfg> [--cells …] [--overwrite] [--incremental]`. Honors `download_from`/`upload_to` (same as `main.py`); `local` reads per-test parquets from `<working_path>/<cell>/*.parquet`, `minio` (default) from `<minio_prefix>/<cell>/`. Legacy `save_local`/`upload_s3` accepted when `upload_to` absent. Skips if the target BRONZE_CU already exists. **`--incremental`** appends only test files not yet in the per-cell manifest sidecar (`<cell>_manifest.json`, beside BRONZE_CU local + MinIO-untagged) instead of re-reading/re-integrating every file: the manifest carries the processed-file list + running Ah total + last sample, so the cumulative Ah integral continues seamlessly and the output equals a full rebuild. **Dead time between test files books no Ah**: any interval whose Δt exceeds the gap cut (auto `50 × median Δt`, or a fixed `ah_gap_threshold_s`) is masked, so a parked cell between files can't interpolate phantom throughput across the gap (`util/add_ah_throughput.py`). Applies on the incremental seed→first-new-row bridge too. Full builds also write the manifest; `--incremental` falls back to a full build when no prior state exists. Assumes new files have later timestamps (a backfilled earlier-`Zeit` file needs `--overwrite`). CU detection (`_is_cu`): per-test file is a check-up when its 4th `=`-delimited filename field contains `procedure_filter`, which is **required** (`process_cell` raises `ValueError` if unset). `procedure_filter` may be a single substring or a **list of substrings** (matches when the field contains **any** entry — e.g. `["jri_CU", "jri_Char"]` to fold a characterization run in with the routine checkup); normalized via `util.procedure_filter.as_filter_list`.
- Tab 3 → `main.py <battery_cfg> [--cells …] [--overwrite] [--clustering …]`. A **Clustering** toggle (Auto / HDBSCAN / Classifier) maps to `--clustering`, so HDBSCAN can be forced even when the config sets `classifier_model_path`. A separate **Interpret clusters (LLM)** button runs `cluster.interpret_clusters` (label-only) on the existing CSVs.
- Tab 4 → `python -m monitor.aging_status <battery_cfg> [-o …]`
- Tab 5 → a checklist of evaluation outputs run in sequence by one "Run evaluation" button: **Fleet-wide capacity aggregation** → `python -m evaluation.export_cap_pulse <battery_cfg>`; **Capacity evaluation (Alterungsmatrix)** → `python -m evaluation.aging_matrix <battery_cfg>`. Pulse / qOCV evaluations are placeholder checkboxes (disabled) for future stages. The "Run all 1→2→3→4→5" chain runs stages 1–4 then every ticked Tab 5 evaluation.
- Tab 6 → `python -m cluster.train_classifier <battery_cfg> [--model-out …] [--meta-out …] [--labels …]`. A **Labels** toggle (Config / target / llm) maps to `--labels`; `Config` omits it so the config's `classifier_label_source` (default `target`) decides.
- Tab 7 → `download/build_bronze_para.py` → `python -m characterize.main_para` → `python -m characterize.fit_characterization`, each gated by its checkbox and run in sequence. The fit stage has **one checkbox per block** (pulse / EIS / qOCV) → `--only`; all three ticked passes no flag, none ticked skips the stage. Outside the chain (one-off BOL step).

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

**Procedure-filter gate**: `_process_cell` peeks at BRONZE `Prozedur` before pulling the payload. `procedure_filter` is a substring or a list of substrings (matches on **any**). If it is set and nothing matches, the cell is skipped (`INFO {cell}: no Prozedur matches filter …`) — dismember never runs and MinIO never calls `fetch_bronze`. Uses `processing_procedure_filter` (pyarrow, reads only `Prozedur` row-group by row-group, short-circuits); on MinIO via `io_router.open_bronze_range` (HTTP-range read of footer + one column).

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

   Routing follows `download_from`/`upload_to`. Export MinIO keys are **untagged** (no `TRACY`).

   - **`output/export_eis.py`** — optional per-`BM_Programm` EIS export (flag `export_eis`, default off). Unlike pulse/qOCV (which read GOLD segments), EIS impedance data lives in **standalone files** in the cell's download folder; the cycler timeline only carries an `EIS` *label* marking when a measurement fired. `export_eis` discovers those files (`util/io_eis.py`), reduces each to a settled per-frequency spectrum, matches each to a `BM_Programm` by nearest EIS-labelled segment time, and bundles **one file per program** with every matched spectrum stacked (tagged by `eis_number`/`Time`/`U`). SOH from the program's CAP, same as pulse/qOCV. File: `25_export_eis/<cell_stem>/<cell_stem>_eis_BM<BM_Programm>_<SOH>SOH.parquet` (MinIO untagged).
     - **EIS files** (`util/io_eis.py`): one measurement per file (csv or parquet), rows = per-second dwell samples of a frequency sweep. Recognised by the `=`-field measurement token `eis_file_marker` (default `(?:EIS|INS)\d+`, e.g. `EIS00017`; cycler tests carry `TS…`) — **not** by "EIS" in the procedure name (over-matches cycler EIS procedures). Reduction keeps the **last settled row per `ActFreq`** (`ActFreq>0 & Betrag>0`), canonical cols `Time, frequency (ActFreq), Z_real (Zreal1), Z_imag (Zimg1), Z_abs (Betrag), phase (Phase), U (U1)`. Filename metadata parsed layout-agnostically (cell_stem = field 1, a datetime field, the marker token). Source per `download_from` (local `<working_path>/<cell_stem>/`, minio `<minio_prefix>/<cell_stem>/`).
     - **Downloader**: `download_from_specimen` detects EIS files (same marker, or `EISkanal` channel) and **keeps all columns**, bypassing the cycler `desired_columns` whitelist that would otherwise strip the impedance-sweep columns to an unusable 6-column stub.
     - **Matching** (`match_spectra_to_programs`): nearest EIS-labelled segment (`target=="EIS"` or `Prozedur` contains `eis_procedure_filter`, default `"EIS"`) within `eis_match_tolerance_minutes` (default 120). BRONZE_CU `Zeit` is tz-aware UTC and EIS times are naive but on the same lab clock, so matching compares tz-naive wall-clock.

7. **`output/export_capacity.py`** — always runs at end of `_process_cell` (no flag). Per-cell capacity summary CSV (one row per BM_Programm) for the aging monitor/matrix. Columns: `BM_Programm, Capacity_py, Ah_throughput, SOH, CAP_start_time` (`Ah_throughput` = cumulative throughput at the CAP-segment start). Flat under the folder:
   - Local: `<working_path>/40_capacity_monitore/<cell_stem>_capacity.csv`
   - MinIO: `<minio_prefix>/40_capacity_monitore/<cell_stem>_capacity.csv` (untagged)

## Learned segment classifier (optional, replaces HDBSCAN)

`cluster/train_classifier.py` + `predict_classifier.py` — a RandomForest drop-in for HDBSCAN + `post_cluster_filter`. **Opt-in**: runs only when `classifier_model_path` is set; otherwise the unchanged HDBSCAN path. Motivation: the rule thresholds (`CAP_Rate ± 5 %`, `cap_temp ± 3 °C`, qOCV window) are per-chemistry and don't transfer to field data.

- **Feature basis** (`FEATURE_COLS`, 12 cols) — scale-/chemistry-portable: current profile (`Current_{mean,std,max,min,range}`, `abs_Current_mean`, normalized ÷ `Nom_Capacity`), voltage **edges** (`Voltage_{max,min,range}`, ÷ `(V_max − V_min)`; `Voltage_range` is the SoC-swing proxy), `Duration_{minutes,quartile}`, and `prev_end_voltage_norm`. Voltage *curve shape* (`Voltage_mean/std`) and `Temperature_mean` are **excluded on purpose** (sensor-/chemistry-bound, no accuracy cost). Contract lives in `_meta.json["feature_columns"]`; `predict_classifier` reads it from there. `create_features` still computes the dropped cols (inert).
- **Context feature** `prev_end_voltage_norm` (`create_features.py`): end-of-segment voltage of the most recent **non-PAU** predecessor (walk back ≤4 steps, skip PAU), normalized as `(V - V_min) / (V_max - V_min)` (same scale as `Voltage_*` features: 0 = bottom rail, 1 = top rail); `-1.0` if no predecessor exists. Mirrors `post_cluster_filter.previous_voltage` — the signal the CAP rule keys on (a true CAP discharge follows a full charge). Inert for HDBSCAN (3-col subset). **CSVs predating this column must be regenerated before training** — `train_classifier` aborts if it's absent.
- **Training** (`python -m cluster.train_classifier <battery_cfg>`): reads every `with_features_post_labeled/*.csv` (post-#37 these carry all segments + `cluster_id`, supplying negatives), weak-labels leftovers via `bootstrap_leftover_labels`, runs leave-one-cell-out CV, then refits on all cells. **Output**: pipeline artifact → `<working_path>/60_classifier/models/<type_cell>_classifier_<UTC-timestamp>.joblib` + `_meta.json` (falls back to `../models/` only if `working_path` unset). Timestamped, so nothing is overwritten — **prune manually**, and `classifier_model_path` must point at a specific file. `--model-out`/`--meta-out` override the stem but a `_<timestamp>` suffix is always appended. Uploaded **untagged** to `<minio_prefix>/60_classifier/models/` when `upload_to` includes minio. **Source routing** (`_load_cell_csvs`): `local` globs `<working_path>/with_features_post_labeled/*.csv`; `minio` lists/fetches the tagged `<prefix>/TRACY/with_features_post_labeled/`. One config = one source.
- **Bootstrap labels** (`bootstrap_leftover_labels`): names the stringified raw leftover clusters by scale-free signature, **without touching** final labels (`CAP`/`PUL`/`PUL*RES`/`qOCV_*`/`-1`). Runs in two places: (a) `main.py` HDBSCAN path — after target sync, applied to `X_silver` and mapped into `df_gold["target"]` so the CSV **and** GOLD carry `PREP_CHA`/`SOC_ADJUST`/`-1` (raw integer cluster kept in `cluster_id`); changes no numeric result. (b) `train_classifier._load_cell_csvs` — idempotent re-derivation for legacy CSVs. Rules:
  - `PREP_CHA` — charge (`Current_mean > 0`) at ≥ `min_prep_current` (default 0.1, ~C/10) ending near full (`Voltage_max > 0.95`).
  - `SOC_ADJUST` — partial charge/discharge at ≥ `min_prep_current` ending at intermediate SoC (`Voltage_max ≤ 0.95` charge, or `Voltage_min > 0.05` discharge). `PREP_CHA` wins ties.
  - Everything else leftover stays `-1` (OTHER). **No `CYCLE`** (BRONZE_CU has no cycling) and **no `PREP_DCH`** — CAP-vs-discharge is carried by `prev_end_voltage_norm`.
  - **`PUL*RES → PUL` for training**: the classifier needn't tell restores apart (the split is recomputed downstream). Merging lifted LOCO 0.952 → 0.985.
- **Inference** (`predict_classifier.predict_targets`, gated in `_process_cell_inner`): maps labels to the tagged form `calculate/` expects (`CAP→CAP*`, `PUL→PUL*`, `qOCV_{DCH,CHA}→QOCV*`); `update_pulse` still runs and re-splits `PUL*` into `PUL`/`PUL*RES`. `PREP_CHA`/`SOC_ADJUST` pass through into GOLD `target` (informational; no numeric effect). NaN-feature rows → `-1`.
  - **CSV routed to `60_classifier/`** (gated on `classifier_model_path`): written to `<working_path>/60_classifier/with_features_post_labeled/<stem>.csv` (local) / `<minio_prefix>/60_classifier/with_features_post_labeled/` (MinIO, untagged) — side by side with the HDBSCAN CSVs, kept out of the training set. **Only the segment CSV is namespaced** to `60_classifier/`. **GOLD + `export_capacity` + pulse/qOCV exports are *not* — the classifier is a drop-in for HDBSCAN and writes them to the shared locations** (`GOLD/`, `40_capacity_monitore/`, `20_/30_export_*`), so a classifier run produces the same downstream outputs as an HDBSCAN run **and overwrites them for that cell**. (Local skip-check keys on the shared GOLD path: a non-`--overwrite` run skips a cell whose GOLD already exists, regardless of which path produced it — so `--overwrite` to re-run.)
  - **`cluster_id` on the classifier path**: holds the **raw predicted label** (`CAP`/`PUL`/`qOCV_*`/`PREP_CHA`/`SOC_ADJUST`/`-1`), captured before the tagged-form map. **Caveat**: it's the classifier's own string label (provenance/audit), **not** a raw HDBSCAN cluster — building new training data still requires an HDBSCAN re-run (`classifier_model_path` unset). Type differs by path: integer (HDBSCAN) vs string (classifier); treat as opaque.

## LLM cluster interpretation (advisory, augment-only)

`cluster/interpret_clusters.py` (`python -m cluster.interpret_clusters <cfg> [--source local|minio] [--cells …] [--overwrite] [--dry-run] [-o …]`) — names HDBSCAN clusters with an LLM so leftover (`-1`) segments get descriptive names and the rule labels get an independent audit. Groups `with_features_post_labeled` rows by `(cell, cluster_id)`, builds a per-cluster signature (mean/min/max of the scale-free classifier features + member count + majority `target` + rule-bootstrap label), dedupes identical rounded signatures across cells (one API call each). The LLM invents its **own free-form snake_case label** (e.g. `full_discharge_c2`, `full_discharge_1c`, `full_charge_c20`, `mixed_*` for sign-mixed clusters) — deliberately *not* constrained to the pipeline taxonomy, so `llm_label` is a second opinion next to `target`, not a forced mapping. The prompt has **no `qocv` label** (nor a capacity-test label): a full charge/discharge at any C-rate — including a very low-rate quasi-OCV sweep — is named `full_charge`/`full_discharge` with its crate, and quasi-OCV-vs-capacity is decided downstream from the measured rate (see below), mirroring how CAP is handled. **Augment-only**: results land in `llm_label` / `llm_confidence` / `llm_rationale` columns written back to the same CSV (local in place / same MinIO key); `target`, `cluster_id`, and all numerics untouched. Audit summary → `<working_path>/50_evaluation/cluster_interpretation.csv` (one row per cluster: n, majority `target`, llm label/confidence/rationale; MinIO untagged); an all-skip run leaves the audit untouched. Cells already carrying `llm_label` are skipped unless `--overwrite`; CSVs predating the `cluster_id` column (pre-#37) are skipped with a regenerate hint — they need an HDBSCAN re-run (`classifier_model_path` unset).

- **Integrated label-only route** (`main.py --interpret`, or config `llm_interpret: true`): folds the same interpretation into the pipeline. After HDBSCAN clusters a cell, it adds `llm_*` to the in-memory `X_silver`, writes the `with_features_post_labeled` CSV, and **returns early — no GOLD, no calc, no capacity/pulse/qOCV exports** (the CSV is the sole output). Unlike the classifier route (which now writes full GOLD + exports), this label-only pass stops at the CSV. `--interpret` forces the HDBSCAN path (`classifier_model_path` ignored), since interpretation names HDBSCAN clusters. Dedup is per-cell here (one cache per cell), vs cross-cell in the standalone CLI. Use the standalone `cluster.interpret_clusters` to label CSVs that already exist; use `--interpret` to cluster-and-label in one pass.
- **Provider seam** (`util/llm_client.py`): `LLMClient` protocol (`interpret_cluster(signature) -> ClusterLabel{label, confidence, rationale}` — Pydantic-validated). Backend via config key `llm_provider` (default `openai`). `OpenAILLMClient`: official `openai` SDK, `responses.parse()` structured output, `reasoning effort: medium`; model via `llm_model` (default `gpt-5.4`). `AnthropicLLMClient`: official `anthropic` SDK, `messages.parse()` structured output, adaptive thinking + `effort: medium`, prompt caching (`cache_control: ephemeral`) on the stable system prompt; `llm_model` default `claude-opus-4-8`. Credentials: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var, fallback `openai_api_key` / `anthropic_api_key` in root `config.json` (gitignored).
- **LLM labels as training targets** (`train_classifier --labels llm`, or config `classifier_label_source: "llm"`; default `"target"`): trains on the `llm_label` column (from `interpret_clusters`), but **canonicalizes it first** (`_canonicalize_llm_labels`) into the resolved space the pipeline acts on — the LLM's raw `label` field is unreliable for training (it sometimes contradicts its own `true_voltage_range ≥ 0.9` rationale on the full-vs-partial cut, and fragments C-rates/pulses into unlearnable singletons: `pulse` vs `pulse_c1`, `full_charge_c20` vs `full_discharge_c20` vs `mixed_*_c20`). Each segment is **rebuilt deterministically from its own features** (sign, `true_voltage_range`, measured `abs_Current_mean`): any `*pulse*`→`pulse`; a full sweep (`true_voltage_range ≥ 0.9`) at/below the qOCV band (either sign)→`qocv`; a full **discharge** in the cap band→`cap`; any other full sweep→`full_charge_<crate>`/`full_discharge_<crate>` (informational, e.g. the C/2 prep charge); a partial→`partial_cha_<crate>`/`partial_dch_<crate>` (kept, with direction+crate); `unknown`/`rest`/`artifact` and sign-ambiguous rows pass through. `mixed_*` clusters **dissolve** because each member is resolved from its own sign. The cap/qocv bands use the **same** `cap_rate`/`qocv_crate` + tolerances (`cap_tol 0.05`, `qocv_tol 0.2`) as inference, so a training label and the inference tag agree; this folds the worst LOCO confusions (within-pulse, within-qocv) into single classes. Segments with no `llm_label` are dropped; no PUL*RES-merge / bootstrap. `_meta.json["label_source"]` records the space. At inference `predict_classifier._map_llm_label_to_tagged` maps the prediction back: `*pulse*→PUL*`, `*qocv*→QOCV*`, bare `cap→CAP*`, and a residual `full_charge`/`full_discharge` is resolved by its **measured** `abs_Current_mean`: `→QOCV*` when ≤ `qocv_crate × (1 + qocv_tol)` (charge or discharge), else `→CAP*` when a `full_discharge` within `cap_rate ± 5 %`. `qocv_crate ≪ cap_rate` so the bands never overlap; qocv is checked first. Every other label passes through into GOLD `target` informationally. `predict_targets` takes `cap_rate` and `qocv_rate` (passed from `main.py` as `cfg["cap_rate"]` / `cfg["qocv_crate"]`); with a rate unset, the corresponding tag is never produced.

## Evaluation

**Fleet-wide capacity** — `evaluation/export_cap_pulse.py` (`python -m evaluation.export_cap_pulse <cfg> [-o …]`): aggregates `40_capacity_monitore/*_capacity.csv` + the procedure names from each cell's BRONZE_CU manifest (`BRONZE_CU/<stem>_manifest.json`) into one table. Aging metadata (`DOD/SOC/C_Rate/Temperature`) parsed from procedure names via `output.add_information_METABATT`. Output (driven by `upload_to`): `<working_path>/50_evaluation/capacity_results.csv` (full CAP history), MinIO `<prefix>/50_evaluation/capacity_results.csv` (untagged).

**Aging matrix** — `evaluation/aging_matrix.py` (`python -m evaluation.aging_matrix <cfg> [-o …]`): per-cell capacity loss normalized by Ah throughput, aggregated over the design space. Reuses `export_cap_pulse.build_capacity_table`; needs `Ah_throughput` (aborts otherwise). Per cell: `capacity_lost` = max−min `Capacity_py`, `Delta_Ah_throughput` = max−min `Ah_throughput`. Matrix: group by `(C_Rate, Temperature, DOD, SOC)` → mean/std, `candidate_count`, cell list, `capacity_lost_norm`. Output: `<working_path>/50_evaluation/aging_matrix.{csv,html}` (interactive plotly), MinIO untagged.

**HDBSCAN vs classifier diff** — `evaluation/compare_labels.py` (`python -m evaluation.compare_labels <cfg> [--source local|minio] [--hdbscan-dir] [--classifier-dir] [-o]`): diffs the two per-segment label sets per cell (joined on `ID`) to see whether the classifier recovers CAP check-ups HDBSCAN missed (invisible to the LOCO report, which scores against HDBSCAN's own labels). Prereq: run the pipeline twice on the same cells (HDBSCAN, then classifier with `--overwrite`). `local` reads `<working_path>/with_features_post_labeled/` (HDBSCAN) + `…/60_classifier/with_features_post_labeled/` (classifier); `minio` reads tagged `TRACY/…` (falling back to the legacy `10_TRACY/…`) + untagged `60_classifier/…`. Reports cell coverage (both / classifier-only / HDBSCAN-only), per-`(cell, BM_Programm)` CAP-count delta, label transitions, and per-segment disagreements with explanatory features. Output (default `<working_path>/50_evaluation`): `label_diff_segments.csv`, `cap_count_diff.csv`, `cap_recovered_cells.csv` (always written locally).

## Initial characterization (BOL parametrization)

A separate, one-off track for the parametrization run that precedes aging.
Parametrization test files sit in the same per-cell download folders as the CU
tests and are selected by `para_procedure_filter` (a **list** of programme-name
substrings; one element is the normal case).

1. **`download/build_bronze_para.py <cfg> [--cells …] [--overwrite]`**
   — the CU builder (`build_bronze_cu_with_ah.run`) parameterized by
   `layer="BRONZE_PARA"` + `filter_key="para_procedure_filter"`. Writes
   `<working_path>/BRONZE_PARA/<cell>.parquet` + manifest sidecar; MinIO
   `<prefix>/BRONZE_PARA/`. **No `--incremental`** — a parametrization run is a
   one-off BOL measurement, not a growing aging series; rebuild with
   `--overwrite`.
2. **`python -m characterize.main_para <cfg> [--cells …] [--overwrite] [--clustering …]`**
   — `main.run_pipeline` with the `CHARACTERIZATION` run context
   (`util/run_context.py`). Same dismember → features → clustering → calculate
   stages; reads BRONZE_PARA and forces the pulse/EIS/qOCV exports on.
3. **`python -m characterize.fit_characterization <cfg> [--cells …]
   [--only {pulse,eis,qocv} …]`** — fits and plots the bundles. Standalone, so
   fits can be repeated without redoing segmentation. The three blocks are
   independently selectable via `--only` (omit = all three); a subset run
   **merges** into the existing `<cell>_parameters.json`, so the blocks left
   out keep their previous results instead of being dropped.

**`RunContext`** (`util/run_context.py`) carries what differs between run
flavours — `bronze_layer`, `procedure_filter_key`, `export_root_prefix`,
`force_exports` — threaded explicitly through `run_pipeline` →
`_process_cell` → `_process_cell_inner` → the export modules, defaulting to
`CU` (today's behaviour, unchanged).

**Layout** — `export_root` covers *everything* the run writes, so a para run
never overwrites the CU `GOLD/<cell>.parquet` or pollutes
`40_capacity_monitore/`:

```
<working_path>/10_initial_characterization/<cell_stem>/
├── <cell>_parameters.json
├── <cell>_{pulse,eis,qocv}_fits.csv   # the JSON's fits[] as flat tables
├── <cell>_eis_drt_peaks.csv          # one row per DRT peak (eis.drt.peaks)
├── GOLD.parquet
├── with_features_post_labeled.csv
├── <cell>_capacity.csv
├── data/   <cell>_{pulse,eis,qocv_dch,qocv_cha}_BM<n>_<SOH>SOH.parquet
└── plots/  pulse_2rc_<cell>_BM<n>_<SOH>SOH_<direction>_<T>degreeC.png
            eis_{2zarc_warburg,raw_spectra,fit_overlay,nyquist}_<stem>_<direction>_<T>degreeC.png
            eis_drt_{gamma,map}_<stem>_<direction>_<T>degreeC.png
            qocv_<T>degreeC.png
```

MinIO mirrors this under `<prefix>/10_initial_characterization/<cell_stem>/`,
**untagged**. The `--overwrite` skip-check keys on this root's GOLD.

**Temperature** — every fit record carries `T_degC` (the measured mean cell
temperature, 2 dp) and every plot filename ends in `_<T>degreeC` (the same
value rounded to whole °C, `NAdegreeC` when unknown). Both R and the qOCV
hysteresis are strongly temperature-dependent, so two runs of the same
programme in different chambers are different measurements and must not
collide on one filename. Sources: **pulse** — mean `Temperature` over the
pulse + its relaxation window (`pulse_fit._window_temperature`); **qOCV** —
mean over both branch parquets; **EIS** — the EIS device file has no
thermocouple channel, so the bundle borrows the temperature of the **pulse
bundle of the same `BM_Programm`**, falling back to the cell's pulse mean.
`eis.settings.bundles[].temperature_source` records which
(`pulse-BM<n>` / `cell-mean` / `eis-bundle` / `unavailable`), so a borrowed
cell-level number is never mistaken for a same-programme measurement. Bundles
exported without a `Temperature` column still fit — `T_degC` is NaN.

**Models are fixed defaults**, not config keys: **2RC** for pulse
(`analysis/fit_2rc_pulse.py`), **2×ZARC + series-L + generalized Warburg** for
EIS (`analysis/eis_vs_soc.fit_zarc_warburg_eis`), and no fit for qOCV (the
curve plus its throughput capacities). The EIS diffusion element has φ
**fitted** (`DIFFUSION_PHI_BOX`) but τ_d **pinned** (`DIFFUSION_TAU_BOX`, 5 s):
`R_d_z` is the amplitude at ω = 1/τ_d and `tau_d_z` a shape constant, not a
result. Those settings — element, pinned τ_d, φ box, `ZARC_ALPHA_MIN` — are
recorded in `parameters.json`'s `eis.settings`; both were tuned on the NFPP
sweep and may need retuning for another cell type or frequency range.

**Two-stage R0** (`eis_two_stage_r0`, default **on** — this is the standard
path; set false only to reproduce a pre-#70 fit) — in the single-stage
full-band fit R0 and the mid-frequency ZARC are correlated: a **depressed** arc
(small α) has a broad high-frequency foot that reaches the real axis and can
absorb part of the series resistance. Harmless while the arc stays round, but
on the NFPP sweep α1 falls to 0.71 below 20 % SOC as the arc grows ~8×, and R0
then gives way — it turns over and *falls* 0.06 mΩ toward the empty end while
the spectrum says it is still rising. With the flag on, `fit_hf_r0` measures R0
first on `f ≥ eis_hf_r0_f_min_hz` (default 100 Hz) with `R0 + jωL + one ZARC` —
a window where the slow arc and diffusion are flat, so R0 is well posed (1σ ≈
0.011–0.013 mΩ, 15 of 48 points) — and `fit_zarc_warburg_eis(pin_r0=…)` then
fixes it, dropping R0 from the free-parameter vector (not boxing it to zero
width; `least_squares` needs `lb < ub`). Result on NFPP_01: R0 becomes monotone
in SOC over the whole sweep, α1 flattens to 0.77–0.90, at a mean RMSE cost of
0.035 → 0.040 mΩ, most of it on the SOC 2.5 % spectrum (0.144 → 0.216) which
the model already fitted worst. Replicated independently on NFPP_02 (a
different fixture): R0 monotone, roughness 0.0102 → 0.0056, headroom spread
4.3× → 1.3×, degenerate fits 3 → 1, RMSE 0.0216 → 0.0233, and the pulse
cross-check `pulse_R0 / EIS(R0+R1+R2)` preserved at 1.009. A failed/NaN HF
stage falls back to fitting R0.
`R0_z`/`L_z` and the HF diagnostics (`R0_hf`, `R0_hf_sigma`, `hf_rmse`, `hf_n`,
`r0_pinned`) are now in `EIS_COLS`, so the fitted series term is exported —
previously only the fit-free crossing was. The `R0` panel of
`plot_zarc_vs_soc` overlays the HF estimate ±1σ.

**`R_cross` is not the ohmic resistance** (called `R_ohm` until the two-stage
work; renamed because the old name invited exactly that misreading, and
`eis_features`' docstring asserted it outright). It is `Z_real` at the
`Z_imag = 0` crossing — and on an inductive cell that crossing is at a
*finite* frequency, not the high-frequency limit: it is wherever ωL cancels
the arcs' reactance, 255–355 Hz on the NFPP sweep (L ≈ 158 nH). The arcs still
contribute 0.11–0.19 mΩ of real part there, so `R_cross` runs **12–16 %
(0.135–0.211 mΩ) above the fitted R0**, by an SOC-dependent margin. The
decomposition closes: `R0 + Re(arcs + diffusion)` at `f_cross` reproduces
`R_cross` to < 0.007 mΩ. What sets the gap is the *inductance*, not the arc
size — corr(gap, ωL at crossing) = 0.88 vs corr(gap, R1+R2) = 0.45, because a
larger arc also has a larger τ and pushes the crossing down in frequency,
compensating. Practically: `R_cross` tracks R0's SOC trend to ±1.5 pp so it is
a usable *relative* proxy, but its ~15 % offset is a property of the rig's
inductance, so never compare it across setups. `f_cross_Hz` is exported beside
it so this is checkable from the CSV. `R_pol = R_tot − R_cross` inherits the
same bias with the opposite sign and **understates** polarisation.
For the series resistance use `R0_z`, or `R0_hf` under `eis_two_stage_r0`.

**Open — SOC-dependent inductive loss.** The pure `jωL` series term is
incomplete: with R0 and the arcs subtracted, `Re(residual)` at 6 kHz is
0.059–0.089 mΩ (a pure inductor predicts exactly 0), rising monotonically with
SOC (corr 0.86) while `Im(residual)` and hence L stay flat at 153 nH ±0.3 %.
So it is a *loss on a constant inductance*, and it is SOC-dependent — not a
fixture constant. Fitting `Re_res = c + A·ω^p` gives **p ≈ 1.10–1.11** (stable
above SOC 45) with offset `c ≈ −0.028 mΩ`, i.e. the two-stage R0 is biased high
by ~2.3 %, varying 0.030 mΩ across the sweep (14 % of R0's own SOC swing).
Neither obvious element fixes it: a **parallel `R_L ∥ jωL`** predicts p = 2
(fitted R_L ≈ 285 mΩ, 3.6σ, and note it only works in the `ωL ≪ R_L` regime —
at small R_L it collapses to a constant resistance degenerate with R0), and a
**fractional inductor `L(jω)^γ`** predicts p = γ ≤ 1 (fitted γ ≈ 0.98, best
HF RMSE of the three at 0.031 vs 0.043 vs 0.049) but shifts R0 by −0.09 mΩ,
3–4× more than the 0.028 mΩ actually there. Choosing on RMSE picks the element
that damages R0 most. **R0 stays monotone in SOC under all three**, so the
two-stage fix is robust to this; left unmodelled deliberately. Untested idea:
narrow the HF window's top instead of adding an element.

**DRT** (`analysis/eis_drt.py`) — model-free companion, **run by default
alongside every EIS fit** (`eis_drt`, default true). `fit_eis` runs it per
bundle on the same raw spectra, writing `plots/eis_drt_{gamma,map}_<stem>_<dir>.png`
and `<cell>_eis_drt_peaks.csv` (one row per peak: `tau_peak`, `gamma_peak`,
`R_peak`, `width_decades`), plus an `eis.drt` block in `parameters.json`. It
answers the one question an ECM cannot ask of itself — how many relaxation
processes are in the spectrum at all — and the γ plot overlays the fitted τ, so
**an ECM τ landing in a DRT *valley* rather than on a peak** flags one element
blanketing a region with more structure than it has parameters for (which is
what `tau1_z` does on NFPP_01; see the bandwidth note). Adds ~0.5 s per bundle.

**λ is fixed** (`eis_drt_lambda`, default `1e-3`), *not* the L-curve corner.
The corner is better for a one-off investigation but is not reproducible enough
to bake in: across one NFPP sweep it picked λ from 2.5e-6 to 1.6e-2 with peak
counts swinging 2–10, so consecutive SOC steps would be smoothed differently
and the vs-SOC structure would be an artefact of λ. It is also ~11× slower
(4.7 s vs 0.4 s per bundle). Peak counts are λ-dependent either way — the same
NFPP_02 data reads "τ1 in a valley" at 4e-3 and "τ1 on a peak" at 1e-3 — so
treat a peak count as a statement about (data, λ), not about the cell.

It also still has its own CLI
(`python -m analysis.eis_drt <eis_export.parquet> [--data-dir …]
[--sweep-direction …] [--lam …]`), which defaults to the L-curve corner and is
not on Tab 7. Its SOC comes from
`util.soc_from_qocv.assign_soc`, same as the ECM fits, so a DRT panel and an
`eis_fits.csv` row can never disagree about a spectrum's SOC (`--data-dir`
defaults to the export's own folder). With no qOCV there it falls back to the
order-based ladder and says so in `meta["soc_source"]` — check that column
before quoting a SOC off a DRT plot. Use it
to ask how many relaxation processes a spectrum actually contains before adding
ECM branches. On NFPP_01 it resolves 2 kinetic peaks at mid/high SOC but only
**one broad** peak (1.15–1.52 decades) below 20 % SOC — which is why a 3rd ZARC
does not help there: it has no discrete peak to attach to, α pins to 1.0 on
19/21 spectra, and branches 2/3 swap roles between adjacent SOC points.

**Sweep direction** — a full-SOC-sweep pulse or EIS run goes one way (start
full and empty, or start empty and fill), and SOC is assigned by measurement
*order*, so a wrong guess inverts the whole SOC axis. **One bundle = one
direction**: a bundle is a single `BM_Programm`, and the parametrization
procedure puts the charge and discharge sweeps in separate programmes. Both
paths detect it the same way, from the first→last trend of a measured voltage
(EIS: per-measurement terminal `U`, `_bundle_direction`; pulse: pre-pulse
`OCV_V`, `pulse_fit._assign_soc_to_bundle`) via the shared turning-point rule
in `analysis/sweep_direction.py`. Config `soc_sweep_direction` overrides
detection for both. The result lands in three places: the plot **filename**,
each fit record (`sweep_direction` + `sweep_direction_source` ∈
`config-override`/`detected`/`assumed`), and `settings.bundles[]`. In
`PULSE_COLS`, `sweep_direction` is distinct from `direction` — the latter is
the pulse's own CHA/DCH polarity. A bundle that reverses mid-sweep is **not**
split: `_bundle_direction` reports the excursion as `reversal_mV` and warns.

**SOC from the qOCV curve** (`util/soc_from_qocv.py` — in `util/`, not
`analysis/`, so *every* module can reach it without re-deriving SOC; the
loading/coulomb-count primitives it needs live beside it in `util/io_qocv.py`
and are re-exported by `analysis/qocv_curve.py`, keeping `util` free of any
dependency on `analysis`). **Use `assign_soc(table, voltage_col, direction,
data_dir, …)`** — the one-call form of `find_sweeps` + `map_table` — rather
than writing an SOC assignment of your own. It is the **only** source
of SOC. The order-based **ladder** (`100 − 5·i` by measurement index) has been
**removed** from `eis_vs_soc.build_eis_table` and `pulse_fit.assign_pulse_soc`,
which now leave `SOC_pct` NaN for this module to fill. It assumed every step
moved the same charge, and was wrong twice over: the measured voltages
contradict it (the first NFPP EIS step drops 155 mV, the next ones ~15 mV, all
labelled "5 %"), and the run puts a **CHA and a DCH pulse on each SOC step**, so
the pulse index advanced twice per real step and the ladder ran **down to
−75 % SOC**. With no same-direction qOCV, `SOC_pct` stays NaN and the vs-SOC
plots skip with a recorded reason — an absent SOC is honest, a fabricated one
is not. Each
measurement's rest voltage (EIS `U`, pulse `OCV_V`) is interpolated onto the
run's own qOCV curve, whose coulomb count gives `SOC(V) = 100·(Q−Q_min)/(Q_max−Q_min)`.
`SOC_pct_ladder` no longer exists.
Sweep selection: **same direction** (from the `_qocv_cha_`/`_qocv_dch_`
filename token — the branches differ by the qOCV hysteresis) and **nearest in
time** (the qOCV usually sits in a different `BM_Programm` than the bundle).
The order-based value is kept as `SOC_pct_ladder` for comparison, and
`settings.bundles[]` records `soc_source_file` / `soc_source_bm` /
`soc_dt_hours` / `n_clipped` / `ir_correction`. With no same-direction qOCV in
the folder the ladder is kept and a warning is logged — a missing qOCV never
blanks the SOC axis.

**IR correction** — the qOCV is measured *under load* at ~C/20 while the mapped
values are **rest** voltages, so the branch's overpotential is removed first.
It is measured from the pair itself, `η(SOC) = [V_cha(SOC) − V_dch(SOC)]/2`,
and each branch shifted toward the middle (`V_dch + η`, `V_cha − η`). An
independently measured resistance is **not** usable: on the NFPP cell η implies
~11.9 mΩ mid-SOC while a 30 s pulse's 2RC fit gives `R0+R1+R2` = 5.6 mΩ (`R0`
alone 3.8 mΩ) — over a 20 h sweep slow diffusion adds resistance a short pulse
never sees, so a pulse/EIS-derived R removes only about half the offset.
Correcting both branches by their own η makes them coincide, so this is
equivalent to mapping on the cha/dch mean, but expressed per SOC and recorded.
Verified by branch convergence: a rest voltage mapped on cha vs dch disagreed
by mean −3.92 / max 14.78 SOC% uncorrected, and mean 0.00 / max 0.05 SOC%
corrected. Falls back to the scalar `qocv_ir_ohm` (`V ∓ I·R`, `I` read from the
sweep) when only one branch was exported, and to no correction when neither is
available.

**Nyquist zoom insets** (`eis_vs_soc.add_hf_inset`) — the full Nyquist view is
dominated by the low-frequency diffusion tail, collapsing the kinetics into a
few pixels at the origin. Both Nyquist axes (`plot_nyquist_by_soc` and the
Nyquist panel of `plot_raw_spectra`) carry two zoom insets along the
bottom-right: **R0 region** (tight on the real-axis intercept, so each SOC
curve's crossing of −Z_imag = 0 is readable) and **MF arc** (the whole
mid-frequency semicircle). Widths come from the fitted ZARC diameters — R0
from `R1_z` alone, MF from `R1_z + R2_z`, since the visible arc is both. The
y-window is scaled to the arc, **not** the data minimum: above the arc the cell
turns inductive and −Z_imag dives to −5.9 mΩ (~14× the arc), which would
flatten the semicircle to a line. Missing SOC (no qOCV) draws grey instead of
a colormap artefact.

**UI**: Tab 7 runs the three stages as a checklist. Like Tab 6 it is **outside**
the "Run all 1→2→3→4→5" chain.

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
| `tolerances.qocv_current_tolerance` | Half-width of `fetch_qOCV`'s **two-sided** current band, in **Amps** (absolute, not a C-rate; default `0.01`). A qOCV's mean current must satisfy `qOCV_CRate × Nom_Capacity ± qocv_current_tolerance`. The lower bound matters: with an upper bound only, a 0 A rest / EIS-dwell segment mislabelled `QOCV*` passed trivially and fell through `sign(0) != 1` to `qOCV_DCH`, exporting a bundle with no current and zero integrated capacity. |
| `tolerances.qocv_std_tolerance` | Max current std for a qOCV in `fetch_qOCV`, as a C-rate (× `Nom_Capacity` → Amps; default `0.002`). Replaces the old absolute `0.001 A` cap that didn't scale with cell size. |
| `pau_duration` | Pause threshold (min) for procedure boundaries (default 9.9) |
| `min_rows` | Min rows to keep a segment (default 20) |
| `qocv_procedure_filter` | Optional substring; inside matching procedures every `Zustand` change cuts a boundary (splits qOCV DCH/CHA). Omit to disable. |
| `target_pulse_duration` | Expected pulse duration (s, default 20) |
| `export_gold` | Write the GOLD parquet to disk/MinIO (default true). Set false to skip the GOLD write (downstream evals no longer read GOLD-on-disk; capacity/pulse/qOCV exports use the in-memory `df_gold`). |
| `export_pulse` / `export_qocv` | Write per-BM_Programm PUL / qOCV parquets (default false) |
| `export_eis` | Write per-BM_Programm EIS spectrum parquets from standalone EIS files (default false) |
| `eis_file_marker` | Regex identifying an EIS file by its `=`-field measurement token (default `(?:EIS|INS)\d+`) |
| `eis_procedure_filter` | Substring marking EIS-labelled segments used as match anchors (default `EIS`) |
| `eis_match_tolerance_minutes` | Max time gap to match an EIS measurement to a segment (default 120) |
| `eis_two_stage_r0` | Measure R0 on the HF window and pin it in the 2×ZARC fit, instead of fitting it against the correlated mid-frequency arc (**default true**). Fixes the spurious low-SOC R0 turnover; set false only to reproduce a pre-#70 fit. |
| `eis_hf_r0_f_min_hz` | Lower frequency bound of the two-stage R0 window (default 100). Needs ≥10 points above it or the stage is skipped and R0 is fitted as before. |
| `eis_drt` | Run the model-free DRT beside every EIS fit (**default true**): γ/map plots + `<cell>_eis_drt_peaks.csv` + an `eis.drt` block. ~0.5 s per bundle. |
| `eis_drt_lambda` | Fixed DRT regularisation (default `1e-3`). Deliberately not the L-curve corner — see the DRT section. |
| (always on) | `export_capacity` writes `<cell_stem>_capacity.csv` to `40_capacity_monitore/` |
| `running_window_days` | Monitor: `running` if last BRONZE_CU `Time` within N days (default 2) |
| `ah_gap_threshold_s` | Optional. BRONZE Ah counter: intervals with Δt above this (seconds) are dead time between test files and book no `Ah_throughput`. Omit (default) to auto-derive the cut as `50 × median Δt` (the cell's sampling cadence) — adapts per cell, no tuning. |
| `classifier_model_path` | Optional `.joblib` (relative to `src/` or absolute). When set, `_process_cell_inner` uses the classifier instead of HDBSCAN. Resolution (`_resolve_classifier_paths`): load locally if it exists; if absent **and** `download_from="minio"`, fetch `.joblib`+`_meta.json` by **basename** from `<minio_prefix>/60_classifier/models/` into a local cache. `local` + missing file → `FileNotFoundError`. Omit to keep HDBSCAN. |
| `classifier_meta_path` | Optional. Defaults to `<model stem>_meta.json`. |
| `classifier_label_source` | `target` (default) or `llm` — training-target space for `train_classifier` (CLI `--labels` overrides). `llm` trains on the free-form `llm_label` column; inference maps it back via measured C-rate. |
| `llm_interpret` | When true (or `main.py --interpret`), the HDBSCAN run is a label-only LLM pass: writes `llm_*` into the per-segment CSV and skips GOLD/exports. Default false. |
| `para_procedure_filter` | **List** of programme-name substrings marking parametrization test files (single-element list is normal). Required by `build_bronze_para.py` and `characterize.main_para`. |
| `qocv_ir_ohm` | Optional (Ω). Fallback IR correction for the qOCV→SOC mapping when only **one** branch was exported, so the pair-derived `η(SOC)` isn't available: the branch is shifted by `I·R` with `I` read from the sweep. Ignored when both cha and dch exist. Omit for no scalar fallback. |

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

`characterize.main_para` takes the same `--cells` / `--overwrite` / `--clustering` flags as `main.py`.

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
