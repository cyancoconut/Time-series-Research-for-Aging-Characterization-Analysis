# TRACY

**T**ime-series **R**esearch for **A**ging **C**haracterization anal**Y**sis

A modular pipeline for processing battery aging check-up data — from raw cycler exports to enriched metrics (capacity, pulse resistance, qOCV). Designed as a companion to **MACY** (MAss CYcling): MACY runs the cycling tests, TRACY analyzes the check-up routines.

## Architecture

TRACY implements a medallion architecture across four parquet layers:

```
BRONZE_CU → preSILVER → SILVER → GOLD
```

| Layer | Description |
|-------|-------------|
| **BRONZE_CU** | Raw cycler export, German columns, unsegmented |
| **preSILVER** | Segmented into discrete procedures, columns renamed to English |
| **SILVER** | preSILVER with HDBSCAN cluster labels merged back |
| **GOLD** | SILVER enriched with capacity, pulse resistance, and qOCV metrics |

**Pipeline stages:**

1. **dismember** — segments raw data into discrete test procedures, reduces long PAU pauses to stubs
2. **feature extraction** — computes per-segment statistical features for clustering
3. **clustering** — two-layer HDBSCAN assigns procedure types (CAP, PUL, QOCV, PAU)
4. **calculate** — trapezoidal capacity integration, pulse resistance (R₀, R_ct), qOCV labeling
5. **output** — uploads GOLD to InfluxDB and MinIO
6. **export (optional)** — writes per-`BM_Programm` PUL and qOCV slices of GOLD as standalone parquet files named with the cell's SOH at that aging point. Gated by the `export_pulse` / `export_qocv` flags in the battery config.

For full details see [`METAbatt_Pipeline_Report.md`](METAbatt_Pipeline_Report.md) and [`METAbatt_Pipeline_Flowchart.svg`](METAbatt_Pipeline_Flowchart.svg).

## Setup

```bash
# Activate virtualenv (Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements_linux.txt   # Linux
pip install -r requirements.txt         # other platforms
```

## Configuration

**Battery parameters** — JSON file passed to `main.py` (e.g. `battery_config_VTC_linux.json`):

| Key | Description |
|-----|-------------|
| `working_path` | Root data directory; required when reading or writing locally. BRONZE_CU parquet files must be under `<working_path>/BRONZE_CU/` |
| `download_from` | `"local"` (default) reads BRONZE_CU from `working_path`; `"minio"` streams from `<bucket>/<minio_prefix>/BRONZE_CU/` |
| `upload_to` | `"local"` (default) writes GOLD/X_silver to `working_path`; `"minio"` uploads them to `<bucket>/<minio_prefix>/TRACY/`; `"both"` does both (objects written before the rename are still read from the legacy `10_TRACY/` prefix) |
| `minio_endpoint`, `bucket_name`, `minio_prefix` | Required when `download_from="minio"` or `upload_to` includes `"minio"` |
| `V_max`, `V_min`, `V_nom` | Voltage limits and nominal voltage |
| `Nom_Capacity` | Nominal capacity in Ah |
| `CAP_Rate` | C-rate for capacity identification |
| `qOCV_CRate` | C-rate threshold for quasi-OCV detection |
| `pau_duration` | Pause threshold in minutes for procedure boundary detection |
| `min_rows` | Minimum rows to keep a procedure segment |
| `target_pulse_duration` | Expected pulse duration in seconds |
| `pulse_keep_per_group` | 1-based positions of test pulses to keep within each group (e.g. `[1,3,5,7,9,11,13,15,17,19,21,23]`); all others are labelled `PUL*RES` |
| `pulse_group_by` | How to group pulses: `"BM_Programm"` (one group per program) or a column name (e.g. `"Temperature"`) combined with `pulse_step_threshold` to split by metric steps |
| `pulse_step_threshold` | Numeric threshold for sub-group detection when `pulse_group_by` is a column name; a new group starts when consecutive pulses differ by more than this value |
| `tolerances.pulse_duration_tolerance` | Safety factor on expected pulse duration for outlier rejection (default `1.08`) |
| `tolerances.restore_current_tolerance` | Fractional current tolerance for restore pulse detection (default `0.05`) |
| `tolerances.qocv_current_tolerance` | Fractional current tolerance for qOCV detection (default `0.01`) |
| `export_pulse` | If `true`, write per-`BM_Programm` PUL parquet files to `<working_path>/20_export_pulse/<cell_stem>/` (and to MinIO under `<minio_prefix>/20_export_pulse/<cell_stem>/` when `upload_to` includes `"minio"`). Default `false`. |
| `export_qocv` | If `true`, write per-`BM_Programm` qOCV_DCH / qOCV_CHA parquet files to `<working_path>/30_export_qocv/<cell_stem>/` (and the matching MinIO path). Default `false`. |

**Credentials** — `config.json` at project root (gitignored). Copy structure from `config_example.json` (a `minio` block: `endpoint` / `access_key` / `secret_key` / `bucket_name`). Also accepts env vars: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `INFLUX_TOKEN`. A full main-pipeline config example lives at `battery_config_example.json`. The MinIO endpoint, bucket and prefix live in the **battery config** alongside the routing keys above.

## Running the pipeline

All scripts must be run from the `src/` directory.

```bash
cd src

# Process all cells
python main.py /path/to/battery_config.json

# Process a subset of cells by name fragment
python main.py /path/to/battery_config.json --cells VTC_cell01 VTC_cell02

# Reprocess cells even if GOLD already exists
python main.py /path/to/battery_config.json --overwrite
```

## Project structure

```
src/
├── main.py                  # pipeline entry point
├── dismember/               # segmentation: BRONZE_CU → preSILVER
├── feature_extraction/      # per-segment statistical features
├── cluster/                 # HDBSCAN clustering and post-cluster filtering
├── calculate/               # capacity, pulse resistance, qOCV
├── output/                  # InfluxDB upload + optional per-program PUL / qOCV exports
└── util/                    # MinIO connection, shared helpers
```
