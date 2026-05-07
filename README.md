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
| `working_path` | Root data directory; BRONZE_CU parquet files must be under `<working_path>/BRONZE_CU/` |
| `V_max`, `V_min`, `V_nom` | Voltage limits and nominal voltage |
| `Nom_Capacity` | Nominal capacity in Ah |
| `CAP_Rate` | C-rate for capacity identification |
| `qOCV_CRate` | C-rate threshold for quasi-OCV detection |
| `pau_duration` | Pause threshold in minutes for procedure boundary detection |
| `min_rows` | Minimum rows to keep a procedure segment |
| `target_pulse_duration` | Expected pulse duration in seconds |

**Credentials** — `config.json` at project root (gitignored). Copy structure from `config_SE_example.json`. Also accepts env vars: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `INFLUX_TOKEN`.

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
├── output/                  # InfluxDB upload
└── util/                    # MinIO connection, shared helpers
```
