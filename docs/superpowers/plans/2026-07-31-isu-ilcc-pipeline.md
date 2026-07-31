# ISU-ILCC Pipeline Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the ISU-ILCC NMC dataset through the unchanged METAbatt classifier pipeline and show the auto-detected per-RPT capacity matches ISU's own `capacity_discharge_C_2` ground truth.

**Architecture:** A standalone adapter (`build_bronze_isu.py`) reads each double-encoded ISU JSON, reassembles the four per-RPT QV sweeps into one BRONZE_CU-shaped parquet (synthesizing the columns the dismember contract needs), and writes it under `<working_path>/BRONZE_CU/`. A new ISU battery config drives `main.py`'s classifier path. A validation script compares pipeline `Capacity_py` to the ISU ground truth.

**Tech Stack:** Python 3, pandas, numpy, pyarrow (parquet), scipy (via `util.add_ah_throughput`), matplotlib (validation plot). No pytest in this repo — tests are standalone `python tests/test_*.py` assert scripts.

## Global Constraints

- All scripts run from `src/` (imports relative to `src/`); venv: `source .venv/bin/activate` from project root.
- Never commit to `J8005_BMWK_METABatt`. Work stays on branch `feat/isu-ilcc-pipeline`.
- Commit messages: no `Co-Authored-By` trailer.
- Config keys are lowercase snake_case (`cap_rate`, `qocv_crate`, `nom_capacity`, `v_max`, …), matching `battery_config_example.json`.
- ISU physical facts (verified): NMC 4.2/3.0 V; `nom_capacity = 0.28` Ah; measured normalized C/2 discharge ≈ 0.45 → `cap_rate = 0.45`; sign convention charge `I>0` / discharge `I<0` (matches pipeline, no negation); no temperature field (synthesize `T1 = 25.0`).
- Dataset root: `/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/` with `Release 1.0/` (238 cells) and `Release 2.0/` (13 cells); files `G<group>C<cell>.json`.
- Working path: `/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline`.
- Do **not** mirror code/doc edits into the data path.

---

### Task 1: ISU JSON reader + per-RPT frame builder

**Files:**
- Create: `src/download/build_bronze_isu.py`
- Test: `tests/test_build_bronze_isu.py`

**Interfaces:**
- Produces:
  - `load_isu_json(path: str) -> dict` — decodes the double-encoded JSON.
  - `SWEEPS: list[tuple[str, str]]` — the four `(qv_key, zustand_tag)` pairs.
  - `build_rpt_frame(cell: str, data: dict, r: int) -> pd.DataFrame | None` — one RPT's
    concatenated 4-sweep frame with columns `Zeit, Strom, Spannung, Zustand, T1, AhAkku,
    Prozedur, Ahjo_Test_ID`. Returns `None` if the RPT has no usable sweep.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_bronze_isu.py
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from download.build_bronze_isu import load_isu_json, build_rpt_frame, SWEEPS


def _synthetic_sweep(v_lo, v_hi, current, t0, n=5):
    """A tiny QV sweep field-set for one RPT."""
    ts = pd.date_range(t0, periods=n, freq="60s").astype(str).tolist()
    return {
        "Q": [list(np.linspace(0, 0.27, n))],
        "V": [list(np.linspace(v_lo, v_hi, n))],
        "t": [ts],
        "E": [list(np.linspace(0, 1.0, n))],
        "I": [[current] * n],
    }


def _synthetic_cell():
    return {
        "capacity_discharge_C_2": [0.27],
        "QV_charge_C_2":    _synthetic_sweep(3.0, 4.2,  0.12, "2021-06-19T16:00:00"),
        "QV_discharge_C_2": _synthetic_sweep(4.2, 3.0, -0.12, "2021-06-19T17:00:00"),
        "QV_charge_C_5":    _synthetic_sweep(3.0, 4.2,  0.05, "2021-06-19T18:00:00"),
        "QV_discharge_C_5": _synthetic_sweep(4.2, 3.0, -0.05, "2021-06-19T20:00:00"),
    }


def test_build_rpt_frame_columns_and_tags():
    frame = build_rpt_frame("G14C1", _synthetic_cell(), 0)
    assert frame is not None
    for col in ["Zeit", "Strom", "Spannung", "Zustand", "T1", "AhAkku",
                "Prozedur", "Ahjo_Test_ID"]:
        assert col in frame.columns, f"missing {col}"
    # four sweeps x 5 rows
    assert len(frame) == 20
    # unique Zustand per sweep
    assert set(frame["Zustand"]) == {"CHA_C2", "DCH_C2", "CHA_C5", "DCH_C5"}
    # synthesized constants
    assert (frame["T1"] == 25.0).all()
    assert frame["AhAkku"].isna().all()
    assert (frame["Prozedur"] == "ISU_RPT").all()
    # zero-padded RPT id so BM_Programm follows RPT order
    assert (frame["Ahjo_Test_ID"] == "G14C1_RPT000").all()
    # sign convention preserved: charge > 0, discharge < 0
    assert (frame.loc[frame["Zustand"] == "CHA_C2", "Strom"] > 0).all()
    assert (frame.loc[frame["Zustand"] == "DCH_C2", "Strom"] < 0).all()


if __name__ == "__main__":
    test_build_rpt_frame_columns_and_tags()
    print("OK test_build_rpt_frame_columns_and_tags")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_build_bronze_isu.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'download.build_bronze_isu'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# src/download/build_bronze_isu.py
"""
Adapter: ISU-ILCC per-cell JSON -> BRONZE_CU-shaped parquet.

Each ISU cell JSON is double-encoded (a JSON string containing JSON). Per cell it
holds N reference-performance-tests (RPTs); each RPT has four QV sweeps
(C/2 & C/5, charge & discharge) as time series {Q, V, t, E, I}. This adapter
reassembles the four sweeps of each RPT into one continuous frame and synthesizes
the columns the dismember contract (read_and_fix_format) needs, so the unchanged
main.py classifier path can process it.

Usage:
    cd src
    python download/build_bronze_isu.py /path/to/battery_config_ISU_linux.json
    python download/build_bronze_isu.py <cfg> --cells G14C1 G4C1 --overwrite
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.add_ah_throughput import add_ah_throughput

# (QV field key, unique Zustand tag). Unique tags make dismember cut a boundary on
# every Zustand change (with qocv_procedure_filter matching), so each RPT splits
# into four clean segments.
SWEEPS = [
    ("QV_charge_C_2", "CHA_C2"),
    ("QV_discharge_C_2", "DCH_C2"),
    ("QV_charge_C_5", "CHA_C5"),
    ("QV_discharge_C_5", "DCH_C5"),
]


def load_isu_json(path: str) -> dict:
    """Decode the double-encoded ISU JSON to a dict."""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, str):
        d = json.loads(d)
    return d


def build_rpt_frame(cell: str, data: dict, r: int):
    """One RPT -> a concatenated 4-sweep frame, or None if no usable sweep."""
    frames = []
    for key, zustand in SWEEPS:
        qv = data.get(key)
        if qv is None or r >= len(qv.get("V", [])):
            continue
        t = pd.to_datetime(qv["t"][r])
        current = np.asarray(qv["I"][r], dtype=float)
        voltage = np.asarray(qv["V"][r], dtype=float)
        n = min(len(t), len(current), len(voltage))
        if n == 0:
            continue
        frames.append(pd.DataFrame({
            "Zeit": t[:n],
            "Strom": current[:n],
            "Spannung": voltage[:n],
            "Zustand": zustand,
        }))
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame["T1"] = 25.0
    frame["AhAkku"] = np.nan
    frame["Prozedur"] = "ISU_RPT"
    frame["Ahjo_Test_ID"] = f"{cell}_RPT{r:03d}"
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_build_bronze_isu.py`
Expected: `OK test_build_rpt_frame_columns_and_tags`

- [ ] **Step 5: Commit**

```bash
cd /home/ann/Documents/Project_METAbatt
git add src/download/build_bronze_isu.py tests/test_build_bronze_isu.py
git commit -m "feat: ISU-ILCC per-RPT frame builder + JSON reader"
```

---

### Task 2: Cell BRONZE assembly (sort + Ah throughput)

**Files:**
- Modify: `src/download/build_bronze_isu.py`
- Test: `tests/test_build_bronze_isu.py`

**Interfaces:**
- Consumes: `build_rpt_frame`, `load_isu_json` from Task 1.
- Produces: `build_cell_bronze(cell: str, data: dict) -> pd.DataFrame | None` — all RPTs
  concatenated, sorted by `Zeit`, with an `Ah_throughput` column added. Returns `None`
  if the cell has no usable RPT.

- [ ] **Step 1: Write the failing test** (append to `tests/test_build_bronze_isu.py`)

```python
from download.build_bronze_isu import build_cell_bronze


def test_build_cell_bronze_orders_and_integrates():
    cell = _synthetic_cell()
    # add a second RPT a week later
    cell["capacity_discharge_C_2"] = [0.27, 0.26]
    for key, _ in SWEEPS:
        base = cell[key]
        cell[key] = {
            k: [base[k][0], base[k][0]] for k in ("Q", "V", "t", "E", "I")
        }
    bronze = build_cell_bronze("G14C1", cell)
    assert bronze is not None
    # two RPTs -> two distinct Ahjo_Test_IDs
    assert set(bronze["Ahjo_Test_ID"]) == {"G14C1_RPT000", "G14C1_RPT001"}
    # sorted ascending by Zeit
    assert bronze["Zeit"].is_monotonic_increasing
    # Ah throughput present and non-decreasing
    assert "Ah_throughput" in bronze.columns
    assert (bronze["Ah_throughput"].diff().dropna() >= -1e-9).all()
    # helper cols removed
    assert "Time_UTC" not in bronze.columns
    assert "Current" not in bronze.columns


if __name__ == "__main__":
    test_build_rpt_frame_columns_and_tags()
    print("OK test_build_rpt_frame_columns_and_tags")
    test_build_cell_bronze_orders_and_integrates()
    print("OK test_build_cell_bronze_orders_and_integrates")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_build_bronze_isu.py`
Expected: FAIL — `ImportError: cannot import name 'build_cell_bronze'`.

- [ ] **Step 3: Write minimal implementation** (add to `src/download/build_bronze_isu.py`)

```python
def build_cell_bronze(cell: str, data: dict):
    """All RPTs -> one BRONZE_CU frame (sorted, Ah-integrated), or None."""
    n_rpt = len(data.get("capacity_discharge_C_2", []))
    frames = [build_rpt_frame(cell, data, r) for r in range(n_rpt)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    bronze = pd.concat(frames, ignore_index=True)
    bronze = bronze.sort_values("Zeit").reset_index(drop=True)

    # Ah throughput over the full cell timeline. add_ah_throughput needs
    # Time_UTC + Current; drop the helpers afterwards.
    bronze["Current"] = bronze["Strom"]
    bronze["Time_UTC"] = bronze["Zeit"]
    if bronze["Time_UTC"].dt.tz is None:
        bronze["Time_UTC"] = bronze["Time_UTC"].dt.tz_localize("UTC")
    bronze = add_ah_throughput(bronze)
    bronze = bronze.drop(columns=["Current", "Time_UTC"])
    return bronze
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_build_bronze_isu.py`
Expected: both `OK …` lines.

- [ ] **Step 5: Commit**

```bash
cd /home/ann/Documents/Project_METAbatt
git add src/download/build_bronze_isu.py tests/test_build_bronze_isu.py
git commit -m "feat: assemble ISU cell BRONZE_CU with Ah throughput"
```

---

### Task 3: CLI wrapper + file discovery + real-cell smoke test

**Files:**
- Modify: `src/download/build_bronze_isu.py`

**Interfaces:**
- Consumes: `build_cell_bronze`, `load_isu_json`.
- Produces:
  - `discover_isu_files(source_dirs: list[str], target_cells: list[str] | None) -> list[tuple[str, str]]`
    — `(cell, json_path)` pairs; `cell` is the filename stem (`G14C1`).
  - `run(cfg: dict, target_cells=None, overwrite=False) -> None` — writes
    `<working_path>/BRONZE_CU/<cell>.parquet` per cell.
  - `__main__` CLI: positional `config`, `--cells`, `--overwrite`.

- [ ] **Step 1: Add discovery + run + CLI** (append to `src/download/build_bronze_isu.py`)

```python
def discover_isu_files(source_dirs, target_cells=None):
    """Return (cell_stem, path) for every G*C*.json under the source dirs."""
    wanted = set(target_cells) if target_cells else None
    out = []
    for d in source_dirs:
        if not os.path.isdir(d):
            print(f"  source dir not found: {d}")
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            cell = name[:-len(".json")]
            if wanted is not None and cell not in wanted:
                continue
            out.append((cell, os.path.join(d, name)))
    return out


def run(cfg, target_cells=None, overwrite=False):
    working_path = cfg["working_path"]
    source_dirs = cfg["isu_source_dirs"]
    out_dir = os.path.join(working_path, "BRONZE_CU")
    os.makedirs(out_dir, exist_ok=True)

    files = discover_isu_files(source_dirs, target_cells)
    if not files:
        print("No ISU JSON files found.")
        return

    for cell, path in files:
        out = os.path.join(out_dir, f"{cell}.parquet")
        if os.path.exists(out) and not overwrite:
            print(f"{cell} - BRONZE_CU exists, skipping.")
            continue
        try:
            data = load_isu_json(path)
        except Exception as e:  # malformed / non-decodable JSON
            print(f"{cell} - JSON decode failed: {e}")
            continue
        bronze = build_cell_bronze(cell, data)
        if bronze is None:
            print(f"{cell} - no usable RPT data.")
            continue
        bronze.to_parquet(out, index=False)
        n_rpt = bronze["Ahjo_Test_ID"].nunique()
        print(f"{cell} - wrote {len(bronze)} rows, {n_rpt} RPTs -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BRONZE_CU from ISU-ILCC JSON")
    parser.add_argument("config", help="Path to ISU battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cell stems")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    run(cfg, target_cells=args.cells, overwrite=args.overwrite)
```

- [ ] **Step 2: Write a temporary smoke config for one cell**

```bash
cd /home/ann/Documents/Project_METAbatt
cat > /tmp/isu_smoke.json <<'JSON'
{
  "working_path": "/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline",
  "isu_source_dirs": [
    "/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/Release 1.0",
    "/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/Release 2.0"
  ]
}
JSON
```

- [ ] **Step 3: Run the adapter on one real cell**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python download/build_bronze_isu.py /tmp/isu_smoke.json --cells G14C1 --overwrite
```
Expected: `G14C1 - wrote <N> rows, <M> RPTs -> …/ISU_pipeline/BRONZE_CU/G14C1.parquet`

- [ ] **Step 4: Verify the written parquet contract**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python - <<'PY'
import pandas as pd
p="/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/BRONZE_CU/G14C1.parquet"
df=pd.read_parquet(p)
need={"Zeit","Strom","Spannung","T1","Zustand","Prozedur","AhAkku","Ahjo_Test_ID","Ah_throughput"}
assert need <= set(df.columns), sorted(need - set(df.columns))
assert df["Zeit"].is_monotonic_increasing
assert (df.loc[df["Zustand"]=="DCH_C2","Strom"]<0).all()
assert (df.loc[df["Zustand"]=="CHA_C2","Strom"]>0).all()
# BM_Programm order will follow the zero-padded id
ids=sorted(df["Ahjo_Test_ID"].unique())
assert ids==sorted(ids), "ids not lexically sorted"
print("rows", len(df), "RPTs", df["Ahjo_Test_ID"].nunique(), "OK")
PY
```
Expected: `rows … RPTs … OK`

- [ ] **Step 5: Commit**

```bash
cd /home/ann/Documents/Project_METAbatt
git add src/download/build_bronze_isu.py
git commit -m "feat: ISU BRONZE_CU CLI + file discovery"
```

---

### Task 4: ISU battery config + classifier model placement

**Files:**
- Create: `battery_config_ISU_linux.json` (repo root)

**Interfaces:**
- Consumes: the classifier model files at
  `/home/ann/media/Sciebo/sciebo/vtc_classifier_20260731T135816.joblib` + `_meta.json`.
- Produces: a config that drives `main.py`'s classifier path on ISU BRONZE_CU.

- [ ] **Step 1: Copy the classifier model into the working path**

Run:
```bash
MODELDIR="/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/60_classifier/models"
mkdir -p "$MODELDIR"
cp "/home/ann/media/Sciebo/sciebo/vtc_classifier_20260731T135816.joblib" "$MODELDIR/"
cp "/home/ann/media/Sciebo/sciebo/vtc_classifier_20260731T135816_meta.json" "$MODELDIR/"
ls -la "$MODELDIR"
```
Expected: both files listed.

- [ ] **Step 2: Write the config** (`battery_config_ISU_linux.json`)

```json
{
    "working_path": "/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline",
    "type_cell": "ISU",

    "isu_source_dirs": [
        "/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/Release 1.0",
        "/home/ann/Documents/Data_Metabatt/comparison data/data_ISU-ILCC/Release 2.0"
    ],

    "nom_capacity": 0.28,
    "v_min": 3.0,
    "v_max": 4.2,
    "v_nom": 3.7,

    "qocv_crate": 0.05,
    "cap_type": "CC",
    "cap_rate": 0.45,
    "cap_temp": 25,

    "target_pulse_duration": 20,
    "pulse_type": 1,
    "pulse_target_unit": "Resistance",
    "pulse_keep_per_group": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
    "pulse_group_by": "BM_Programm",
    "pulse_step_threshold": null,

    "tolerances": {
        "pulse_duration_tolerance": 1.08,
        "pulse_cluster_tolerance": 3.0,
        "restore_current_tolerance": 0.05,
        "qocv_current_tolerance": 0.01,
        "qocv_duration_tolerance": 1.2,
        "qocv_std_tolerance": 0.002
    },

    "min_rows": 20,
    "pau_duration": 9.9,
    "procedure_filter": "ISU_RPT",
    "qocv_procedure_filter": "ISU_RPT",

    "export_gold": true,
    "export_pulse": false,
    "export_qocv": false,
    "running_window_days": 2,

    "feature_columns": ["Voltage", "Current", "Temperature"],

    "hdbscan_para_layer_1": {
        "min_samples": 8,
        "cluster_selection_epsilon": 0.3,
        "allow_single_cluster": false
    },
    "hdbscan_para_layer_2": {
        "min_cluster_size": 3,
        "min_samples": 3,
        "cluster_selection_epsilon": 0.001,
        "allow_single_cluster": false
    },

    "classifier_model_path": "/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/60_classifier/models/vtc_classifier_20260731T135816.joblib",
    "classifier_meta_path": "/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/60_classifier/models/vtc_classifier_20260731T135816_meta.json",

    "download_from": "local",
    "upload_to": "local",
    "minio_endpoint": "unused",
    "minio_access_key": "unused",
    "minio_secret_key": "unused",
    "bucket_name": "unused",
    "minio_prefix": "unused"
}
```

- [ ] **Step 3: Validate the config loads and keys are present**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python - <<'PY'
import json, os
c=json.load(open("battery_config_ISU_linux.json"))
for k in ["working_path","nom_capacity","cap_rate","qocv_crate","procedure_filter",
          "qocv_procedure_filter","classifier_model_path","classifier_meta_path",
          "isu_source_dirs"]:
    assert k in c, k
assert os.path.exists(c["classifier_model_path"]), "model missing"
assert os.path.exists(c["classifier_meta_path"]), "meta missing"
print("config OK, cap_rate", c["cap_rate"])
PY
```
Expected: `config OK, cap_rate 0.45`

- [ ] **Step 4: Commit**

```bash
cd /home/ann/Documents/Project_METAbatt
git add battery_config_ISU_linux.json
git commit -m "config: ISU-ILCC battery config for classifier pipeline"
```

---

### Task 5: 3-cell pilot pipeline run

**Files:** none (integration run of existing `main.py`).

**Interfaces:**
- Consumes: BRONZE_CU parquets from Task 3's adapter + config from Task 4.
- Produces: `<working_path>/40_capacity_monitore/<cell>_capacity.csv` for the pilot cells,
  and `<working_path>/GOLD/<cell>.parquet`.

- [ ] **Step 1: Build BRONZE_CU for three pilot cells**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python download/build_bronze_isu.py ../battery_config_ISU_linux.json \
  --cells G4C1 G14C1 G1C1 --overwrite
```
Expected: three `… wrote … rows …` lines. (G4C1 has ~31 RPTs, G1C1/G14C1 fewer.)

- [ ] **Step 2: Run the pipeline (classifier path) on the pilot cells**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python main.py ../battery_config_ISU_linux.json --cells G4C1 G14C1 G1C1 --overwrite
```
Expected: log lines including `classifying segments via …vtc_classifier…joblib` and no
tracebacks. Cells with no detected CAP log a `no proper checkup detected … skipping GOLD`
warning (acceptable, not a failure).

- [ ] **Step 3: Verify capacity CSVs were produced and are sane**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python - <<'PY'
import pandas as pd, glob, os
d="/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/40_capacity_monitore"
files=sorted(glob.glob(os.path.join(d,"*_capacity.csv")))
assert files, "no capacity CSVs written"
for f in files:
    df=pd.read_csv(f)
    print(os.path.basename(f), "rows", len(df),
          "Capacity_py range", round(df["Capacity_py"].min(),3), "-", round(df["Capacity_py"].max(),3))
    assert (df["Capacity_py"]>0).all()
    # ISU cells are ~0.27 Ah; sanity bound
    assert df["Capacity_py"].max() < 0.5, "capacity implausibly large"
print("pilot capacity CSVs OK")
PY
```
Expected: per-cell row counts + `pilot capacity CSVs OK`. **If a pilot cell produced no CSV
(no CAP detected), STOP and report** — the classifier may not be locking onto the C/2
discharge; revisit `cap_rate` / inspect the segment labels before continuing.

- [ ] **Step 4: Inspect what label the C/5 sweep received (informational)**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python - <<'PY'
import pandas as pd, glob, os
d="/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/with_features_post_labeled"
alt="/home/ann/Documents/Data_Metabatt/comparison data/ISU_pipeline/60_classifier/with_features_post_labeled"
d = alt if os.path.isdir(alt) else d
for f in sorted(glob.glob(os.path.join(d,"*.csv")))[:1]:
    df=pd.read_csv(f)
    cols=[c for c in ["ID","target","cluster_id","abs_Current_mean"] if c in df.columns]
    print(os.path.basename(f))
    print(df[cols].to_string())
PY
```
Expected: prints per-segment labels. Note which label the C/5 (`abs_Current_mean` ≈ 0.18
normalized) segments got — for the write-up. No assertion.

- [ ] **Step 5: No commit** (pilot produces data-path artifacts only, which are not tracked).

---

### Task 6: ISU capacity validation script

**Files:**
- Create: `src/evaluation/validate_isu.py`
- Test: `tests/test_validate_isu.py`

**Interfaces:**
- Consumes: pipeline `40_capacity_monitore/<cell>_capacity.csv`; ISU JSON via
  `download.build_bronze_isu.load_isu_json`.
- Produces:
  - `match_caps_to_rpts(cap_df: pd.DataFrame, rpt_windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[int | None]`
    — for each capacity row (by `CAP_start_time`), the RPT index whose C/2-discharge time
    window contains it, else `None`.
  - `build_validation_table(cfg: dict, cells: list[str] | None) -> pd.DataFrame`
    — columns `cell, rpt, Capacity_py, cap_truth_Ah, abs_err_Ah, pct_err`.
  - `main`/`run(cfg, cells, out_dir)` writing `isu_capacity_validation.csv` + `.png` and
    printing MAPE/bias.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_isu.py
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.validate_isu import match_caps_to_rpts


def test_match_caps_to_rpts_by_window():
    windows = [
        (pd.Timestamp("2021-06-20 00:00"), pd.Timestamp("2021-06-20 03:00")),
        (pd.Timestamp("2021-06-28 00:00"), pd.Timestamp("2021-06-28 03:00")),
        (pd.Timestamp("2021-07-06 00:00"), pd.Timestamp("2021-07-06 03:00")),
    ]
    # CAP rows land in RPT 0 and RPT 2 (RPT 1 missing -> unmatched middle stays put)
    cap_df = pd.DataFrame({
        "CAP_start_time": [
            pd.Timestamp("2021-06-20 01:00"),
            pd.Timestamp("2021-07-06 01:30"),
        ]
    })
    assert match_caps_to_rpts(cap_df, windows) == [0, 2]


if __name__ == "__main__":
    test_match_caps_to_rpts_by_window()
    print("OK test_match_caps_to_rpts_by_window")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_validate_isu.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation.validate_isu'`.

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/validate_isu.py
"""
Validate the ISU-ILCC pipeline run: compare pipeline-extracted Capacity_py per RPT
against ISU's own capacity_discharge_C_2 ground truth.

Match key: each capacity row's CAP_start_time is placed into the RPT whose
C/2-discharge time window contains it (robust to a mid-run RPT yielding no CAP).

Usage:
    cd src
    python -m evaluation.validate_isu /path/to/battery_config_ISU_linux.json
    python -m evaluation.validate_isu <cfg> --cells G4C1 -o /custom/out
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from download.build_bronze_isu import load_isu_json, discover_isu_files


def _rpt_windows(data: dict):
    """C/2-discharge (min, max) timestamp per RPT."""
    qv = data["QV_discharge_C_2"]
    windows = []
    for r in range(len(qv["t"])):
        ts = pd.to_datetime(qv["t"][r])
        windows.append((ts.min(), ts.max()))
    return windows


def match_caps_to_rpts(cap_df: pd.DataFrame, rpt_windows):
    """For each capacity row (CAP_start_time), the RPT index whose window contains
    it, else None."""
    starts = pd.to_datetime(cap_df["CAP_start_time"])
    if starts.dt.tz is not None:
        starts = starts.dt.tz_localize(None)
    out = []
    for t in starts:
        hit = None
        for r, (lo, hi) in enumerate(rpt_windows):
            if lo <= t <= hi:
                hit = r
                break
        out.append(hit)
    return out


def build_validation_table(cfg: dict, cells=None) -> pd.DataFrame:
    working_path = cfg["working_path"]
    cap_dir = os.path.join(working_path, "40_capacity_monitore")
    file_map = dict(discover_isu_files(cfg["isu_source_dirs"], cells))

    rows = []
    for cell, path in sorted(file_map.items()):
        cap_csv = os.path.join(cap_dir, f"{cell}_capacity.csv")
        if not os.path.exists(cap_csv):
            continue
        cap_df = pd.read_csv(cap_csv)
        if "CAP_start_time" not in cap_df.columns or cap_df.empty:
            continue
        data = load_isu_json(path)
        truth = data["capacity_discharge_C_2"]
        windows = _rpt_windows(data)
        matched = match_caps_to_rpts(cap_df, windows)
        for i, r in enumerate(matched):
            if r is None or r >= len(truth):
                continue
            cap_py = float(cap_df["Capacity_py"].iloc[i])
            cap_truth = float(truth[r])
            rows.append({
                "cell": cell,
                "rpt": r,
                "Capacity_py": cap_py,
                "cap_truth_Ah": cap_truth,
                "abs_err_Ah": abs(cap_py - cap_truth),
                "pct_err": 100.0 * (cap_py - cap_truth) / cap_truth,
            })
    return pd.DataFrame(rows)


def run(cfg: dict, cells=None, out_dir=None) -> pd.DataFrame:
    working_path = cfg["working_path"]
    out_dir = out_dir or os.path.join(working_path, "50_evaluation")
    os.makedirs(out_dir, exist_ok=True)

    table = build_validation_table(cfg, cells)
    csv_path = os.path.join(out_dir, "isu_capacity_validation.csv")
    table.to_csv(csv_path, index=False)

    if table.empty:
        print("No matched (cell, RPT) capacities — nothing to score.")
        return table

    mape = table["pct_err"].abs().mean()
    bias = table["pct_err"].mean()
    print(f"matched pairs: {len(table)}  cells: {table['cell'].nunique()}")
    print(f"MAPE: {mape:.2f}%   bias: {bias:+.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(table["cap_truth_Ah"], table["Capacity_py"], s=12, alpha=0.6)
    lim = [0, max(table["cap_truth_Ah"].max(), table["Capacity_py"].max()) * 1.05]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlabel("ISU capacity_discharge_C_2 (Ah)")
    ax1.set_ylabel("pipeline Capacity_py (Ah)")
    ax1.set_title(f"ISU CAP parity (MAPE {mape:.2f}%)")
    for cell, g in table.groupby("cell"):
        ax2.plot(g["rpt"], g["Capacity_py"], marker="o", ms=3, label=cell)
    ax2.set_xlabel("RPT index")
    ax2.set_ylabel("pipeline Capacity_py (Ah)")
    ax2.set_title("SOH timeline (pipeline)")
    if table["cell"].nunique() <= 12:
        ax2.legend(fontsize=7)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "isu_capacity_validation.png")
    fig.savefig(png_path, dpi=120)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ISU pipeline capacities")
    parser.add_argument("config")
    parser.add_argument("--cells", nargs="*")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    run(cfg, cells=args.cells, out_dir=args.out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python tests/test_validate_isu.py`
Expected: `OK test_match_caps_to_rpts_by_window`

- [ ] **Step 5: Run validation on the pilot cells**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python -m evaluation.validate_isu ../battery_config_ISU_linux.json --cells G4C1 G14C1 G1C1
```
Expected: `matched pairs: …`, a `MAPE: …%` line, and two `wrote …` lines. **Target MAPE
< ~5%.** If MAPE is large, STOP and report — likely the CAP segment is picking up the wrong
sweep or a partial discharge; inspect `isu_capacity_validation.csv` and the segment labels
from Task 5 Step 4.

- [ ] **Step 6: Commit**

```bash
cd /home/ann/Documents/Project_METAbatt
git add src/evaluation/validate_isu.py tests/test_validate_isu.py
git commit -m "feat: ISU capacity validation vs ground truth"
```

---

### Task 7: Full-dataset build, run, and validation

**Files:** none (bulk run of the tools built above).

**Interfaces:** consumes everything from Tasks 1–6.

- [ ] **Step 1: Build BRONZE_CU for all cells**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python download/build_bronze_isu.py ../battery_config_ISU_linux.json
```
Expected: ~251 `… wrote …` / `… skipping` lines, no tracebacks.

- [ ] **Step 2: Run the pipeline on all cells**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python main.py ../battery_config_ISU_linux.json
```
Expected: per-cell classifier logs; some `no proper checkup detected` warnings are
acceptable. Note any cell that errors with a traceback — that's a real failure to report.

- [ ] **Step 3: Run full validation**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt/src && source ../.venv/bin/activate
python -m evaluation.validate_isu ../battery_config_ISU_linux.json
```
Expected: `matched pairs`, `MAPE`, `bias`, and the two `wrote …` outputs under
`50_evaluation/`.

- [ ] **Step 4: Summarize coverage and accuracy**

Run:
```bash
cd /home/ann/Documents/Project_METAbatt && source .venv/bin/activate && python - <<'PY'
import pandas as pd, json
c=json.load(open("battery_config_ISU_linux.json"))
t=pd.read_csv(f"{c['working_path']}/50_evaluation/isu_capacity_validation.csv")
print("cells validated:", t["cell"].nunique())
print("matched (cell,RPT) pairs:", len(t))
print("MAPE %.2f%%  bias %+.2f%%" % (t["pct_err"].abs().mean(), t["pct_err"].mean()))
print("worst 5 by |pct_err|:")
print(t.reindex(t["pct_err"].abs().sort_values(ascending=False).index).head().to_string())
PY
```
Expected: a coverage + accuracy summary for the paper. **This is the deliverable** —
report the numbers back.

- [ ] **Step 5: No code commit** (bulk run produces data-path artifacts only). Optionally
      record the headline numbers in the design/report docs in a follow-up.

---

## Self-Review notes

- **Spec coverage:** adapter (Task 1–3), config + model placement (Task 4), pilot
  (Task 5), validation script + deliverable (Task 6), full rollout (Task 7). Out-of-scope
  items (aging matrix, MinIO, HDBSCAN comparison) are intentionally absent.
- **CAP-lock risk** is gated: Task 5 Step 3 halts if a pilot cell yields no CAP; Task 6
  Step 5 halts on high MAPE — both point at `cap_rate` / segment labels as the knob.
- **Type consistency:** `build_rpt_frame`/`build_cell_bronze`/`discover_isu_files`/
  `load_isu_json`/`SWEEPS` names are used identically across tasks and by the validator.
