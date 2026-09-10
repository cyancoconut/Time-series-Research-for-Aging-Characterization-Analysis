# Initial Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a per-cell BOL characterization artefact set (pulse/EIS/qOCV bundles, fitted parameters, plots) from parametrization test files, via their own BRONZE build and a separate pipeline entry point, without disturbing the check-up (CU) aging outputs.

**Architecture:** A frozen `RunContext` dataclass carries the three things that differ between a CU run and a characterization run — which BRONZE layer to read, which config key names the procedure filter, and which output root to write under. It is threaded explicitly through `run_pipeline` → `_process_cell` → `_process_cell_inner` → the export modules, defaulting to the CU context so existing behaviour is bit-for-bit unchanged. Fitting is a separate CLI that imports the existing `src/analysis/` fitters.

**Tech Stack:** Python 3, pandas, pyarrow, minio, scipy/numpy (fitters), matplotlib (Agg), customtkinter (UI).

## Global Constraints

- Everything runs from `src/` — imports are relative to `src/`. Venv: `source .venv/bin/activate` from project root.
- Branch is `feat/initial-characterization`, already created. **Never commit to `J8005_BMWK_METABatt`.** PR at the end.
- **Do not add a `Co-Authored-By` trailer to commit messages.**
- The repo has **no test suite and no test framework**. Verification steps in this plan are self-contained `python - <<'PY'` assertion snippets and `--help` smoke runs that need no lab data. Data-dependent verification is explicitly the user's, deferred.
- Every signature change in this plan is **additive with a default equal to today's behaviour**. A CU run must produce byte-identical output afterwards.
- The MinIO prefix tag is `TRACY` after Task 1; reads fall back to `10_TRACY`.
- Characterization output root: `10_initial_characterization/<cell_stem>/`, MinIO **untagged**.
- EIS/pulse model choices are fixed defaults, not config keys: 2RC (pulse), 2×ZARC + generalized Warburg (EIS).
- Spec: `docs/superpowers/specs/2026-08-21-initial-characterization-design.md`.

---

## File Structure

**Created:**
- `src/util/run_context.py` — the `RunContext` dataclass + the two module-level contexts. One responsibility: describe what varies between run flavours.
- `src/download/build_bronze_para.py` — thin CLI over the parameterized bronze builder.
- `src/characterize/__init__.py`
- `src/characterize/main_para.py` — thin wrapper over `main.run_pipeline` with the characterization context.
- `src/characterize/fit_characterization.py` — fit + plot CLI over the exported bundles.

**Modified:**
- `src/util/io_router.py` — `TRACY` tag + legacy fallback; `layer=` on the bronze helpers; `root=` on the export/gold/x_silver key helpers.
- `src/main.py` — thread `run_ctx` through `run_pipeline`, `_process_cell`, `_process_cell_inner`, `_build_paths`, `_write_gold`, `_write_x_silver`.
- `src/output/export_pulse.py`, `export_qocv.py`, `export_eis.py`, `export_capacity.py` — accept `run_ctx`, pass `root=` to the key helper.
- `src/download/build_bronze_cu_with_ah.py` — `layer` + `filter_key` parameters.
- `src/evaluation/compare_labels.py` — tagged dir via the io_router helper.
- `src/pipeline_ui.py` — Tab 7.
- `CLAUDE.md`, `README.md` — docs.

---

### Task 1: TRACY rename with legacy read-fallback

**Files:**
- Modify: `src/util/io_router.py:8` (docstring), `:20` (tag), `:60-68` (`list_gold_cells`), `:160-164` (`open_gold_range`), `:274-287` (`list_csv_objects`), `:312-325` (`list_x_silver_cells`), `:328-337` (`fetch_x_silver_bytes`)
- Modify: `src/evaluation/compare_labels.py:47`

**Interfaces:**
- Produces: `io_router.UPLOAD_PREFIX_TAG == "TRACY"`, `io_router.LEGACY_PREFIX_TAG == "10_TRACY"`, `io_router.tagged_rel(rel: str) -> str` returning `f"TRACY/{rel}"`, and `io_router.resolve_tagged_rel(client, cfg, rel: str) -> str` which returns the legacy rel when the new prefix lists nothing.

- [ ] **Step 1: Rename the tag and add the legacy constant**

In `src/util/io_router.py`, replace line 20:

```python
UPLOAD_PREFIX_TAG = "TRACY"

#: Pre-rename tag. Objects uploaded before the rename still live under this
#: prefix; readers fall back to it when the new prefix is empty, so nothing
#: already on MinIO becomes unreachable. Writes always use UPLOAD_PREFIX_TAG.
LEGACY_PREFIX_TAG = "10_TRACY"
```

And update the module docstring line 8 from `<bucket>/<minio_prefix>/10_TRACY/<layer>/...` to `<bucket>/<minio_prefix>/TRACY/<layer>/...`.

- [ ] **Step 2: Add the two rel-path helpers**

Insert after the `LEGACY_PREFIX_TAG` definition:

```python
def tagged_rel(rel: str) -> str:
    """Relative object dir under the current tag, e.g. `TRACY/GOLD`."""
    return f"{UPLOAD_PREFIX_TAG}/{rel.strip('/')}"


def _prefix_has_objects(client: Minio, cfg: dict, rel: str) -> bool:
    base = f"{cfg['minio_prefix']}/{rel.strip('/')}/"
    objs = client.list_objects(cfg["bucket_name"], prefix=base, recursive=False)
    return any(True for _ in objs)


def resolve_tagged_rel(client: Minio, cfg: dict, rel: str) -> str:
    """`TRACY/<rel>` if anything is there, else the legacy `10_TRACY/<rel>`.

    Read-side only. Lets a bucket written before the rename keep working
    without a bulk server-side copy.
    """
    current = tagged_rel(rel)
    if client is None:
        return current
    if _prefix_has_objects(client, cfg, current):
        return current
    legacy = f"{LEGACY_PREFIX_TAG}/{rel.strip('/')}"
    if _prefix_has_objects(client, cfg, legacy):
        logging.info("MinIO: %s empty, falling back to %s", current, legacy)
        return legacy
    return current
```

Add `import logging` to the imports at the top of the file (after `import io`).

- [ ] **Step 3: Route the tagged readers through the resolver**

`list_gold_cells` (line 60) — replace its `base` line:

```python
def list_gold_cells(client: Minio, cfg: dict) -> list:
    bucket = cfg["bucket_name"]
    rel = resolve_tagged_rel(client, cfg, "GOLD")
    base = f"{cfg['minio_prefix']}/{rel}/"
```

`list_x_silver_cells` (line 312) — same treatment:

```python
    rel = resolve_tagged_rel(client, cfg, "with_features_post_labeled")
    base = f"{cfg['minio_prefix']}/{rel}/"
```

`fetch_x_silver_bytes` (line 328):

```python
    rel = resolve_tagged_rel(client, cfg, "with_features_post_labeled")
    key = f"{cfg['minio_prefix']}/{rel}/{name}"
```

`open_gold_range` (line 160):

```python
    rel = resolve_tagged_rel(client, cfg, "GOLD")
    key = f"{cfg['minio_prefix']}/{rel}/{cell}"
```

`list_csv_objects` and `fetch_csv_object` take an explicit `rel_dir` from their callers, so they need no change here — the caller in Task 1 Step 4 supplies the resolved dir.

- [ ] **Step 4: Fix the compare_labels caller**

`src/evaluation/compare_labels.py:47` currently reads:

```python
_HDBSCAN_REL = f"{io_router.UPLOAD_PREFIX_TAG}/with_features_post_labeled"  # 10_TRACY/...
```

Replace with a resolved lookup at call time. Change the constant to the bare rel and resolve where the MinIO client exists:

```python
_HDBSCAN_REL_BASE = "with_features_post_labeled"  # resolved to TRACY/ or 10_TRACY/
```

Then find every use of `_HDBSCAN_REL` in the file (`grep -n "_HDBSCAN_REL" src/evaluation/compare_labels.py`) and, in the MinIO branch, replace it with:

```python
hdbscan_rel = io_router.resolve_tagged_rel(client, cfg, _HDBSCAN_REL_BASE)
```

using `hdbscan_rel` in the `list_csv_objects` / `fetch_csv_object` calls.

- [ ] **Step 5: Verify the pure helpers**

Run from `src/`:

```bash
python - <<'PY'
from util import io_router as r
assert r.UPLOAD_PREFIX_TAG == "TRACY"
assert r.LEGACY_PREFIX_TAG == "10_TRACY"
assert r.tagged_rel("GOLD") == "TRACY/GOLD"
assert r.tagged_rel("/with_features_post_labeled/") == "TRACY/with_features_post_labeled"
# No client -> current tag, no network touched.
assert r.resolve_tagged_rel(None, {"minio_prefix": "p", "bucket_name": "b"}, "GOLD") == "TRACY/GOLD"
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 6: Verify compare_labels still imports**

```bash
python -c "import evaluation.compare_labels; print('OK')"
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/util/io_router.py src/evaluation/compare_labels.py
git commit -m "refactor(io): rename MinIO prefix tag 10_TRACY -> TRACY

Reads fall back to the legacy 10_TRACY prefix when the new one is empty,
so objects uploaded before the rename stay reachable without a bulk copy.
Writes always use the new tag."
```

---

### Task 2: RunContext and root/layer-aware object keys

**Files:**
- Create: `src/util/run_context.py`
- Modify: `src/util/io_router.py:49-57` (`list_bronze_cells`), `:167-176` (`open_bronze_range`), `:183-218` (`bronze_object_key`, `bronze_exists_on_minio`, `fetch_bronze`), `:254-255` (`gold_object_key`), `:302-309` (`x_silver_object_key`), `:340-357` (the four export key helpers)

**Interfaces:**
- Produces: `run_context.RunContext(bronze_layer, procedure_filter_key, export_root_prefix, force_exports)` with method `export_root(cell) -> str | None`; module constants `run_context.CU` and `run_context.CHARACTERIZATION`.
- Produces: `io_router.bronze_object_key(cell, layer="BRONZE_CU")`, `bronze_exists_on_minio(client, cfg, cell, layer="BRONZE_CU")`, `fetch_bronze(client, cfg, cell, layer="BRONZE_CU")`, `open_bronze_range(client, cfg, cell, layer="BRONZE_CU")`, `list_bronze_cells(client, cfg, layer="BRONZE_CU")`, `gold_object_key(cell, root=None)`, `x_silver_object_key(cell, classifier=False, root=None)`, `export_pulse_object_key(cell, filename, root=None)` and the `qocv` / `eis` / `capacity` counterparts.

- [ ] **Step 1: Write `src/util/run_context.py`**

```python
"""What varies between a check-up run and a characterization run.

The pipeline in :mod:`main` is shared. Three things differ:

* which BRONZE layer supplies the payload (``BRONZE_CU`` vs ``BRONZE_PARA``),
* which config key names the procedure filter that selects the test files,
* where every artefact is written.

They travel together as one explicit parameter rather than as config keys, so
any function's output destination is readable from its own signature.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    #: BRONZE layer folder / object prefix the run reads.
    bronze_layer: str = "BRONZE_CU"
    #: Battery-config key holding the procedure filter for this run.
    procedure_filter_key: str = "procedure_filter"
    #: When set, every artefact goes under `<prefix>/<cell_stem>/` instead of
    #: the shared GOLD/20_/25_/30_/40_ folders. MinIO keys are untagged.
    export_root_prefix: str | None = None
    #: Run the pulse/EIS/qOCV exports regardless of the config's export_* flags.
    force_exports: bool = False

    def export_root(self, cell: str) -> str | None:
        """Per-cell output root as a POSIX-style relative path, or None."""
        if not self.export_root_prefix:
            return None
        return f"{self.export_root_prefix}/{cell.split('.')[0]}"


#: The check-up pipeline — today's behaviour in every respect.
CU = RunContext()

#: Initial characterization (BOL parametrization).
CHARACTERIZATION = RunContext(
    bronze_layer="BRONZE_PARA",
    procedure_filter_key="para_procedure_filter",
    export_root_prefix="10_initial_characterization",
    force_exports=True,
)
```

- [ ] **Step 2: Verify the dataclass**

```bash
python - <<'PY'
from util.run_context import CU, CHARACTERIZATION, RunContext
assert CU.bronze_layer == "BRONZE_CU"
assert CU.procedure_filter_key == "procedure_filter"
assert CU.export_root("cell01.parquet") is None
assert CU.force_exports is False
assert CHARACTERIZATION.bronze_layer == "BRONZE_PARA"
assert CHARACTERIZATION.export_root("VTC6_003.parquet") == "10_initial_characterization/VTC6_003"
assert CHARACTERIZATION.export_root("VTC6_003") == "10_initial_characterization/VTC6_003"
try:
    CU.bronze_layer = "x"; raise SystemExit("frozen dataclass should reject writes")
except Exception:
    pass
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 3: Add `layer=` to the bronze helpers in io_router**

Replace the five functions with these versions (the only change is the parameter and its use in the key/path):

```python
def list_bronze_cells(client: Minio, cfg: dict, layer: str = "BRONZE_CU") -> list:
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/{layer}/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    return sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith(".parquet")
    )


def open_bronze_range(
    client: Minio, cfg: dict, cell: str, layer: str = "BRONZE_CU"
) -> _MinioRangeFile:
    """Open a BRONZE parquet on MinIO as a range-read file-like object.

    Used to peek at a single column (e.g. Prozedur) without downloading the
    whole bronze file — the procedure-filter gate can then skip cells that
    don't match before fetch_bronze pulls the full payload.
    """
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/{layer}/{cell}"
    return _MinioRangeFile(client, bucket, key)


def bronze_object_key(cell: str, layer: str = "BRONZE_CU") -> str:
    return f"{layer}/{cell}"


def bronze_exists_on_minio(
    client: Minio, cfg: dict, cell: str, layer: str = "BRONZE_CU"
) -> bool:
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/{layer}/{cell}"
    try:
        client.stat_object(bucket, key)
        return True
    except S3Error:
        return False


@contextmanager
def fetch_bronze(client: Minio, cfg: dict, cell: str, layer: str = "BRONZE_CU"):
    """Stream a BRONZE object from MinIO into a tempfile; yield its path."""
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/{layer}/{cell}"
    response = client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
```

- [ ] **Step 4: Add `root=` to the artefact key helpers**

Replace `gold_object_key`, `x_silver_object_key` and the four export key helpers:

```python
def gold_object_key(cell: str, root: str | None = None) -> str:
    # Under a characterization root the layer collapses to a single file, so a
    # para run can never overwrite the shared GOLD/<cell>.parquet of a CU run.
    if root:
        return f"{root}/GOLD.parquet"
    return f"GOLD/{cell}"


def x_silver_object_key(
    cell: str, classifier: bool = False, root: str | None = None
) -> str:
    stem = cell.split(".")[0]
    if root:
        return f"{root}/with_features_post_labeled.csv"
    # Classifier-path CSVs go to 60_classifier/ (untagged, caller passes
    # include_tag=False) so they sit beside the model and stay out of the tagged
    # TRACY/with_features_post_labeled/ that train_classifier consumes.
    if classifier:
        return f"60_classifier/with_features_post_labeled/{stem}.csv"
    return f"with_features_post_labeled/{stem}.csv"


def export_pulse_object_key(cell: str, filename: str, root: str | None = None) -> str:
    stem = cell.split(".")[0]
    if root:
        return f"{root}/data/{filename}"
    return f"20_export_pulse/{stem}/{filename}"


def export_qocv_object_key(cell: str, filename: str, root: str | None = None) -> str:
    stem = cell.split(".")[0]
    if root:
        return f"{root}/data/{filename}"
    return f"30_export_qocv/{stem}/{filename}"


def export_eis_object_key(cell: str, filename: str, root: str | None = None) -> str:
    stem = cell.split(".")[0]
    if root:
        return f"{root}/data/{filename}"
    return f"25_export_eis/{stem}/{filename}"


def export_capacity_object_key(cell: str, filename: str, root: str | None = None) -> str:
    if root:
        return f"{root}/{filename}"
    return f"40_capacity_monitore/{filename}"
```

Note: keep the existing body of `export_capacity_object_key` (line 355 onward) as the `root is None` branch — read it first and preserve whatever it returns today rather than assuming the string above; if it differs, keep the original and only prepend the `if root:` branch.

- [ ] **Step 5: Verify the key helpers**

```bash
python - <<'PY'
from util import io_router as r
c, f = "VTC6_003.parquet", "VTC6_003_pulse_BM13_98.4SOH.parquet"
root = "10_initial_characterization/VTC6_003"
# defaults unchanged
assert r.bronze_object_key(c) == "BRONZE_CU/VTC6_003.parquet"
assert r.gold_object_key(c) == "GOLD/VTC6_003.parquet"
assert r.x_silver_object_key(c) == "with_features_post_labeled/VTC6_003.csv"
assert r.x_silver_object_key(c, classifier=True) == "60_classifier/with_features_post_labeled/VTC6_003.csv"
assert r.export_pulse_object_key(c, f) == f"20_export_pulse/VTC6_003/{f}"
assert r.export_qocv_object_key(c, f) == f"30_export_qocv/VTC6_003/{f}"
assert r.export_eis_object_key(c, f) == f"25_export_eis/VTC6_003/{f}"
# rooted
assert r.bronze_object_key(c, layer="BRONZE_PARA") == "BRONZE_PARA/VTC6_003.parquet"
assert r.gold_object_key(c, root=root) == f"{root}/GOLD.parquet"
assert r.x_silver_object_key(c, root=root) == f"{root}/with_features_post_labeled.csv"
assert r.x_silver_object_key(c, classifier=True, root=root) == f"{root}/with_features_post_labeled.csv"
for fn in (r.export_pulse_object_key, r.export_qocv_object_key, r.export_eis_object_key):
    assert fn(c, f, root=root) == f"{root}/data/{f}"
assert r.export_capacity_object_key(c, "VTC6_003_capacity.csv", root=root) == f"{root}/VTC6_003_capacity.csv"
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/util/run_context.py src/util/io_router.py
git commit -m "feat(io): RunContext plus root/layer-aware object keys

RunContext carries the BRONZE layer, the procedure-filter config key and the
per-cell output root. io_router's bronze helpers take a layer and the artefact
key helpers take an optional root; both default to today's CU behaviour."
```

---

### Task 3: Thread RunContext through main.py

**Files:**
- Modify: `src/main.py:62-106` (`run_pipeline`), `:135-171` (`_process_cell`), `:174` + `:177-181` + `:361-378` (`_process_cell_inner`), `:427-444` (`_write_x_silver`, `_write_gold`), `:541-560` (`_build_paths`), `:563-607` (CLI)

**Interfaces:**
- Consumes: `util.run_context.RunContext`, `CU`; the `layer=` / `root=` io_router parameters from Task 2.
- Produces: `main.run_pipeline(cfg, target_specimen=None, overwrite=False, run_ctx=CU)`, `main._build_paths(cell, working_path, classifier=False, run_ctx=CU)`, and `_process_cell` / `_process_cell_inner` each taking a trailing `run_ctx=CU`.

- [ ] **Step 1: Import the context in main.py**

Add near the other `util` imports at the top of `src/main.py`:

```python
from util.run_context import CU, RunContext
```

- [ ] **Step 2: Rewrite `_build_paths` (line 541)**

```python
def _build_paths(
    cell: str, working_path: str, classifier: bool = False, run_ctx: RunContext = CU
) -> dict:
    stem = cell.split(".")[0]
    bronze = os.path.join(working_path, run_ctx.bronze_layer, cell)

    export_root = run_ctx.export_root(cell)
    if export_root:
        # Characterization: every artefact under one per-cell root, so a para
        # run never collides with the CU GOLD / capacity CSV of the same cell.
        base = os.path.join(working_path, *export_root.split("/"))
        data = os.path.join(base, "data")
        return {
            "bronze": bronze,
            "X_silver": os.path.join(base, "with_features_post_labeled.csv"),
            "gold": os.path.join(base, "GOLD.parquet"),
            "export_pulse_dir": data,
            "export_eis_dir": data,
            "export_qocv_dir": data,
            "export_capacity_dir": base,
            "root": base,
        }

    # The classifier path's CSVs are *not* valid training data, so they are
    # routed to 60_classifier/ (away from the HDBSCAN with_features_post_labeled/
    # that train_classifier reads) — keeping the two label sets side by side and
    # the training set uncontaminated.
    x_silver_dir = (
        os.path.join("60_classifier", "with_features_post_labeled")
        if classifier
        else "with_features_post_labeled"
    )
    return {
        "bronze": bronze,
        "X_silver": os.path.join(working_path, x_silver_dir, stem + ".csv"),
        "gold": os.path.join(working_path, "GOLD", cell),
        "export_pulse_dir": os.path.join(working_path, "20_export_pulse", stem),
        "export_eis_dir": os.path.join(working_path, "25_export_eis", stem),
        "export_qocv_dir": os.path.join(working_path, "30_export_qocv", stem),
        "export_capacity_dir": os.path.join(working_path, "40_capacity_monitore"),
        "root": None,
    }
```

- [ ] **Step 3: Thread through `run_pipeline` (line 62)**

Change the signature and the three places that hardcode `BRONZE_CU` / call downstream:

```python
def run_pipeline(
    cfg: dict,
    target_specimen: list = None,
    overwrite: bool = False,
    run_ctx: RunContext = CU,
):
```

Inside, replace the cell listing (lines 71-76):

```python
    if download_from == "minio":
        cells = io_router.list_bronze_cells(minio_client, cfg, layer=run_ctx.bronze_layer)
    else:
        if not working_path:
            raise ValueError("working_path required when download_from='local'")
        cells = glob.glob1(os.path.join(working_path, run_ctx.bronze_layer), "*.parquet")
```

The skip-check (line 90) and the call (line 96):

```python
            gold_path = _build_paths(cell, working_path, run_ctx=run_ctx)["gold"]
```

```python
            _process_cell(cell, cfg, minio_client, exceptions, run_ctx=run_ctx)
```

- [ ] **Step 4: Thread through `_process_cell` (line 135)**

```python
def _process_cell(
    cell: str, cfg: dict, minio_client, exceptions: dict, run_ctx: RunContext = CU
):
```

Its `paths` call (line 140):

```python
        _build_paths(
            cell,
            working_path,
            classifier=bool(cfg.get("classifier_model_path")),
            run_ctx=run_ctx,
        )
```

Its procedure-filter gate (lines 150-156) reads the context's key, and the range read passes the layer:

```python
    procedure_filter = cfg.get(run_ctx.procedure_filter_key, None)
    if procedure_filter is not None:
        if download_from == "minio":
            with io_router.open_bronze_range(
                minio_client, cfg, cell, layer=run_ctx.bronze_layer
            ) as src:
                matched = processing_procedure_filter(src, procedure_filter)
        else:
            matched = processing_procedure_filter(paths["bronze"], procedure_filter)
```

Its fetch (line 164) and inner call (lines 169-171):

```python
        bronze_ctx = io_router.fetch_bronze(
            minio_client, cfg, cell, layer=run_ctx.bronze_layer
        )
```

```python
        return _process_cell_inner(
            cell, cfg, bronze_path, paths, minio_client, exceptions, run_ctx=run_ctx
        )
```

- [ ] **Step 5: Thread through `_process_cell_inner` (line 174)**

Signature:

```python
def _process_cell_inner(
    cell, cfg, bronze_path, paths, minio_client, exceptions, run_ctx: RunContext = CU
):
```

Line 177, the dismember filter:

```python
    procedure_filter = cfg.get(run_ctx.procedure_filter_key, None)
```

The write calls (lines 341, 362) gain the context:

```python
    _write_x_silver(X_silver, cell, cfg, paths, minio_client, run_ctx=run_ctx)
```

```python
        _write_gold(df_gold, cell, cfg, paths, minio_client, run_ctx=run_ctx)
```

And the export block (lines 366-378) honours `force_exports` and passes the context:

```python
    df_export = df_gold[
        df_gold["target"].isin(["CAP", "PUL", "qOCV_DCH", "qOCV_CHA", "PAU"])
    ]
    soh = _build_soh_map(df_export, cfg["nom_capacity"])
    export_capacity(df_export, soh, cell, cfg, paths, minio_client, run_ctx=run_ctx)
    if cfg.get("export_pulse") or run_ctx.force_exports:
        export_pulse(
            df_export, soh, cell, cfg, paths, minio_client, bronze_path, run_ctx=run_ctx
        )
    if cfg.get("export_qocv") or run_ctx.force_exports:
        export_qocv(df_export, soh, cell, cfg, paths, minio_client, run_ctx=run_ctx)
    if cfg.get("export_eis") or run_ctx.force_exports:
        # df_gold (not df_export): EIS spectra are matched to the cell's EIS-
        # labelled segments, which df_export filters out.
        export_eis(df_gold, soh, cell, cfg, paths, minio_client, run_ctx=run_ctx)
```

- [ ] **Step 6: Update the two writers (lines 427-444)**

```python
def _write_x_silver(df, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
    classifier = bool(cfg.get("classifier_model_path"))
    root = run_ctx.export_root(cell)
    if io_router.writes_local(cfg) and paths:
        os.makedirs(os.path.dirname(paths["X_silver"]), exist_ok=True)
        df.to_csv(paths["X_silver"], index=False)
        logging.info(f"{cell}: X_silver -> {paths['X_silver']}")
    if io_router.writes_minio(cfg):
        key = io_router.x_silver_object_key(cell, classifier=classifier, root=root)
        # Rooted runs are untagged: they live beside their own data/ and plots/,
        # not under the shared TRACY/ layer tree.
        io_router.upload_csv(
            minio_client, cfg, df, key, include_tag=not (classifier or root)
        )


def _write_gold(df, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
    root = run_ctx.export_root(cell)
    if io_router.writes_local(cfg) and paths:
        os.makedirs(os.path.dirname(paths["gold"]), exist_ok=True)
        df.to_parquet(paths["gold"], index=False)
        logging.info(f"{cell}: GOLD -> {paths['gold']}")
    if io_router.writes_minio(cfg):
        io_router.upload_parquet(
            minio_client,
            cfg,
            df,
            io_router.gold_object_key(cell, root=root),
            include_tag=not root,
        )
```

- [ ] **Step 7: Verify defaults are untouched**

```bash
python - <<'PY'
import main
from util.run_context import CU, CHARACTERIZATION
import inspect

# every threaded function defaults to the CU context
for fn in (main.run_pipeline, main._process_cell, main._process_cell_inner,
           main._build_paths, main._write_gold, main._write_x_silver):
    assert inspect.signature(fn).parameters["run_ctx"].default is CU, fn.__name__

wp = "/tmp/wp"
cu = main._build_paths("VTC6_003.parquet", wp)
assert cu["bronze"] == "/tmp/wp/BRONZE_CU/VTC6_003.parquet"
assert cu["gold"] == "/tmp/wp/GOLD/VTC6_003.parquet"
assert cu["X_silver"] == "/tmp/wp/with_features_post_labeled/VTC6_003.csv"
assert cu["export_pulse_dir"] == "/tmp/wp/20_export_pulse/VTC6_003"
assert cu["export_capacity_dir"] == "/tmp/wp/40_capacity_monitore"

clf = main._build_paths("VTC6_003.parquet", wp, classifier=True)
assert clf["X_silver"] == "/tmp/wp/60_classifier/with_features_post_labeled/VTC6_003.csv"

ch = main._build_paths("VTC6_003.parquet", wp, run_ctx=CHARACTERIZATION)
base = "/tmp/wp/10_initial_characterization/VTC6_003"
assert ch["bronze"] == "/tmp/wp/BRONZE_PARA/VTC6_003.parquet"
assert ch["gold"] == base + "/GOLD.parquet"
assert ch["X_silver"] == base + "/with_features_post_labeled.csv"
assert ch["export_pulse_dir"] == ch["export_eis_dir"] == ch["export_qocv_dir"] == base + "/data"
assert ch["export_capacity_dir"] == base
print("OK")
PY
```

Expected: `OK`. (This task's export calls will still fail at runtime until Task 4 lands — that is expected; only the path logic is verified here.)

- [ ] **Step 8: Commit**

```bash
git add src/main.py
git commit -m "refactor(pipeline): thread RunContext through main.py

run_pipeline/_process_cell/_process_cell_inner/_build_paths and the two
writers take an explicit run_ctx defaulting to CU, so the check-up path is
unchanged while a characterization run can redirect its BRONZE layer,
procedure-filter key and output root."
```

---

### Task 4: Export modules accept the run context

**Files:**
- Modify: `src/output/export_pulse.py:77` + `:131`
- Modify: `src/output/export_qocv.py:17` + `:45`
- Modify: `src/output/export_eis.py:83` + `:140`
- Modify: `src/output/export_capacity.py:19` + `:53`

**Interfaces:**
- Consumes: `util.run_context.RunContext`, `CU`; `io_router.export_*_object_key(..., root=...)`.
- Produces: `export_pulse(df_export, soh, cell, cfg, paths, minio_client, bronze_path=None, run_ctx=CU)`, `export_qocv(df_export, soh, cell, cfg, paths, minio_client, run_ctx=CU)`, `export_eis(df_gold, soh, cell, cfg, paths, minio_client, run_ctx=CU)`, `export_capacity(df_export, soh, cell, cfg, paths, minio_client, run_ctx=CU)`.

- [ ] **Step 1: export_pulse**

Add to the imports of `src/output/export_pulse.py`:

```python
from util.run_context import CU, RunContext
```

Signature (line 77):

```python
def export_pulse(
    df_export, soh, cell, cfg, paths, minio_client, bronze_path=None,
    run_ctx: RunContext = CU,
):
```

Key call (line 131-132):

```python
            key = io_router.export_pulse_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
            io_router.upload_parquet(minio_client, cfg, group, key, include_tag=False)
```

- [ ] **Step 2: export_qocv**

Same import. Signature (line 17):

```python
def export_qocv(df_export, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
```

Key call (line 45):

```python
            key = io_router.export_qocv_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
```

- [ ] **Step 3: export_eis**

Same import. Signature (line 83):

```python
def export_eis(df_gold, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
```

Key call (line 140):

```python
            key = io_router.export_eis_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
```

- [ ] **Step 4: export_capacity**

Same import. Signature (line 19):

```python
def export_capacity(df_export, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
```

Key call (line 53):

```python
        key = io_router.export_capacity_object_key(
            cell, filename, root=run_ctx.export_root(cell)
        )
```

- [ ] **Step 5: Verify signatures and imports**

```bash
python - <<'PY'
import inspect
from util.run_context import CU
from output.export_pulse import export_pulse
from output.export_qocv import export_qocv
from output.export_eis import export_eis
from output.export_capacity import export_capacity
for fn in (export_pulse, export_qocv, export_eis, export_capacity):
    p = inspect.signature(fn).parameters
    assert "run_ctx" in p, fn.__name__
    assert p["run_ctx"].default is CU, fn.__name__
assert inspect.signature(export_pulse).parameters["bronze_path"].default is None
import main  # the call sites now match
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/output/export_pulse.py src/output/export_qocv.py src/output/export_eis.py src/output/export_capacity.py
git commit -m "feat(output): route exports through the run context

Each export takes run_ctx (default CU) and passes its per-cell root to the
io_router key helper, so a characterization run writes under
10_initial_characterization/<cell>/data/ instead of the shared export folders."
```

---

### Task 5: BRONZE_PARA builder

**Files:**
- Modify: `src/download/build_bronze_cu_with_ah.py:347-359` (manifest helpers), `:433-465` (`process_cell`), `:573-650` (`run`, CLI)
- Create: `src/download/build_bronze_para.py`

**Interfaces:**
- Consumes: `io_router.bronze_exists_on_minio(..., layer=...)` from Task 2.
- Produces: `build_bronze_cu_with_ah.run(cfg, target_cells=None, overwrite=False, incremental=False, layer="BRONZE_CU", filter_key="procedure_filter")` and `process_cell(..., layer="BRONZE_CU", filter_key="procedure_filter")`.
- New config key: `para_procedure_filter` — a list of substrings (single-element list is the normal case).

- [ ] **Step 1: Parameterize the manifest helpers**

In `src/download/build_bronze_cu_with_ah.py`, replace `_manifest_minio_key` (line 358):

```python
def _manifest_minio_key(cell: str, layer: str = "BRONZE_CU") -> str:
    return f"{layer}/{cell}_manifest.json"
```

`_manifest_local_path` (line 353) derives from the output path and needs no change.

- [ ] **Step 2: Parameterize `process_cell`**

Signature (line 433) gains two trailing parameters:

```python
def process_cell(
    cfg: dict,
    cell: str,
    out_bronze_cu: str | None,
    overwrite: bool = False,
    upload_minio: bool = False,
    minio_client: Minio | None = None,
    download_from: str = "minio",
    incremental: bool = False,
    layer: str = "BRONZE_CU",
    filter_key: str = "procedure_filter",
) -> None:
```

Its filter lookup (lines 449-456):

```python
    # Test-file detection follows the config's procedure filter (the programme
    # name in the 4th '='-delimited filename field).
    cu_marker = cfg.get(filter_key)
    if not cu_marker:
        raise ValueError(
            f"{filter_key} must be set in the battery config — it is the "
            f"programme name used to detect {layer} test files."
        )
```

Its MinIO skip-check (line ~464):

```python
        if upload_minio and io_router.bronze_exists_on_minio(
            minio_client, cfg, cell_file, layer=layer
        ):
            print(f"{cell} - MinIO {layer} already exists, skipping.")
            return
```

Also update the local skip-check message just above it from `local BRONZE_CU already exists` to `local {layer} already exists`.

Then `grep -n "_manifest_minio_key\|BRONZE_CU" src/download/build_bronze_cu_with_ah.py` and, for every remaining occurrence **inside `process_cell` and the functions it calls with a layer-dependent key** (`_load_incremental_state`, `_save_manifest`), pass `layer` through and use it in the object key. Those two helpers take `cell` and build `f"{cfg['minio_prefix']}/{_manifest_minio_key(cell)}"` — add a `layer: str = "BRONZE_CU"` parameter to each and forward it to `_manifest_minio_key`, then pass `layer=layer` at their call sites in `process_cell`.

- [ ] **Step 3: Parameterize `run`**

Signature (line 573):

```python
def run(
    cfg: dict,
    target_cells: list = None,
    overwrite: bool = False,
    incremental: bool = False,
    layer: str = "BRONZE_CU",
    filter_key: str = "procedure_filter",
) -> None:
```

Its `process_cell` call at the bottom:

```python
        process_cell(
            cfg=cfg,
            cell=cell,
            out_bronze_cu=os.path.join(working_path, layer, f"{cell}.parquet") if save_local else None,
            overwrite=overwrite,
            upload_minio=upload_minio,
            minio_client=minio_client,
            download_from=download_from,
            incremental=incremental,
            layer=layer,
            filter_key=filter_key,
        )
```

Add `"BRONZE_PARA"` to the `reserved` set at line 560 so a `BRONZE_PARA/` folder is never mistaken for a cell folder by `_list_cells_local`. While there, also add `"10_initial_characterization"`, `"25_export_eis"` and `"60_classifier"` — the same class of bug, and `25_export_eis`/`60_classifier` are already missing.

- [ ] **Step 4: Write `src/download/build_bronze_para.py`**

```python
"""Build BRONZE_PARA — the parametrization (initial characterization) layer.

Same builder as :mod:`build_bronze_cu_with_ah`, pointed at a different set of
test files: the ones whose 4th '='-delimited filename field matches
``para_procedure_filter`` (a list of substrings; one element is the normal
case). Output goes to ``<working_path>/BRONZE_PARA/`` and, when uploading,
``<minio_prefix>/BRONZE_PARA/``.

    python download/build_bronze_para.py <battery_cfg> [--cells …]
                                         [--overwrite] [--incremental]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download.build_bronze_cu_with_ah import run

LAYER = "BRONZE_PARA"
FILTER_KEY = "para_procedure_filter"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BRONZE_PARA (initial-characterization layer) and Ah sidecar"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cells")
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild even if output exists"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Append only new test files to an existing BRONZE_PARA (uses the "
             "manifest sidecar); full build if no prior state exists.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if not cfg.get(FILTER_KEY):
        parser.error(
            f"{FILTER_KEY} must be set in the battery config — it lists the "
            "programme-name substrings that mark a parametrization test file."
        )

    run(
        cfg,
        target_cells=args.cells,
        overwrite=args.overwrite,
        incremental=args.incremental,
        layer=LAYER,
        filter_key=FILTER_KEY,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify parameterization and the CLI guard**

```bash
python - <<'PY'
import inspect
from download import build_bronze_cu_with_ah as b
for fn in (b.run, b.process_cell):
    p = inspect.signature(fn).parameters
    assert p["layer"].default == "BRONZE_CU", fn.__name__
    assert p["filter_key"].default == "procedure_filter", fn.__name__
assert b._manifest_minio_key("c") == "BRONZE_CU/c_manifest.json"
assert b._manifest_minio_key("c", layer="BRONZE_PARA") == "BRONZE_PARA/c_manifest.json"
print("OK")
PY

python download/build_bronze_para.py --help
```

Expected: `OK`, then the argparse help listing `--cells`, `--overwrite`, `--incremental`.

- [ ] **Step 6: Verify the missing-filter error**

```bash
python - <<'PY'
import json, subprocess, sys, tempfile, os
cfg = {"working_path": "/tmp/wp", "bucket_name": "b", "minio_prefix": "p"}
p = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, p); p.close()
r = subprocess.run([sys.executable, "download/build_bronze_para.py", p.name],
                   capture_output=True, text=True)
assert r.returncode != 0
assert "para_procedure_filter" in r.stderr, r.stderr
os.unlink(p.name)
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/download/build_bronze_cu_with_ah.py src/download/build_bronze_para.py
git commit -m "feat(download): BRONZE_PARA build for parametrization tests

The bronze builder takes a layer and a procedure-filter config key, both
defaulting to today's BRONZE_CU values. build_bronze_para.py is a thin CLI
selecting para_procedure_filter files into BRONZE_PARA/."
```

---

### Task 6: Characterization pipeline entry point

**Files:**
- Create: `src/characterize/__init__.py`, `src/characterize/main_para.py`

**Interfaces:**
- Consumes: `main.run_pipeline(cfg, target_specimen, overwrite, run_ctx)`, `util.run_context.CHARACTERIZATION`, `main.load_config`.
- Produces: CLI `python -m characterize.main_para <cfg> [--cells …] [--overwrite] [--clustering {auto,hdbscan,classifier}]`.

- [ ] **Step 1: Create the package**

```bash
touch src/characterize/__init__.py
```

- [ ] **Step 2: Write `src/characterize/main_para.py`**

```python
"""Initial characterization pipeline — the BOL parametrization run.

Same pipeline as :mod:`main` (dismember → features → clustering → calculate),
reading BRONZE_PARA instead of BRONZE_CU and writing every artefact under
``10_initial_characterization/<cell_stem>/`` instead of the shared GOLD /
20_export_pulse / 25_export_eis / 30_export_qocv / 40_capacity_monitore
folders. Pulse, EIS and qOCV exports are forced on — they are the point of the
run — and the BOL capacity CSV stays out of the aging monitor's folder.

    python -m characterize.main_para <battery_cfg> [--cells …]
                                     [--overwrite] [--clustering …]

Fitting is a separate step: see :mod:`characterize.fit_characterization`.
"""

import argparse
import logging

from main import load_config, run_pipeline
from util.run_context import CHARACTERIZATION


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial characterization pipeline (BRONZE_PARA → "
                    "10_initial_characterization/)"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "--cells", nargs="*", help="Optional subset of cell names to process"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess cells even if the characterization GOLD already exists",
    )
    parser.add_argument(
        "--clustering",
        choices=["auto", "hdbscan", "classifier"],
        default="auto",
        help="Clustering path: 'auto' (config classifier_model_path decides), "
             "'hdbscan' (force HDBSCAN, ignore classifier_model_path), or "
             "'classifier' (require classifier_model_path).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.clustering == "hdbscan":
        cfg["classifier_model_path"] = None
    elif args.clustering == "classifier" and not cfg.get("classifier_model_path"):
        parser.error("--clustering classifier needs classifier_model_path in the config")

    if not cfg.get(CHARACTERIZATION.procedure_filter_key):
        parser.error(
            f"{CHARACTERIZATION.procedure_filter_key} must be set in the battery "
            "config — it selects the parametrization procedures."
        )

    # Interpretation is a CU-path affordance (label-only, no GOLD/exports) and
    # would defeat the purpose here, where the exports are the deliverable.
    cfg["llm_interpret"] = False

    logging.info(
        "initial characterization: %s -> %s/<cell>/",
        CHARACTERIZATION.bronze_layer, CHARACTERIZATION.export_root_prefix,
    )
    run_pipeline(
        cfg,
        target_specimen=args.cells,
        overwrite=args.overwrite,
        run_ctx=CHARACTERIZATION,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the CLI**

```bash
python -m characterize.main_para --help
```

Expected: help text listing `config`, `--cells`, `--overwrite`, `--clustering`.

- [ ] **Step 4: Verify the missing-filter guard**

```bash
python - <<'PY'
import json, subprocess, sys, tempfile, os
cfg = {"working_path": "/tmp/wp", "type_cell": "X", "nom_capacity": 3.0}
p = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, p); p.close()
r = subprocess.run([sys.executable, "-m", "characterize.main_para", p.name],
                   capture_output=True, text=True)
assert r.returncode != 0
assert "para_procedure_filter" in r.stderr, r.stderr
os.unlink(p.name)
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/characterize/__init__.py src/characterize/main_para.py
git commit -m "feat(characterize): BOL characterization pipeline entry point

Wraps main.run_pipeline with the CHARACTERIZATION run context: reads
BRONZE_PARA, forces the pulse/EIS/qOCV exports on, and writes everything
under 10_initial_characterization/<cell>/."
```

---

### Task 7: Fit and plot CLI

**Files:**
- Create: `src/characterize/fit_characterization.py`

**Interfaces:**
- Consumes: `analysis.fit_2rc_pulse.fit_folder(folder, nom_capacity, remove_before_min, exclude_zc) -> DataFrame` (columns include `pulse_segment_id`, `ID`, `SOH`, `SOC`, `direction`, `I_A`, `C_rate`, `OCV_V`, `R0_ohm`, `R1_ohm`, `tau1_s`, `R2_ohm`, `tau2_s`, `rmse_mV`, `BM_Programm`, `SOH_num`, `pulse_type`); `analysis.fit_2rc_pulse.plot_vs_soc(results, out_png, title="")`; module defaults `REMOVE_PULSE_BEFORE_MIN = 0`, `EXCLUDE_ZUSTAND_CURRENT = ["DCH/-1.5"]`.
- Consumes: `analysis.eis_vs_soc.build_eis_table(df, direction=None, step=None, fit_2rc=True, fit_warburg=True, fit_zarc=True) -> DataFrame` (ZARC columns `R_ohm`, `R1_z`, `tau1_z`, `alpha1_z`, `R2_z`, `tau2_z`, `alpha2_z`, `R_d_z`, `tau_d_z`, `phi_d_z`, `zarc_rmse`, `zarc_degenerate`, plus `eis_number`, `SOC_pct`, `U`, `Time`); `analysis.eis_vs_soc.plot_zarc_vs_soc(table, out_png, title="")`; constants `ZARC_DIFFUSION_ELEMENT`, `DIFFUSION_TAU_BOX`, `DIFFUSION_PHI_BOX`, `ZARC_ALPHA_MIN`.
- Consumes: `analysis.qocv_curve.find_pairs(folder) -> {bm: {"cha": path, "dch": path, "soh": float}}`, `_pick_pair(pairs, soh=None) -> (bm, dict)`, `load_sweep(path, discharge=False) -> (v, q)`, `plot_qocv(cha_path, dch_path, out_png, nom_capacity)`.
- Produces: CLI `python -m characterize.fit_characterization <cfg> [--cells …]`; function `fit_cell(cell_dir: str, nom_capacity: float) -> dict` returning the parameters payload.

- [ ] **Step 1: Write `src/characterize/fit_characterization.py`**

```python
"""Fit the characterization bundles and plot them.

Reads ``<working_path>/10_initial_characterization/<cell>/data/`` — the pulse,
EIS and qOCV parquets written by :mod:`characterize.main_para` — and writes
``<cell>_parameters.json`` plus ``plots/`` beside it. Runs standalone, so fits
can be repeated without redoing segmentation.

Models are fixed defaults:

* **pulse** — 2RC (:mod:`analysis.fit_2rc_pulse`).
* **EIS** — 2×ZARC + series-L + generalized Warburg
  (:func:`analysis.eis_vs_soc.fit_zarc_warburg_eis`). φ is fitted; τ_d is
  **pinned** by ``DIFFUSION_TAU_BOX``, so ``R_d_z`` is the amplitude at
  ω = 1/τ_d and ``tau_d_z`` is a shape constant, not a result. Those settings
  are recorded in the params file so the numbers stay interpretable.
* **qOCV** — no fit; the curve and its throughput-normalised capacities.

    python -m characterize.fit_characterization <battery_cfg> [--cells …]
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis import eis_vs_soc, qocv_curve
from analysis import fit_2rc_pulse as pulse_fit
from main import load_config
from util.run_context import CHARACTERIZATION

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PULSE_MODEL = "2rc"
EIS_MODEL = "2zarc_warburg"

#: Columns lifted from the 2RC results table into the params file.
PULSE_COLS = [
    "pulse_segment_id", "ID", "BM_Programm", "SOH", "SOC", "direction",
    "I_A", "C_rate", "OCV_V", "R0_ohm", "R1_ohm", "tau1_s", "R2_ohm",
    "tau2_s", "rmse_mV",
]

#: Columns lifted from the EIS table into the params file.
EIS_COLS = [
    "eis_number", "SOC_pct", "U", "R_ohm", "R1_z", "tau1_z", "alpha1_z",
    "R2_z", "tau2_z", "alpha2_z", "R_d_z", "tau_d_z", "phi_d_z",
    "zarc_rmse", "zarc_degenerate",
]


def _jsonable(value):
    """NumPy/pandas scalars -> plain Python; NaN/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.str_, str)):
        return str(value)
    return value


def _records(table: pd.DataFrame, cols: list) -> list:
    present = [c for c in cols if c in table.columns]
    return [
        {c: _jsonable(row[c]) for c in present}
        for _, row in table[present].iterrows()
    ]


def fit_pulse(data_dir: str, plots_dir: str, nom_capacity: float) -> dict:
    """2RC fit of every pulse bundle in ``data_dir``."""
    files = sorted(glob.glob(os.path.join(data_dir, "*_pulse_BM*.parquet")))
    block = {"model": PULSE_MODEL, "fits": [], "sources": [os.path.basename(f) for f in files]}
    if not files:
        logging.info("no pulse bundles in %s", data_dir)
        return block
    try:
        results = pulse_fit.fit_folder(
            data_dir,
            nom_capacity,
            pulse_fit.REMOVE_PULSE_BEFORE_MIN,
            pulse_fit.EXCLUDE_ZUSTAND_CURRENT,
        )
    except Exception as exc:                      # one bad bundle ≠ dead cell
        logging.warning("pulse fit failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block
    if results.empty:
        block["error"] = "no pulses fit"
        return block

    block["fits"] = _records(results, PULSE_COLS)
    out_png = os.path.join(plots_dir, "pulse_2rc.png")
    try:
        pulse_fit.plot_vs_soc(results, out_png, title="initial characterization")
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("pulse plot failed: %s", exc)
    return block


def fit_eis(data_dir: str, plots_dir: str) -> dict:
    """2×ZARC + generalized-Warburg fit of every spectrum in every bundle."""
    files = sorted(glob.glob(os.path.join(data_dir, "*_eis_BM*.parquet")))
    block = {
        "model": EIS_MODEL,
        "settings": {
            "element": eis_vs_soc.ZARC_DIFFUSION_ELEMENT,
            "tau_d_pinned_s": list(eis_vs_soc.DIFFUSION_TAU_BOX)[0],
            "tau_d_is_fitted": eis_vs_soc.DIFFUSION_TAU_BOX[0]
                               != eis_vs_soc.DIFFUSION_TAU_BOX[1],
            "phi_box": list(eis_vs_soc.DIFFUSION_PHI_BOX),
            "alpha_min": eis_vs_soc.ZARC_ALPHA_MIN,
        },
        "fits": [],
        "sources": [os.path.basename(f) for f in files],
    }
    if not files:
        logging.info("no EIS bundles in %s", data_dir)
        return block

    tables = []
    for path in files:
        try:
            table = eis_vs_soc.build_eis_table(pd.read_parquet(path))
            table["source"] = os.path.basename(path)
            tables.append(table)
        except Exception as exc:
            logging.warning("EIS fit failed for %s: %s", os.path.basename(path), exc)
            block.setdefault("errors", []).append(
                {"source": os.path.basename(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    if not tables:
        return block

    combined = pd.concat(tables, ignore_index=True)
    block["fits"] = _records(combined, EIS_COLS + ["source"])
    n_deg = int(combined.get("zarc_degenerate", pd.Series(dtype=bool)).sum())
    if n_deg:
        logging.warning("%d/%d EIS fits flagged degenerate", n_deg, len(combined))
    block["n_degenerate"] = n_deg

    out_png = os.path.join(plots_dir, "eis_2zarc_warburg.png")
    try:
        eis_vs_soc.plot_zarc_vs_soc(combined, out_png, title="initial characterization")
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("EIS plot failed: %s", exc)
    return block


def summarize_qocv(data_dir: str, plots_dir: str, nom_capacity: float) -> dict:
    """qOCV curve + throughput capacities. No fit — the curve is the result."""
    pairs = qocv_curve.find_pairs(data_dir)
    block = {"fits": [], "sources": []}
    if not pairs:
        logging.info("no complete qOCV cha/dch pair in %s", data_dir)
        return block

    bm, pair = qocv_curve._pick_pair(pairs)
    block["sources"] = [os.path.basename(pair["cha"]), os.path.basename(pair["dch"])]
    try:
        _, qc = qocv_curve.load_sweep(pair["cha"], discharge=False)
        _, qd = qocv_curve.load_sweep(pair["dch"], discharge=True)
        block["fits"] = [{
            "BM_Programm": int(bm),
            "SOH": _jsonable(pair.get("soh")),
            "capacity_cha_ah": round(float(np.nanmax(qc)), 4),
            "capacity_dch_ah": round(float(np.nanmax(qd)), 4),
        }]
    except Exception as exc:
        logging.warning("qOCV read failed: %s", exc)
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block

    out_png = os.path.join(plots_dir, "qocv.png")
    try:
        qocv_curve.plot_qocv(pair["cha"], pair["dch"], out_png, nom_capacity=nom_capacity)
        block["plot"] = os.path.relpath(out_png, os.path.dirname(plots_dir))
    except Exception as exc:
        logging.warning("qOCV plot failed: %s", exc)
    return block


def fit_cell(cell_dir: str, nom_capacity: float) -> dict:
    """Fit one `10_initial_characterization/<cell>/` folder; return the payload."""
    stem = os.path.basename(os.path.normpath(cell_dir))
    data_dir = os.path.join(cell_dir, "data")
    plots_dir = os.path.join(cell_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "cell": stem,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nom_capacity": nom_capacity,
        "pulse": fit_pulse(data_dir, plots_dir, nom_capacity),
        "eis": fit_eis(data_dir, plots_dir),
        "qocv": summarize_qocv(data_dir, plots_dir, nom_capacity),
    }


def run(cfg: dict, target_cells: list = None) -> None:
    working_path = cfg.get("working_path")
    if not working_path:
        raise ValueError("working_path required in the battery config")
    root = os.path.join(working_path, CHARACTERIZATION.export_root_prefix)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found — run `python -m characterize.main_para <cfg>` first"
        )

    cells = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if target_cells:
        cells = [c for c in cells if any(t in c for t in target_cells)]
    if not cells:
        logging.warning("no characterization cell folders under %s", root)
        return

    for stem in cells:
        cell_dir = os.path.join(root, stem)
        logging.info("fitting %s", stem)
        payload = fit_cell(cell_dir, float(cfg["nom_capacity"]))
        out_json = os.path.join(cell_dir, f"{stem}_parameters.json")
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        logging.info("%s: parameters -> %s", stem, out_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit + plot the initial-characterization bundles"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cell stems")
    args = parser.parse_args()
    run(load_config(args.config), target_cells=args.cells)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI and the empty-folder paths**

```bash
python -m characterize.fit_characterization --help
```

Expected: help text listing `config` and `--cells`.

- [ ] **Step 3: Verify `fit_cell` on an empty bundle folder**

This exercises every "nothing to fit" branch and the JSON serialization without needing lab data:

```bash
python - <<'PY'
import json, os, tempfile
from characterize.fit_characterization import fit_cell
d = tempfile.mkdtemp()
cell = os.path.join(d, "VTC6_003"); os.makedirs(os.path.join(cell, "data"))
payload = fit_cell(cell, 3.0)
assert payload["cell"] == "VTC6_003"
assert payload["pulse"]["model"] == "2rc" and payload["pulse"]["fits"] == []
assert payload["eis"]["model"] == "2zarc_warburg" and payload["eis"]["fits"] == []
s = payload["eis"]["settings"]
assert s["element"] == "generalized" and s["tau_d_is_fitted"] is False
assert s["tau_d_pinned_s"] == 5.0 and s["phi_box"] == [0.2, 0.9]
assert payload["qocv"]["fits"] == []
assert os.path.isdir(os.path.join(cell, "plots"))
json.dumps(payload)          # must be serializable
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 4: Verify the missing-root error**

```bash
python - <<'PY'
import tempfile
from characterize.fit_characterization import run
try:
    run({"working_path": tempfile.mkdtemp(), "nom_capacity": 3.0})
    raise SystemExit("expected FileNotFoundError")
except FileNotFoundError as exc:
    assert "main_para" in str(exc)
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/characterize/fit_characterization.py
git commit -m "feat(characterize): fit + plot CLI for the BOL bundles

Runs the existing analysis fitters over 10_initial_characterization/<cell>/
data/ — 2RC for pulse, 2xZARC + generalized Warburg for EIS, qOCV curve — and
writes <cell>_parameters.json plus plots/. Records the pinned tau_d and the
phi/alpha boxes so the fitted values stay interpretable."
```

---

### Task 8: UI Tab 7

**Files:**
- Modify: `src/pipeline_ui.py:232-244` (tab registration), after `:480` (new tab builder), `:526-567` (state I/O), after `:812` (argv builders + step collector), after `:875` (run path)

**Interfaces:**
- Consumes: the three CLIs from Tasks 5–7.
- Produces: `_build_characterization_tab`, `_build_bronze_para_argv`, `_build_char_pipeline_argv`, `_build_char_fit_argv`, `_collect_characterization_steps`, `_run_characterization`; widgets `ch_cells`, `ch_overwrite`, `ch_incremental`, `ch_clustering`, `ch_build`, `ch_pipeline`, `ch_fit`.

- [ ] **Step 1: Register the tab**

In `_build_widgets`, after line 237 add:

```python
        self.tabs.add("7. Initial Characterization")
```

and after line 244:

```python
        self._build_characterization_tab(self.tabs.tab("7. Initial Characterization"))
```

- [ ] **Step 2: Add the tab builder**

Insert after `_build_train_tab` (i.e. after line 522):

```python
    def _build_characterization_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        s = _Section(parent, "Initial characterization (BOL parametrization)")
        s.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.ch_cells = s.add_entry(
            "Cells filter (--cells):",
            placeholder="optional, space-separated, e.g. VTC_cell01 VTC_cell02",
        )
        self.ch_build = s.add_checkbox(
            "Build BRONZE_PARA   →   BRONZE_PARA/<cell>.parquet"
        )
        self.ch_pipeline = s.add_checkbox(
            "Run characterization pipeline   →   10_initial_characterization/<cell>/data/"
        )
        self.ch_fit = s.add_checkbox(
            "Fit + plot (pulse / EIS / qOCV)   →   <cell>_parameters.json + plots/"
        )
        self.ch_overwrite = s.add_checkbox(
            "--overwrite (rebuild / reprocess even if output exists)"
        )
        self.ch_incremental = s.add_checkbox(
            "--incremental (BRONZE_PARA build: append only new test files)"
        )

        clu = ctk.CTkFrame(s, fg_color="transparent")
        clu.grid(sticky="w", padx=4, pady=(2, 2))
        ctk.CTkLabel(clu, text="Clustering:").pack(side="left", padx=(0, 6))
        self.ch_clustering = ctk.CTkSegmentedButton(
            clu, values=["Auto (config)", "HDBSCAN", "Classifier"],
        )
        self.ch_clustering.set("Auto (config)")
        self.ch_clustering.pack(side="left")

        ctk.CTkLabel(
            parent,
            text=(
                "Each ticked stage runs in sequence. Needs para_procedure_filter in "
                "the battery config — the list of programme-name substrings marking "
                "the parametrization test files. Outputs go to <working_path>/"
                "10_initial_characterization/<cell>/ (data/, plots/, "
                "<cell>_parameters.json); the BOL capacity CSV stays out of "
                "40_capacity_monitore/. Fixed models: 2RC (pulse), 2×ZARC + "
                "generalized Warburg (EIS). Not part of 'Run all' — this is a "
                "one-off BOL step, not part of the aging loop."
            ),
            text_color="#888",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        self.ch_run_btn = ctk.CTkButton(
            btns, text="▶ Run characterization", width=200,
            command=self._run_characterization,
        )
        self.ch_run_btn.pack(side="right", padx=4)
```

- [ ] **Step 3: Persist the tab's state**

In `_restore_state`, after line 545 (`self.tr_labels.set(...)`):

```python
        self.ch_cells.insert(0, s.get("ch_cells", ""))
        # All three stages default to ticked — the usual run is end to end.
        if s.get("ch_build", True):
            self.ch_build.select()
        if s.get("ch_pipeline", True):
            self.ch_pipeline.select()
        if s.get("ch_fit", True):
            self.ch_fit.select()
        if s.get("ch_overwrite"):
            self.ch_overwrite.select()
        if s.get("ch_incremental"):
            self.ch_incremental.select()
        self.ch_clustering.set(s.get("ch_clustering", "Auto (config)"))
```

In `_persist_state`, inside the `self._state.update({...})` dict after `"tr_labels"`:

```python
            "ch_cells": self.ch_cells.get(),
            "ch_build": bool(self.ch_build.get()),
            "ch_pipeline": bool(self.ch_pipeline.get()),
            "ch_fit": bool(self.ch_fit.get()),
            "ch_overwrite": bool(self.ch_overwrite.get()),
            "ch_incremental": bool(self.ch_incremental.get()),
            "ch_clustering": self.ch_clustering.get(),
```

- [ ] **Step 4: Add the argv builders and the step collector**

Insert after `_collect_evaluation_steps` (line 812):

```python
    def _build_bronze_para_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "download/build_bronze_para.py", cfg]
        cells = self.ch_cells.get().strip().split()
        if cells:
            argv += ["--cells", *cells]
        if self.ch_overwrite.get():
            argv.append("--overwrite")
        if self.ch_incremental.get():
            argv.append("--incremental")
        return argv

    def _build_char_pipeline_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "-m", "characterize.main_para", cfg]
        cells = self.ch_cells.get().strip().split()
        if cells:
            argv += ["--cells", *cells]
        if self.ch_overwrite.get():
            argv.append("--overwrite")
        clustering = {"HDBSCAN": "hdbscan", "Classifier": "classifier"}.get(
            self.ch_clustering.get()
        )
        if clustering:
            argv += ["--clustering", clustering]
        return argv

    def _build_char_fit_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "-m", "characterize.fit_characterization", cfg]
        cells = self.ch_cells.get().strip().split()
        if cells:
            argv += ["--cells", *cells]
        return argv

    def _collect_characterization_steps(self) -> list[tuple[str, list[str]]] | None:
        """(label, argv) for each ticked stage; None if the config is invalid."""
        steps: list[tuple[str, list[str]]] = []
        for ticked, label, builder in (
            (self.ch_build.get(), "build_bronze_para", self._build_bronze_para_argv),
            (self.ch_pipeline.get(), "characterization pipeline",
             self._build_char_pipeline_argv),
            (self.ch_fit.get(), "characterization fit", self._build_char_fit_argv),
        ):
            if not ticked:
                continue
            argv = builder()
            if argv is None:
                return None
            steps.append((label, argv))
        return steps
```

- [ ] **Step 5: Add the run path**

Insert after `_run_train` (line 875):

```python
    def _run_characterization(self):
        if self._runner.is_running:
            messagebox.showwarning("Busy", "A stage is already running.")
            return
        steps = self._collect_characterization_steps()
        if steps is None:
            return  # invalid config — error already shown
        if not steps:
            messagebox.showinfo(
                "Nothing selected", "Tick at least one characterization stage to run."
            )
            return
        self._append_console("=== Running initial characterization ===\n")
        self._launch_chain(steps)
```

- [ ] **Step 6: Verify the module parses and the new methods exist**

`pipeline_ui` needs a display and `python3-tk`, so check by AST rather than import:

```bash
python - <<'PY'
import ast, pathlib
src = pathlib.Path("pipeline_ui.py").read_text()
tree = ast.parse(src)                      # syntax check
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and any(
               isinstance(b, ast.FunctionDef) and b.name == "_build_evaluation_tab"
               for b in n.body))
names = {b.name for b in cls.body if isinstance(b, ast.FunctionDef)}
for m in ("_build_characterization_tab", "_build_bronze_para_argv",
          "_build_char_pipeline_argv", "_build_char_fit_argv",
          "_collect_characterization_steps", "_run_characterization"):
    assert m in names, m
assert '7. Initial Characterization' in src
for key in ("ch_cells", "ch_build", "ch_pipeline", "ch_fit",
            "ch_overwrite", "ch_incremental", "ch_clustering"):
    assert src.count(f'"{key}"') >= 2, key   # restore + persist
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline_ui.py
git commit -m "feat(ui): tab 7 for initial characterization

Checklist of BRONZE_PARA build / characterization pipeline / fit+plot, run in
sequence by one button, with the shared cells filter, overwrite, incremental
and clustering controls. Outside the 'Run all' chain like the classifier tab."
```

---

### Task 9: Documentation and PR

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update `README.md`**

Line 52 mentions the tag. Change `<bucket>/<minio_prefix>/10_TRACY/` to `<bucket>/<minio_prefix>/TRACY/` and append to that row: `(objects written before the rename are still read from the legacy 10_TRACY/ prefix)`.

- [ ] **Step 2: Update `CLAUDE.md` — tag references**

Replace every `10_TRACY` occurrence with `TRACY` (in the architecture, classifier, interpret-clusters and compare-labels sections), noting the read-fallback once, in the compare-labels bullet: `minio` reads tagged `TRACY/…` (falling back to the legacy `10_TRACY/…`) + untagged `60_classifier/…`.

- [ ] **Step 3: Update `CLAUDE.md` — new section**

Add a section after "Evaluation", before "Aging-status monitor":

```markdown
## Initial characterization (BOL parametrization)

A separate, one-off track for the parametrization run that precedes aging.
Parametrization test files sit in the same per-cell download folders as the CU
tests and are selected by `para_procedure_filter` (a **list** of programme-name
substrings; one element is the normal case).

1. **`download/build_bronze_para.py <cfg> [--cells …] [--overwrite] [--incremental]`**
   — the CU builder (`build_bronze_cu_with_ah.run`) parameterized by
   `layer="BRONZE_PARA"` + `filter_key="para_procedure_filter"`. Writes
   `<working_path>/BRONZE_PARA/<cell>.parquet` + manifest sidecar; MinIO
   `<prefix>/BRONZE_PARA/`.
2. **`python -m characterize.main_para <cfg> [--cells …] [--overwrite] [--clustering …]`**
   — `main.run_pipeline` with the `CHARACTERIZATION` run context
   (`util/run_context.py`). Same dismember → features → clustering → calculate
   stages; reads BRONZE_PARA and forces the pulse/EIS/qOCV exports on.
3. **`python -m characterize.fit_characterization <cfg> [--cells …]`** — fits
   and plots the bundles. Standalone, so fits can be repeated without redoing
   segmentation.

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
├── GOLD.parquet
├── with_features_post_labeled.csv
├── <cell>_capacity.csv
├── data/   <cell>_{pulse,eis,qocv_dch,qocv_cha}_BM<n>_<SOH>SOH.parquet
└── plots/  pulse_2rc.png · eis_2zarc_warburg.png · qocv.png
```

MinIO mirrors this under `<prefix>/10_initial_characterization/<cell_stem>/`,
**untagged**. The `--overwrite` skip-check keys on this root's GOLD.

**Models are fixed defaults**, not config keys: **2RC** for pulse
(`analysis/fit_2rc_pulse.py`), **2×ZARC + series-L + generalized Warburg** for
EIS (`analysis/eis_vs_soc.fit_zarc_warburg_eis`), and no fit for qOCV (the
curve plus its throughput capacities). The EIS diffusion element has φ
**fitted** (`DIFFUSION_PHI_BOX`) but τ_d **pinned** (`DIFFUSION_TAU_BOX`, 5 s):
`R_d_z` is the amplitude at ω = 1/τ_d and `tau_d_z` a shape constant, not a
result. Those settings — element, pinned τ_d, φ box, `ZARC_ALPHA_MIN` — are
recorded in `parameters.json`'s `eis.settings`; both were tuned on the NFPP
sweep and may need retuning for another cell type or frequency range.

**UI**: Tab 7 runs the three stages as a checklist. Like Tab 6 it is **outside**
the "Run all 1→2→3→4→5" chain.
```

- [ ] **Step 4: Update `CLAUDE.md` — key parameters table**

Add these rows to the "Key parameters" table:

```markdown
| `para_procedure_filter` | **List** of programme-name substrings marking parametrization test files (single-element list is normal). Required by `build_bronze_para.py` and `characterize.main_para`. |
```

And to the "CLI flags" section, note that `characterize.main_para` takes the same `--cells` / `--overwrite` / `--clustering` flags as `main.py`.

- [ ] **Step 5: Update the UI section of `CLAUDE.md`**

In the "Unified UI" bullet list, add:

```markdown
- Tab 7 → `download/build_bronze_para.py` → `python -m characterize.main_para` → `python -m characterize.fit_characterization`, each gated by its checkbox and run in sequence. Outside the chain (one-off BOL step).
```

and change the opening sentence from "six stages" to "seven stages".

- [ ] **Step 6: Commit and push**

```bash
git add CLAUDE.md README.md
git commit -m "docs: initial characterization track and TRACY rename"
git push -u origin feat/initial-characterization
```

- [ ] **Step 7: Open the PR**

```bash
gh pr create --base J8005_BMWK_METABatt \
  --title "feat: initial characterization (BOL parametrization) + TRACY rename" \
  --body "$(cat <<'EOF'
## Summary
- New BOL characterization track: `BRONZE_PARA` build → `characterize.main_para` → `characterize.fit_characterization`, writing data/, `<cell>_parameters.json` and plots/ under `10_initial_characterization/<cell>/`.
- `RunContext` (`util/run_context.py`) carries the BRONZE layer, procedure-filter key and output root, threaded explicitly through the pipeline with CU defaults — the check-up path is unchanged.
- MinIO prefix tag renamed `10_TRACY` → `TRACY`, with a read-fallback to the legacy prefix so nothing already uploaded is orphaned.
- New UI tab 7, outside the "Run all" chain.

## Design
`docs/superpowers/specs/2026-08-21-initial-characterization-design.md`

## Verification
Path/key/signature logic verified with self-contained assertion snippets (no lab data). **End-to-end runs against real data are outstanding and left to the maintainer** — in particular a cell with a complete parametrization run, since the local Namey folder only exercises the EIS branch.
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| 1. `10_TRACY` → `TRACY` + fallback | Task 1 |
| 2. BRONZE_PARA + `para_procedure_filter` | Task 5 |
| 3. Characterization pipeline, export routing A′, layout | Tasks 2, 3, 4, 6 |
| 4. Parametrization, plots, `parameters.json`, EIS settings | Task 7 |
| 5. UI Tab 7 | Task 8 |
| 6. Verification deferred, PR | Task 9 |
| Docs (`CLAUDE.md`, `README.md`) | Task 9 |

**Deviation from the spec, flagged:** the spec described `export_root` as a single threaded parameter. The plan threads a `RunContext` carrying `export_root_prefix` **plus** `bronze_layer` and `procedure_filter_key`, because the para run needs all three and threading them as three separate parameters through the same four frames triples the signature churn for no gain. It remains an explicit parameter, not config state — the property the spec's decision turned on.

**Placeholder scan:** no TBD/TODO; every code step carries the actual code; the one judgement call (`export_capacity_object_key`'s existing body, Task 2 Step 4) is called out explicitly with instructions to read before replacing.

**Type consistency:** `run_ctx` is the parameter name in every frame (Tasks 3, 4); `root=` is the io_router keyword (Tasks 2, 3, 4); `layer=` is the bronze keyword (Tasks 2, 3, 5); `RunContext.export_root(cell)` returns `str | None` and is the only producer of `root` values; `_build_paths` returns the same seven keys in both branches plus `"root"`.
