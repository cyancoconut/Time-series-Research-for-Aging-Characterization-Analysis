# Initial characterization — design

Date: 2026-08-21
Branch: `feat/initial-characterization`

## Goal

Produce a per-cell **initial characterization** (BOL parametrization) artefact set:
the measurement bundles, the fitted model parameters, and the plots for pulse,
EIS and qOCV. The parametrization test files run through their own BRONZE build
and a separate pipeline entry point, so nothing collides with the recurring
check-up (CU) aging outputs.

Secondary: rename the MinIO prefix tag `10_TRACY` to `TRACY`.

## 1. `10_TRACY` → `TRACY`

`UPLOAD_PREFIX_TAG` (`util/io_router.py:20`) becomes `"TRACY"`. The tag is
MinIO-only — local paths never carry it — so the change touches uploads and the
tagged readers: `list_x_silver_cells`, `list_csv_objects` / `fetch_csv_object`
(used by `train_classifier`), and `compare_labels._HDBSCAN_REL`.

Continuity without a bulk copy: add `LEGACY_PREFIX_TAG = "10_TRACY"` and a
`_tagged_dir(rel)` helper. Listing functions try `TRACY/<rel>` first and fall
back to `10_TRACY/<rel>` when the new prefix returns nothing. **Writes always go
to the new tag.** Objects already uploaded stay readable.

Docs updated alongside: `CLAUDE.md`, `README.md`.

## 2. BRONZE_PARA

Parametrization files sit in the same per-cell download folders as the CU tests
and are selected by a different token in the 4th `=`-delimited filename field.
So this is the existing builder with two things swapped: which filter key
selects a file, and which layer it writes to.

`download/build_bronze_cu_with_ah.py` gains two parameters, defaulted to
today's values so the CU path is unchanged:

- `run(cfg, …, layer="BRONZE_CU", filter_key="procedure_filter")`, threaded into
  `process_cell` (which reads `cfg["procedure_filter"]` at line 449 and raises
  if unset) and into the manifest helpers `_manifest_minio_key` /
  `_manifest_local_path`.
- `io_router.bronze_object_key(cell, layer="BRONZE_CU")`, and the same optional
  `layer` on `bronze_exists_on_minio`, `fetch_bronze`, `open_bronze_range`.

`download/build_bronze_para.py` is a thin CLI calling
`run(cfg, layer="BRONZE_PARA", filter_key="para_procedure_filter")`, with the
same `--cells` / `--overwrite` / `--incremental` flags.

New config key **`para_procedure_filter`** — a list of substrings; the single
element case is normal. Normalized by `util.procedure_filter.as_filter_list`
(which also tolerates a bare string). Required by the para CLI exactly as
`procedure_filter` is required today.

Output: `<working_path>/BRONZE_PARA/<cell>.parquet` + `<cell>_manifest.json`;
MinIO `<prefix>/BRONZE_PARA/`. Ah integration stays enabled — harmless at BOL
and it keeps a single code path.

## 3. Characterization pipeline

`characterize/main_para.py` is a thin wrapper. It loads the battery config and
calls `main.run_pipeline` with two new arguments, both defaulted so the CU path
is unchanged:

- `bronze_layer="BRONZE_PARA"`
- `export_root="10_initial_characterization/<cell_stem>"`

Segmentation, feature extraction, clustering (`auto` / `hdbscan` / `classifier`,
same `--clustering` flag and semantics as `main.py`) and `calculate/` all run
unchanged.

### Export routing (approach A′ — explicit parameter, not config state)

`export_root` is threaded as one optional parameter, `None` = today's behaviour:

```
run_pipeline(cfg, …, export_root=None)
  └─ _process_cell(cell, cfg, client, exceptions, export_root)
       └─ _process_cell_inner(…, export_root)
            ├─ _build_paths(cell, working_path, classifier, export_root)
            └─ export_pulse / export_eis / export_qocv / export_capacity(…, export_root)
                 └─ io_router.export_*_object_key(cell, filename, root=export_root)
```

Chosen over stashing the root in `cfg` (which flows everywhere for free, but
hides the destination of every artefact behind a config key) and over
export-then-relocate (which duplicates data into the CU export tree). Cost is
~a dozen mechanical signature edits.

### Layout

`export_root` covers **everything the run writes**, so a para run can never
overwrite the CU `GOLD/<cell>.parquet` for the same cell:

```
<working_path>/10_initial_characterization/<cell_stem>/
├── <cell>_parameters.json
├── GOLD.parquet                     ← not GOLD/<cell>.parquet
├── with_features_post_labeled.csv
├── <cell>_capacity.csv              ← stays out of 40_capacity_monitore/
├── data/
│   ├── <cell>_pulse_BM<n>_<SOH>SOH.parquet
│   ├── <cell>_eis_BM<n>_<SOH>SOH.parquet
│   └── <cell>_qocv_{dch,cha}_BM<n>_<SOH>SOH.parquet
└── plots/
    ├── <cell>_pulse_2rc.png
    ├── <cell>_eis_2zarc_warburg.png
    └── <cell>_qocv.png
```

MinIO mirrors this under `<prefix>/10_initial_characterization/<cell_stem>/`,
untagged (consistent with the other `NN_` export folders).

Consequences, both intended:

- The BOL capacity CSV lands here instead of `40_capacity_monitore/`, so the
  aging monitor and aging matrix are unaffected by characterization runs.
- The skip-check (`--overwrite`) keys on **this root's** GOLD, so a para run
  never skips merely because a CU GOLD exists.

`export_pulse` / `export_eis` / `export_qocv` are forced on for this run
regardless of the config flags — they are the point of it.

## 4. Parametrization and plots

`characterize/fit_characterization.py`, own CLI
(`python -m characterize.fit_characterization <cfg> [--cells …]`), reads
`10_initial_characterization/<stem>/data/` and writes beside it. Runnable
independently of the pipeline, so fits can be repeated without redoing
segmentation.

It imports the existing fitters rather than reimplementing any maths:

| Input bundle | Fitter | Output |
|---|---|---|
| `*_pulse_BM*.parquet` | `analysis.fit_2rc_pulse` (2RC) | R0, R1, τ1, R2, τ2, RMSE per pulse |
| `*_eis_BM*.parquet` | `analysis.eis_vs_soc.fit_zarc_warburg_eis` (2×ZARC + generalized Warburg) | R_ohm, (R, τ, α)×2, R_d, τ_d, φ_d, RMSE, degenerate flag per spectrum |
| `*_qocv_{dch,cha}*.parquet` | `analysis.qocv_curve` | no fit — curve + dQ/dV plot |

Both fitters are argparse scripts today, so the runner calls their importable
functions (`fit_2rc_pulse.run_cell_folder`, `eis_vs_soc.fit_zarc_warburg_eis`)
through a small adapter supplying the options they currently read off `args`.

Models are **fixed defaults**, not config keys: 2RC for pulse, 2×ZARC +
generalized Warburg for EIS.

### EIS diffusion element

The default `ZARC_DIFFUSION_ELEMENT = "generalized"` is
`R·coth((jωτ)^φ)/((jωτ)^φ)` with **φ fitted**, boxed to
`DIFFUSION_PHI_BOX = (0.2, 0.9)`. φ = 0.5 is ideal Fickian; NFPP fits land at
0.32–0.42 (sub-diffusive).

Two properties the params file must record so the fitted values stay
interpretable:

- **τ_d is pinned, not fitted** — `DIFFUSION_TAU_BOX = (5.0, 5.0)`. With φ free
  the branch has three parameters but the data constrains two, so a free τ_d
  bifurcates and `R_d` stops being comparable across spectra. `R_d_z` is the
  amplitude at ω = 1/τ_d, `phi_d_z` the slope; `tau_d_z` is a shape constant.
- **5 s and `ZARC_ALPHA_MIN = 0.6` were tuned on the NFPP sweep** and may need
  retuning for another cell type or frequency range.

These stay module constants; `parameters.json` makes them visible.

### `<cell>_parameters.json`

```json
{ "cell": "...", "generated_utc": "...", "nom_capacity": 28.0,
  "pulse": {"model": "2rc",
            "fits": [{"pulse_id": "13_16", "soc": 0.9, "i_pulse": -14.0,
                      "r0": 0.0021, "r1": 0.0009, "tau1": 3.2,
                      "r2": 0.0014, "tau2": 41.0, "rmse": 0.00042}]},
  "eis":   {"model": "2zarc_warburg",
            "settings": {"element": "generalized", "tau_d_pinned_s": 5.0,
                         "phi_box": [0.2, 0.9], "alpha_min": 0.6},
            "fits": [{"eis_number": 5, "U": 3.31, "R_ohm": 0.0018,
                      "R1_z": 0.0007, "tau1_z": 0.004, "alpha1_z": 0.82,
                      "R2_z": 0.0021, "tau2_z": 0.9, "alpha2_z": 0.71,
                      "R_d_z": 0.0035, "tau_d_z": 5.0, "phi_d_z": 0.38,
                      "zarc_rmse": 0.00011, "degenerate": false}]},
  "qocv":  {"capacity_cha_ah": 27.8, "capacity_dch_ah": 27.6},
  "sources": {"pulse": ["data/..."], "eis": ["data/..."], "qocv": ["data/..."]} }
```

A bundle that fails to fit is recorded with `null` parameters and an `error`
string rather than aborting the cell.

## 5. UI — Tab 7 "Initial Characterization"

Reuses Tab 5's pattern: a checklist whose `_collect_*_steps()` returns
`[(label, argv), …]` run in sequence by one button. Three checkboxes, all
ticked by default:

| Checkbox | Command |
|---|---|
| Build BRONZE_PARA | `download/build_bronze_para.py <cfg> [--cells …] [--overwrite] [--incremental]` |
| Run characterization pipeline | `python -m characterize.main_para <cfg> [--cells …] [--overwrite] [--clustering …]` |
| Fit + plot (pulse / EIS / qOCV) | `python -m characterize.fit_characterization <cfg> [--cells …]` |

Plus a shared `--cells` entry, an `--overwrite` checkbox and a Clustering
toggle, mirroring Tabs 2 and 3. New methods follow existing naming:
`_build_characterization_tab`, `_build_bronze_para_argv`,
`_build_char_pipeline_argv`, `_build_char_fit_argv`,
`_collect_characterization_steps`, `_run_characterization`. State persists via
the existing `_restore_state` / `_persist_state`.

Like Tab 6, Tab 7 sits **outside** the "Run all 1→2→3→4→5" chain —
characterization is a one-off BOL step, not part of the recurring aging loop.

## 6. Verification

Deferred to the user. On completion the work is committed and pushed, and a PR
opened into `J8005_BMWK_METABatt`.

Known data limitation for whoever runs it: the local
`J8049_Namey_NFPP_28Ah_01` folder holds a single EIS file and no pulse/qOCV
parquets, so it exercises the EIS branch only. A cell with a complete
parametrization run is needed to exercise pulse and qOCV.

## Out of scope

- Migrating existing `10_TRACY/*` MinIO objects (read-fallback instead).
- New fitting maths — every model already exists in `src/analysis/`.
- Configurable model choice (fixed defaults for now).
- Aggregating characterization results across cells (per-cell artefacts only).
