# Manuscript — signal-based check-up decomposition

Target: *Journal of Power Sources* / *Journal of Energy Storage*.
Strategy and scope decisions: `.claude/handoffs/2026-07-23-paper-scope-checkup-segmentation.md`.

## Build

```bash
latexmk -pdf main.tex
```

Needs `elsarticle`. On Ubuntu 24.04:

```bash
sudo apt install -y --no-install-recommends \
  latexmk texlive-latex-recommended texlive-latex-extra \
  texlive-publishers texlive-science texlive-fonts-recommended
```

`texlive-publishers` carries `elsarticle.cls` and `elsarticle-num.bst`;
`texlive-science` carries `siunitx`. Those two are the easy ones to miss.

Builds clean (11 pages, 5 references, no undefined citations, no overfull boxes
over 20 pt). Citations need two `latexmk` passes on a cold build — run it twice
if you see `Citation ... undefined`.

Build artifacts are gitignored. Note that Elsevier wants `main.bbl` included at
submission time, so un-ignore it when you get there.

## Draft conventions

Unrun experiments are wrapped in `\todobox{}{}`; inline gaps use `\todo{}`.
Both render as visible red text so a placeholder cannot be mistaken for a
result. Set `\draftfalse` in the preamble to hide them all before circulating
externally — but note that hides the fact that those sections are empty, so
prefer to fill them.

No number in this draft is invented. Every figure quoted is traceable to the
source below; anything not yet measured is a TODO box, not a placeholder value.

## Where the numbers come from

| Section | Content | Source |
|---|---|---|
| 5.1 | Segmentation metadata ablation (L0/L1/L2) | `docs/segmentation_metadata_audit.md`; regenerate with `evaluation.segmentation_audit` |
| 5.2 | Cross-chemistry transfer | Run of 2026-07-23. **Per-cell artefacts overwritten**; the finding is reproduced under controlled folds in 5.5 |
| 5.3 | Hand-labelled segment F1 | Not run — needs a human labeller |
| 5.4 | Detection recall + fixed-threshold baseline | `evaluation.detection_recall` → `50_evaluation/detection_recall_fleet_*.csv` |
| 5.5 | Label space vs features | `evaluation.feature_ablation` → `50_evaluation/feature_ablation_*.csv` |
| 5.7 | Field data (20 on-road EVs) | `field.extract_capacity` then `field.benchmark_shiyunliu --our-dir <dir>`. Needs the metric fix on branch `fix/field-benchmark-trend-residual` |

The three `figures/field_*.pdf` are checked in as build artifacts. They and
§5.7's numbers reflect the **dSOC-alone** selection (`field-data`, commit
`12eb905`); anything generated before 2026-09-08 used the earlier circular
selection and does not match the text. Their
generator, `field/plot_field.py`, lives on the **`field-data`** branch, not
here — field code is kept off `main`. To regenerate:
`git checkout field-data -- src/field/plot_field.py` then
`python -m field.plot_field --out-dir ../paper/figures --vehicle 3`.
| 5.6 | Cross-laboratory (ISU-ILCC, UConn-ILCC) | `evaluation.external_validation` → `50_evaluation/external_validation_*.csv`. Needs the layer-2 fix on branch `fix/layer2-capacity-cluster-selection` |

Regenerate 5.4 and 5.5:

```bash
cd src
for c in VTC APR Hina; do
  python -m evaluation.detection_recall \
    /home/ann/Documents/Data_Metabatt/battery_config_${c}_linux.json \
    -o /home/ann/Documents/Data_Metabatt/50_evaluation
done
python -m evaluation.detection_recall --pool -o /home/ann/Documents/Data_Metabatt/50_evaluation

python -m evaluation.feature_ablation \
  /home/ann/Documents/Data_Metabatt/battery_config_{VTC,APR,Hina}_linux.json \
  -o /home/ann/Documents/Data_Metabatt/50_evaluation
```

Regenerate 5.6 (external laboratories):

```bash
python -m evaluation.external_validation \
  --external ../battery_config_ISU_linux.json ../battery_config_UConn_linux.json \
  --in-house /home/ann/Documents/Data_Metabatt/battery_config_{VTC,APR,Hina}_linux.json \
  -o /home/ann/Documents/Data_Metabatt/50_evaluation
```

**The UConn epsilon workaround.** `battery_config_UConn_linux.json` had
`cluster_selection_epsilon: 0.0` on both HDBSCAN layers, set to dodge the
sklearn/numpy crash described at the bottom of this file. That crash is an
environment problem, not a code one, so the config now carries the in-house
values (0.3 / 0.001). These configs are gitignored, so that change lives only on
disk — check it before reproducing 5.6.

**Caveat on the ISU clustering row.** `external_validation` evaluates whichever
routes a dataset has been run through. ISU-ILCC had only been run through the
classifier route, so its clustering number (89.3 %) came from a separate HDBSCAN
run on a *copy* of `ISU_pipeline/BRONZE_CU` — done on a copy because an in-place
run would have overwritten the classifier-route GOLD and capacity exports. To
reproduce it, copy `BRONZE_CU` to a scratch `working_path`, drop
`classifier_model_path` from the config, and run
`main.py <scratch_cfg> --clustering hdbscan --overwrite`; the resulting
`with_features_post_labeled/*.csv` are what the row is computed from. Those CSVs
are **not** currently in the ISU tree.

Comparing a laboratory under one route against a laboratory under another
confounds route with laboratory — which routes a dataset happens to have been
processed with is an accident of its history. The module now loads every
available route separately for this reason.

## Status

Written from real results: **5.1, 5.2, 5.4, 5.5, 5.6, 5.7**. Scaffolding: 5.3,
related work, conclusion.

Before submission, in rough priority order:
1. **Hand-labelled F1 (5.3)** — the only place labels meet ground truth rather
   than another automated route.
2. The field capacity tables in
   `field_data/shiyunliu_20ev/40_capacity_monitore/` were regenerated on
   2026-09-08; the previous files dated from 2026-06-09 and were produced by
   older code, which made session retention look bimodal (24 to 2701 rows per
   vehicle against a true 155-457). A backup of the stale set is not kept in the
   repo. Regenerate with `python -m field.extract_capacity --vehicle N`.
3. Widen 5.6 beyond capacity: both external sources give an independent ground
   truth for the capacity test only, and both are NMC-type, so laboratory and
   chemistry are not fully crossed. Pozzato–Onori carries the full taxonomy.
4. Either re-run and archive the per-cell 5.2 comparison, or cut its cell-level
   counts and let 5.5 carry the finding. Do not ship numbers whose artefacts
   are gone.
5. The add-them-back feature ablation owed by §3.3 (voltage-shape and
   temperature features).
6. Related work (§2) and the conclusion.
7. Decide whether to bring in the full fleet (200 LFP, 80 Na-ion) — Na-ion is
   n=1 here and a reviewer will notice.

## A note on reproducing the numbers

The venv must match `requirements_linux.txt`. It had drifted to numpy 2.5.1
against the pinned 2.0.0, which makes sklearn 1.5.1's HDBSCAN raise
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
whenever `cluster_selection_epsilon > 0` reaches its epsilon search — silently
failing whole cells. Upstream issue scikit-learn#33355 is open, so upgrading
does not help; restore the pin instead.
