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

Builds clean (8 pages, 5 references, no undefined citations, no overfull boxes
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
| 5.6 | External dataset (Pozzato–Onori) | Not run |

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

## Status

Written from real results: **5.1, 5.2, 5.4, 5.5**. Scaffolding: 5.3, 5.6,
related work, conclusion.

Before submission, in rough priority order:
1. **External-dataset validation (5.6)** — otherwise the portability claim must
   be scoped to a single laboratory in the abstract, not just the limitations.
2. **Hand-labelled F1 (5.3)** — the only place labels meet ground truth rather
   than another automated route.
3. Either re-run and archive the per-cell 5.2 comparison, or cut its cell-level
   counts and let 5.5 carry the finding. Do not ship numbers whose artefacts
   are gone.
4. The add-them-back feature ablation owed by §3.3 (voltage-shape and
   temperature features).
5. Related work (§2) and the conclusion.
6. Decide whether to bring in the full fleet (200 LFP, 80 Na-ion) — Na-ion is
   n=1 here and a reviewer will notice.

## A note on reproducing the numbers

The venv must match `requirements_linux.txt`. It had drifted to numpy 2.5.1
against the pinned 2.0.0, which makes sklearn 1.5.1's HDBSCAN raise
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
whenever `cluster_selection_epsilon > 0` reaches its epsilon search — silently
failing whole cells. Upstream issue scikit-learn#33355 is open, so upgrading
does not help; restore the pin instead.
