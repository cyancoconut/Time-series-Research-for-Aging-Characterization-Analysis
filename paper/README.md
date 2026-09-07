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
| 5.1 | Segmentation metadata ablation (L0/L1/L2) | `docs/segmentation_metadata_audit.md`; regenerate with `evaluation.segmentation_audit` (see that file) |
| 5.2 | Cross-chemistry transfer | Run of 2026-07-23. **Artefacts overwritten** — must be re-run and archived per chemistry before submission (see the TODO box in that section) |
| 5.3–5.7 | Hand-label F1, detection recall, rule baseline, feature ablation, external dataset | Not yet run |

## Status

Sections written from real results: 5.1 (complete), 5.2 (complete but needs its
artefacts regenerated). Everything else is scaffolding.

Before submission, in rough priority order:
1. Re-run and archive the cross-chemistry comparison (5.2).
2. External-dataset validation (5.7) — otherwise the portability claim must be
   scoped to a single laboratory in the abstract, not just the limitations.
3. Hand-labelled F1 (5.3) — the only ground-truth comparison in the paper.
4. Detection recall against the test plan (5.4).
5. Rule baseline (5.5) and feature ablations (5.6).
6. Decide whether to bring in the full fleet (200 LFP, 80 Na-ion) — see the
   TODO in Section 4.
