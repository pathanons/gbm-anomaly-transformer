# Manuscript Workspace

This folder contains bilingual-ready LaTeX starters for the GBM-aware Anomaly Transformer research project.

## Layout

```text
manuscript/
  thesis-th/        Thai thesis draft with Thai and English abstracts
  paper-en/         English paper draft
  shared/
    figures/        Shared figures for both drafts
    tables/         Shared tables for both drafts
    references.bib  Shared BibTeX database
```

## Overleaf Setup

Recommended workflow:

1. Create one Overleaf project for `thesis-th`.
2. Create one Overleaf project for `paper-en`.
3. Upload the relevant project folder plus the `shared` folder.
4. Set the compiler to `XeLaTeX`.
5. Keep figures, tables, and references in `shared/` so both drafts use the same assets.

For the thesis, upload:

```text
thesis-th/
shared/
```

For the paper, upload:

```text
paper-en/
shared/
```

## Syncing Changes To Overleaf

For repeat updates from this local repo, use Overleaf's Git integration. Get the
Git URL from the Overleaf project menu, then run one of:

```bash
scripts/manuscript/sync_overleaf.sh paper-en https://git.overleaf.com/PROJECT_ID
scripts/manuscript/sync_overleaf.sh thesis-th https://git.overleaf.com/PROJECT_ID
```

Use a separate Overleaf project URL for `paper-en` and `thesis-th`. The script
pushes only the selected manuscript folder plus `shared/`, so the research code
and generated experiment files stay out of Overleaf.

If your Overleaf project uses a branch other than `master`, set:

```bash
REMOTE_BRANCH=main scripts/manuscript/sync_overleaf.sh paper-en https://git.overleaf.com/PROJECT_ID
```

## Research Claim Guardrails

Current evidence supports careful claims about:

- Dataset construction and leakage-aware split protocol.
- GBM-implied distribution consistency as the selected modeling direction.
- Score-change dynamic thresholding as a high-precision but low-sensitivity operational alert rule.

Do not claim the proposed model outperforms baselines until the project has:

- Baseline comparison under the same split and tuning budget.
- Multi-seed evaluation.
- Confidence intervals and statistical tests.
- Lead-time analysis if making early-warning claims.
- Explanation faithfulness or stability tests if making interpretability claims.
