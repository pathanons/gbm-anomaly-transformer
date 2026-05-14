# Research Journey

**Project**: Interpretable hybrid anomaly detection for U.S. equities  
**Research Focus**: Publication-ready AI/ML paper with finance relevance  
**Timeline**: 4 phases  
**Date Started**: April 2, 2026

---

## Mission Statement

Build a research paper around a daily U.S. equity anomaly detector that is early, explainable, and robust across market regimes. The framework keeps the Anomaly Transformer backbone and extends it for stock-level relations, multi-scale time dynamics, and anomaly explanations.

## What We Are Studying
- How daily stock anomalies appear in OHLCV and returns.
- Which historical windows are most useful for early warning.
- Which features and stock relations explain the anomaly signal.
- Whether regime-aware modeling improves usefulness and robustness.

## Formal Research Claims
1. Financial anomalies can be defined through a composite rule taxonomy.
2. Early warning recall is more important than delayed detection.
3. Feature attribution and cross-sectional relation changes can explain alerts.
4. Regime-aware modeling improves stability across different market periods.

## Research Phases
### Phase 1: Data and Label Design
- Use daily U.S. large-cap equity data.
- Keep OHLCV and returns as the core input features.
- Build composite labels from return jumps/drops, volume spikes, and volatility shocks.
- Prepare multiple universes for ablation.

### Phase 2: Model Design
- Start with Anomaly Transformer.
- Add multi-scale temporal modeling.
- Add cross-sectional stock relation modeling.
- Add an explanation head for feature and stock-to-stock attribution.

### Phase 3: Experimental Validation
- Use train/validation/test splits by time.
- Run rolling window backtests.
- Evaluate pre-crisis, crisis, and post-crisis periods separately.
- Compare against vanilla Anomaly Transformer, LSTM/GRU autoencoders, forecasting-based detectors, and classical baselines.

### Phase 4: Publication Preparation
- Select the most prominent anomalies as case studies.
- Report early warning recall first, then ROC-AUC/F1, then interpretability.
- Package the work as an IEEE-friendly AI/ML paper.

## Success Criteria
- Better early warning than benchmark methods.
- Competitive detection quality.
- Clear explanation of feature and stock-relation drivers.
- Evidence that the model remains useful across regimes.

## Timeline
### Week 1
- Finalize anomaly taxonomy.
- Confirm features and universes.
- Validate the data split and baseline pipeline.

### Week 2
- Implement model extensions.
- Add explanation outputs.
- Start baseline comparisons.

### Week 3
- Run rolling backtests.
- Run regime-split evaluation.
- Complete ablation studies.

### Week 4
- Write results and case studies.
- Finalize paper structure and tables.

## End State
The final deliverable is a publishable paper with a defensible methodology, measurable early warning improvement, and explanations that are meaningful to analysts or regulators.

