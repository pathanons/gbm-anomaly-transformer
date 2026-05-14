# Anomaly Transformer for U.S. Equity Market Anomaly Detection

**Version**: 3.0 - Interview-Driven Research Framework  
**Date**: April 2, 2026  
**Status**: Aligned with publication plan

---

## Research Objective

Build an interpretable hybrid anomaly detection system for daily U.S. large-cap equities. The project uses Anomaly Transformer as the backbone and extends it with regime awareness, multi-scale temporal modeling, cross-sectional stock relations, and an anomaly explanation head.

## Core Research Questions
1. Can the model detect financial anomalies earlier than standard baselines?
2. Can it explain which features and stock-to-stock relations triggered the alert?
3. Does it remain robust across different market regimes?
4. Can the framework support publication in an IEEE-style AI/ML venue with finance relevance?

## Data Setting
- Daily U.S. equity data.
- Large-cap stocks, especially S&P 500 names.
- Inputs: OHLCV and returns.
- Multiple stock universes may be used for ablation.

## Composite Anomaly Taxonomy
### Level 1: Point Anomalies
- return jump/drop
- volume spike
- realized volatility shock

### Level 2: Collective Anomalies
- unusual multi-day windows
- persistent stress patterns

### Level 3: Regime Anomalies
- structural breaks
- market-wide phase shifts

## Model Design
### Backbone
- Anomaly Transformer

### Extensions
- Multi-scale temporal module for short and medium lag structure.
- Cross-sectional stock relation module for stock-to-stock dependencies.
- Explanation head for feature attribution and relation attribution.
- Regime-aware behavior for changing market conditions.

## Evaluation Framework
- Time-based train/validation/test split.
- Rolling window backtest.
- Regime-split evaluation: pre-crisis, crisis, post-crisis.
- Case studies selected from the most prominent anomalies found by the model.

## Benchmarks
- Vanilla Anomaly Transformer
- LSTM/GRU autoencoder
- Forecasting-based detectors
- Additional classical anomaly baselines

## Metrics
Primary order of importance:
1. Early warning recall within the event window
2. Accuracy, F1, and ROC-AUC
3. Interpretability quality

## Expected Contribution
The paper should argue that the model is not only a detector but also a financial explanation system that is suitable for regulatory and analytical use.

## Paper Framing
The work should be positioned as a balanced methodology + application paper that remains credible for IEEE/AI audiences while still being meaningful for financial anomaly analysis.

