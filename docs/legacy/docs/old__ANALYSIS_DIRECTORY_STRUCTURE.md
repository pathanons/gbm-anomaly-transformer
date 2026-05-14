## ANALYTICAL RESULTS DIRECTORY STRUCTURE
==========================================

```
results/
├── eda/                                  [EDA Analysis Results]
│   ├── AAPL/
│   │   ├── AAPL_autocorrelation.png
│   │   ├── AAPL_powerlaw_fit.png
│   │   ├── AAPL_eda_summary.csv
│   │   └── ...
│   ├── MSFT/
│   ├── ... (111 tickers total)
│   
├── analysis/                             [DEEP ANALYTICAL RESULTS]
│   │
│   ├── anomaly_patterns/                 [Script 1: Anomaly Frequency & Severity]
│   │   ├── AAPL/
│   │   │   ├── AAPL_anomaly_timeline.png             # Anomalies over time by severity
│   │   │   ├── AAPL_severity_distribution.png        # Count of each severity tier
│   │   │   ├── AAPL_price_change_dist.png            # Price changes on anomaly vs normal days
│   │   │   ├── AAPL_analysis.json                    # All metrics (JSON)
│   │   │   └── AAPL_summary.csv                      # Quick summary table
│   │   ├── MSFT/
│   │   ├── ... (111 tickers total)
│   │
│   ├── statistical_insights/             [Script 2: Statistical Relationships]
│   │   ├── AAPL/
│   │   │   ├── AAPL_volatility_anomaly.png          # Volatility vs Anomaly correlation
│   │   │   ├── AAPL_volume_distribution.png          # Volume ratio on anomaly days
│   │   │   ├── AAPL_insights.json                    # Correlations & relationships
│   │   │   └── ... (more analysis plots)
│   │   ├── MSFT/
│   │   ├── ... (111 tickers total)
│   │
│   ├── COMPREHENSIVE_ANALYSIS_REPORT.md  [Master Report with all findings]
│
├── experiments/
├── interpretability/
├── logs/
├── models/
├── plots/
├── reports/
└── scores/
```

---

## KEY OUTPUT FILES EXPLAINED

### Anomaly Pattern Analysis Files
**File**: `{TICKER}_anomaly_timeline.png`
- **What**: Time-series plot showing all anomalies by severity
- **X-axis**: Date
- **Y-axis**: Close Price  
- **Colors**: Yellow (score 1), Orange (score 2), Red (score 3)
- **Use**: Visual inspection of when anomalies occur

**File**: `{TICKER}_severity_distribution.png`
- **What**: Bar chart showing count of each severity level
- **X-axis**: Severity tier (no_anomaly, low, medium, high)
- **Y-axis**: Count (number of trading days)
- **Use**: Understand prevalence of each severity class

**File**: `{TICKER}_price_change_dist.png`
- **What**: Histogram comparing price changes
- **Blue bars**: Normal days (no anomaly)
- **Red bars**: Anomaly days (at least one criterion met)
- **Use**: Show if anomalies coincide with large price moves

**File**: `{TICKER}_analysis.json`
- **What**: Raw numerical results
- **Contents**:
  ```
  {
    "frequency": {
      "total_days": 3500,
      "anomaly_days": 245,
      "anomaly_percentage": 7.0,
      "avg_days_between_anomalies": 14.3,
      ...
    },
    "severity_tiers": {...},
    "temporal_patterns": {...},
    "rolling_statistics": {...}
  }
  ```

### Statistical Insights Files
**File**: `{TICKER}_volatility_anomaly.png`
- **What**: Scatter plot of volatility vs anomaly score
- **X-axis**: 5-day rolling volatility (%)
- **Y-axis**: Anomaly score (0-3)
- **Color**: Anomaly score shade
- **Use**: Detect correlation between volatility and anomalies

**File**: `{TICKER}_volume_distribution.png`
- **What**: Histogram comparing volume ratios
- **Blue bars**: Volume ratio on normal days
- **Red bars**: Volume ratio on anomaly days
- **Use**: Show if anomalies have unusual volume

**File**: `{TICKER}_insights.json`
- **What**: Statistical relationships
- **Contents**:
  ```
  {
    "volatility_correlation": {
      "correlation_volatility_5d_anomaly": 0.34,
      ...
    },
    "volume_correlation": {...},
    "return_distribution": {...},
    "lead_lag_effects": {...}
  }
  ```

### Master Report
**File**: `COMPREHENSIVE_ANALYSIS_REPORT.md`
- **What**: Markdown document with full analysis + interpretation
- **Sections**:
  1. Executive Summary
  2. Variable Definitions
  3. Anomaly Pattern Results
  4. Statistical Relationship Results
  5. Key Findings
  6. Portfolio Implications
  7. Recommendations

---

## METRICS EXPLAINED

### From Anomaly Patterns

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| `anomaly_percentage` | (anomaly_days / total_days) × 100 | What % of trading days had anomalies? |
| `avg_days_between_anomalies` | Mean gap between consecutive anomalies | How frequently do anomalies occur? |
| `median_days_between_anomalies` | Median gap (less affected by outliers) | Typical spacing between anomalies |

### From Severity Tiers

| Tier | Criteria | Meaning |
|------|----------|---------|
| `no_anomaly` (0) | Count of days with score 0 | Normal trading days |
| `low_severity` (1) | Count of days with score 1 | One detection criterion triggered |
| `medium_severity` (2) | Count of days with score 2 | Two criteria triggered |
| `high_severity` (3) | Count of days with score 3 | All three criteria triggered |

### From Temporal Patterns

| Metric | Definition |
|--------|-----------|
| `avg_price_change_on_anomaly_days` | Mean of \|return\| when anomaly occurs |
| `avg_price_change_normal_days` | Mean of \|return\| on normal days |
| `price_change_ratio` | Anomaly_change / Normal_change |
| `volume_ratio` | Anomaly_volume / Normal_volume |

**Interpretation**:
- `price_change_ratio > 2.0`: Anomalies involve 2x larger price swings
- `volume_ratio > 1.3`: Higher trading activity on anomaly days

### From Statistical Insights

| Metric | Definition |
|--------|-----------|
| `correlation_volatility_5d_anomaly` | Pearson correlation(-1 to +1) |
| `ttest_p_value` | Statistical significance of distribution difference |
| `lead_lag_interpretation` | Whether anomalies lead, lag, or coincide with movements |

---

## EXAMPLE INTERPRETATION WORKFLOW

**Scenario**: Analyzing AAPL results

1. **Check frequency**:
   - "AAPL has 7% anomaly percentage"
   - → Roughly 1 in 14 trading days
   - → Not unusual, but regular enough to matter

2. **Check severity**:
   - "High severity: 2%, Medium: 5%, Low: 8%"
   - → Most anomalies are mild
   - → High severity events rare (potentially very important)

3. **Check temporal patterns**:
   - "Price change ratio: 2.5"
   - → Anomalies involve 2.5x larger price moves
   - → Anomalies ARE correlated with price volatility

4. **Check statistical insights**:
   - "Volatility correlation: 0.62" (fairly strong)
   - "Volume ratio: 1.15" (slightly higher volume)
   - "Lead-lag: concurrent"
   - → Anomalies occur during volatility spikes
   - → Volume slightly elevated but not dramatic
   - → Anomalies coincide with movement (not predictive)

5. **Conclusion**:
   - AAPL anomalies are moderately frequent
   - They systematically coincide with volatility bursts
   - Useful as contemporaneous indicator, not predictive signal
   - Worth monitoring for risk management

---
