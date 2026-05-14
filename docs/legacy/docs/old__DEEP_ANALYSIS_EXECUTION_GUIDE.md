## DEEP ANALYTICAL SUITE - COMPLETE EXECUTION GUIDE
====================================================

**Overview**: 3 independent analysis scripts + 1 master runner for comprehensive anomaly insights

---

## 🎯 QUICK START

```bash
# ONE COMMAND TO RUN EVERYTHING:
python EXECUTE_DEEP_ANALYSIS.py --ticker ALL

# FOR SPECIFIC TICKERS:
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL MSFT GOOGL

# FOR SINGLE TICKER (DEBUG):
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL
```

---

## 📋 FULL EXECUTION ROADMAP

### PHASE 1: DATA PREPARATION (First time only)
```bash
# 1.1 Update stock prices to present date
python update_sp500_data.py --all

# 1.2 Update anomaly labels
python update_anomaly_labels.py --all
```

### PHASE 2: BASIC EDA
```bash
# 2.1 Run Week 1 exploratory analysis (all 3 scripts per ticker)
python EXECUTE_WEEK1_EDA.py --ticker ALL

# 2.2 OR run by sector
python EXECUTE_SECTOR_EDA.py

# 2.3 OR test single ticker first
python EXECUTE_WEEK1_EDA.py --ticker AAPL
```

### PHASE 3: DEEP ANALYTICAL INSIGHTS (NEW!)
```bash
# 3.1 Run comprehensive deep analysis
python EXECUTE_DEEP_ANALYSIS.py --ticker ALL

# This internally runs:
#   - ANALYZE_ANOMALY_PATTERNS.py
#   - ANALYZE_STATISTICAL_INSIGHTS.py
#   - Generates COMPREHENSIVE_ANALYSIS_REPORT.md
```

---

## 🔧 INDIVIDUAL SCRIPT COMMANDS

### 1. ANOMALY PATTERN ANALYSIS
**Purpose**: Understand frequency, severity, and clustering of anomalies

```bash
# All 111 stocks
python ANALYZE_ANOMALY_PATTERNS.py --all

# Specific tickers
python ANALYZE_ANOMALY_PATTERNS.py --ticker AAPL MSFT GOOGL

# Single ticker (debug)
python ANALYZE_ANOMALY_PATTERNS.py --ticker AAPL
```

**Outputs**:
- `results/analysis/anomaly_patterns/{TICKER}/`
  - `{TICKER}_anomaly_timeline.png` - Time series with anomalies marked
  - `{TICKER}_severity_distribution.png` - Distribution of severity tiers
  - `{TICKER}_price_change_dist.png` - Price change histogram
  - `{TICKER}_analysis.json` - All numerical metrics
  - `{TICKER}_summary.csv` - Quick lookup table

**Key Metrics**:
- Anomaly frequency (% of days)
- Days between anomalies (spacing)
- Severity tier distribution
- Price changes on anomaly vs normal days
- Volume patterns
- Daily price ranges

---

### 2. STATISTICAL INSIGHTS ANALYSIS
**Purpose**: Find correlations and relationships between anomalies and market metrics

```bash
# All stocks
python ANALYZE_STATISTICAL_INSIGHTS.py --all

# Specific tickers
python ANALYZE_STATISTICAL_INSIGHTS.py --ticker AAPL MSFT

# Single ticker
python ANALYZE_STATISTICAL_INSIGHTS.py --ticker AAPL
```

**Outputs**:
- `results/analysis/statistical_insights/{TICKER}/`
  - `{TICKER}_volatility_anomaly.png` - Scatter plot of volatility vs anomaly
  - `{TICKER}_volume_distribution.png` - Volume comparison histogram
  - `{TICKER}_insights.json` - All correlation statistics

**Key Metrics**:
- Volatility correlation (do anomalies occur during volatile periods?)
- Volume correlation (is volume elevated on anomaly days?)
- Price range analysis
- Return distribution statistics (t-tests)
- Lead-lag effects (do anomalies predict future moves?)

---

### 3. MASTER COMPREHENSIVE ANALYSIS
**Purpose**: Run all analyses + generate unified report

```bash
# Run everything for all tickers
python EXECUTE_DEEP_ANALYSIS.py --ticker ALL

# Run everything for specific tickers
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL MSFT GOOGL

# Run everything for single ticker
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL
```

**Outputs**:
- All outputs from scripts 1 & 2
- `results/analysis/COMPREHENSIVE_ANALYSIS_REPORT.md` - Master report

---

## 📊 OUTPUT LOCATIONS

```
results/
├── eda/                          [Week 1 EDA results]
│   ├── AAPL/
│   ├── MSFT/
│   └── ... (111 tickers)
│
└── analysis/                     [Deep Analysis Results]
    ├── anomaly_patterns/
    │   ├── AAPL/
    │   └── ... (111 tickers)
    │
    ├── statistical_insights/
    │   ├── AAPL/
    │   └── ... (111 tickers)
    │
    └── COMPREHENSIVE_ANALYSIS_REPORT.md (Master report)
```

---

## 🔍 UNDERSTANDING THE OUTPUTS

### Anomaly Patterns Analysis

**Files Generated**:
1. `{TICKER}_anomaly_timeline.png`
   - Visual: Times series with color-coded anomaly severity
   - What it shows: When anomalies occur relative to price
   - How to read: Red dots = severe, Yellow = mild

2. `{TICKER}_severity_distribution.png`
   - Visual: Bar chart of severity tiers
   - What it shows: Are anomalies rare or common? How severe are they?
   - How to read: Mostly green? Mostly red? This tells the story

3. `{TICKER}_price_change_dist.png`
   - Visual: Histogram comparing price changes
   - What it shows: Do anomalies coincide with large price moves?
   - How to read: If red (anomaly) histogram is right-shifted = yes

4. `{TICKER}_analysis.json`
   - Contains all numerical metrics
   - Key fields:
     - `anomaly_percentage`: % of days with anomalies
     - `avg_days_between_anomalies`: How far apart they are
     - `severity_tiers`: Count of each tier
     - `temporal_patterns`: Volume/price relationships

---

### Statistical Insights Analysis

**Files Generated**:
1. `{TICKER}_volatility_anomaly.png`
   - Visual: Scatter plot
   - X-axis: Volatility (5-day rolling std)
   - Y-axis: Anomaly Score (0-3)
   - Color: Intensity of anomaly score
   - Interpretation:
     - If points move up-right = volatility predicts anomalies
     - If scattered = no clear relationship

2. `{TICKER}_volume_distribution.png`
   - Visual: Two overlaid histograms
   - Blue = volume ratio on normal days
   - Red = volume ratio on anomaly days
   - Interpretation:
     - If red right-shifted = anomalies have higher volume
     - If overlapped = volume not different

3. `{TICKER}_insights.json`
   - Contains statistical test results
   - Key fields:
     - `correlation_volatility_5d_anomaly`: Correlation coefficient (-1 to +1)
     - `correlation_volume_ratio_anomaly`: Does volume predict anomalies?
     - `ttest_p_value`: Are return distributions significantly different?
     - `lead_lag_interpretation`: Does anomaly lead/lag/coincide with movement?

---

### Comprehensive Report

**File**: `COMPREHENSIVE_ANALYSIS_REPORT.md`
- Complete written analysis
- Sections:
  1. Executive Summary
  2. Variable Definitions (what each metric means)
  3. Anomaly Pattern Results (frequency, severity, clustering)
  4. Statistical Relationships (correlations, causality)
  5. Key Findings (main insights)
  6. Portfolio Implications (how to use this)
  7. Recommendations (next steps)

---

## 💡 INTERPRETATION FRAMEWORK

### Question 1: "How often do anomalies occur?"
**Answer from**: `{TICKER}_analysis.json` → `frequency.anomaly_percentage`
- < 1%: Very rare, likely tail events
- 1-5%: Occasional, worth monitoring
- 5-10%: Regular, systematic component
- > 10%: Very common, needs investigation (data quality?)

### Question 2: "How severe are typical anomalies?"
**Answer from**: `{TICKER}_analysis.json` → `severity_tiers`
- Mostly score 1: Mild anomalies, borderline cases
- Mix of 1 & 2: Moderate, varied severity
- Many score 3: Severe, dramatic moves
- Compare distribution percentages

### Question 3: "Are anomalies clustered or scattered?"
**Answer from**: `{TICKER}_analysis.json` → `median_days_between_anomalies`
- < 5 days: Very clustered, regimes
- 5-15 days: Moderate spacing, some regimes
- > 30 days: Scattered, truly rare events

### Question 4: "Do anomalies predict future price moves?"
**Answer from**: `{TICKER}_insights.json` → `lead_lag_effects`
- `lead_correlation > lag_correlation`: Anomalies lead (predictive!)
- `lag_correlation > lead_correlation`: Anomalies lag (reactive)
- `concurrent_movement`: Anomalies coincide (not predictive)

### Question 5: "Should I trade on these anomalies?"
**Decision Tree**:
- If anomaly_percentage < 2% AND lead_correlation > 0.3 → **YES, rare signals**
- If anomaly_percentage > 8% AND lead_correlation < 0.1 → **NO, too common & not predictive**
- If volume_ratio > 1.3 AND lead_correlation > 0.2 → **MAYBE, investigate further**

---

## ⚡ EXECUTION SCENARIOS

### Scenario 1: Quick Test (5 min)
```bash
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL
```
- Tests all 3 analysis scripts on single stock
- Quick check if everything works
- Generates sample results

### Scenario 2: Sector Deep Dive (30 min)
```bash
python EXECUTE_DEEP_ANALYSIS.py --ticker AAPL MSFT GOOGL NVDA INTC ADBE
```
- Analyzes 6 tech stocks deeply
- Generates sector-specific insights
- Good for focused analysis

### Scenario 3: Full Portfolio Analysis (2-4 hours)
```bash
python EXECUTE_DEEP_ANALYSIS.py --ticker ALL
```
- Analyzes all 111 S&P 500 stocks
- Comprehensive market-wide insights
- Generates master report
- **Fire and forget** - can run overnight

---

## 📈 NEXT STEPS WITH RESULTS

**Once you have results**, you can:

1. **Compare across tickers**:
   - Which sectors have highest anomaly frequency?
   - Which stocks are most predictive?

2. **Combine with EDA results**:
   - Do anomalies correlate with autocorrelation (H1)?
   - Do anomalies correlate with power-law behavior?

3. **Build screening tool**:
   - Filter stocks by anomaly characteristics
   - Create trading signals based on patterns

4. **Risk management**:
   - Increase position size for low-anomaly stocks
   - Hedge high-anomaly-frequency positions

5. **Machine learning**:
   - Use insights as features in anomaly detection model
   - Predict anomalies before they occur

---

## 🛠️ TROUBLESHOOTING

**Error**: "ModuleNotFoundError: No module named 'statsmodels'"
```bash
pip install statsmodels
```

**Error**: "No data found for AAPL"
```bash
# Update data first
python update_sp500_data.py --all
python update_anomaly_labels.py --all
```

**Error**: Script hangs on "Loading EDA scripts"
- Check that `scripts/research/` directory exists
- Check that script files exist (01_, 02_, 03_)

**Output folder empty**
- Check `results/analysis/` directory was created
- Check console for error messages
- Run single ticker to debug

---

## 📞 SUMMARY TABLE

| Script | Purpose | Time | Output | Command |
|--------|---------|------|--------|---------|
| `ANALYZE_ANOMALY_PATTERNS.py` | Frequency/severity | 30s/ticker | Patterns, PNG, JSON | `--all` or `--ticker X` |
| `ANALYZE_STATISTICAL_INSIGHTS.py` | Correlations | 15s/ticker | Insights, Correlations | `--all` or `--ticker X` |
| `EXECUTE_DEEP_ANALYSIS.py` | Run both + report | 45s/ticker | All above + MD report | `--ticker ALL` |
| `EXECUTE_WEEK1_EDA.py` | Basic EDA | 2min/ticker | EDA plots, CSV | `--ticker ALL` |

---

**You're All Set!** 🚀 Run the analysis and discover deep insights in your anomaly data!
