# Anomaly Transformer - Project Status Report

**Last Updated**: March 22, 2026  
**Report Type**: Complete Project Analysis & Reorganization Guide  

---

## 📊 Project Overview

### What This Project Does
**Anomaly Transformer** is a deep learning model designed to detect anomalies in time series data, specifically optimized for **financial stock market data** (S&P 500). The model uses a transformer-based architecture with an attention mechanism that learns to identify unusual patterns in stock price movements.

### Primary Use Case
- Detect abnormal trading patterns in stock market data
- Analyze multiple stocks simultaneously
- Support multi-modal analysis (price + volume)
- Benchmark anomaly detection against baseline methods

---

## 🔧 What Work Has Been Done

### Phase 1: Power-law Prior Implementation ✅ COMPLETED
**Timeframe**: Latest implementation phase

#### What Was Changed (Origin & Reasoning)
The original Anomaly Transformer used a **Gaussian Prior** in its attention mechanism. This assumes that temporal relationships follow a normal distribution, which works well for general time series but may not capture **long-memory dependencies** common in financial markets.

**Solution Implemented**: Added a **Power-law Prior** option that models temporal relationships using power-law decay (1/distance^alpha), which better represents long-range dependencies in financial data.

#### Implementation Details

**1. Modified Files:**

| File | Changes | Purpose |
|------|---------|---------|
| `model/attn.py` | Added `prior_type` parameter to `AnomalyAttention` and `AttentionLayer` classes | Toggle between prior types |
| `model/AnomalyTransformer.py` | Added `prior_type` parameter, propagated to all layers | Pass prior type through model |
| `test_prior_types.py` | Created comprehensive test suite | Validate both prior implementations |

**2. Key Code Additions:**

```python
# Gaussian Prior (Original - Preserved)
prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

# Power-law Prior (New)
alpha = F.softplus(alpha) + 1e-4
distances = torch.clamp(distances, min=1.0)
prior = torch.exp(-alpha * torch.log(distances))  # Numerically stable
prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)  # Normalized
```

#### Results Achieved
✅ **5/5 tests passed**
- Backward compatibility maintained (default Gaussian)
- Power-law prior computes correctly
- Proper normalization verified
- Error handling working
- Works with/without output attention

#### Backward Compatibility
- ✅ Fully backward compatible - all existing code works without changes
- ✅ Default behavior unchanged (`prior_type="gaussian"`)
- ✅ New models can be created with `prior_type="powerlaw"`

---

### Phase 2: Data Pipeline Development ✅ COMPLETED

#### Data Sources Integrated
1. **S&P 500 Dataset**: Multiple stock tickers with OHLCV data
2. **Exchange Format**: Yahoo Finance data via yfinance
3. **Data Preprocessing**: Batch normalization and normalization per ticker

#### Training Scripts Created

| Script | Purpose | Features |
|--------|---------|----------|
| `train_test_pipeline.py` | Unified pipeline | Single ticker, configurable features |
| `train_multi_ticker.py` | Multi-ticker learning | Train on multiple stocks simultaneously |
| `train_multi_ticker_with_threshold.py` | Advanced multi-ticker | With anomaly thresholding |
| `train_sp500_all_features.py` | Full pipeline | All OHLCV features |
| `train_sp500_price_only.py` | Price analysis | Price data only |
| `train_sp500_volume_only.py` | Volume analysis | Volume data only |

#### Data Loading Capabilities
- ✅ Supports multiple datasets: SP500, SAW, credit card, etc.
- ✅ Flexible feature selection: all, price_only, volume_only
- ✅ Batch-wise and time-series normalization
- ✅ Configurable train/val/test splits (default: 70/15/15)

---

### Phase 3: Model Architecture ✅ STABLE

#### Core Components

```
Model Architecture:
├── Input Embedding (Time Series Features)
├── Encoder (Multi-head Transformer)
│   ├── Self-Attention Layer
│   │   ├── Gaussian Prior OR
│   │   └── Power-law Prior (NEW)
│   └── Position-wise Feed Forward
├── Reconstruction Decoder
└── Anomaly Score Output
```

#### Model Capabilities
- Multi-head attention with learnable priors
- Reconstruction-based anomaly detection
- Attention weight outputs for interpretability
- Support for different input feature dimensions

---

### Phase 4: Evaluation & Visualization ✅ IMPLEMENTED

#### Evaluation Outputs
1. **Checkpoints**: Best models saved (`checkpoints/best_model_*.pt`)
2. **Anomaly Scores**: CSV files with scores and ground truth
3. **Visualizations**: Per-window anomaly plots
4. **Metrics**: Reconstruction loss, anomaly score distribution

#### Result Directories
```
results/
├── plots/          - Visualization plots
├── scores/         - Anomaly score CSVs
└── *_winsize_test/ - Hyperparameter experiments
```

---

## 📁 Current Repository Structure

### Organization Issues Identified

**Problems:**
1. ❌ **Root is cluttered** - 20+ training scripts at root level
2. ❌ **Mixed concerns** - Research scripts and production code mixed
3. ❌ **Unclear purpose** - Hard to distinguish main pipeline from experiments
4. ❌ **BAT files** - Windows batch scripts for running experiments (legacy)
5. ❌ **Documentation scattered** - Multiple guide files could be centralized
6. ❌ **Checkpoints unorganized** - 30+ model files without clear naming

### Current Structure
```
Anomaly-Transformer/
├── [ROOT CLUTTER]
│   ├── BATCH_NORMALIZATION_GUIDE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── POWERLAW_PRIOR_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   ├── 20+ training scripts (*.py)
│   ├── 4 batch scripts (*.bat)
│   ├── Utility scripts (fetch_sp500_*.py, analyze_*.py, etc.)
│   └── output.csv, stock_name.txt
│
├── model/              ✅ Well organized
│   ├── AnomalyTransformer.py
│   ├── attn.py
│   └── embed.py
│
├── data_factory/       ✅ Well organized
│   └── data_loader.py
│
├── datasets/           ✅ Well organized
│   ├── SP500/
│   └── SAW/
│
├── checkpoints/        ❌ Needs organization
│   ├── 30+ .pt files
│   └── SP500/
│
├── results/            ⚠️ Needs organization
│   ├── plots/
│   ├── scores/
│   └── *_winsize_test/
│
├── utils/              ✅ Exists (not explored)
└── scripts/            ✅ Exists (shell scripts)
```

---

## 🛠️ Recommended Repository Organization

### Proposed Structure
```
Anomaly-Transformer/
├── README.md                           # Main entry point
├── CHANGELOG.md                        # Track changes
│
├── docs/                               # ✨ NEW - Centralized documentation
│   ├── ARCHITECTURE.md                 # Model details
│   ├── IMPLEMENTATION.md               # Power-law prior implementation
│   ├── DATA_FORMAT.md                  # Data structure guide
│   ├── TRAINING.md                     # Training procedures
│   └── API_REFERENCE.md                # Function documentation
│
├── src/                                # ✨ NEW - Source code
│   ├── model/
│   │   ├── AnomalyTransformer.py
│   │   ├── attn.py
│   │   └── embed.py
│   │
│   ├── data_factory/
│   │   └── data_loader.py
│   │
│   └── utils/
│       └── utils.py
│
├── notebooks/                          # ✨ NEW - Jupyter notebooks
│   ├── exploration.ipynb
│   ├── results_analysis.ipynb
│   └── prior_comparison.ipynb
│
├── scripts/                            # ✨ REORGANIZED - Pipelines
│   ├── train/                          # Main training scripts
│   │   ├── train_single_ticker.py
│   │   ├── train_multi_ticker.py
│   │   └── train_sp500_all.py
│   │
│   ├── evaluate/                       # Evaluation scripts
│   │   ├── evaluate_model.py
│   │   ├── compute_metrics.py
│   │   └── compare_priors.py
│   │
│   ├── data/                           # Data utilities
│   │   ├── fetch_sp500_data.py
│   │   ├── preprocess.py
│   │   └── analyze_data.py
│   │
│   └── legacy/                         # Archive old batch files
│       ├── run_all_baseline.bat
│       ├── run_all_powerprior.bat
│       └── run_aom.bat
│
├── experiments/                        # ✨ NEW - Research experiments
│   ├── baseline/                       # Run baseline comparisons
│   ├── hyperparameter_tuning/
│   ├── prior_comparison/
│   └── README.md                       # How to run experiments
│
├── checkpoints/                        # ✨ REORGANIZED
│   ├── README.md                       # Checkpoint guide
│   ├── pretrained/                     # Pretrained models
│   │   ├── gaussian_prior/
│   │   │   └── best_model_sp500.pt
│   │   └── powerlaw_prior/
│   │       └── best_model_sp500.pt
│   │
│   ├── experiments/                    # Experimental checkpoints
│   │   ├── baseline_20251203/
│   │   ├── powerprior_20251203/
│   │   └── thresholding_20251205/
│   │
│   └── archive/                        # Old checkpoint archives
│       └── [dated folders]
│
├── results/                            # ✨ REORGANIZED
│   ├── baseline/
│   │   ├── plots/
│   │   └── scores/
│   │
│   ├── powerprior/
│   │   ├── plots/
│   │   └── scores/
│   │
│   └── comparison/                     # A/B testing results
│       ├── metrics.csv
│       └── visualization/
│
├── data/                               # ✨ NEW - Data storage
│   ├── processed/
│   │   └── sp500/
│   └── raw/
│       └── sp500/
│
├── tests/                              # ✨ NEW - Unit tests
│   ├── test_prior_types.py
│   ├── test_data_loading.py
│   ├── test_model_forward.py
│   └── test_training_pipeline.py
│
├── requirements.txt                    # Dependencies
├── requirements_dev.txt                # Dev dependencies
├── setup.py                            # ✨ NEW - Package installation
└── .gitignore                          # ✨ NEW - Ignore rules
```

---

## 🎯 Key Findings & Status

### What's Working Well ✅
1. **Core Model** - Implementation complete and tested
2. **Prior Types** - Both Gaussian and Power-law working
3. **Data Pipeline** - Multiple dataset support
4. **Training** - Scripts for single and multi-ticker training
5. **Documentation** - Comprehensive guides exist
6. **Backward Compatibility** - No breaking changes

### What Needs Attention ⚠️
1. **Repository Organization** - Messy root directory
2. **Checkpoint Management** - 30+ unnamed checkpoints
3. **Script Organization** - 20+ scripts scattered at root
4. **Documentation Consolidation** - Multiple .md files
5. **Experiment Tracking** - No systematic tracking of experiments
6. **Testing** - Limited unit testing infrastructure

### Improvements Done in This Session 🔄
1. ✅ Created this comprehensive status report
2. ✅ Analyzed project architecture
3. ✅ Identified organization issues
4. ✅ Proposed reorganization plan

---

## 📋 Next Steps for Organization

### Priority 1: Critical Organization (Do First)
```bash
1. Create src/ directory and move model files
2. Create scripts/ subdirectories and move training scripts
3. Create docs/ and consolidate all markdown files
4. Create tests/ and consolidate test files
5. Organize checkpoints/ into pretrained/ and experiments/
```

### Priority 2: Documentation (Do Second)
```bash
1. Create comprehensive README.md
2. Update ARCHITECTURE.md with diagrams
3. Create QUICKSTART.md for new users
4. Document experiment tracking system
```

### Priority 3: Infrastructure (Nice to Have)
```bash
1. Add setup.py for package installation
2. Add CI/CD pipeline (GitHub Actions)
3. Add MLflow for experiment tracking
4. Add comprehensive unit tests
```

---

## 💡 Summary

### What the Code Does
- **Main Function**: Detects anomalies in financial time series using transformer-based architecture
- **Innovation**: Supports both Gaussian (traditional) and Power-law (new) prior distributions
- **Data Source**: S&P 500 stock data with configurable feature selection
- **Output**: Anomaly scores, visualizations, and model checkpoints

### Why It's Structured This Way
- **Modular Design**: Separates data, model, and training concerns
- **Flexibility**: Supports multiple prior types for A/B testing
- **Scalability**: Can train on multiple stocks simultaneously
- **Reproducibility**: Multiple training scripts for different configurations

### Current State
- ✅ **Functionally Complete**: All core features implemented and tested
- ⚠️ **Organizationally Messy**: Root directory cluttered with scripts
- 📈 **Ready for Scaling**: Infrastructure exists for parameter tuning and experiments

---

## 📞 Questions & Clarifications

**Q: Should we preserve the batch scripts (.bat files)?**  
A: Recommend moving to `scripts/legacy/` - they're Windows-specific, better to have shell scripts for portability

**Q: What about the old checkpoint files?**  
A: Recommend archiving by date (e.g., `checkpoints/archive/20251215/`) for reference, keep only best models in `pretrained/`

**Q: Should each training script be a separate tool or consolidated?**  
A: Consolidate into one configurable script in `scripts/train/` with arguments for ticker, features, prior type, etc.

**Q: How should experiments be tracked?**  
A: Recommend using MLflow in `experiments/` directory with structured naming conventions

---

*Generated: March 22, 2026*
