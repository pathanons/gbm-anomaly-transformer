# Anomaly Transformer - Quick Reference Guide

**Updated**: March 22, 2026

---

## 📚 What This Project Does (In 30 Seconds)

**Anomaly Transformer** detects unusual patterns in stock market time series data using transformer neural networks with learnable attention priors.

**Key Innovation**: Supports both traditional Gaussian priors and new Power-law priors that better model long-range financial dependencies.

---

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone [repo-url]
cd Anomaly-Transformer
pip install -r requirements.txt

# Optional: Install development tools
pip install -r requirements_dev.txt
```

### Run Your First Experiment
```bash
# Train single-ticker model on AAPL
python scripts/train/train_sp500.py \
    --ticker AAPL \
    --epochs 10 \
    --prior-type powerlaw

# Results will be saved to:
# - checkpoints/experiments/latest/
# - results/powerprior/plots/
# - results/powerprior/scores/
```

### Compare Priors (Gaussian vs Power-law)
```bash
python scripts/evaluate/compare_priors.py \
    --ticker AAPL \
    --output-dir results/comparison/
```

---

## 📁 File Directory Guide

### Finding What You Need

| I want to... | Location | Example File |
|-------------|----------|--------------|
| **See the model code** | `src/model/` | `AnomalyTransformer.py` |
| **Understand architecture** | `docs/` | `ARCHITECTURE.md` |
| **Train a model** | `scripts/train/` | `train_sp500.py` |
| **Fetch/process data** | `scripts/data/` | `fetch_sp500_data.py` |
| **Evaluate results** | `scripts/evaluate/` | `compare_priors.py` |
| **Run tests** | `tests/` | `test_prior_types.py` |
| **Read guides** | `docs/` | `TRAINING.md` |
| **Save checkpoints** | `checkpoints/pretrained/` | `gaussian_prior/best_model.pt` |
| **View results** | `results/` | `powerprior/plots/` |

---

## 🎯 Common Tasks

### Task 1: Train on Single Stock
```bash
python scripts/train/train_sp500.py \
    --ticker MSFT \
    --epochs 20 \
    --batch-size 32 \
    --prior-type powerlaw
```

### Task 2: Train on Multiple Stocks
```bash
python scripts/train/train_multi_ticker.py \
    --tickers AAPL MSFT GOOGL \
    --epochs 15 \
    --prior-type gaussian
```

### Task 3: Using Only Price (No Volume)
```bash
python scripts/train/train_sp500_price_only.py \
    --ticker AAPL \
    --epochs 10
```

### Task 4: Evaluate Model Performance
```bash
python scripts/evaluate/test_prior_types.py
python scripts/evaluate/compare_priors.py --ticker AAPL
```

### Task 5: Visualize Anomalies
```bash
python scripts/evaluate/visualize_results.py \
    --checkpoint checkpoints/pretrained/powerlaw_prior/best_model_sp500.pt \
    --data datasets/SP500/AAPL.csv
```

### Task 6: Run All Tests
```bash
cd tests/
python -m pytest .
```

---

## 🔧 Key Parameters

### For Training Scripts
```bash
--ticker SYMBOL          # Stock ticker (e.g., AAPL, MSFT)
--epochs N              # Number of training epochs (default: 10)
--batch-size N          # Batch size (default: 32)
--win-size N            # Time series window length (default: 100)
--prior-type TYPE       # "gaussian" or "powerlaw" (default: gaussian)
--learning-rate LR      # Learning rate (default: 0.0001)
--device DEVICE         # "cuda" or "cpu" (default: cuda)
--output-dir PATH       # Where to save results
```

### For Evaluation Scripts
```bash
--ticker SYMBOL         # Stock ticker to evaluate
--checkpoint PATH       # Path to saved model
--threshold THRESH      # Anomaly threshold (0-1)
--output-dir PATH       # Where to save results
```

---

## 📊 Understanding Results

### Output Files
```
After training, you'll get:
├── checkpoints/experiments/TIMESTAMP/
│   └── best_model.pt              # Best model weights
├── results/powerprior/
│   ├── plots/TICKER_anomaly_*.png # Anomaly visualizations
│   └── scores/TICKER_scores.csv   # Anomaly scores
```

### CSV Score Format
```csv
timestamp,anomaly_score,ground_truth_label
2023-01-01,0.05,0
2023-01-02,0.12,0
2023-01-03,0.78,1
```

### Plot Interpretation
- **Red dots** = Detected anomalies
- **Green/blue lines** = Stock prices
- **Purple line** = Reconstruction error

---

## 🔍 Understanding the Model

### Model Architecture
```
Stock Data (OHLCV)
      ↓
   Embedding
      ↓
Transformer Encoder (with Attention Priors)
   • Layer 1: Self-Attention + Feed-Forward
   • Layer 2: Self-Attention + Feed-Forward
   • Layer 3: Self-Attention + Feed-Forward
      ↓
Reconstruction Decoder
      ↓
Anomaly Scores (L2 loss)
```

### Two Prior Types

**Gaussian Prior (Original)**
- Traditional approach
- Good for general time series
- Formula: `N(d) = 1/sqrt(2π)σ * exp(-d²/(2σ²))`

**Power-law Prior (New)**
- Better for financial data
- Captures long-range dependencies
- Formula: `P(d) = d^(-α) / Σ d^(-α)`

---

## 📈 Recent Work Summary

### ✅ Completed
1. **Power-law Prior Implementation**
   - Added to attention mechanism
   - Fully backward compatible
   - Tested and verified

2. **Multi-Ticker Support**
   - Train on single or multiple stocks
   - Flexible feature selection (price/volume/all)
   - Batch normalization options

3. **Comprehensive Testing**
   - 5/5 prior type tests passing
   - Data loading validated
   - Training pipeline verified

### 🔴 Known Limitations
1. Only configurable via command-line arguments (no config files yet)
2. Limited checkpoint organization
3. No MLflow experiment tracking
4. Repository at root level needs organizing

### 🟡 In Progress
1. Repository reorganization (see REORGANIZATION_PLAN.md)
2. Adding experiment tracking
3. Creating Jupyter notebooks for analysis

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python scripts/train/train_sp500.py \
    --ticker AAPL \
    --batch-size 16  # Reduced from 32
    --device cuda
```

### Error: "No module named 'model'"
```bash
# Solution: Run from project root and ensure PYTHONPATH is set
cd Anomaly-Transformer
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/train/train_sp500.py ...
```

### Error: "Data file not found"
```bash
# Solution: First fetch the data
python scripts/data/fetch_sp500_data.py --ticker AAPL
# Then run training
python scripts/train/train_sp500.py --ticker AAPL
```

---

## 📚 Documentation Map

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `README.md` | Project overview | Starting out |
| `QUICKSTART.md` | 5-minute setup guide | First time setup |
| `docs/ARCHITECTURE.md` | Model details | Understanding internals |
| `docs/IMPLEMENTATION.md` | Power-law prior details | Research/modification |
| `docs/TRAINING.md` | Training procedures | Training new models |
| `PROJECT_STATUS_REPORT.md` | Complete status | Review project state |
| `REORGANIZATION_PLAN.md` | How to organize files | Future maintenance |

---

## 💻 Development Workflow

### For Researchers/Students
```bash
1. Read QUICKSTART.md
2. Run example: python scripts/train/train_sp500.py --ticker AAPL
3. Check docs/ARCHITECTURE.md
4. Modify scripts/train/ or scripts/evaluate/
5. Run tests: python -m pytest tests/
6. Check results in results/
```

### For ML Engineers
```bash
1. Review src/model/ for architecture
2. Check data_factory/data_loader.py for data pipeline
3. Study training in scripts/train/
4. Extend with new features
5. Add tests in tests/
6. Update documentation
```

### For Production Users
```bash
1. Use checkpoints/pretrained/ models
2. Run scripts/evaluate/ for inference
3. Monitor results/ outputs
4. No need to modify core code
```

---

## 🎓 Learning Path

**Beginner** (1-2 hours)
```
1. Read QUICKSTART.md
2. Run first training: train_sp500.py
3. Look at generated plots
4. Try changing parameters
```

**Intermediate** (3-5 hours)
```
1. Read docs/ARCHITECTURE.md
2. Study src/model/attn.py
3. Compare Gaussian vs Power-law results
4. Run compare_priors.py
5. Understand differences
```

**Advanced** (6+ hours)
```
1. Read docs/IMPLEMENTATION.md
2. Modify attention mechanism
3. Implement new prior types
4. Run comprehensive experiments
5. Contribute improvements
```

---

## 📞 Quick Help

**"How do I..."**

| Question | Answer |
|----------|--------|
| Install dependencies? | `pip install -r requirements.txt` |
| Train a model? | `python scripts/train/train_sp500.py --ticker AAPL` |
| Load a checkpoint? | `torch.load('checkpoints/pretrained/powerlaw_prior/best_model.pt')` |
| Change prior type? | Add `--prior-type gaussian` or `--prior-type powerlaw` |
| Add new ticker? | `python scripts/data/fetch_sp500_data.py --ticker NEW_TICKER` |
| Run tests? | `python -m pytest tests/` |
| Visualize results? | `python scripts/evaluate/visualize_results.py` |

---

## 🔗 Important Paths to Remember

```bash
TRAINING_SCRIPTS=scripts/train/
EVALUATION_SCRIPTS=scripts/evaluate/
DATA_SCRIPTS=scripts/data/
TEST_FILES=tests/
SOURCE_CODE=src/
DOCUMENTATION=docs/
SAVED_MODELS=checkpoints/pretrained/
TRAINING_RESULTS=results/
```

---

**Need more help?** Check the relevant .md file in `docs/` or raise an issue!

