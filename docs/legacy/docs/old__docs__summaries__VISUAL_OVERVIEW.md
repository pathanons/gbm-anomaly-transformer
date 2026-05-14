# Project Summary - Visual Overview

---

## 🎯 What is Anomaly Transformer?

### Problem It Solves
```
📊 Stock Market Data
           ↓
    Time Series with
    Unusual Patterns
           ↓
  🤔 How to detect them?
           ↓
  ✨ Anomaly Transformer
(Transformer + Attention Priors)
           ↓
  🚨 Anomaly Alerts
```

### What Makes It Special
- ✅ **Transformer-based** - Modern deep learning approach
- ✅ **Learnable Attention** - Priors adapt to data
- ✅ **Flexible Priors** - Gaussian (classic) or Power-law (financial)
- ✅ **Multi-stock Support** - Train on single or multiple tickers
- ✅ **Interpretable** - Attention weights show what model focused on

---

## 🏗️ High-Level Architecture

```
INPUT: Stock Data (Open, High, Low, Close, Volume)
                    ↓
          ┌─────────────────────┐
          │ Embedding Layer     │
          │ (Time series → vec) │
          └──────────┬──────────┘
                     ↓
     ┌───────────────────────────────┐
     │  Transformer Encoder (3x)     │
     │ ┌─────────────────────────────┤
     │ │ Multi-Head Self-Attention   │
     │ │ ├─ Query, Key, Value        │
     │ │ ├─ Attention Prior (NEW!)   │
     │ │ │  ├─ Gaussian (original)   │
     │ │ │  └─ Power-law (new)       │
     │ │ └─ Attention Weights        │
     │ ├─────────────────────────────┤
     │ │ Feed-Forward Network        │
     │ └─────────────────────────────┤
     └───────────────────────────────┘
                     ↓
          ┌─────────────────────┐
          │ Reconstruction      │
          │ Decoder             │
          └──────────┬──────────┘
                     ↓
OUTPUT: Reconstructed Values + Anomaly Scores
         (Difference = Anomaly)
```

---

## 📊 Key Metrics Computed

### For Each Time Step
```
Anomaly Score = ||Predicted Values - Original Values||²

If score > threshold → ANOMALY DETECTED
If score < threshold → NORMAL
```

### For Each Model
```
Reconstruction Loss = MSE(predicted, actual)
Attention Coherence = How well priors explain attention
Prior Distribution = How parameters differ between tickers
```

---

## 🔬 Core Components

### 1. Anomaly Attention Layer
```python
class AnomalyAttention:
    - Computes attention between time steps
    - Uses learnable prior (Gaussian or Power-law)
    - Returns: attention_weights, priors
    
Prior Type: "gaussian" (original) or "powerlaw" (new)
```

**Gaussian Prior**
```
σ(i) = learned per-timestep
P(i,j) = bell-curve centered at each point
Good for: General patterns
```

**Power-law Prior**
```
α(i) = learned per-timestep
P(i,j) ∝ |i-j|^(-α(i))
Good for: Long-range dependencies (financial)
```

### 2. Attention Layer
```python
class AttentionLayer:
    - Wraps AnomalyAttention
    - For Power-law: learns alpha parameters
    - For Gaussian: uses existing sigma learning
    - Returns: attention output
```

### 3. Anomaly Transformer Model
```python
class AnomalyTransformer:
    - Encoder: 3x AttentionLayer
    - Decoder: Reconstructs sequence
    - Parameters:
        * d_model: 512
        * n_heads: 8
        * e_layers: 3
        * prior_type: "gaussian" or "powerlaw"
```

---

## 💾 Data Pipeline

### Data Flow
```
Yahoo Finance / CSV
      ↓
Raw Data (OHLCV)
      ↓
┌──────────────────┐
│ Preprocessing    │
├──────────────────┤
│ - Normalization  │
│ - Batch handling │
│ - Sliding window │
└────────┬─────────┘
         ↓
Training Set (70%)
Validation Set (15%)  → DataLoader
Test Set (15%)
         ↓
Model Training
```

### Supported Datasets
```
SP500/        - S&P 500 stocks
SAW/          - Synthetic data
credit/       - Credit card transactions
+ Any CSV format with numeric data
```

### Data Features
```
Can use:
- Price only (Open, High, Low, Close)
- Volume only
- All features (OHLCV)
- Add/remove custom features
```

---

## 🚀 Training Process

### Single Ticker Training
```bash
python scripts/train/train_sp500.py --ticker AAPL

Step 1: Load AAPL data
Step 2: Create train/val/test splits
Step 3: Initialize Anomaly Transformer
Step 4: Loop over epochs:
   - Forward pass on train batch
   - Compute reconstruction loss
   - Backward pass & update weights
   - Evaluate on validation set
Step 5: Save best model checkpoint
Step 6: Compute anomaly scores on test set
Step 7: Visualize results
```

### Multi-Ticker Training
```bash
python scripts/train/train_multi_ticker.py --tickers AAPL MSFT GOOGL

Step 1: Load data for 3 tickers
Step 2: Create combined train/val/test
Step 3: Initialize shared model
Step 4: Train on all 3 stocks together
Step 5: Extract per-ticker metrics
Step 6: Compare anomaly patterns across stocks
```

---

## 📈 Example Results

### Training Curves
```
Loss over epochs:
│
│  ╱╲   ╱╲
│ ╱  ╲ ╱  ╲
│╱    ╲    ╲___   ← Loss decreasing = good training
└─────────────────→ Epoch
```

### Anomaly Detection
```
Stock Price
│        *ANOMALY*    *ANOMALY*
│       /     ╲      /     ╲
│      /       ╲    /       ╲
│ ────         ────        ────  ← Normal periods
└──────────────────────────────→ Time
```

### Prior Comparison
```
Gaussian Prior         Power-law Prior
(Bell curve)          (Long-tail decay)

σ = 0.5               α = 2.0

σ*.*.*.*.*.*.*        ╲╲╲│╱╱╱
  *.***.***.*          ╲╲│╱╱
    ***|***               │
  .***...***            ╱╱│╲╲
.*.*.*.*.*.*           ╱╱╱│╲╲╲╲
```

---

## 🎓 Implementation Timeline

### Phase 1: Baseline (Original)
```
✅ Original Gaussian Prior
   - AnomalyAttention with sigma learning
   - Basic transformer encoder/decoder
   - Tested on credit card dataset
```

### Phase 2: Power-law Extension
```
✅ New Power-law Prior
   - Added alpha_projection layers
   - Implemented numerically stable computation
   - Backward compatible (default: Gaussian)
   - Tested on SP500 stocks
```

### Phase 3: Multi-Stock Support
```
✅ Multi-ticker training
   - DataLoader for multiple stocks
   - Batch normalization options
   - Per-ticker evaluation
```

### Phase 4: Current - Organization
```
🔴 Repository reorganization
   - Move files to logical structure
   - Consolidate documentation
   - Add experiment tracking
```

---

## 📊 Comparison: Gaussian vs Power-law

### Gaussian Prior (Original)

**Advantages**
- ✅ Well-established approach
- ✅ Symmetric pattern (natural for oscillations)
- ✅ Mathematically simple

**Best For**
- General time series
- Symmetric patterns
- Cyclical data

**Formula**: `P(i,j) = exp(-(i-j)² / 2σ²)`

### Power-law Prior (New)

**Advantages**
- ✅ Long-range dependencies
- ✅ Asymmetric (realistic for markets)
- ✅ Heavy-tailed (captures outliers)

**Best For**
- Financial markets
- Long-range correlations
- Heavy-tailed distributions

**Formula**: `P(i,j) = (i-j)^(-α)`

### When to Use Each
```
Use GAUSSIAN when:
- Data is normally distributed
- Patterns repeat regularly
- Want simpler, faster inference

Use POWER-LAW when:
- Long-range dependencies matter
- Data has heavy tails
- Financial/real-world data
- Anomalies are rare events
```

---

## 🔄 Typical Workflow

### Week 1: Setup & Baseline
```
Day 1-2: Install, run example
Day 3-4: Train with Gaussian prior
Day 5:   Evaluate baseline
```

### Week 2: Experimentation
```
Day 1-2: Train with Power-law prior
Day 3-4: Compare results
Day 5:   Analyze differences
```

### Week 3: Tuning & Analysis
```
Day 1-2: Hyperparameter tuning
Day 3-4: Multi-ticker testing
Day 5:   Documentation & reporting
```

---

## 💡 Key Insights

### Why Transformers?
```
1. Self-attention captures long-range dependencies
2. Parallel computation (faster than RNN)
3. Interpretable attention weights
4. State-of-art performance on sequences
```

### Why Multiple Priors?
```
1. Different data has different patterns
2. Gaussian good for oscillations
3. Power-law good for heavy-tails
4. A/B testing validates approach
```

### Why Reconstruction-based?
```
1. Unsupervised learning (no labels needed)
2. Can detect any deviation from pattern
3. Interpretable: anomaly score = error
4. Works for different feature dimensions
```

---

## 📈 Scalability Roadmap

### Current (Works Well)
```
✅ Single GPU training
✅ 100+ stocks in sequence
✅ 100-1000 time steps per sequence
✅ 5 features (OHLCV)
```

### Near Future (Can Add)
```
🟡 Distributed training (multi-GPU)
🟡 Online learning (streaming data)
🟡 Explainability (SHAP/attention visualization)
🟡 Real-time inference
```

### Future (Research)
```
⚪ Graph neural networks (stock relationships)
⚪ Transfer learning (domain adaptation)
⚪ Ensemble methods (multiple models)
⚪ Reinforcement learning (trading signals)
```

---

## 🎯 Success Criteria

### Model Evaluation
```
✅ Reconstruction loss decreases over epochs
✅ Attention weights are interpretable
✅ Anomaly scores distinguish normal/abnormal
✅ Both prior types work correctly
```

### Robustness Testing
```
✅ Works on unseen tickers
✅ Handles different time periods
✅ Robust to data noise
✅ Consistent across runs
```

### Real-World Application
```
✅ Detects known market anomalies
✅ Reasonable false positive rate
✅ Fast inference (< 100ms)
✅ Explainable decisions
```

---

## 🔗 Key References

### Files to Understand
1. `src/model/AnomalyTransformer.py` - Main model
2. `src/model/attn.py` - Attention with priors
3. `src/data_factory/data_loader.py` - Data pipeline
4. `scripts/train/train_sp500.py` - Training example
5. `docs/ARCHITECTURE.md` - Detailed explanation

### Papers Referenced
- "Attention is All You Need" (Vaswani et al., 2017)
- Power-law principles in financial markets
- Anomaly detection in time series

---

**This visualization provides a holistic understanding of the project. For details, refer to specific documentation files.**

