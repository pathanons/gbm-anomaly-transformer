# Training and Testing Pipeline for Anomaly Transformer

## ภาพรวม

โปรแกรม `train_test_pipeline.py` จะทำการ:
1. โหลดข้อมูล SP500 ที่เรากำลังเตรียม
2. แบ่งข้อมูลเป็น train/val/test (70%/15%/15%)
3. Train Anomaly Transformer model
4. Evaluate บน test set
5. คำนวณ anomaly scores
6. Visualize ผลลัพธ์ด้วยกราฟ

## วิธีการใช้

### 1. ติดตั้ง dependencies (ครั้งแรก)
```bash
pip install torch matplotlib pandas scikit-learn tqdm
```

### 2. รัน training pipeline พื้นฐาน
```bash
python train_test_pipeline.py \
    --ticker AAPL \
    --epochs 10 \
    --batch-size 32
```

### 3. ตัวอย่างขั้นสูง (ปรับแต่งพารามิเตอร์)
```bash
python train_test_pipeline.py \
    --ticker AAPL \
    --features all \
    --epochs 20 \
    --batch-size 16 \
    --win-size 100 \
    --device cuda \
    --plot-dir results/plots
```

### 4. โหลด checkpoint ที่บันทึกไว้
```bash
python train_test_pipeline.py \
    --ticker AAPL \
    --checkpoint checkpoints/best_model_epoch_5.pt \
    --epochs 0  # ข้ามการ train
```

## Argument เบื้องต้น

| Argument | ค่าเริ่มต้น | คำอธิบาย |
|----------|-----------|---------|
| `--ticker` | `AAPL` | Stock ticker (e.g., AAPL, MSFT, TSLA) |
| `--data-path` | `datasets/SP500` | โฟลเดอร์ข้อมูล |
| `--features` | `all` | Features: `all`, `price_only`, `volume_only` |
| `--batch-size` | `32` | จำนวน samples ต่อ batch |
| `--win-size` | `100` | ความยาวของ time series window |
| `--epochs` | `10` | จำนวนรอบการ train |
| `--device` | `cuda` หรือ `cpu` | Device ที่ใช้ |
| `--plot-dir` | `results/plots` | โฟลเดอร์บันทึกกราฟ |

## Output

### 1. Model Checkpoints
- บันทึกที่ `checkpoints/best_model_epoch_X.pt`
- เก็บ model weights, optimizer state, และ loss

### 2. Visualization Plots
- บันทึกที่ `results/plots/{TICKER}/anomaly_window_X.png`
- แสดง:
  - แต่ละ feature (Open, High, Low, Close, Volume)
  - Anomaly scores (reconstruction error)
  - Ground truth labels (สิ่งที่เรารู้)
  - ทำเครื่องหมาย anomalies ด้วยจุด red 'x'

### 3. Anomaly Scores CSV
- บันทึกที่ `results/scores/{TICKER}_scores.csv`
- มี 2 คอลัมน์:
  - `anomaly_score`: คะแนนที่โมเดลคำนวณ (0-1)
  - `ground_truth_label`: ป้ายจริง (0 or 1)

## ตัวอย่าง Output

```
================================================================================
Anomaly Transformer - Training and Testing Pipeline
================================================================================
Ticker: AAPL
Features: all
Batch size: 32
Window size: 100
Epochs: 10
Device: cuda
================================================================================

Loading AAPL data...
AAPL (all) [batch-wise] - train: (7054, 100, 5), val: (1509, 100, 5), test: (1509, 100, 5)

Initializing Anomaly Transformer (n_features=5)...

Starting training...

Epoch 1/10
--------------------------------------------
Training: 100%|████████| 220/220 [00:45<00:00, 4.89it/s, loss=0.00235]
Train Loss: 0.002350
Validating: 100%|████████| 47/47 [00:08<00:00, 5.82it/s, loss=0.00198]
Val Loss: 0.001982
Checkpoint saved: checkpoints/best_model_epoch_0.pt

...

Testing on AAPL...
Testing: 100%|████████| 47/47 [00:06<00:00, 7.41it/s]
Test data shape: (1509, 100, 5)
Test labels shape: (1509, 100)
Anomaly scores shape: (1509, 100)

Generating visualization plots...
Figure saved: results/plots/AAPL/anomaly_window_0.png
Figure saved: results/plots/AAPL/anomaly_window_754.png
Figure saved: results/plots/AAPL/anomaly_window_1508.png

Scores saved: results/scores/AAPL_scores.csv

================================================================================
Training and testing completed!
================================================================================
```

## ความหมายของกราฟ

### Plot แต่ละ feature (Open, High, Low, Close, Volume)
- **เส้นสีน้ำเงิน**: ค่า feature ตามเวลา
- **จุด X สีแดง**: จุดที่ถูกป้ายว่าเป็น anomaly (ground truth)

### Anomaly Score Plot
- **บาร์สีฟ้า**: คะแนนความผิดปกติที่โมเดลคำนวณ
- **สูงกว่า = ผิดปกติมากขึ้น**

### Ground Truth Label Plot
- **สีเขียว**: Normal (label = 0)
- **สีแดง**: Anomaly (label = 1)

## เคล็ดลับการใช้งาน

### 1. เพิ่มจำนวน epochs หากไม่ converge
```bash
python train_test_pipeline.py --ticker AAPL --epochs 50
```

### 2. ลด batch size หากขาด memory
```bash
python train_test_pipeline.py --ticker AAPL --batch-size 8
```

### 3. เทียบหุ้นที่ต่างกัน
```bash
# AAPL
python train_test_pipeline.py --ticker AAPL --plot-dir results/AAPL

# MSFT
python train_test_pipeline.py --ticker MSFT --plot-dir results/MSFT

# TSLA
python train_test_pipeline.py --ticker TSLA --plot-dir results/TSLA
```

### 4. ใช้ CPU หากไม่มี GPU
```bash
python train_test_pipeline.py --ticker AAPL --device cpu
```

## การวิเคราะห์ผลลัพธ์

สมบูรณ์ของการทำงาน จะแสดงประเมิณวา:
1. **ค่า loss ลดลง**: โมเดลกำลังเรียนรู้อยู่
2. **Anomaly scores สูง**: ที่จุดที่มี 'x' แดง - โมเดลทำงานถูกต้อง
3. **กราฟราคา**: ควรจะมีรูปแบบ candlestick แบบปกติ

## หากมีข้อผิดพลาด

### "No data found for AAPL"
- ยังไม่ได้ดาวน์โหลดข้อมูล: `python fetch_sp500_batch.py`

### "CUDA out of memory"
- ลด batch size: `--batch-size 8`

### "Model checkpoint not found"
- ให้ train จากเริ่มต้น (ไม่ใส่ `--checkpoint`)
