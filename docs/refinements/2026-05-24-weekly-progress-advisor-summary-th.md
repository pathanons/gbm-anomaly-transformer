# สรุปความคืบหน้าประจำสัปดาห์: GBM-aware Anomaly Transformer Refinement

วันที่สรุป: 2026-05-24

เอกสารนี้สรุปงานที่ทำในสัปดาห์นี้สำหรับใช้คุยกับอาจารย์ที่ปรึกษา โดยเน้นว่าเราเปลี่ยนอะไร ทำไมจึงเปลี่ยน ผลที่ได้เป็นอย่างไร และค่าหรือวิธีใดดีที่สุดในตอนนี้

## 1. ภาพรวมสั้นที่สุด

สัปดาห์นี้เราเปลี่ยนงานจากการดูผลของ `canonical_gbm` แบบ raw score ไปสู่การทดลอง decision/calibration layer อย่างเป็นระบบมากขึ้น โดยเพิ่มและทดสอบวิธี threshold หลายแบบ ได้แก่ global conformal, per-ticker conformal, EVT threshold, tail probability และ VaR-style tail breach

ผลสำคัญที่สุดคือ `per_ticker_conformal` ที่ threshold quantile `0.90` ให้ผลดีที่สุดในกลุ่มวิธีที่ยังตีความได้สมเหตุสมผล:

```text
precision   = 0.9951
sensitivity = 0.3479
specificity = 0.8338
F1          = 0.5155
ROC-AUC     = 0.6254
PR-AUC      = 0.9933
TP/FP/TN/FN = 35413 / 175 / 878 / 66389
```

เมื่อเทียบกับผล canonical GBM เดิมที่ใช้ global conformal q=0.95:

```text
precision   = 0.9907
sensitivity = 0.1114
specificity = 0.8993
F1          = 0.2002
```

แปลว่า sensitivity เพิ่มจากประมาณ 11% เป็น 35% ขณะที่ precision ยังสูงมากเกือบ 99.5% นี่เป็น improvement ที่สำคัญที่สุดของสัปดาห์นี้

## 2. ปัญหาเดิมที่เราพยายามแก้

ก่อนหน้านี้ pipeline มี canonical GBM score ที่ดูมีสัญญาณบางอย่างจากกราฟ เช่น score ยกตัวขึ้นใกล้ช่วงราคามี jump, gap หรือ volatility burst แต่ปัญหาคือการใช้ raw score หรือ global threshold เดียวกันข้ามหุ้นทั้งหมดไม่เหมาะกับข้อมูลหุ้นจริง

เหตุผลหลักคือหุ้นแต่ละตัวมี baseline score ไม่เท่ากัน บางตัว score แกว่งกว้างมาก บางตัวนิ่งกว่า และบางตัวมี normal regime ที่ต่างกันชัดเจน ถ้าใช้ threshold กลางตัวเดียว จะเกิดสองปัญหาพร้อมกัน:

1. หุ้นที่ score scale สูงอาจถูก flag ง่ายเกินไป
2. หุ้นที่ score scale ต่ำอาจแทบไม่ถูก flag แม้มี event จริง

ดังนั้นโจทย์ของสัปดาห์นี้ไม่ใช่แค่ "โมเดล score สูงไหม" แต่เปลี่ยนเป็น "score นี้ extreme แค่ไหนเมื่อเทียบกับ calibration distribution ของหุ้นตัวนั้นหรือ validation distribution ที่เหมาะสม"

## 3. สิ่งที่ปรับปรุงใน code

เราเพิ่ม shared calibration layer ใหม่:

```text
src/gbm/calibration.py
```

หน้าที่ของไฟล์นี้คือรวม logic การ fit threshold และการคำนวณ calibrated score ให้เป็นศูนย์กลาง ได้แก่:

- empirical validation quantile
- global conformal p-value
- per-ticker conformal p-value
- EVT threshold ด้วย Generalized Pareto Distribution
- fallback กรณี EVT fit ไม่ได้

เราแก้ scoring ให้รองรับ predictive tail diagnostics เพิ่มใน:

```text
src/gbm/scoring.py
```

ตอนนี้ score output มีคอลัมน์เพิ่ม:

```text
tail_probability
tail_surprise = -log10(tail_probability)
```

สำหรับ Gaussian ใช้ normal CDF ส่วน Student-t ใช้ SciPy Student-t CDF และมี fallback เป็น normal approximation ถ้า SciPy ใช้ไม่ได้

เราแก้ validation/test pipeline:

```text
scripts/gbm/validate_joint.py
scripts/gbm/test_joint.py
```

ให้รองรับ threshold methods ใหม่:

```text
quantile
conformal
per_ticker_conformal
evt
tail_probability
var
```

เราเพิ่ม runner สำหรับ decision sweep โดยไม่ retrain:

```text
scripts/gbm/run_decision_sweep.py
```

และเพิ่ม config ตัวอย่าง:

```text
configs/train1/review_decision_sweep_canonical_price.yaml
```

ข้อดีคือเราไม่ต้อง train โมเดลใหม่ทุกครั้ง แต่ใช้ checkpoint เดิมของ `review_canonical_price_studentt_conformal` แล้ว vary เฉพาะ decision layer ได้

## 4. ทำไมจึงเลือกวิธีเหล่านี้

### 4.1 Conformal p-value

Conformal calibration เหมาะกับโจทย์นี้เพราะเปลี่ยน raw score ให้เป็นค่าที่ตีความง่ายขึ้นว่า score ของ window ปัจจุบัน extreme แค่ไหนเมื่อเทียบกับ calibration set

ใน setting นี้เราใช้ validation-normal windows เป็น calibration distribution เป็นหลัก เพื่อเลี่ยงการ tune จาก test set และสอดคล้องกับ research rule ที่ว่า validation ใช้เลือก threshold ได้ แต่ test ใช้รายงานผลสุดท้ายเท่านั้น

### 4.2 Per-ticker conformal

Per-ticker conformal เป็นวิธีที่สำคัญที่สุด เพราะมันแก้ปัญหา score scale ต่างกันข้ามหุ้นโดยตรง แทนที่จะถามว่า score นี้สูงเมื่อเทียบกับหุ้นทั้งหมดไหม เราถามว่า score นี้สูงเมื่อเทียบกับ normal calibration windows ของหุ้นตัวเดียวกันไหม

นี่เข้ากับข้อสังเกตจาก deep research report ว่า canonical GBM score ไม่ควรถูกใช้ด้วย raw global threshold เดียวข้ามทุก ticker

### 4.3 EVT threshold

EVT ถูกเพิ่มเพื่อทดสอบแนวคิดว่า anomaly detection ควรสนใจ tail behavior ของ score มากกว่า quantile ธรรมดา เรา fit Generalized Pareto Distribution บน validation-normal score exceedances เหนือ tail base quantile เช่น 0.85, 0.90, 0.95 แล้ว extrapolate threshold สำหรับ target quantile

วิธีนี้มีความหมายทางสถิติด้าน extreme events มากกว่า hard threshold ธรรมดา แต่ผลรอบนี้ยังไม่ชนะ per-ticker conformal

### 4.4 Tail probability / VaR-style decision

เราเพิ่ม tail probability และ VaR-style rule เพื่อทดสอบแนวทาง predictive distribution โดยตรง ถ้า predictive distribution calibrated ดี tail probability ควรเป็น score ที่มีความหมายทาง financial risk เช่น realized return อยู่ใน tail ของ distribution หรือไม่

แต่ผลรอบนี้แสดงว่า raw tail probability ยังใช้ตรง ๆ ไม่ได้ เพราะ distribution ยังไม่ calibrated พอ ทำให้ tail_probability เล็กมากเกือบทุก window และ rule จึง flag แทบทุกอย่าง

## 5. ชุดทดลองที่รัน

เราใช้ checkpoint จาก:

```text
D:\AnomalyTransformerRuns\experiments\review_canonical_price_studentt_conformal\models\gbm_joint.pt
```

configuration หลัก:

```text
association_mode         = canonical_gbm
features                 = price_only
predictive_distribution  = student_t
window_size              = 100
step                     = 1
seed                     = 42
tickers                  = 111
```

เรา vary:

```text
threshold_methods   = conformal, per_ticker_conformal, evt, tail_probability, var
threshold_quantiles = 0.90, 0.95, 0.975, 0.99
evt_tail_quantiles  = 0.85, 0.90, 0.95
```

ผลรวมถูกเก็บที่:

```text
D:\AnomalyTransformerRuns\experiments\review_canonical_price_studentt_decision_sweep\reports\decision_sweep_summary.csv
```

## 6. ผลลัพธ์หลักตามวิธี

### 6.1 Per-ticker conformal

ผลดีที่สุด:

```text
method      = per_ticker_conformal
quantile    = 0.90
precision   = 0.9951
sensitivity = 0.3479
specificity = 0.8338
F1          = 0.5155
ROC-AUC     = 0.6254
PR-AUC      = 0.9933
```

ผลนี้ดีที่สุดในแง่ balance ที่ยังสมเหตุสมผล เพราะ precision สูงมาก sensitivity เพิ่มขึ้นมาก และ specificity ยังไม่พัง

ถ้าใช้ quantile สูงขึ้น:

```text
q=0.95  F1=0.4221, sensitivity=0.2678, precision=0.9955
q=0.975 F1=0.2610, sensitivity=0.1502, precision=0.9960
q=0.99  F1=0.0231, sensitivity=0.0117, precision=1.0000
```

ดังนั้น q=0.90 เป็น operating point ที่ดีที่สุดตอนนี้ เพราะ q สูงกว่านี้ conservative เกินไปและ miss event จำนวนมาก

### 6.2 EVT

ผลดีที่สุด:

```text
method            = evt
threshold_quantile = 0.90
evt_tail_quantile  = 0.85
precision          = 0.9909
sensitivity        = 0.1985
specificity        = 0.8243
F1                 = 0.3307
```

EVT ดีกว่า global conformal เล็กน้อย แต่ยังแพ้ per-ticker conformal อย่างชัดเจน

### 6.3 Global conformal

ผลดีที่สุด:

```text
method      = conformal
quantile    = 0.90
precision   = 0.9905
sensitivity = 0.1824
specificity = 0.8310
F1          = 0.3081
```

global conformal q=0.90 ดีกว่า q=0.95 เดิม แต่ยังแก้ปัญหาความต่างข้าม ticker ได้ไม่ดีเท่า per-ticker conformal

### 6.4 Tail probability และ VaR

ตัวเลขภายนอกดูดีมาก:

```text
method      = tail_probability / var
q=0.90
precision   = 0.9898
sensitivity = 1.0000
specificity = 0.0000
F1          = 0.9949
```

แต่ผลนี้ไม่ควรถูกตีความว่าเป็นผลดีที่สุด เพราะโมเดล flag ทุก window เป็น anomaly:

```text
TP = 101802
FP = 1053
TN = 0
FN = 0
```

นี่เป็น artifact จาก class imbalance และ tail_probability ที่เล็กมากเกือบทั้งหมด ไม่ใช่ detector ที่ใช้งานได้จริง

## 7. Event-type results

ผล event catch rate ของวิธีที่ดีที่สุด `per_ticker_conformal q=0.90`:

```text
drop              = 0.3758
jump              = 0.3608
volatility_shock  = 0.3791
volume_spike      = 0.3499
regime_shift      = 0.0000
```

เทียบกับ canonical global conformal q=0.95 เดิมที่จับ event ได้ประมาณ 11-12% เท่านั้น วิธีใหม่เพิ่ม event catch rate เป็นประมาณ 35-38%

ข้อสังเกตคือ catch rate เพิ่มขึ้นทุก event type หลัก ไม่ใช่ดีขึ้นเฉพาะ event ใด event หนึ่ง จึงดูเป็น improvement ของ decision calibration มากกว่าการ overfit ไปที่ event เฉพาะกลุ่ม

regime_shift ยังตอบไม่ได้ เพราะ test windows ไม่มี regime_shift event

## 8. ทำไม per-ticker conformal ถึงดีที่สุดตอนนี้

เหตุผลเชิง intuition:

1. Score distribution ของหุ้นแต่ละตัวต่างกัน
2. Global threshold ไม่รู้ว่าหุ้นตัวไหน baseline สูงหรือต่ำ
3. Per-ticker calibration ทำให้ threshold ปรับตามพฤติกรรมของหุ้นแต่ละตัว
4. ใช้ validation-normal windows จึงยังรักษาหลักไม่ tune บน test
5. เมื่อ threshold ไม่ bias ไปตาม ticker scale แล้ว sensitivity เพิ่มขึ้นมาก โดย precision ยังสูง

พูดสั้น ๆ คือ improvement รอบนี้ไม่ได้มาจากการเปลี่ยน model architecture แต่มาจากการตัดสิน anomaly ให้สอดคล้องกับ financial time series มากขึ้น

นี่สอดคล้องกับข้อเสนอจาก deep research reports ที่บอกว่า canonical GBM score ควรถูกใช้เป็น structural signal และต้องมี calibration layer ที่เหมาะสม ไม่ใช่ใช้ raw/global threshold โดยตรง

## 9. สิ่งที่ควร claim ตอนนี้

Claim ที่ปลอดภัย:

- เราเพิ่ม decision/calibration layer ที่มีหลักการมากขึ้นให้กับ GBM-aware pipeline
- Per-ticker conformal calibration ช่วยเพิ่ม sensitivity และ event catch rate อย่างชัดเจนใน single-seed experiment นี้
- Canonical GBM + Student-t + price-only + per-ticker conformal q=0.90 เป็น configuration ที่ดีที่สุดในตอนนี้ในแง่ F1 และ event catch rate ที่ยังรักษา precision สูง
- EVT เป็น baseline tail-thresholding ที่ใช้ได้ แต่ยังไม่ชนะ per-ticker conformal
- Raw tail probability / VaR-style rule ยังไม่พร้อมใช้ เพราะ predictive distribution ยัง calibrate ไม่ดีพอ

Claim ที่ยังไม่ควรพูด:

- ยังไม่ควร claim ว่า canonical GBM outperform ทุก baseline
- ยังไม่ควร claim statistical significance เพราะยังเป็น single seed
- ยังไม่ควร claim early warning เพราะยังไม่ได้วัด lead time
- ยังไม่ควร claim regime-shift detection เพราะ test set ไม่มี regime_shift event
- ยังไม่ควร claim ว่า VaR/tail probability ใช้ได้จริงจนกว่าจะ calibrate predictive distribution ดีขึ้น

## 10. สรุปสำหรับพูดกับอาจารย์

สัปดาห์นี้เราเปลี่ยนจากการดู raw canonical GBM anomaly score ไปสู่การสร้าง calibrated decision layer ที่เหมาะกับหุ้นรายตัวมากขึ้น เพราะพบว่า global threshold ทำให้โมเดล conservative เกินไปและ miss anomaly windows จำนวนมาก

เราเพิ่มและทดสอบหลายวิธี ได้แก่ conformal p-value, per-ticker conformal, EVT threshold, tail probability และ VaR-style decision โดยใช้ checkpoint เดิมและ sweep เฉพาะ decision layer ทำให้เปรียบเทียบได้เร็วโดยไม่ต้อง retrain

ผลดีที่สุดตอนนี้คือ per-ticker conformal ที่ q=0.90 ซึ่งเพิ่ม F1 จาก 0.2002 เป็น 0.5155 และเพิ่ม sensitivity จาก 0.1114 เป็น 0.3479 ในขณะที่ precision ยังสูงมากที่ 0.9951 นอกจากนี้ event catch rate ของ drop, jump, volatility shock และ volume spike เพิ่มขึ้นเป็นประมาณ 35-38%

ผลนี้สนับสนุนสมมติฐานว่า canonical GBM score มีประโยชน์ในฐานะ structural anomaly signal แต่การตัดสิน anomaly ต้อง calibrate ตาม ticker ไม่ใช่ใช้ threshold กลางตัวเดียว

อย่างไรก็ตาม ผลนี้ยังเป็น single-seed และยังไม่มี baseline ภายนอกหรือ lead-time evaluation ดังนั้นข้อสรุปที่เหมาะสมคือ "per-ticker conformal calibration เป็น decision layer ที่ดีที่สุดในตอนนี้ และควรนำไปทดสอบซ้ำด้วย multi-seed/baseline/lead-time evaluation" ไม่ใช่ claim ว่าโมเดลชนะทุกวิธีแล้ว

## 11. Next steps ที่ควรทำต่อ

1. รัน multi-seed สำหรับ configuration สำคัญ:
   - canonical price Student-t per-ticker conformal q=0.90
   - canonical price Student-t conformal q=0.90
   - canonical price Student-t EVT q=0.90 tail=0.85
   - no-prior price Student-t per-ticker conformal q=0.90
2. เพิ่ม baseline ภายนอก:
   - rolling volatility/z-score
   - forecasting error baseline
   - LSTM/GRU autoencoder
   - vanilla Anomaly Transformer
3. เพิ่ม lead-time / detection-delay evaluation
4. เพิ่ม per-ticker metric distribution เพื่อดูว่าหุ้นไหนดี/แย่
5. ปรับ tail probability ให้เป็น calibrated score เช่น conformal บน `tail_surprise` แทนใช้ raw `tail_probability < alpha`
6. เพิ่ม change-point layer บน smoothed calibrated score สำหรับ regime anomaly
7. ทำรายงานผลแบบ confidence interval และ significance test ก่อนนำไปเขียนเป็น claim หลัก

