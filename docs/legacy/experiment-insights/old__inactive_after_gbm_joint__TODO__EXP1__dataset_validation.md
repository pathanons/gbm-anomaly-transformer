# EXP1 Dataset Validation Checklist

Use this checklist to decide whether the SP500 dataset is fit for training, validation, and final testing in Experiment 1.

## 1. Label Provenance and Event Mapping
- [x] ระบุ event taxonomy หลักของงานให้ชัดเจนก่อนว่า label ที่ paper จะใช้คือ jump, drop, volume spike, volatility shock, regime shift
- [x] ระบุ rule หรือ threshold ที่ใช้สร้าง label ของแต่ละ event type ให้ตรวจสอบได้
- [x] ระบุไฟล์หรือสคริปต์ที่สร้าง label โดยตรง และอธิบายตัวแปรหรือสูตรที่ใช้
- [x] ตรวจว่า label ไม่ได้สร้างจากข้อมูลอนาคต
- [x] ตรวจว่า label ไม่ได้อาศัย model output หรือผลการ infer ภายหลัง
- [x] ถ้ามีคอลัมน์ภายในอย่าง `criterion_a`, `criterion_b`, `criterion_c` ให้ถือว่าเป็น implementation detail และ map กลับไปยัง event type ที่ชัดเจน ไม่ใช่ใช้เป็นชื่อหลักใน paper
- [x] ถ้ายัง map ไม่ได้ ต้องหยุดการเขียนผลเชิงตีความ เพราะยังไม่มี semantic label ที่ชัดพอ

## 2. File Integrity and Temporal Alignment
- [x] ตรวจว่าแต่ละ ticker มีคู่ไฟล์ OHLCV และ anomaly_label ครบ
- [x] ตรวจว่า date ใน OHLCV และ label ตรงกันทุกแถว
- [x] ตรวจว่า date เรียงลำดับถูกต้องและไม่มี duplicate row
- [x] ตรวจว่าไม่มี missing date gap ที่ไม่ได้อธิบาย
- [x] ตรวจว่า merge ข้อมูลไม่มี row หลุดหรือ row ซ้ำ
- [x] ตรวจว่า label file และ price file ใช้ ticker เดียวกันจริง
- [x] ตรวจว่า label ตรงกับวันที่ของ OHLCV จริง และไม่มี date leak

## 3. Split Policy and Leakage Control
- [x] ตรวจว่า split เป็น time-based เท่านั้น
- [x] ตรวจว่า train, validation, test ไม่ทับซ้อนกัน
- [x] ตรวจว่า rolling backtest ไม่ดึงข้อมูลอนาคตมาใช้
- [x] ตรวจว่า threshold selection ใช้ validation เท่านั้น ไม่ใช้ test
- [x] ตรวจว่า normalization fit บน train เท่านั้น ถ้าใช้ global scaling
- [x] ถ้าใช้ batch-wise normalization ให้ระบุผลต่อ amplitude ของ jump และ spike (see `TODO/EXP1/validation_reports_preprocessing/preprocessing_sensitivity_report.md`)
- [x] ตรวจว่า preprocessing ทุกชนิดทำก่อนหรือหลัง split อย่างถูกลำดับ
- [x] ตรวจว่า no preprocessing step ดึงสถิติจากทั้ง dataset ก่อน split

## 4. Event Coverage and Class Balance
- [x] นับจำนวน anomaly ทั้งหมดต่อ ticker
- [x] นับจำนวน anomaly ต่อ regime
- [x] นับจำนวน anomaly ต่อช่วงเวลา เช่น pre-crisis, crisis, post-crisis
- [x] ถ้า label มีหลาย criterion ให้ report จำนวนของแต่ละ criterion แยกกัน
- [x] รายงานสัดส่วน anomaly เทียบกับ normal
- [x] รายงาน imbalance ratio และ class prevalence
- [x] ตรวจว่ามี event พอสำหรับให้โมเดลเรียนรู้จริงหรือไม่
- [x] ตรวจว่า ticker ใดมี anomaly น้อยผิดปกติจนใช้ประเมินยาก
- [ ] ตรวจว่าแต่ละ event type มีจำนวนพอสำหรับ train และ evaluation หรือไม่

## 5. Window Size and Sampling Logic
- [x] ระบุ default window size และ step size ใน CLI และ config
- [x] ตรวจว่า window size มีผลต่อจำนวน sample ที่สร้างจาก dataset อย่างไร
- [x] ตรวจว่า window size มีผลต่อจำนวน window ที่มี anomaly อย่างไร
- [x] ทดลองหลายค่า window size และบันทึกผลเทียบกัน
- [ ] ตรวจว่า window สั้นเกินไปทำให้ context หายหรือไม่
- [ ] ตรวจว่า window ยาวเกินไปทำให้ anomaly signal diluted หรือไม่
- [x] ตรวจว่า sampling step ทำให้ anomaly หายไปหรือไม่
- [ ] ตรวจว่า sensitivity ต่อ window size เปลี่ยน coverage ของ event แต่ละชนิดอย่างไร

## 6. Feature and Preprocessing Validation
- [x] ตรวจว่า input features ตรงกับ requirement ของงาน: OHLCV และ returns
- [x] ตรวจว่า feature extraction จาก code ตรงกับที่อธิบายใน paper
- [x] ตรวจว่าใช้ price_only, volume_only, หรือ all ในแต่ละ experiment ชัดเจน
- [x] ตรวจผลของ standardization ต่อ return jump/drop (see `TODO/EXP1/validation_reports_preprocessing/preprocessing_sensitivity_report.md`)
- [x] ตรวจผลของ normalization ต่อ volume spike (see `TODO/EXP1/validation_reports_preprocessing/preprocessing_sensitivity_report.md`)
- [ ] ตรวจว่าการเติม missing value ด้วย zero หรือ `nan_to_num` ไม่ทำให้สัญญาณเพี้ยน
- [ ] ตรวจว่า preprocessing ไม่สร้าง artifact ที่ทำให้ label ดูง่ายเกินจริง
- [ ] ตรวจว่า preprocessing ไม่ bias ตลาดบาง ticker หรือบาง regime

## 7. Dataset-Level Statistical Validation
- [x] เปรียบเทียบ distribution ของ anomaly vs normal ด้วย statistical test ที่เหมาะสม เช่น Mann-Whitney U, KS test, หรือ chi-square
- [x] ทดสอบว่า anomaly days มีความต่างจาก normal days จริงหรือไม่สำหรับ return, volume, volatility, และ daily range
- [x] รายงาน p-value ทุกครั้ง
- [x] รายงาน effect size ไม่ใช่แค่ p-value
- [x] ใช้ bootstrap confidence interval สำหรับ metric หรือ statistic หลัก
- [ ] ตรวจว่า hypothesis test ถูกนิยามก่อนรัน ไม่ใช่กำหนดตามผลลัพธ์

## 8. Programming and Pipeline Checks
- [x] ตรวจว่า dataloader ใช้ไฟล์ dataset path ถูกต้อง
- [x] ตรวจว่า loader คืน shape ของ input ถูกต้องตาม model
- [x] ตรวจว่า train/val/test loader ใช้ code path เดียวกันแต่แยก mode ถูกต้อง
- [x] ตรวจว่า batch-wise normalization อยู่ใน `__getitem__` หรือจุดที่ตั้งใจจริง
- [x] ตรวจว่า global scaling ไม่ fit บนข้อมูลทั้งชุดก่อน split
- [x] ตรวจว่า label extraction ใน loader ไม่ทำให้ test label ผิดตำแหน่ง
- [x] ตรวจว่า script train, validate, test รับ argument ตรงกัน
- [x] ตรวจว่า experiment runner ส่ง `dataset`, `data-path`, `features`, `seed`, `exp-name` ถูกต้อง
- [x] ตรวจว่า output path แยกตาม experiment และไม่ทับกัน
- [x] ตรวจว่า checkpoint, score file, และ result file ถูกเขียนครบ
- [x] ตรวจว่า metric computation ใช้ `y_true` และ `y_pred` จากไฟล์ที่ถูกต้อง
- [x] ตรวจว่า no silent failure ใน pipeline stage ใด stage หนึ่ง

## 9. Hypothesis Tests for Model Results
- [ ] ตั้ง hypothesis ให้ชัดก่อนทดสอบ
- [ ] ใช้ paired test across seeds หรือ across tickers สำหรับเปรียบเทียบโมเดล
- [ ] ใช้ bootstrap confidence interval สำหรับ mean performance
- [ ] ห้ามสรุป improvement จาก run เดียว
- [ ] ถ้าผลไม่ significant ให้รายงานเป็น trend ไม่ใช่ conclusion

## 10. Benchmark Fairness
- [ ] ใช้ split เดียวกันกับทุก baseline
- [ ] ใช้ tuning budget ใกล้เคียงกันทุก baseline
- [ ] ใช้ feature set เดียวกันเมื่อเปรียบเทียบแบบ fair
- [ ] ใช้ seed หลายค่าและรายงาน mean, std
- [ ] ไม่ tune บน test
- [ ] ไม่เลือก threshold บน test
- [ ] ไม่เปรียบเทียบโมเดลจากคนละ preprocessing โดยไม่อธิบาย

## 11. Robustness and Generalization
- [ ] ตรวจ robustness across seeds
- [ ] ตรวจ robustness across tickers
- [ ] ตรวจ robustness across market regimes
- [ ] ตรวจว่า performance ไม่พังเมื่อเปลี่ยน ticker universe
- [ ] ตรวจว่า event coverage ยังพอเมื่อแยก regime

## 12. Interpretability Claims
- [ ] ถ้าจะอ้าง interpretability ต้องมี faithfulness test หรือ stability test
- [ ] ถ้าจะอ้าง interpretability ต้องมี evidence ว่า attribution สอดคล้องกับ event จริง
- [ ] ถ้าจะอ้าง interpretability ไม่ควรใช้แค่ case study อย่างเดียว
- [ ] ถ้าจะอ้าง explanation quality ต้องมี quantitative evaluation

## 13. Required Reports Before Using Dataset
- [x] ตารางจำนวน event ต่อ ticker แยกตาม jump, drop, volume spike, volatility shock, regime shift
- [x] ตารางสัดส่วน anomaly ต่อ normal
- [x] ตาราง mapping ระหว่าง event type และ label column ภายใน
- [x] ตารางแยกตาม regime
- [x] ตาราง sensitivity ต่อ window size
- [x] ตาราง sensitivity ต่อ preprocessing แบบต่าง ๆ
- [x] ตาราง statistical test พร้อม p-value และ effect size

## 14. Go / No-Go Decision
- [ ] Go ถ้า semantic label ชัดว่าแต่ละ event type คืออะไร, leakage ปิดแล้ว, event coverage พอ, และ test ทางสถิติครบ
- [ ] No-go ถ้ายังมี global normalization บน full dataset
- [ ] No-go ถ้ายังไม่รู้ว่า event taxonomy ถูกนิยามอย่างไร
- [ ] No-go ถ้ายังไม่มี evidence ว่า dataset ครอบคลุม jump, drop, volume spike, volatility shock, regime shift
- [ ] No-go ถ้ายังไม่มี variance และ hypothesis test สำหรับผล benchmark
