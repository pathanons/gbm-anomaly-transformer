# Research Conclusion 2: ใช้ Attention Map เป็น Explanation Layer ไม่ใช่ Anomaly Score หลัก

วันที่สรุป: 2026-06-20

## ข้อสรุปหลัก

ควรใช้ Student-t next-day NLL เป็น anomaly score หลักต่อไป และใช้ Transformer attention map เป็นชั้นอธิบายผลหลังจากรู้แล้วว่าจุดใดเป็น anomaly

กล่าวคือ:

```text
NLL score บอกว่า จุดนี้แปลกหรือไม่น่าจะเกิด
Attention map บอกว่า ตอนโมเดลประเมินจุดนี้ มันมอง historical context วันไหนบ้าง
```

วิธี framing นี้ดีกว่าการพยายามอ้างว่า association-based score ชนะ NLL เพราะหลักฐานล่าสุดสนับสนุนชัดว่า Student-t NLL เป็น ranking signal ที่ใช้งานได้ ส่วน attention เหมาะกว่าในบทบาท interpretability/diagnostic layer

## Research Framing

ใน paper สามารถวางบทบาทของ self-attention map ได้แบบนี้:

Self-attention map ของแต่ละ time point มองเป็น distribution ของความสัมพันธ์ระหว่าง time point ปัจจุบันกับ time point อื่นใน window ได้ distribution นี้ให้ข้อมูล temporal context เช่น pattern, trend, regime memory, หรือ historical analog ที่โมเดลใช้ประกอบการสร้าง representation

ดังนั้นเราไม่จำเป็นต้องบอกว่า attention เป็น anomaly detector ที่ดีกว่า NLL แต่บอกว่า:

```text
NLL identifies anomalous return realizations.
Attention maps explain which historical time points the Transformer used as context when assigning that anomaly score.
```

## Interpretation Logic

เมื่อเจอวันที่ anomaly จาก NLL:

1. ดูว่า attention กระจุกอยู่ใกล้ endpoint หรือกระจายไปอดีตไกล
2. เทียบ attention pattern ของ anomaly day กับ normal day
3. ดูว่า anomaly day ไปมองวันเก่าที่มี return shock, volatility cluster, หรือ regime-like pattern หรือไม่
4. ถ้ามี prior/association model ให้เทียบ learned attention กับ GBM prior เพื่อดูว่าความสัมพันธ์ที่โมเดลเรียนต่างจาก expected temporal relation ตรงไหน

ตัวอย่างคำอธิบายที่เหมาะ:

```text
NLL flags this endpoint as unlikely under the predicted Student-t return distribution.
The attention map shows that the endpoint representation relies on a narrow recent context rather than the broader historical pattern seen in normal windows.
```

หรือ:

```text
The anomaly score comes from predictive surprise, while the attention map provides temporal context for that surprise.
```

## What Not To Claim Yet

ยังไม่ควร claim ว่า:

- attention map เป็น causal explanation
- attention-based score ชนะ NLL
- association discrepancy เป็น final anomaly score ที่ดีกว่า
- attention pattern พิสูจน์สาเหตุของ anomaly

จนกว่าจะมี faithfulness checks เช่น deletion/insertion, perturbation, seed stability, หรือ comparison กับ normal matched windows

## Practical Diagnostic Plan

สำหรับ follow-up ควรทำ attention diagnostic แบบไม่เปลี่ยน score หลัก:

1. ใช้ NLL + validation-quantile threshold เลือก anomaly endpoints
2. เลือก matched normal endpoints จาก ticker เดียวกันและช่วงเวลาใกล้กัน
3. Export attention maps ของ anomaly และ normal windows
4. Plot learned attention heatmap, endpoint attention profile, และ top-attended historical dates
5. สรุปว่า anomaly attention ต่างจาก normal อย่างไร

Minimal comparison groups:

- true overlap anomalies: NLL flag และ log-return หลุด +/-3 std
- false alarms: NLL flag แต่ log-return ไม่หลุด +/-3 std
- missed anomalies: log-return หลุด +/-3 std แต่ NLL ไม่ flag
- normal controls: neither NLL flag nor +/-3 std

## Current Evidence Link

Conclusion 1 สนับสนุนว่า Student-t next-day NLL เป็น anomaly-ranking signal ที่ใช้ต่อได้ โดยเฉพาะ validation-quantile thresholds แถว `delta=0.02` และ `delta=0.015`

Conclusion 2 นี้เพิ่ม framing ต่อจากนั้น:

```text
Use NLL for detection.
Use attention for interpretation.
```

## Next Actions

1. สร้าง attention visualization สำหรับ NLL-selected anomaly endpoints
2. เพิ่ม matched normal comparison plots
3. เก็บ per-window top-attended historical dates เป็น CSV
4. แยก qualitative interpretability ออกจาก quantitative detector performance
5. ค่อยเพิ่ม faithfulness tests ก่อนใช้คำว่า explanation แบบแข็งแรงใน paper
